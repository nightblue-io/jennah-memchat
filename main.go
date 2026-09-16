// Command memchat is a demo agent: a chatbot that REMEMBERS ACROSS SESSIONS by
// consuming Jennah's public memory APIs exactly the way any external agent would
// — plain HTTP/JSON through the jennah-proxy gateway, authenticated with a
// jennah_sk_ API key. It is deliberately a standalone Go module (its own go.mod,
// not part of the server build) so it models a real outside consumer and keeps
// the Anthropic SDK dependency out of the server tree.
//
// Each turn does query → think → commit:
//
//  1. memory:query for semantic recall of past exchanges (the server embeds the
//     query text via managed embeddings), plus memory:inspect to read the whole
//     knowledge graph back as triples. See recallFacts for why the graph side
//     enumerates rather than traverses: edge direction is the model's phrasing
//     choice, and only inspect reports it.
//  2. Claude answers, and calls the remember_fact tool to persist durable facts
//     as (subject)-[relationship]->(object) triples. The subject is the model's
//     to choose, so a fact relating two entities the user merely mentioned is
//     expressible as itself rather than as a spoke off the user.
//  3. memory:commit — writes the exchange as a vector chunk, any new facts as
//     graph nodes/edges, and a turn record to the execution log, atomically. The
//     receipt is then CHECKED, not discarded: it names any chunk whose embedding
//     the model truncated, which is a memory-quality problem no error code
//     reports (see printReceipt).
//
// Cross-session memory is simply reusing the same agent_instance_id, persisted to
// a small state file. Nothing else needs tracking: graph writes are idempotent
// upserts on caller-supplied ids, so re-asserting a fact across sessions converges
// on the same node and edge instead of duplicating or failing.
//
// The chat brain is pluggable (see brain.go): it talks to Claude or Gemini
// depending on which API key is present, or an explicit --provider. Only the LLM
// differs — every memory call is identical, which is the whole point of the demo.
//
// Setup (Jennah key + one chat provider). Keys come from env or flags:
//
//	export JENNAH_API_KEY=jennah_sk_...      # from POST /v1/apikeys
//	export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic key, OR
//	export GEMINI_API_KEY=...                # Google AI Studio key
//	go run .                                 # talks to https://jennah.alphaus.cloud
//
// or pass them explicitly:
//
//	go run . --jennah-api-key jennah_sk_... --anthropic-api-key sk-ant-...
//
// The agent's home region can be chosen at first launch with --region (or
// $JENNAH_REGION); it's applied only when the workspace is created, since an
// agent is pinned to one region for its lifetime. Empty uses the platform default.
// List the available regions with 'jnh agents regions'.
package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"time"

	agentpb "github.com/alphauslabs/jennah-sdk-go/jennah/agent/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// verbose makes the memory activity visible on screen: recalled triples and
// snippets before each reply, and the commit receipt after. Set by --verbose.
var verbose bool

// userNode is the stable anchor the knowledge graph is reachable from, so a
// traversal always has a known place to start. It is the one node with a fixed
// id rather than a content hash; every other entity gets one from nodeID.
// Seeded once at bootstrap and never re-sent: node writes are idempotent
// upserts, so re-sending it would silently overwrite its label.
const userNode = "user"

// demoPrefix namespaces every workspace this demo creates under a "demo." subtree.
// '.' is the agent-selector hierarchy separator server-side, and selector matching
// is segment-anchored, so one role selector "demo.*" reaches every id minted here
// (and nothing else). That keeps a demo run scopable to a throwaway role instead of
// needing blanket agent access. Only interior '.' is legal in an agent id, so the
// prefix must be followed by a real name — never used on its own.
const demoPrefix = "demo."

func main() {
	var (
		endpoint     = flag.String("endpoint", envOr("JENNAH_ENDPOINT", "https://jennah.alphaus.cloud"), "Jennah proxy origin (http/https)")
		statePath    = flag.String("state", "memchat-state.json", "path to the local state file (agent id + committed graph ids)")
		provider     = flag.String("provider", "auto", "chat LLM: auto|gemini|anthropic (auto prefers Anthropic, else Gemini, by which API key is set)")
		region       = flag.String("region", envOr("JENNAH_REGION", ""), "Jennah home region for the agent (e.g. us-central1); empty uses the platform default. Only applied when creating a new agent workspace. List regions with 'jnh agents regions'")
		jennahKey    = flag.String("jennah-api-key", "", "Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY")
		anthropicKey = flag.String("anthropic-api-key", "", "Anthropic API key (sk-ant-...); falls back to $ANTHROPIC_API_KEY")
	)
	flag.BoolVar(&verbose, "verbose", false, "print recalled memory and commit receipts each turn")
	flag.Parse()

	// Fall back to the env vars, but keep them OUT of the flag defaults so --help
	// never prints the actual secret. A flag wins over its env var when both are set.
	*jennahKey = envOr2(*jennahKey, "JENNAH_API_KEY")
	*anthropicKey = envOr2(*anthropicKey, "ANTHROPIC_API_KEY")

	apiKey := *jennahKey
	if apiKey == "" {
		fatal("a Jennah API key is required: pass --jennah-api-key or set JENNAH_API_KEY (a jennah_sk_ key for an approved, entitled enterprise)")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// The one part that varies by provider: the chat brain. Everything below is
	// provider-agnostic; the memory APIs don't care which LLM is thinking.
	br, err := newBrain(ctx, *provider, *anthropicKey)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("chat model: %s\n", br.label())

	jc := &jennahClient{
		endpoint: strings.TrimRight(*endpoint, "/"),
		token:    apiKey,
		hc:       &http.Client{Timeout: 60 * time.Second},
	}

	st, err := loadState(*statePath)
	if err != nil {
		fatal("load state: %v", err)
	}

	// One-time bootstrap: create the agent workspace and seed the stable "user"
	// anchor node, then persist the id. Save only after the seed succeeds, so a
	// crash mid-bootstrap doesn't leave a saved agent without its anchor node.
	if st.AgentID == "" {
		id, err := createAgent(ctx, jc, *region)
		if err != nil {
			fatal("create agent: %v", err)
		}
		if _, err := commit(ctx, jc, id, &agentpb.CommitMemoryRequest{
			AgentInstanceId: id,
			Graph:           &agentpb.GraphWrite{Nodes: []*agentpb.GraphNode{{NodeId: userNode, Label: "User"}}},
		}); err != nil {
			fatal("seed user node: %v", err)
		}
		st.AgentID = id
		save(*statePath, st)
		if *region != "" {
			fmt.Printf("created agent workspace %s (region %s)\n", id, *region)
		} else {
			fmt.Printf("created agent workspace %s (platform default region)\n", id)
		}
	} else {
		fmt.Printf("reusing agent workspace %s (memory carries over)\n", st.AgentID)
	}

	fmt.Println("\nmemchat — a chatbot that remembers across sessions (Ctrl-D or /exit to quit).")
	fmt.Println("Tell it about yourself, quit, run it again, and it'll recall.")

	// Session-local conversation history lives inside the brain; long-term memory
	// lives in Jennah. This loop just wires stdin → recall → think → commit.
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for {
		fmt.Print("\nyou> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		if line == "/exit" || line == "/quit" {
			break
		}

		reply, facts, err := turn(ctx, jc, br, st.AgentID, line)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				break
			}
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}
		fmt.Printf("\nmemo> %s\n", reply)

		if err := commitTurn(ctx, jc, st.AgentID, line, reply, facts); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not persist this turn's memory: %v\n", err)
		}
	}
	if err := in.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}
	fmt.Println("\nbye — your memory is saved in Jennah.")
}

// turn runs one query → think loop: recall from Jennah, ask the brain (letting it
// call remember_fact), and return the reply plus any facts it chose to store. The
// brain is whichever LLM was selected; this function is provider-agnostic.
func turn(ctx context.Context, jc *jennahClient, br brain, agentID, userMsg string) (string, []fact, error) {
	snippets, err := recallSemantic(ctx, jc, agentID, userMsg)
	if err != nil {
		return "", nil, fmt.Errorf("semantic recall: %w", err)
	}
	knownFacts, err := recallFacts(ctx, jc, agentID)
	if err != nil {
		return "", nil, fmt.Errorf("graph recall: %w", err)
	}
	vlog("recalled %d fact(s), %d past snippet(s):", len(knownFacts), len(snippets))
	for _, f := range knownFacts {
		vlog("  · %s", f)
	}
	for _, s := range snippets {
		vlog("  ~ %s", s)
	}
	if !verbose {
		fmt.Printf("  \033[2m[recalled %d fact(s), %d past snippet(s)]\033[0m\n", len(knownFacts), len(snippets))
	}

	return br.chat(ctx, buildSystemPrompt(knownFacts, snippets), userMsg)
}

func buildSystemPrompt(knownFacts, snippets []string) string {
	var b strings.Builder
	b.WriteString("You are Memo, a warm, concise assistant with long-term memory that persists across sessions. ")
	b.WriteString("Personalize using the remembered context below and refer back to it naturally. ")
	b.WriteString("Whenever the user shares a durable fact, call remember_fact to store it as one (subject, relationship, object) triple: their name, preferences, job, location and goals, and also the people, organizations and teams they mention and how those relate to one another. One call per fact, one entity per field: a team with three members is three calls, not one call listing three names. Do not store transient chit-chat.\n\n")

	b.WriteString("# What you already know (knowledge graph)\n")
	if len(knownFacts) == 0 {
		b.WriteString("(nothing yet — this may be your first conversation)\n")
	} else {
		for _, f := range knownFacts {
			b.WriteString("- ")
			b.WriteString(f)
			b.WriteString("\n")
		}
	}
	b.WriteString("\n# Relevant snippets from past conversations\n")
	if len(snippets) == 0 {
		b.WriteString("(none retrieved)\n")
	} else {
		for _, s := range snippets {
			b.WriteString("- ")
			b.WriteString(s)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// ---- Jennah memory API calls (HTTP/JSON through the proxy) ----

func recallSemantic(ctx context.Context, jc *jennahClient, agentID, query string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Semantic:        &agentpb.SemanticQuery{QueryText: query, Limit: 6},
	}, &resp); err != nil {
		return nil, err
	}
	var out []string
	for _, m := range resp.GetSemantic().GetMatches() {
		if c := strings.TrimSpace(m.GetRawContent()); c != "" {
			out = append(out, singleLine(c))
		}
	}
	return out, nil
}

// recallPageLimit is the page size for the graph read-out, and maxRecallPages
// bounds the walk so a runaway workspace can't stall a chat turn.
const (
	recallPageLimit = 200
	maxRecallPages  = 10
)

// recallFacts reads back the whole knowledge graph as readable triples.
//
// This uses memory:inspect rather than a memory:query traversal, and the reason is
// edge DIRECTION. A traversal row projects the edge's id, type and valid-time but
// not its endpoints, so orientation is only known when the step pins a direction:
// an OUTGOING walk from the user anchor reads "user is named Hajime" correctly and
// never sees "Chew is cto of Alphaus", because that edge points AT Alphaus rather
// than away from it. Which way a fact points is the model's phrasing choice, so
// half the graph would silently vanish from the prompt. Pinning INCOMING instead
// just loses the other half, and a path that changes direction mid-walk (user ->
// Hajime -> Alphaus <- Chew) is not expressible as one query at all, since steps
// fix a direction per hop.
//
// Inspect returns edges with source_node_id and target_node_id, so every fact is
// rendered the way it was asserted, in one request instead of a query per depth.
// The trade is that it enumerates the workspace rather than walking from the user,
// which for a personal graph is what "what do you already know" actually means:
// it also recovers anything the model asserted without a path back to the anchor.
func recallFacts(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	labels := map[string]string{}
	var edges []*agentpb.GraphEdge

	var nodeTok, edgeTok string
	for page := 0; page < maxRecallPages; page++ {
		var resp agentpb.InspectMemoryResponse
		if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "inspect"), &agentpb.InspectMemoryRequest{
			AgentInstanceId: agentID,
			Graph: &agentpb.InspectGraph{
				NodeLimit:     recallPageLimit,
				EdgeLimit:     recallPageLimit,
				NodePageToken: nodeTok,
				EdgePageToken: edgeTok,
			},
		}, &resp); err != nil {
			return nil, err
		}
		for _, n := range resp.GetGraph().GetNodes() {
			labels[n.GetNodeId()] = n.GetLabel()
		}
		edges = append(edges, resp.GetGraph().GetEdges()...)

		// The two listings exhaust independently, so keep going while EITHER has
		// more. An empty token means that listing is done, not merely this page.
		nodeTok, edgeTok = resp.GetNextNodeToken(), resp.GetNextEdgeToken()
		if nodeTok == "" && edgeTok == "" {
			break
		}
	}

	// Join each edge to its endpoints' labels. A node id with no label means the
	// node listing was cut short by maxRecallPages while its edges came back;
	// showing the raw id is more useful to the model than dropping the fact.
	label := func(id string) string {
		if l := strings.TrimSpace(labels[id]); l != "" {
			return l
		}
		return id
	}
	var out []string
	seen := map[string]bool{}
	for _, e := range edges {
		line := tripleText(label(e.GetSourceNodeId()), e.GetRelationshipType(), label(e.GetTargetNodeId()))
		if !seen[line] {
			out = append(out, line)
			seen[line] = true
		}
	}
	return out, nil
}

// commitTurn writes the exchange (vector chunk), any facts Claude chose to store
// (graph nodes/edges), and a turn record (log) in one atomic commit. Graph writes
// are idempotent server-side, so re-asserting a fact across turns just converges —
// no client-side id tracking, no skip list. Deterministic content-hashed ids give
// each entity/relationship a stable identity so the graph doesn't fragment.
func commitTurn(ctx context.Context, jc *jennahClient, agentID, userMsg, reply string, facts []fact) error {
	var nodes []*agentpb.GraphNode
	var edges []*agentpb.GraphEdge
	seenN, seenE := map[string]bool{}, map[string]bool{}
	addNode := func(label string) string {
		id := nodeID(label)
		// The user anchor is seeded once at bootstrap and carries its own label;
		// re-writing it here would overwrite "User" with whatever the model typed.
		if id != userNode && !seenN[id] {
			nodes = append(nodes, &agentpb.GraphNode{NodeId: id, Label: strings.TrimSpace(label)})
			seenN[id] = true
		}
		return id
	}
	var stored []string
	for _, f := range facts {
		// Both endpoints get a node regardless of which way the edge ends up
		// pointing, so the flip below only reorients the edge.
		src, dst := addNode(f.subj), addNode(f.obj)
		srcLabel, dstLabel := subjectLabel(f.subj), strings.TrimSpace(f.obj)
		rel := normRel(f.rel)
		if inv, ok := inverseRel[rel]; ok {
			src, dst = dst, src
			srcLabel, dstLabel = dstLabel, srcLabel
			rel = inv
		}
		// Keyed on the canonical ids and the normalized relationship, so the same
		// fact phrased differently ("me"/"user", "has CTO"/"has cto", "Chew is CTO
		// of Alphaus"/"Alphaus has CTO Chew") converges on one edge instead of
		// accumulating near-duplicates.
		eid := "e_" + hash(src+"|"+rel+"|"+dst)
		// Dedup within this single commit: a mutation set can't carry two writes
		// for the same key. Idempotency across commits is the server's job.
		if !seenE[eid] {
			edges = append(edges, &agentpb.GraphEdge{
				EdgeId: eid, SourceNodeId: src, TargetNodeId: dst, RelationshipType: rel,
			})
			seenE[eid] = true
		}
		stored = append(stored, tripleText(srcLabel, rel, dstLabel))
	}

	req := &agentpb.CommitMemoryRequest{
		AgentInstanceId: agentID,
		Log: &agentpb.ExecutionLogStep{
			StepId:         randID("step"),
			ThoughtProcess: "conversation turn",
			ToolUsed:       "memchat",
			ToolInput:      truncate(userMsg, 500),
			ToolOutput:     truncate(reply, 1000),
		},
		Vectors: []*agentpb.VectorChunk{{
			ChunkId:    randID("chunk"),
			RawContent: "User: " + userMsg + "\nAssistant: " + reply,
		}},
	}
	if len(nodes) > 0 || len(edges) > 0 {
		req.Graph = &agentpb.GraphWrite{Nodes: nodes, Edges: edges}
	}

	resp, err := commit(ctx, jc, agentID, req)
	if err != nil {
		return err
	}
	for _, s := range stored {
		vlog("stored fact: %s", s)
	}
	printReceipt(resp)
	return nil
}

// createAgent provisions the agent workspace. region is the optional Jennah home
// region ("" = platform default); it's honored only at creation time because an
// agent instance is pinned to one home region for its lifetime.
func createAgent(ctx context.Context, jc *jennahClient, region string) (string, error) {
	id := demoPrefix + randID("memchat")
	var resp agentpb.CreateAgentResponse
	if _, err := jc.do(ctx, http.MethodPost, "/v1/agents", &agentpb.CreateAgentRequest{
		AgentInstanceId: id,
		AgentName:       "memchat-demo",
		Region:          region,
	}, &resp); err != nil {
		return "", err
	}
	if got := resp.GetAgent().GetAgentInstanceId(); got != "" {
		return got, nil
	}
	return id, nil
}

func commit(ctx context.Context, jc *jennahClient, agentID string, req *agentpb.CommitMemoryRequest) (*agentpb.CommitMemoryResponse, error) {
	var resp agentpb.CommitMemoryResponse
	_, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "commit"), req, &resp)
	return &resp, err
}

func printReceipt(r *agentpb.CommitMemoryResponse) {
	ts := "?"
	if t := r.GetCommitTimestamp(); t != nil {
		ts = t.AsTime().UTC().Format(time.RFC3339)
	}
	vlog("committed: log=%d vec=%d nodes=%d edges=%d @ %s",
		r.GetExecutionLogRows(), r.GetVectorRows(), r.GetGraphNodeRows(), r.GetGraphEdgeRows(), ts)

	// The commit SUCCEEDED, but Jennah is reporting that the embedding model
	// truncated content past its input limit (~2048 tokens): the chunk's text is
	// stored in full, while its vector covers only the beginning. Semantic recall
	// can no longer find this turn by anything said in the part that was cut.
	//
	// Printed unconditionally, NOT under vlog. This is a silent memory-quality
	// problem, and the reason the receipt reports it at all is so a caller doesn't
	// have to be in verbose mode to find out. Any agent consuming these APIs should
	// check this much.
	//
	// We deliberately do NOT set reject_on_truncation on the request: for a chatbot,
	// losing the turn is worse than remembering most of it. A retrieval-critical
	// ingest pipeline should make the opposite choice and split the content instead.
	if ids := r.GetTruncatedChunkIds(); len(ids) > 0 {
		fmt.Printf("  \033[33m[memory] that message was too long to embed in full (%s). "+
			"It is stored, but recall may miss the end of it.\033[0m\n", strings.Join(ids, ", "))
	}
}

// ---- HTTP client (protojson over the gateway, Bearer auth) ----

type jennahClient struct {
	endpoint string
	token    string
	hc       *http.Client
}

func (c *jennahClient) do(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	var body io.Reader
	if in != nil {
		b, err := protojson.Marshal(in)
		if err != nil {
			return 0, err
		}
		body = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return 0, err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 300 {
		return resp.StatusCode, fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, gatewayMessage(raw))
	}
	if out != nil {
		if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
		}
	}
	return resp.StatusCode, nil
}

func agentPath(id string) string        { return "/v1/agents/" + url.PathEscape(id) }
func memoryPath(id, verb string) string { return agentPath(id) + "/memory:" + verb }

func gatewayMessage(body []byte) string {
	var gw struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &gw); err == nil && strings.TrimSpace(gw.Message) != "" {
		return strings.TrimSpace(gw.Message)
	}
	return strings.TrimSpace(string(body))
}

// ---- local state ----

// state persists only the agent id — that's what makes memory carry across
// sessions. No graph id tracking is needed: graph writes are idempotent server
// side, so re-asserting a node/edge across turns just converges.
type state struct {
	AgentID string `json:"agent_id"`
}

// fact is one (subject)-[relationship]->(object) triple the model chose to store.
// An empty subj means the user themselves, the one entity with a fixed node id.
type fact struct{ subj, rel, obj string }

func loadState(path string) (*state, error) {
	st := &state{}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, st); err != nil {
		return nil, err
	}
	return st, nil
}

func save(path string, st *state) {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: marshal state: %v\n", err)
		return
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "warning: write state %s: %v\n", path, err)
	}
}

// ---- helpers ----

func hash(s string) string {
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func randID(prefix string) string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

// nodeID maps an entity label to its stable node id, content-hashed so the same
// entity named twice converges on one node instead of fragmenting.
//
// The user is the exception: they get the fixed "user" id, because a traversal
// needs one known place to start. Anything the model offers as a synonym for the
// user (an omitted subject, or "me", "I", "the user") folds onto that anchor, so
// a phrasing choice can't strand facts on a rival node the traversal never visits.
func nodeID(label string) string {
	if isUserRef(label) {
		return userNode
	}
	return "n_" + hash(strings.ToLower(strings.TrimSpace(label)))
}

func isUserRef(label string) bool {
	switch strings.ToLower(strings.TrimSpace(label)) {
	case "", "user", "the user", "me", "i", "myself":
		return true
	}
	return false
}

// subjectLabel renders a fact's subject for display, naming the user when the
// model left the subject implicit.
func subjectLabel(subj string) string {
	if isUserRef(subj) {
		return "user"
	}
	return strings.TrimSpace(subj)
}

// tripleText renders one triple as the readable line the prompt and --verbose show.
func tripleText(subj, rel, obj string) string {
	return fmt.Sprintf("%s %s %s", subj, prettyRel(rel), obj)
}

// inverseRel canonicalizes edge DIRECTION. A key is a relationship the model emits
// pointing the "wrong" way; its value is the canonical relationship to store once
// source and target are swapped. So "Chew is CTO of Alphaus" and "Alphaus has CTO
// Chew" both land as Alphaus -[HAS_CTO]-> Chew, one edge with one id.
//
// The convention is container first: the organization, department or owner is the
// source, and the person or part it contains is the target. That direction is the
// one the model already picks most often, and it makes an outward walk from an org
// node enumerate its people.
//
// This is NOT an ontology. The relationship vocabulary stays open, and a predicate
// absent from this table is stored exactly as the model phrased it. The table only
// lists pairs actually observed being emitted BOTH ways across repeated runs of the
// same conversation; predicates seen in only one direction (REPORTS_TO always runs
// person -> manager, IS_NAMED always user -> name) are left alone rather than given
// an invented opposite. Synonyms that agree on direction (OWNS vs OWNS_DEPARTMENT)
// are a separate axis this deliberately does not touch.
var inverseRel = map[string]string{
	"IS_CEO_OF":          "HAS_CEO",
	"IS_CTO_OF":          "HAS_CTO",
	"IS_COO_OF":          "HAS_COO",
	"IS_CFO_OF":          "HAS_CFO",
	"IS_MEMBER_OF":       "HAS_MEMBER",
	"IS_PART_OF":         "HAS_PART",
	"IS_A_DEPARTMENT_OF": "HAS_DEPARTMENT",
	"IS_OWNED_BY":        "OWNS",
	"BELONGS_TO":         "HAS_MEMBER",
}

// normRel turns a verb phrase into an edge RelationshipType, e.g. "is named" -> "IS_NAMED".
func normRel(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(strings.TrimSpace(s)) {
		switch {
		case r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "RELATED_TO"
	}
	return out
}

// prettyRel renders a stored RelationshipType back as a readable phrase.
func prettyRel(s string) string {
	if s == "" {
		return "→"
	}
	return strings.ToLower(strings.ReplaceAll(s, "_", " "))
}

func rowStr(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func singleLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envOr2 returns val when it's non-empty (a flag was passed), otherwise the value
// of env var key. Unlike envOr, the caller-supplied value wins — so an explicit
// flag overrides the env var, and the env var is never a flag default (keeping
// secrets out of --help).
func envOr2(val, key string) string {
	if strings.TrimSpace(val) != "" {
		return val
	}
	return os.Getenv(key)
}

// vlog prints a dim diagnostic line to stdout, only when --verbose is set.
func vlog(format string, args ...any) {
	if verbose {
		fmt.Printf("  \033[2m%s\033[0m\n", fmt.Sprintf(format, args...))
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "memchat: "+format+"\n", args...)
	os.Exit(1)
}
