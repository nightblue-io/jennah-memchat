// Command memchat is a demo agent: a chatbot that REMEMBERS ACROSS SESSIONS by
// consuming Jennah's public memory APIs exactly the way any external agent would
// — plain HTTP/JSON through the jennah-proxy gateway, authenticated with a
// jennah_sk_ API key. It is deliberately a standalone Go module (its own go.mod,
// not part of the server build) so it models a real outside consumer and keeps
// the Anthropic SDK dependency out of the server tree.
//
// Each turn does query → think → commit:
//
//  1. memory:query — semantic recall of past exchanges (the server embeds the
//     query text via managed embeddings), plus a graph traversal from the
//     stable "user" node to list everything already known about the user.
//  2. Claude answers, and calls the remember_fact tool to persist durable facts
//     as (user)-[relationship]->(value) triples.
//  3. memory:commit — writes the exchange as a vector chunk, any new facts as
//     graph nodes/edges, and a turn record to the execution log, atomically.
//
// Cross-session memory is simply reusing the same agent_instance_id: the id (and
// the set of graph ids already written, since graph writes are insert-only and
// re-committing an id fails the whole commit) is persisted to a small state file.
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
	"google.golang.org/protobuf/types/known/structpb"
)

// verbose makes the memory activity visible on screen: recalled triples and
// snippets before each reply, and the commit receipt after. Set by --verbose.
var verbose bool

// userNode is the stable anchor every fact hangs off, so a traversal from it
// recalls the whole per-user knowledge graph. Created once, then never re-sent
// (graph nodes are insert-only server-side).
const userNode = "user"

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
	b.WriteString("Whenever the user shares a durable fact about themselves — their name, preferences, job, location, goals, or important people/things — call remember_fact to store it (one call per fact). Do not store transient chit-chat.\n\n")

	b.WriteString("# What you already know about the user (knowledge graph)\n")
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

func recallFacts(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: &agentpb.GraphNodeMatch{
				Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(userNode)}},
			},
			Steps: []*agentpb.GraphStep{{
				Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING,
				Node:      &agentpb.GraphNodeMatch{},
			}},
			Limit: 100,
		},
	}, &resp); err != nil {
		return nil, err
	}
	// A start-node + one-hop traversal returns rows keyed n0_id, n0_label,
	// e0_type (relationship), n1_id, n1_label (the value we stored as Label).
	var out []string
	for _, row := range resp.GetGraph().GetRows() {
		m := row.AsMap()
		rel := prettyRel(rowStr(m, "e0_type"))
		val := rowStr(m, "n1_label")
		if val == "" {
			continue
		}
		out = append(out, fmt.Sprintf("user %s %s", rel, val))
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
	for _, f := range facts {
		nid := "n_" + hash(strings.ToLower(f.value))
		eid := "e_" + hash(strings.ToLower(f.rel)+"|"+strings.ToLower(f.value))
		// Dedup within this single commit: a mutation set can't carry two writes
		// for the same key. Idempotency across commits is the server's job.
		if !seenN[nid] {
			nodes = append(nodes, &agentpb.GraphNode{NodeId: nid, Label: f.value})
			seenN[nid] = true
		}
		if !seenE[eid] {
			edges = append(edges, &agentpb.GraphEdge{
				EdgeId: eid, SourceNodeId: userNode, TargetNodeId: nid, RelationshipType: normRel(f.rel),
			})
			seenE[eid] = true
		}
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
	for _, f := range facts {
		vlog("stored fact: user %s %s", prettyRel(normRel(f.rel)), f.value)
	}
	printReceipt(resp)
	return nil
}

// createAgent provisions the agent workspace. region is the optional Jennah home
// region ("" = platform default); it's honored only at creation time because an
// agent instance is pinned to one home region for its lifetime.
func createAgent(ctx context.Context, jc *jennahClient, region string) (string, error) {
	id := randID("agent")
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

type fact struct{ rel, value string }

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
