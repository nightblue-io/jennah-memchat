// Command memchat is a demo agent: a chatbot that REMEMBERS ACROSS SESSIONS by
// consuming Jennah's public memory APIs exactly the way any external agent would:
// plain HTTP/JSON through the jennah-proxy gateway, authenticated with a
// jennah_sk_ API key. It is deliberately a standalone Go module (its own go.mod,
// not part of the server build) so it models a real outside consumer and keeps
// the Anthropic SDK dependency out of the server tree.
//
// Each turn does query → think → form:
//
//  1. memory:query for semantic recall of past exchanges (the server embeds the
//     query text via managed embeddings), plus memory:inspect to read the whole
//     knowledge graph back as triples. See recallFacts for why the graph side
//     enumerates rather than traverses: edge direction is the model's phrasing
//     choice, and only inspect reports it.
//  2. The chat model answers the user, and that is ALL it is asked for. It is
//     given no memory tool and no instruction about what to remember.
//  3. memory:form: the platform extracts candidate memories from the turns,
//     recalls what this workspace already holds, reconciles the two, and commits
//     the result as one atomic write. The receipt says what it decided about every
//     candidate, and what it RETIRED to make room for a correction, which is the
//     part no commit receipt can report (see printFormationReceipt).
//
// --authored is the other arm: steps 2 and 3 go back to the way this demo worked
// before formation existed, and the way any client works that owns its own memory
// extraction. The model is handed a remember_fact tool, and the demo turns the
// triples it emits into graph nodes and edges itself (its own ids, its own
// direction convention, its own relationship normalizer), then writes them with
// memory:commit. Both arms are real and both write into the same workspace, since
// formed memory and authored memory are the same substrate rows either way. Run
// one, then the other, and the diff is the cognition layer.
//
// Cross-session memory is simply reusing the same agent_instance_id, persisted to
// a small state file. Nothing else needs tracking: graph writes are idempotent
// upserts on caller-supplied ids, so re-asserting a fact across sessions converges
// on the same node and edge instead of duplicating or failing.
//
// The chat brain is pluggable (see brain.go): it talks to Claude or Gemini
// depending on which API key is present, or an explicit --provider. Only the LLM
// differs: every memory call is identical, which is the whole point of the demo.
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
// snippets before each reply, and the formation receipt after. Set by --verbose.
var verbose bool

// authored selects the arm that extracts memory in THIS process (the remember_fact
// tool plus a hand-built memory:commit) instead of the default, which hands the
// turns to memory:form and lets the platform decide what is worth remembering.
// Set by --authored. It is off by default because the decision of what to remember
// is the thing Jennah is for; the arm is kept so the difference is runnable.
var authored bool

// userNode is the AUTHORED ARM's anchor: the stable node its graph is reachable
// from, so a traversal always has a known place to start. It is the one node with
// a fixed id rather than a content hash; every other entity gets one from nodeID.
//
// It is a CLIENT CONVENTION, not a platform concept, which is exactly why the
// default arm neither seeds nor uses it. Formation extracts relationships between
// entities the conversation NAMES, so "my name is Chew" becomes facts about an
// entity called Chew and nothing links them to the person typing. The authored arm
// only gets away with anchoring because it owns the tool schema and can rule that
// an omitted subject means the user.
const userNode = "user"

// demoPrefix namespaces every workspace this demo creates under a "demo." subtree.
// '.' is the agent-selector hierarchy separator server-side, and selector matching
// is segment-anchored, so one role selector "demo.*" reaches every id minted here
// (and nothing else). That keeps a demo run scopable to a throwaway role instead of
// needing blanket agent access. Only interior '.' is legal in an agent id, so the
// prefix must be followed by a real name, never used on its own.
const demoPrefix = "demo."

func main() {
	var (
		endpoint     = flag.String("endpoint", envOr("JENNAH_ENDPOINT", "https://jennah.alphaus.cloud"), "Jennah proxy origin (http/https)")
		statePath    = flag.String("state", "memchat-state.json", "path to the local state file (agent id + committed graph ids)")
		agentID      = flag.String("agent", "", "use this EXISTING agent workspace instead of the one in the state file, for a workspace provisioned out of band (e.g. one with a vocabulary declared on it). Never creates and never writes the state file; an id that is not there is an error naming it")
		provider     = flag.String("provider", "auto", "chat LLM: auto|gemini|anthropic (auto prefers Anthropic, else Gemini, by which API key is set)")
		region       = flag.String("region", envOr("JENNAH_REGION", ""), "Jennah home region for the agent (e.g. us-central1); empty uses the platform default. Only applied when creating a new agent workspace. List regions with 'jnh agents regions'")
		jennahKey    = flag.String("jennah-api-key", "", "Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY")
		anthropicKey = flag.String("anthropic-api-key", "", "Anthropic API key (sk-ant-...); falls back to $ANTHROPIC_API_KEY")
	)
	flag.BoolVar(&verbose, "verbose", false, "print recalled memory and formation receipts each turn")
	flag.BoolVar(&authored, "authored", false, "extract and author the memory writes in this client (remember_fact + memory:commit) instead of letting memory:form do it. Writes into the same workspace as the default arm")
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
	// provider-agnostic; the memory APIs don't care which LLM is thinking. Only the
	// authored arm gives it a memory tool.
	br, err := newBrain(ctx, *provider, *anthropicKey, authored)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("chat model: %s\n", br.label())
	if authored {
		fmt.Println("memory: authored in this client (remember_fact + memory:commit)")
	} else {
		fmt.Println("memory: formed by Jennah (memory:form)")
	}

	jc := &jennahClient{
		endpoint: strings.TrimRight(*endpoint, "/"),
		token:    apiKey,
		hc:       &http.Client{Timeout: 60 * time.Second},
		formHC:   &http.Client{Timeout: formTimeout},
	}

	// --agent names a workspace someone else provisioned, which is the ordinary way
	// to run against one that has a VOCABULARY declared on it: declaring is
	// management-class, so it happens out of band and the id has to come in from
	// outside (see vocabularySummary).
	//
	// It deliberately does NOT create, and does not touch the state file. Creating
	// would mean a mistyped id silently mints a second, empty workspace and the
	// demo then reports remembering nothing, which reads as a platform fault. And
	// writing the id to the state file would leave a later flagless run pointed at
	// the operator's workspace, so the flag would quietly become permanent.
	st := &state{}
	switch {
	case strings.TrimSpace(*agentID) != "":
		st.AgentID = strings.TrimSpace(*agentID)
		if err := requireAgent(ctx, jc, st.AgentID); err != nil {
			fatal("%v", err)
		}
		fmt.Printf("using agent workspace %s (--agent; the state file is untouched)\n", st.AgentID)

	default:
		if st, err = loadState(*statePath); err != nil {
			fatal("load state: %v", err)
		}
		if st.AgentID == "" {
			// One-time bootstrap: create the workspace and persist the id.
			id, err := createAgent(ctx, jc, *region)
			if err != nil {
				fatal("create agent: %v", err)
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
	}

	// The "user" anchor belongs to the AUTHORED ARM ONLY, and seeding it in the
	// default arm was actively misleading. Formation names its own entities from
	// what the conversation says, and it has no concept of the caller, so nothing
	// it writes ever touches this node: a workspace that has only ever been formed
	// ends up holding an orphan node labelled "User" next to the real entities,
	// which reads as a broken graph when it is just an unused convention.
	//
	// Seeded on every authored start rather than once at bootstrap, because the arm
	// can change between runs: a workspace created by the default arm has no anchor,
	// and an authored commit naming an absent node is rejected. The write is an
	// idempotent upsert of a fixed label, so re-sending it costs one row and cannot
	// drift.
	if authored {
		if _, err := commit(ctx, jc, st.AgentID, &agentpb.CommitMemoryRequest{
			AgentInstanceId: st.AgentID,
			Graph:           &agentpb.GraphWrite{Nodes: []*agentpb.GraphNode{{NodeId: userNode, Label: "User"}}},
		}); err != nil {
			fatal("seed user node: %v", err)
		}
	} else {
		fmt.Printf("vocabulary: %s\n", vocabularySummary(ctx, jc, st.AgentID))
	}

	fmt.Println("\nmemchat: a chatbot that remembers across sessions (Ctrl-D or /exit to quit).")
	fmt.Println("Tell it about yourself, quit, run it again, and it'll recall.")

	// Session-local conversation history lives inside the brain; long-term memory
	// lives in Jennah. This loop just wires stdin → recall → think → form.
	//
	// transcript is the formation arm's own copy of the recent turns, in the wire
	// type memory:form takes. The brain's history can't serve: it's in whichever
	// vendor SDK's message type answered. sessionID plus the turn ordinal is what
	// names each formation, see formTurn.
	var transcript []*agentpb.ConversationTurn
	sessionID, turnNo := randID("sess"), 0

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

		// The reply is on screen BEFORE memory is written, in both arms but for
		// different reasons. A commit is milliseconds; a formation runs model
		// inference and a retrieval before it writes, so it is a seconds-class
		// call by contract, and making the user wait on it to read an answer the
		// model already produced would be a self-inflicted latency.
		turnNo++
		transcript = append(transcript,
			&agentpb.ConversationTurn{Role: agentpb.TurnRole_TURN_ROLE_USER, Content: line},
			&agentpb.ConversationTurn{Role: agentpb.TurnRole_TURN_ROLE_ASSISTANT, Content: reply},
		)
		if authored {
			if err := commitTurn(ctx, jc, st.AgentID, line, reply, facts); err != nil {
				fmt.Fprintf(os.Stderr, "warning: could not persist this turn's memory: %v\n", err)
			}
			continue
		}
		if err := formTurn(ctx, jc, st.AgentID, window(transcript), formationKey(sessionID, turnNo)); err != nil {
			if errors.Is(err, context.Canceled) {
				break
			}
			fmt.Fprintf(os.Stderr, "warning: could not form this turn's memory: %v\n", err)
		}
	}
	if err := in.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}
	fmt.Println("\nbye. Your memory is saved in Jennah.")
}

// turn runs one query → think loop: recall from Jennah, ask the brain (letting it
// call remember_fact), and return the reply plus any facts it chose to store. The
// brain is whichever LLM was selected; this function is provider-agnostic.
func turn(ctx context.Context, jc *jennahClient, br brain, agentID, userMsg string) (string, []fact, error) {
	snippets, err := recallSemantic(ctx, jc, agentID, userMsg)
	if err != nil {
		return "", nil, fmt.Errorf("semantic recall: %w", err)
	}
	knownFacts, retired, err := recallFacts(ctx, jc, agentID)
	if err != nil {
		return "", nil, fmt.Errorf("graph recall: %w", err)
	}
	vlog("recalled %d fact(s)%s, %d past snippet(s):", len(knownFacts), retiredNote(retired), len(snippets))
	for _, f := range knownFacts {
		vlog("  · %s", f)
	}
	for _, s := range snippets {
		vlog("  ~ %s%s", s.text, s.prov)
	}
	if !verbose {
		fmt.Printf("  \033[2m[recalled %d fact(s)%s, %d past snippet(s)]\033[0m\n",
			len(knownFacts), retiredNote(retired), len(snippets))
	}

	return br.chat(ctx, buildSystemPrompt(knownFacts, snippets), userMsg)
}

// buildSystemPrompt assembles the turn's system prompt from what Jennah recalled.
//
// The instruction about STORING memory appears only in the authored arm, and its
// absence by default is not a simplification: a prompt that told the model what to
// remember while the platform was independently deciding the same thing would be
// two extractors with one workspace, disagreeing at the caller's expense. The
// default prompt says what the model is for (answering) and nothing about memory.
func buildSystemPrompt(knownFacts []string, snippets []snippet) string {
	var b strings.Builder
	b.WriteString("You are Memo, a warm, concise assistant with long-term memory that persists across sessions. ")
	b.WriteString("Personalize using the remembered context below and refer back to it naturally. ")
	if authored {
		b.WriteString("Whenever the user shares a durable fact, call remember_fact to store it as one (subject, relationship, object) triple: their name, preferences, job, location and goals, and also the people, organizations and teams they mention and how those relate to one another. One call per fact, one entity per field: a team with three members is three calls, not one call listing three names. Do not store transient chit-chat.")
	}
	b.WriteString("\n\n")

	b.WriteString("# What you already know (knowledge graph)\n")
	if len(knownFacts) == 0 {
		b.WriteString("(nothing yet, this may be your first conversation)\n")
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
			b.WriteString(s.text)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// ---- Jennah memory API calls (HTTP/JSON through the proxy) ----

func recallSemantic(ctx context.Context, jc *jennahClient, agentID, query string) ([]snippet, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Semantic:        &agentpb.SemanticQuery{QueryText: query, Limit: 6},
	}, &resp); err != nil {
		return nil, err
	}
	var out []snippet
	for _, m := range resp.GetSemantic().GetMatches() {
		if c := strings.TrimSpace(m.GetRawContent()); c != "" {
			out = append(out, snippet{text: singleLine(c), prov: provenance(m)})
		}
	}
	return out, nil
}

// snippet is one recalled chunk: the text that goes into the prompt, and separately
// where it came from. They are kept apart on purpose. Provenance is for the person
// watching --verbose, and splicing it into the prompt would put ids in front of the
// model that it has no use for and may well repeat back.
type snippet struct{ text, prov string }

// provenance renders where a recalled chunk came from, when the chunk was formed
// rather than authored.
//
// Formation stamps the memory it writes with the formation that produced it and the
// turns within it, under reserved jennah.* metadata keys that a caller cannot set
// itself. A chunk this demo authored carries none of them, so this returns "" and
// the authored arm's recall looks exactly as it did before. Reading it back is the
// audit trail working: a formed memory can always be traced to the conversation it
// was drawn from.
func provenance(m *agentpb.SemanticMatch) string {
	md := m.GetMetadata()
	step := md["jennah.source_step"]
	if step == "" {
		return ""
	}
	if turns := md["jennah.source_turns"]; turns != "" {
		return fmt.Sprintf("  [formed by %s, turn(s) %s]", step, turns)
	}
	return fmt.Sprintf("  [formed by %s]", step)
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
//
// RETIRED EDGES ARE FILTERED HERE, because inspect enumerates: it hands back
// superseded assertions alongside current ones, where semantic recall does not.
// A query answers "what is true" and an inspector answers "what is stored". Left
// in, a correction would leave the prompt holding both "lives in Osaka" and
// "lives in Tokyo" as current facts, which is the confusion supersession exists
// to prevent. The retired edge is still stored and still readable, which is what
// makes it history rather than a deletion; this function just declines to present
// it as current, and returns the count so the caller can say so out loud rather
// than dropping facts silently.
//
// KNOWN GAP, and the reason this looks like it is not doing anything: the inspect
// EDGE listing does not currently project valid_at/invalid_at (the chunk listing
// does, so the retired chunk behind the same correction is filtered, and the
// snippets half of the prompt is clean). Until the edge listing carries them, a
// superseded relationship still reaches the prompt and this filter reports zero.
// The code is written against the contract rather than against that gap, so it
// starts working the moment the projection lands, and the count is the thing that
// will show it.
func recallFacts(ctx context.Context, jc *jennahClient, agentID string) ([]string, int, error) {
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
			return nil, 0, err
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
	var retired int
	seen := map[string]bool{}
	for _, e := range edges {
		if isRetired(e, time.Now()) {
			retired++
			continue
		}
		line := tripleText(label(e.GetSourceNodeId()), e.GetRelationshipType(), label(e.GetTargetNodeId()))
		if !seen[line] {
			out = append(out, line)
			seen[line] = true
		}
	}
	return out, retired, nil
}

// isRetired reports whether an edge's valid-time window has closed as of now.
//
// An unset invalid_at means "still current" (validity is null-means-current), and
// a future one means the assertion is still live for the moment, so only a bound
// at or before now retires an edge.
func isRetired(e *agentpb.GraphEdge, now time.Time) bool {
	iv := e.GetInvalidAt()
	return iv != nil && !iv.AsTime().After(now)
}

// ---- memory:form (the default arm) ----

// formWindow is how many recent turns each formation submits: the current exchange
// plus the two before it.
//
// Not just the current exchange, and the reason is the order of formation's stages.
// Extraction runs BEFORE recall, so nothing the workspace already holds can resolve
// "she", "there" or "the second one": only the submitted turns can, and a lone
// exchange would hand the extractor a pronoun with no antecedent. The cost of the
// overlap is that already-formed content gets re-extracted, which reconciliation
// then reports as KNOWN instead of writing twice. That is a token bill, not a
// correctness problem, and it is why the number is small.
const formWindow = 6

// window returns the last formWindow turns, the slice each formation submits.
func window(turns []*agentpb.ConversationTurn) []*agentpb.ConversationTurn {
	if len(turns) <= formWindow {
		return turns
	}
	return turns[len(turns)-formWindow:]
}

// formationKey names one formation so that resending it is safe.
//
// It matters more here than an idempotency key usually does. Extraction is
// nondeterministic, so a blind resend does not repeat the first attempt: it forms a
// second, different set of memory alongside the first. Under this key a resend
// replays the original receipt and extracts nothing.
//
// The identity of a formation in this demo is (this workspace, this session, this
// turn ordinal). The session id is needed because the turn ordinal alone would make
// the first turn of every session the same formation, and the ordinal is needed
// because hashing the text would make a user who says "thanks" twice lose the
// second one to a replay.
func formationKey(sessionID string, turnNo int) string {
	return fmt.Sprintf("frm_%s_%d", sessionID, turnNo)
}

// formTurn hands the recent turns to memory:form and reports what came back.
//
// This is the whole write path of the default arm. There is no extraction here, no
// node ids, no direction convention and no normalizer, because the platform runs
// all four stages (extract, recall, reconcile, commit) behind this one call. What
// the demo is left holding is the part that is genuinely its own: the transcript.
func formTurn(ctx context.Context, jc *jennahClient, agentID string, turns []*agentpb.ConversationTurn, key string) error {
	// A formation is a seconds-class call by contract (model inference plus a
	// retrieval before it writes), so say that it is running rather than leaving a
	// silent pause after the reply.
	vlog("forming memory from %d turn(s), key %s ...", len(turns), key)
	if !verbose {
		fmt.Print("  \033[2m[forming memory ...]\033[0m\n")
	}

	var resp agentpb.FormMemoryResponse
	if _, err := jc.doForm(ctx, http.MethodPost, memoryPath(agentID, "form"), &agentpb.FormMemoryRequest{
		ScopeId:      agentID,
		Turns:        turns,
		FormationKey: key,
		// ObservedAt is deliberately left unset, which means "the instant this
		// formation is received". For a live conversation that IS when it happened.
		// The field is for the other case: a backfill or an import of old
		// transcripts, where valid time is not ingest time, and where getting it
		// wrong would date every fact to the day of the migration.
	}, &resp); err != nil {
		return err
	}
	printFormationReceipt(&resp)
	return nil
}

// vocabularySummary reports the vocabulary formation will classify this
// workspace's memory against, for the startup banner.
//
// READ ONLY, AND NOT BECAUSE IT WAS SIMPLER. Declaring a vocabulary is
// management-class, so no API key can do it and this demo therefore cannot: an
// operator declares it out of band (see vocabulary.yaml), and the agent lives with
// what resolves. That split is worth showing rather than hiding, because it is the
// shape of the feature: the vocabulary is an enterprise's decision about how its
// memory is typed, not a knob each agent sets for itself.
//
// Even the READ needs `agent.vocabulary:read`, which is data-plane but is NOT in
// the member default bundle, so a key minted without it asks in vain. That is
// reported as the ordinary outcome it is, not as a failure: the vocabulary is not
// this demo's to manage, and a chatbot that refused to start because it could not
// read one would be making the wrong thing essential.
func vocabularySummary(ctx context.Context, jc *jennahClient, agentID string) string {
	var resp agentpb.GetMemoryVocabularyResponse
	code, err := jc.do(ctx, http.MethodGet,
		"/v1/memory/vocabulary?scope_id="+url.QueryEscape(agentID), nil, &resp)
	switch {
	case code == http.StatusForbidden:
		return "not readable with this key (needs the agent.vocabulary:read scope); " +
			"formation still classifies against whatever is declared"
	case err != nil:
		return fmt.Sprintf("could not be read (%v); formation still classifies against whatever is declared", err)
	}
	// RESOLVED, not the declaration: what a formation actually classifies against
	// is this scope's own declaration if it has one and the enterprise default
	// otherwise, and only the resolved view answers that in one field.
	v := resp.GetResolved()
	classes, relations := len(v.GetEntityClasses()), len(v.GetRelationTypes())
	if classes == 0 && relations == 0 {
		return "none declared, so entities are extracted untyped " +
			"(declare one with: jnh vocabulary declare --scope " + agentID + " --from-file vocabulary.yaml)"
	}
	return fmt.Sprintf("%d entity class(es), %d relation type(s) in effect", classes, relations)
}

// printFormationReceipt renders what the formation decided.
//
// A formation receipt is not a commit receipt with extra fields. A commit reports
// what it wrote because the caller already knew what it asked for; a formation
// reports DECISIONS, because the caller asked for nothing in particular and the
// platform chose. So every candidate is shown with what became of it, and the
// three things a caller cannot infer from row counts are printed whether or not
// --verbose is set: what was retired, what was dropped, and what was flattened.
func printFormationReceipt(r *agentpb.FormMemoryResponse) {
	if verbose {
		for _, c := range r.GetCandidates() {
			vlog("%-8s %s%s", decisionWord(c.GetDecision()), candidateText(c), decisionNote(c))
		}
		ts := "nothing committed"
		if t := r.GetCommitTimestamp(); t != nil {
			ts = t.AsTime().UTC().Format(time.RFC3339)
		}
		// The replacement a REVISED candidate writes is counted as a supersession,
		// NOT in vector_rows or graph_edge_rows, because closing one window and
		// opening another is not the same act as inserting a new assertion. So a
		// turn that only corrected things legitimately reports zero rows, and
		// printing the two counts side by side is the difference between "nothing
		// was stored" and "what was stored replaced something".
		vlog("formed: log=%d vec=%d(+%d superseded) nodes=%d edges=%d(+%d superseded) @ %s",
			r.GetExecutionLogRows(), r.GetVectorRows(), r.GetChunkSupersessions(),
			r.GetGraphNodeRows(), r.GetGraphEdgeRows(), r.GetEdgeSupersessions(), ts)
	} else {
		fmt.Printf("  \033[2m[formed: %s]\033[0m\n", decisionSummary(r))
	}

	// SUPERSESSION. The formation found a candidate that contradicts something the
	// workspace already held, and retired the old assertion in favour of the new
	// one. Printed unconditionally because it is the one outcome that changes what
	// the workspace says it knows, and because "retired" is not "deleted": the old
	// assertion keeps its validity window and stays readable as history.
	if n := r.GetEdgeSupersessions() + r.GetChunkSupersessions(); n > 0 {
		fmt.Printf("  \033[36m[memory] %d earlier assertion(s) retired by a correction in this turn "+
			"(superseded, not overwritten - the previous value stays readable as history)\033[0m\n", n)
	}

	// The cap was hit, so the formation considered fewer candidates than it found.
	// Same class of problem as an embedding truncation in the authored arm: the call
	// succeeded and something was silently left out, and a caller that never looks
	// at the receipt never learns it.
	if d := r.GetCandidatesDropped(); d > 0 {
		fmt.Printf("  \033[33m[memory] %d candidate(s) past the per-formation cap of %d were dropped, "+
			"so not everything in that exchange was considered.\033[0m\n", d, r.GetCandidateCap())
	}

	// Structure that could not survive as individual assertions was summarized
	// instead, and the receipt discloses it rather than letting the caller believe
	// each item was stored on its own terms.
	for _, s := range r.GetSummarizedStructures() {
		fmt.Printf("  \033[33m[memory] turn %d: %s (%d item(s) summarized rather than stored individually)\033[0m\n",
			s.GetTurnIndex(), s.GetDescription(), s.GetSummarizedCount())
	}

	// Something in the turns was masked by the platform's egress scrubber before the
	// extraction model saw it, so the memory formed from that turn was formed from
	// the masked text. Only ever printed when a value actually was masked.
	for _, rd := range r.GetRedactions() {
		fmt.Printf("  \033[33m[memory] turn %d: %d value(s) masked before the extraction model saw them\033[0m\n",
			rd.GetTurnIndex(), rd.GetMaskedCount())
	}
}

// candidateText renders a candidate the way it will be remembered: a relationship
// as the triple it becomes, anything else as the text that gets stored.
func candidateText(c *agentpb.FormedCandidate) string {
	if c.GetKind() == agentpb.CandidateKind_CANDIDATE_KIND_RELATIONSHIP && c.GetSourceEntity() != "" {
		return tripleText(c.GetSourceEntity(), c.GetRelationshipType(), c.GetTargetEntity())
	}
	return singleLine(c.GetText())
}

// decisionWord is the short label for a decision, e.g. MEMORY_DECISION_NEW -> "new".
func decisionWord(d agentpb.MemoryDecision) string {
	if d == agentpb.MemoryDecision_MEMORY_DECISION_UNSPECIFIED {
		return "?"
	}
	return strings.ToLower(strings.TrimPrefix(d.String(), "MEMORY_DECISION_"))
}

// decisionNote adds the part of a decision that is only meaningful with its
// subject: what a revision replaced, what a known candidate matched, and why a
// rejected one was refused. A rejection reason in particular is worth surfacing:
// it is how a caller learns that a conversation recounting history was not allowed
// to retire the current fact.
func decisionNote(c *agentpb.FormedCandidate) string {
	switch c.GetDecision() {
	case agentpb.MemoryDecision_MEMORY_DECISION_REVISED:
		if id := c.GetMatchedId(); id != "" {
			return "  (retired " + id + ")"
		}
	case agentpb.MemoryDecision_MEMORY_DECISION_KNOWN:
		if id := c.GetMatchedId(); id != "" {
			return "  (matches " + id + ")"
		}
	case agentpb.MemoryDecision_MEMORY_DECISION_REJECTED:
		if why := strings.TrimSpace(c.GetRejectionReason()); why != "" {
			return "  (" + singleLine(why) + ")"
		}
	}
	return ""
}

// decisionSummary counts the decisions for the one-line non-verbose report. Zero
// candidates is a legitimate outcome, not a failure: a turn can contain nothing
// worth remembering, and saying so is more useful than printing "formed: ".
func decisionSummary(r *agentpb.FormMemoryResponse) string {
	if len(r.GetCandidates()) == 0 {
		return "nothing worth remembering in that exchange"
	}
	counts := map[agentpb.MemoryDecision]int{}
	for _, c := range r.GetCandidates() {
		counts[c.GetDecision()]++
	}
	var parts []string
	for _, d := range []agentpb.MemoryDecision{
		agentpb.MemoryDecision_MEMORY_DECISION_NEW,
		agentpb.MemoryDecision_MEMORY_DECISION_REVISED,
		agentpb.MemoryDecision_MEMORY_DECISION_KNOWN,
		agentpb.MemoryDecision_MEMORY_DECISION_REJECTED,
	} {
		if n := counts[d]; n > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", n, decisionWord(d)))
		}
	}
	return strings.Join(parts, ", ")
}

// ---- memory:commit (the --authored arm) ----

// commitTurn writes the exchange (vector chunk), any facts Claude chose to store
// (graph nodes/edges), and a turn record (log) in one atomic commit. Graph writes
// are idempotent server-side, so re-asserting a fact across turns just converges:
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

// requireAgent checks that a workspace named with --agent is really there, so a
// mistyped id fails at startup naming itself rather than surfacing as a memory API
// error three lines into a conversation.
//
// A not-found here is indistinguishable from a workspace this key cannot reach,
// because the platform collapses the two on purpose: a refusal that confirmed the
// id exists would be a disclosure. The message therefore names both possibilities
// instead of guessing.
func requireAgent(ctx context.Context, jc *jennahClient, agentID string) error {
	var resp agentpb.GetAgentResponse
	code, err := jc.do(ctx, http.MethodGet, agentPath(agentID), nil, &resp)
	if code == http.StatusNotFound {
		return fmt.Errorf("agent workspace %q is not there, or this API key cannot reach it "+
			"(the platform answers both the same way). Create it with 'jnh agents create %s', "+
			"or drop --agent to let this demo mint its own", agentID, agentID)
	}
	if err != nil {
		return fmt.Errorf("check agent workspace %q: %w", agentID, err)
	}
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
	hc       *http.Client // the millisecond-class calls: query, inspect, commit
	formHC   *http.Client // memory:form only, see formTimeout
}

// formTimeout is the client-side deadline for a formation, and it is deliberately
// five times the one the other calls get.
//
// A formation is not slow by accident, it is slow by contract: it runs two bounded
// generative calls plus a retrieval before it writes, each call held to the
// region's own bound (120s by default), and the platform validates at startup that
// the whole thing fits the 300s its load balancer allows a request. A 60s client
// timeout would therefore abandon formations the server goes on to finish and
// commit, and report memory that WAS written as a failure. The formation key makes
// that recoverable rather than corrupting, but the recovery path should not be the
// ordinary one.
const formTimeout = 300 * time.Second

// do issues an ordinary call. Formation goes through doForm instead, because the
// two have deadlines an order of magnitude apart and one client cannot hold both.
func (c *jennahClient) do(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	return c.send(ctx, c.hc, method, path, in, out)
}

// doForm issues the formation call under formTimeout.
func (c *jennahClient) doForm(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	return c.send(ctx, c.formHC, method, path, in, out)
}

func (c *jennahClient) send(ctx context.Context, hc *http.Client, method, path string, in, out proto.Message) (int, error) {
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

	resp, err := hc.Do(req)
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

// state persists only the agent id, which is what makes memory carry across
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

// retiredNote names the superseded facts recall left out, so that a correction is
// visible as a correction rather than as facts quietly going missing.
func retiredNote(n int) string {
	if n == 0 {
		return ""
	}
	return fmt.Sprintf(" (+%d retired, not shown)", n)
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
// of env var key. Unlike envOr, the caller-supplied value wins, so an explicit
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
