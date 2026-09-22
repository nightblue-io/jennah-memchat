# memchat

A CLI chatbot demonstrating cross-session persistent memory using Jennah's public
memory APIs over HTTP/JSON via `jennah-proxy`.

By default the chatbot does not decide what to remember. It hands each exchange to
`memory:form` and Jennah's **cognition** layer does the deciding: it extracts
candidate memories from the turns, recalls what the workspace already holds,
reconciles the two, and commits the result as one atomic write. The chat model is
asked for one thing only, a reply.

`-authored` runs the other arm, the way this demo worked before formation existed
and the way any client works that owns its own extraction: the model is handed a
`remember_fact` tool and this process turns the triples it emits into graph nodes
and edges itself, then writes them with `memory:commit`. Both arms are real, both
write into the same workspace, and the difference between them is the point of the
demo.

This repository is a standalone Go module that interacts exclusively with the
public HTTP/JSON gateway authenticated by a `jennah_sk_` API key.

## The two arms

| | default (`memory:form`) | `-authored` (`memory:commit`) |
| --- | --- | --- |
| Who decides what is worth remembering | Jennah | your prompt and your tool schema |
| Who writes the extraction prompt | Jennah | you |
| Entity ids, edge direction, relationship names | Jennah | you |
| Contradiction handling | supersession, old assertion retired and kept as history | last write wins, or you build it |
| Tool calls in your model loop | none | one per fact |
| Cost | server-side inference, metered as formation | your own tokens |
| Latency per turn | seconds (inference before the write) | milliseconds |

## Architecture

Each turn executes a query-think-form cycle:

```
[User Input]
     │
     ├── 1. Query ───┬─ memory:query   (semantic search over past turns)
     │               └─ memory:inspect (full knowledge graph retrieval)
     │
     ├── 2. Think ────  LLM answers the user. No memory tool, no memory instruction.
     │
     └── 3. Form ────   memory:form (extract → recall → reconcile → commit, atomic)
```

1. **Query**:
   - `memory:query`: Performs semantic vector search across past turns. Query text
     is embedded by Jennah using server-managed embeddings.
   - `memory:inspect`: Retrieves known knowledge graph triples. Full inspection is
     used instead of traversal queries because edge endpoints (`source_node_id`,
     `target_node_id`) are needed to reconstruct direction regardless of phrasing.
2. **Think**: The LLM generates a response given the user message and recalled
   memory context. Under `-authored` it is additionally given the `remember_fact`
   tool and told to use it.
3. **Form**: `memory:form` receives the recent turns and returns a receipt naming
   every candidate it extracted and what it decided about each one. Under
   `-authored` this step is instead a `memory:commit` the client composes itself.

### What each turn submits

A formation carries a **rolling window** of the last 6 turns, not just the current
exchange. Extraction runs before recall, so the only thing that can resolve "she",
"there" or "the second one" is the submitted turns themselves. The overlap means
already-formed content gets re-extracted, which reconciliation reports as `known`
rather than writing twice.

Each formation is sent under a **formation key** (`frm_<session>_<turn>`). Extraction
is nondeterministic, so a blind resend would form a second, different set of memory
alongside the first; under the key a resend replays the original receipt instead.

## Reading the receipt

Every candidate comes back with a decision, and `-verbose` prints all of them:

```
you> Actually I moved to Tokyo last week.

memo> Oh, congratulations on the move! ...
  [forming memory from 4 turn(s), key frm_sess_d310a0d4_2 ...]
  known    Hajime works as a backend engineer at Alphaus.  (matches fct_b862ea8b...)
  revised  Hajime lives in Tokyo, having recently moved there from Osaka.  (retired fct_85ff9222...)
  known    Hajime works at Alphaus  (matches rel_e52a9fc8...)
  revised  Hajime lives in Tokyo  (retired rel_99f7081d...)
  formed: log=1 vec=0(+1 superseded) nodes=2 edges=0(+1 superseded) @ 2026-09-22T03:34:12Z
  [memory] 2 earlier assertion(s) retired by a correction in this turn
           (superseded, not overwritten - the previous value stays readable as history)
```

- `new` / `revised` / `known` / `rejected` is the decision taxonomy. A `rejected`
  candidate carries the reason, which is how you learn that, for example, a
  conversation recounting history was not allowed to retire the current fact.
- A **revision's replacement row is counted as a supersession**, not in
  `vec`/`edges`, because closing one validity window and opening another is not the
  same act as inserting a new assertion. A turn that only corrected things reports
  zero rows and non-zero supersessions.
- Retired assertions are **not deleted**. They keep their validity window and stay
  readable, which is what makes them history.
- Without `-verbose` each turn prints one line: `[formed: 2 new, 3 known]`.

Recalled chunks that were formed carry their provenance, shown under `-verbose`:

```
  ~ Hajime lives in Tokyo, having recently moved there from Osaka.  [formed by frm_1f68a103..., turn(s) 0,2]
```

Those are reserved `jennah.*` metadata keys that a caller cannot set itself, so a
formed memory can always be traced back to the conversation it came from. Memory
written by the `-authored` arm carries none of them.

## Typing the formed memory: vocabulary

Left alone, formation names entities and relations however the extraction model
phrases them, which is workable but loose. Declaring a **vocabulary** renders your
entity classes and relation types into the extraction prompt, so the model picks
from what you declared instead of inventing. It steers generation only: nothing is
ever refused for being off vocabulary.

`vocabulary.yaml` in this repo is a starting vocabulary for a personal assistant
(6 classes, 9 relation types). Because a vocabulary is declared on a workspace and
the agent cannot declare its own, provision the workspace first and then point the
demo at it with `-agent`:

```sh
# 1. create the workspace and type it (management credentials, not an API key)
jnh agents create demo.memchat_vocab01 --name memchat-vocab-demo --wait
jnh vocabulary declare --scope demo.memchat_vocab01 --from-file vocabulary.yaml
jnh vocabulary get --scope demo.memchat_vocab01

# 2. run the demo against that exact workspace
./memchat -verbose -agent demo.memchat_vocab01
```

`-agent` uses the workspace you name and nothing else: it never creates one, and it
never writes the state file. Both are deliberate. A mistyped id that silently minted
a fresh empty workspace would leave the demo reporting that it remembers nothing,
which reads as a platform fault rather than a typo, so an unknown id is an error at
startup that names itself. And an id written back to the state file would leave a
later flagless run still pointed at the operator's workspace, quietly making the
flag permanent.

To declare the same vocabulary for every workspace in the enterprise instead, omit
`--scope`. A scope-level declaration replaces the enterprise default for that one
scope rather than merging with it.

**It cannot be declared by the agent.** `agent.vocabulary:manage` is
management-class, so no API key carries it: an operator declares the vocabulary and
the agent lives with what resolves. Even *reading* it needs `agent.vocabulary:read`,
which is data-plane but is not in the member default bundle, so a key minted without
it gets a note in the banner rather than a vocabulary. That split is the shape of
the feature, not a limitation of this demo: how an enterprise's memory is typed is
an enterprise decision, not a per-agent one.

The difference, measured on two identical runs of the same two turns ("I am Kenji
and I lead the platform team at Nightblue." / "I live in Sapporo, and Nightblue is
based in Tokyo. I am also into bouldering."), same model, two fresh workspaces:

| | no vocabulary | with `vocabulary.yaml` |
| --- | --- | --- |
| nodes | 5: Kenji, Nightblue, Sapporo, Tokyo, Bouldering | 6: the same plus `platform team` |
| edges | 4: `WORKS_AT`, `LIVES_IN`, `LOCATED_IN`, `INTERESTED_IN` | 6: the same plus `Kenji -LEADS-> platform team` and `platform team -PART_OF-> Nightblue` |
| classes | none | Person, Organization, Team, Place, Interest |

**The team is the whole story.** Untyped, "I lead the platform team at Nightblue"
produced only `Kenji -WORKS_AT-> Nightblue`: the team he leads was not represented
at all, so neither the leadership nor the team's place in the organization survived.
Typed, declaring `Team` as a class distinct from `Organization` made the team an
entity, and its two relationships followed.

Extraction is nondeterministic, so the untyped side fails differently from run to
run rather than the same way each time. An earlier run of the first sentence alone
kept the team but welded it to its employer as one node labelled
`platform team at Nightblue`, which is unfollowable in a different way. What was
consistent across runs is the typed side: `Team` plus `LEADS` plus `PART_OF`, every
time. Judge a vocabulary on that, not on any single untyped graph.

The descriptions carry more weight than the names. `LIVES_IN` is described as "use
this for people, never for organizations" and `LOCATED_IN` as its mirror, which is
what keeps a person's city and an employer's city on separate relations. Untyped,
one workspace was observed holding both relations for the same person; another got
it right unaided. The vocabulary is what makes it not a coin toss.

## Memory Model & State

- **Session Continuity**: The agent workspace ID is persisted locally in
  `memchat-state.json`. Re-running the application resumes the existing agent
  memory. Deleting `memchat-state.json` creates a fresh agent workspace.
- **Workspace Namespacing**: Workspaces are created with the prefix
  `demo.memchat_<random>`. Role-based access policies targeting `demo.*`
  automatically cover any workspace created by this tool.

Under `-authored` the client also owns all of the following, which is exactly what
the default arm does not have to build:

- **Graph Triples**: Entity names map to node labels; relationships map to edge
  relationship types.
- **Deterministic IDs**: Entity node IDs are SHA-1 content hashes of their labels,
  ensuring that re-asserting an entity converges on the same node.
- **Subject Resolution**: The user anchor has a dedicated node ID (`user`). When the
  user refers to themselves, the subject is omitted and resolves to `user`. When
  facts describe third-party entities, both subject and object are explicitly named.
- **Relationship Normalization**: Reverse relationship phrases are normalized before
  commit (for example, `Chew IS_CTO_OF Alphaus` is normalized to
  `Alphaus HAS_CTO Chew`).

## Prerequisites

Formation replaces the extraction, not the reply, so both arms still need a chat
provider.

- **Jennah API Key**: A `jennah_sk_` key for an entitled enterprise (generate via
  console or `jnh`: `POST /v1/apikeys {"label":"memchat"}`).
- **LLM Provider Credentials**: Either Anthropic Claude or Google Gemini:
  - **Anthropic**: `ANTHROPIC_API_KEY`
  - **Gemini (Google AI Studio)**: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - **Gemini (Vertex AI)**: `GOOGLE_GENAI_USE_VERTEXAI=true`,
    `GOOGLE_CLOUD_PROJECT`, and Application Default Credentials

Submitted turns reach an external inference service exactly as you write them.

## Usage

### Quick Start

With Anthropic Claude:
```sh
export JENNAH_API_KEY=jennah_sk_...
export ANTHROPIC_API_KEY=sk-ant-...
go run .
```

With Google Gemini via AI Studio:
```sh
export JENNAH_API_KEY=jennah_sk_...
export GEMINI_API_KEY=...
go run .
```

With Google Gemini via Vertex AI:
```sh
gcloud auth application-default login
export JENNAH_API_KEY=jennah_sk_...
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1 # optional, defaults to global
go run .
```

### Try the thing the demo exists for

Tell it something, correct yourself, and watch the correction land as a
supersession rather than an overwrite:

```text
you> Hi, I am Hajime. I live in Osaka and I work at Alphaus as a backend engineer.
you> Actually I moved to Tokyo last week.
```

Then quit, run it again, and ask where you live. Run the same two lines under
`-authored` to see what your own client has to do to get there.

### CLI Flags

| Flag | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `-endpoint` | `JENNAH_ENDPOINT` | `https://jennah.alphaus.cloud` | Jennah proxy origin URL |
| `-authored` | | `false` | Extract and author the memory writes in this client (`remember_fact` + `memory:commit`) instead of letting `memory:form` do it |
| `-provider` | | `auto` | LLM backend: `auto`, `gemini`, or `anthropic` |
| `-region` | `JENNAH_REGION` | (platform default) | Region for agent creation (e.g. `us-central1`). List with `jnh agents regions` |
| `-agent` | | | Use this existing agent workspace (provisioned out of band, e.g. with a vocabulary declared on it). Never creates one and never writes the state file |
| `-state` | | `memchat-state.json` | Path to local state file storing agent ID |
| `-verbose` | | `false` | Print recalled snippets, facts, and formation receipts |
| `-jennah-api-key` | `JENNAH_API_KEY` | | Jennah API key |
| `-anthropic-api-key` | `ANTHROPIC_API_KEY` | | Anthropic API key |

Exit a chat session with `/exit`, `/quit`, or Ctrl-D.

## Model Configuration

The chat model here is the one that answers the user. The model that forms memory
is Jennah's, configured per region on the platform, and is not selectable from a
client.

Default chat models are chosen for low latency:
- Anthropic: `claude-sonnet-5` (configured in `brain_anthropic.go`)
- Gemini: `gemini-3.8-flash` (configured in `brain_gemini.go`)

To use other models (such as `claude-opus-4-8` or `gemini-2.5-pro`), update the
model constant in the corresponding provider file.

## Implementation Notes

- **Timeouts**: formation gets its own HTTP client with a 300s deadline, five times
  the one the other calls use. A formation runs two bounded generative calls plus a
  retrieval before it writes, and a 60s client timeout would abandon formations the
  server goes on to finish and commit.
- **Graph Recall**: Uses `memory:inspect` instead of graph traversal queries
  (`memory:query`). Traversal rows project edge IDs and types without endpoint IDs,
  whereas `memory:inspect` returns full `source_node_id` and `target_node_id`
  fields.
- **Retired assertions in the prompt**: `memory:query` excludes superseded chunks,
  so the snippets half of the prompt is current by construction. The graph half is
  read with `memory:inspect`, which enumerates rather than answers, and its edge
  listing does not yet project `valid_at`/`invalid_at`. The filter for retired
  edges is written against the contract and reports `(+N retired, not shown)` once
  the projection lands; until then a superseded relationship can still reach the
  prompt even though its chunk counterpart does not.
- **Vocabulary**: see "Typing the formed memory" above. The startup banner reports
  what is in effect, and reading even that needs `agent.vocabulary:read` on the key.
- **Link Fusion**: Link fusion (`link: true`) is not enabled because the endpoint is
  currently unimplemented.

## Inspecting Stored Memory

To inspect the raw knowledge graph and memory stored in Jennah for an agent
workspace:

```sh
jnh scopes memory inspect <agent-id> --graph --all
```
