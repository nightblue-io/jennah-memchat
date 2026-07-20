# memchat - a chatbot that remembers across sessions

A small demo **agent** that consumes Jennah's public memory APIs the way any
external agent would: plain HTTP/JSON through the `jennah-proxy` gateway,
authenticated with a `jennah_sk_` API key. No Jennah server internals are
imported - this is a standalone Go module, so it doubles as a reference for
outside integrators.

It shows off **unified memory**: semantic recall of past conversation *and* a
per-user **knowledge graph**, both over the one memory transport.

## What it does each turn

```
you> …                       ┌─ memory:query (semantic) ── recall past exchanges
                             ├─ memory:query (graph)   ── traverse user's fact graph
     (query → think → commit)│
                             ├─ Claude answers, calling remember_fact(...) for
                             │  durable facts → (user)-[relationship]->(value)
                             └─ memory:commit ── vector chunk + new graph nodes/edges
                                                 + an execution-log entry (atomic)
```

Cross-session memory is just **reusing the same `agent_instance_id`**, persisted
to `memchat-state.json`. Graph writes are idempotent server side, so re-asserting
a fact across turns just converges - the client keeps no id ledger. Delete the
state file to start a fresh persona.

## Prerequisites

1. A Jennah API key for an **approved, entitled** enterprise. Mint one after
   logging in (console or `jnh`):
   `POST /v1/apikeys {"label":"memchat"}` → copy the `secret` (shown once).
2. A chat model - Anthropic, or Gemini (via **Google AI Studio** with an API
   key, or via **Vertex AI** with a GCP project + ADC).

The chat brain is pluggable: only the LLM differs, every Jennah memory call is
identical. `-provider auto` (the default) picks **Anthropic** when an Anthropic
key is configured, otherwise **Gemini** (a Gemini key or a Vertex/GCP env); force
it with `-provider gemini|anthropic`. Within Gemini, Vertex is used when
`GOOGLE_GENAI_USE_VERTEXAI=true` or when `GOOGLE_CLOUD_PROJECT` is set and no
Studio key is present.

The agent's home region is chosen at creation with `-region` (or `$JENNAH_REGION`);
it's applied only on first launch, since an agent is pinned to one region for its
lifetime, and empty uses the platform default. List the available regions with
`jnh agents regions`. The target region must have managed embeddings configured
(prod `db0001` / `us-central1` does) - the demo sends plain text and lets the
server embed it.

## Run

```sh
export JENNAH_API_KEY=jennah_sk_...

# Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
go run .                                        # auto-selects Anthropic

# …or Gemini via Google AI Studio (API key):
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
go run .

# …or Gemini via Vertex AI (GCP project + ADC, no API key):
gcloud auth application-default login          # once
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1       # optional; defaults to "global"
go run .

go run . -provider gemini   # force a provider regardless of which keys are set
go run . -verbose           # show recalled triples/snippets + commit receipts each turn
go run . -endpoint http://127.0.0.1:8090   # against a local proxy instead
go run . -region us-central1               # pin the agent's home region (or $JENNAH_REGION)

# …or pass the keys as flags instead of env vars:
go run . -jennah-api-key jennah_sk_... -anthropic-api-key sk-ant-...
```

On start it prints the chosen brain, e.g. `chat model: anthropic/claude-sonnet-5`
or `chat model: gemini/gemini-2.5-flash (vertex:my-gcp-project/us-central1)`.

Then talk to it, quit (`/exit` or Ctrl-D), run it again - it recalls what you
told it. Try: *"Hi, I'm Alice, I'm a backend engineer in Berlin and I'm learning
to sail."* … quit … relaunch … *"what do you remember about me?"*

## Notes

- Each provider defaults to a snappy/cheap model (`claude-sonnet-5`,
  `gemini-2.5-flash`); edit `anthropicModel` in `brain_anthropic.go`
  (→ `anthropic.ModelClaudeOpus4_8`) or `geminiModel` in `brain_gemini.go`
  (→ `gemini-2.5-pro`) for max capability. Backends live behind the `brain`
  interface in `brain.go`.
- `-verbose` surfaces the memory activity live: the recalled `user <rel> <value>`
  triples and past snippets before each reply, and the commit receipt
  (`log=… vec=… nodes=… edges=…`) after - handy when demoing.
- `remember_fact` stores the fact's human value in the graph node **Label** and
  the predicate as the edge **RelationshipType**, so a one-hop traversal from
  the `user` node reads back as readable `user <relationship> <value>` triples.
- Fusion (`link:true`) is intentionally not used - it returns Unimplemented.
