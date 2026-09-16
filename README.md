# memchat

A CLI chatbot demonstrating cross-session persistent memory using Jennah's public memory APIs over HTTP/JSON via `jennah-proxy`.

The client maintains state across sessions using two complementary memory models in Jennah:
- **Semantic recall**: Vector search over previous conversation exchanges.
- **Knowledge graph**: Entity and relationship extraction stored as graph nodes and edges.

This repository is a standalone Go module that interacts exclusively with the public HTTP/JSON gateway authenticated by a `jennah_sk_` API key.

## Architecture

Each turn executes a query-think-commit cycle:

```
[User Input]
     │
     ├── 1. Query ───┬─ memory:query   (semantic search over past turns)
     │               └─ memory:inspect (full knowledge graph retrieval)
     │
     ├── 2. Think ────  LLM receives context and extracts facts via remember_fact tool
     │
     └── 3. Commit ───  memory:commit  (atomic write: vector chunk + graph nodes/edges + log)
```

1. **Query**:
   - `memory:query`: Performs semantic vector search across past turns. Query text is embedded by Jennah using server-managed embeddings.
   - `memory:inspect`: Retrieves known knowledge graph triples. Full inspection is used instead of traversal queries because edge endpoints (`source_node_id`, `target_node_id`) are needed to reconstruct direction regardless of phrasing.
2. **Think**:
   - The LLM generates a response given the user message and recalled memory context.
   - When durable facts are mentioned, the model invokes the `remember_fact` tool to emit `(subject)-[relationship]->(object)` triples.
3. **Commit**:
   - `memory:commit`: Atomically writes the turn exchange as a vector chunk, persists extracted graph nodes and edges, and records an execution log entry. The commit receipt is checked for embedding truncations.

## Memory Model & State

- **Graph Triples**: Entity names map to node labels; relationships map to edge relationship types.
- **Deterministic IDs**: Entity node IDs are SHA-1 content hashes of their labels, ensuring that re-asserting an entity converges on the same node.
- **Subject Resolution**: The user anchor has a dedicated node ID (`user`). When the user refers to themselves, the subject is omitted and resolves to `user`. When facts describe third-party entities, both subject and object are explicitly named.
- **Relationship Normalization**: Reverse relationship phrases are normalized before commit (for example, `Chew IS_CTO_OF Alphaus` is normalized to `Alphaus HAS_CTO Chew`).
- **Session Continuity**: The agent workspace ID is persisted locally in `memchat-state.json`. Re-running the application resumes the existing agent memory. Deleting `memchat-state.json` creates a fresh agent workspace.
- **Workspace Namespacing**: Workspaces are created with the prefix `demo.memchat_<random>`. Role-based access policies targeting `demo.*` automatically cover any workspace created by this tool.

## Prerequisites

- **Jennah API Key**: A `jennah_sk_` key for an entitled enterprise (generate via console or `jnh`: `POST /v1/apikeys {"label":"memchat"}`).
- **LLM Provider Credentials**: Either Anthropic Claude or Google Gemini:
  - **Anthropic**: `ANTHROPIC_API_KEY`
  - **Gemini (Google AI Studio)**: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - **Gemini (Vertex AI)**: `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and Application Default Credentials

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

### CLI Flags

| Flag | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `-endpoint` | `JENNAH_ENDPOINT` | `https://jennah.alphaus.cloud` | Jennah proxy origin URL |
| `-provider` | | `auto` | LLM backend: `auto`, `gemini`, or `anthropic` |
| `-region` | `JENNAH_REGION` | (platform default) | Region for agent creation (e.g. `us-central1`). List with `jnh agents regions` |
| `-state` | | `memchat-state.json` | Path to local state file storing agent ID |
| `-verbose` | | `false` | Print recalled snippets, facts, and commit receipts |
| `-jennah-api-key` | `JENNAH_API_KEY` | | Jennah API key |
| `-anthropic-api-key` | `ANTHROPIC_API_KEY` | | Anthropic API key |

Exit a chat session with `/exit`, `/quit`, or Ctrl-D.

## Model Configuration

Default models are chosen for low latency:
- Anthropic: `claude-sonnet-5` (configured in `brain_anthropic.go`)
- Gemini: `gemini-3.8-flash` (configured in `brain_gemini.go`)

To use other models (such as `claude-opus-4-8` or `gemini-2.5-pro`), update the model constant in the corresponding provider file.

## Implementation Notes

- **Graph Recall**: Uses `memory:inspect` instead of graph traversal queries (`memory:query`). Traversal rows project edge IDs and types without endpoint IDs, whereas `memory:inspect` returns full `source_node_id` and `target_node_id` fields.
- **Commit Receipts**: When running with `-verbose`, commit receipts (`log`, `vec`, `nodes`, `edges`) are printed after each turn.
- **Link Fusion**: Link fusion (`link: true`) is not enabled because the endpoint is currently unimplemented.

## Inspecting Stored Memory

To inspect the raw knowledge graph and memory stored in Jennah for an agent workspace:

```sh
jnh scopes memory inspect <agent-id> --graph --all
```
