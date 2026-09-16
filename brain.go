package main

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// brain is the pluggable chat LLM. It owns the session-local conversation history
// (so the current chat stays coherent) and, given a freshly-built system prompt
// reflecting what Jennah remembers, returns the assistant's reply plus any facts
// the model chose to persist via the remember_fact tool. Everything Jennah-facing
// in this demo is identical regardless of which brain answers — that's the point.
type brain interface {
	chat(ctx context.Context, system, userMsg string) (reply string, facts []fact, err error)
	label() string // short "provider/model" string for the startup banner
}

// The remember_fact tool, described once and mapped into each SDK's own tool type
// so the two backends stay in lockstep. It stores ONE (subject)-[relationship]->(object)
// triple per call; commitTurn turns those into graph nodes/edges.
//
// The subject is a FIELD, not an assumption. An earlier version of this tool took
// only (relationship, value) and hardwired the user as the subject, which made
// every fact a spoke off one hub: anything the user said about how two other
// entities relate had nowhere to go, so the model packed that structure into the
// two strings it did have ("Gucci, Haruka, Suna-kun" as one object, a whole
// clause as one relationship). The model was not extracting badly; the schema
// could not carry what it had extracted. Hence the emphasis below on exactly one
// entity per field.
const (
	toolName     = "remember_fact"
	toolDesc     = "Store ONE durable fact in long-term memory as a (subject)-[relationship]->(object) triple. Call once per fact, and call as many times as a message needs: a fact mentioning several entities is several calls, never one call with a list crammed into a field. Use for stable facts worth recalling in future sessions (the user's name, preferences, job, location and goals, and the people, organizations, teams and things they tell you about, including how those relate to each other); do NOT store transient chit-chat or questions."
	toolSubjDesc = "the single entity the fact is about, e.g. 'Alice', 'Alphaus', 'FinOps Consulting'. OMIT it whenever the fact is the user talking about themselves ('my name is X', 'I live in Y', 'I work at Z'), and keep omitting it once you know their name: the user already has a dedicated node, so naming them here creates a duplicate of them. Name a subject only for facts about someone or something else."
	toolRelDesc  = "short verb phrase linking subject to object, e.g. 'is named', 'likes', 'lives in', 'works at', 'has cto', 'owns', 'reports to', 'has member'"
	toolObjDesc  = "the single entity or value the relationship points at, e.g. 'Alice', 'hiking', 'Tokyo', 'Alphaus'. Exactly one, never a list: three members of a team is three calls, and two roles held by one person is two calls."
)

// newBrain selects the chat provider. "auto" prefers Anthropic when an Anthropic
// key is present, else Gemini — so someone with only one key set just runs `go run .`.
// anthropicKey, when non-empty, is the Anthropic API key from --anthropic-api-key
// (already defaulted to $ANTHROPIC_API_KEY); it overrides the SDK's own env lookup.
func newBrain(ctx context.Context, provider, anthropicKey string) (brain, error) {
	if provider == "auto" {
		switch {
		case anthropicKey != "":
			provider = "anthropic"
		case os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" || useVertexAI():
			provider = "gemini"
		default:
			return nil, fmt.Errorf("no chat credentials found: set GEMINI_API_KEY / Vertex AI env (Gemini) or pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), or pass --provider")
		}
	}
	switch strings.ToLower(provider) {
	case "gemini":
		return newGeminiBrain(ctx)
	case "anthropic", "claude":
		return newAnthropicBrain(anthropicKey), nil
	default:
		return nil, fmt.Errorf("unknown --provider %q (want auto|gemini|anthropic)", provider)
	}
}
