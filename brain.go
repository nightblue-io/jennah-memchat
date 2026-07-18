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
// so the two backends stay in lockstep. It stores ONE (user)-[relationship]->(value)
// triple per call; commitTurn turns those into graph nodes/edges.
const (
	toolName    = "remember_fact"
	toolDesc    = "Store ONE durable fact about the user in long-term memory as a (user)-[relationship]->(value) triple. Call once per fact. Use for stable facts worth recalling in future sessions (their name, preferences, job, location, goals, important people or things); do NOT store transient chit-chat or questions."
	toolRelDesc = "short verb phrase linking user to value, e.g. 'is named', 'likes', 'works as', 'lives in', 'is learning'"
	toolValDesc = "the object of the fact, e.g. 'Alice', 'hiking', 'software engineer', 'Tokyo'"
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
