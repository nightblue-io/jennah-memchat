package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// geminiModel keeps the interactive demo snappy and cheap; swap to
// "gemini-2.5-pro" for maximum capability. The same id works on both the AI
// Studio (API-key) and Vertex AI backends.
const geminiModel = "gemini-3.8-flash"

// geminiBrain is the Google Gemini backend. It talks to either AI Studio (an
// API key in GEMINI_API_KEY / GOOGLE_API_KEY) or Vertex AI (GCP project +
// location + Application Default Credentials), and keeps the session transcript
// in the SDK's Content type.
type geminiBrain struct {
	client  *genai.Client
	backend string // for the startup banner, e.g. "vertex:my-proj/us-central1"
	history []*genai.Content
	config  *genai.GenerateContentConfig
}

// useVertexAI reports whether to route Gemini through Vertex AI rather than the
// AI Studio API-key path: either explicitly (GOOGLE_GENAI_USE_VERTEXAI=1|true),
// or implicitly when a GCP project is configured and no Studio key is set.
func useVertexAI() bool {
	switch strings.ToLower(os.Getenv("GOOGLE_GENAI_USE_VERTEXAI")) {
	case "1", "true":
		return true
	}
	return os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" &&
		os.Getenv("GOOGLE_CLOUD_PROJECT") != ""
}

// newGeminiBrain builds the Gemini backend. offerTool declares remember_fact,
// and only the --authored arm passes true: by default formation does the
// extracting, so the model is given no tool and the tool-call branch of chat
// below never fires.
func newGeminiBrain(ctx context.Context, offerTool bool) (*geminiBrain, error) {
	cc := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	backend := "ai-studio"
	if useVertexAI() {
		// Vertex uses Application Default Credentials (run once:
		//   gcloud auth application-default login
		// or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key), not an
		// API key. Project/location come from the standard GCP env vars.
		project := os.Getenv("GOOGLE_CLOUD_PROJECT")
		if project == "" {
			return nil, fmt.Errorf("Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set (and run: gcloud auth application-default login)")
		}
		location := envOr("GOOGLE_CLOUD_LOCATION", os.Getenv("GOOGLE_CLOUD_REGION"))
		if location == "" {
			location = "global" // Gemini 3.8 is served on the global endpoint
		}
		cc.Backend = genai.BackendVertexAI
		cc.Project = project
		cc.Location = location
		backend = "vertex:" + project + "/" + location
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	config := &genai.GenerateContentConfig{MaxOutputTokens: 2048}
	if offerTool {
		config.Tools = []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        toolName,
				Description: toolDesc,
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"subject":      {Type: genai.TypeString, Description: toolSubjDesc},
						"relationship": {Type: genai.TypeString, Description: toolRelDesc},
						"object":       {Type: genai.TypeString, Description: toolObjDesc},
					},
					// subject is optional: omitted means the user, which keeps the
					// common case ("my name is Sabrina") a two-field call as before.
					Required: []string{"relationship", "object"},
				},
			}},
		}}
	}
	return &geminiBrain{client: client, backend: backend, config: config}, nil
}

func (b *geminiBrain) label() string { return "gemini/" + geminiModel + " (" + b.backend + ")" }

func (b *geminiBrain) chat(ctx context.Context, system, userMsg string) (string, []fact, error) {
	// The system prompt changes every turn (it embeds freshly-recalled memory), so
	// set it per-request rather than baking it into the client.
	b.config.SystemInstruction = genai.NewContentFromText(system, genai.RoleUser)
	b.history = append(b.history, genai.NewContentFromText(userMsg, genai.RoleUser))

	var facts []fact
	var reply strings.Builder
	for {
		resp, err := b.client.Models.GenerateContent(ctx, geminiModel, b.history, b.config)
		if err != nil {
			return "", nil, err
		}
		if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
			break
		}
		model := resp.Candidates[0].Content
		b.history = append(b.history, model)

		var toolResults []*genai.Part
		for _, part := range model.Parts {
			switch {
			case part.FunctionCall != nil:
				fc := part.FunctionCall
				subj, _ := fc.Args["subject"].(string)
				rel, _ := fc.Args["relationship"].(string)
				obj, _ := fc.Args["object"].(string)
				if strings.TrimSpace(rel) != "" && strings.TrimSpace(obj) != "" {
					facts = append(facts, fact{subj: subj, rel: rel, obj: obj})
				}
				toolResults = append(toolResults, genai.NewPartFromFunctionResponse(
					fc.Name, map[string]any{"result": "noted"}))
			case part.Text != "" && !part.Thought:
				reply.WriteString(part.Text)
			}
		}
		if len(toolResults) == 0 {
			break
		}
		// Feed the tool results back so the model can produce its final reply.
		b.history = append(b.history, genai.NewContentFromParts(toolResults, genai.RoleUser))
	}
	return strings.TrimSpace(reply.String()), facts, nil
}
