package main

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicModel keeps the interactive demo snappy and cheap; swap to
// anthropic.ModelClaudeOpus4_8 for maximum capability.
const anthropicModel = anthropic.Model("claude-sonnet-5")

// anthropicBrain is the Claude backend. It reads ANTHROPIC_API_KEY (or an ambient
// ant profile) and keeps the session transcript in the SDK's message type.
type anthropicBrain struct {
	client  anthropic.Client
	history []anthropic.MessageParam
}

// newAnthropicBrain builds the Claude backend. When apiKey is non-empty (from
// --anthropic-api-key) it's passed explicitly; otherwise the SDK falls back to
// its usual ANTHROPIC_API_KEY / ambient profile resolution.
func newAnthropicBrain(apiKey string) *anthropicBrain {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &anthropicBrain{client: anthropic.NewClient(opts...)}
}

func (b *anthropicBrain) label() string { return "anthropic/" + string(anthropicModel) }

func (b *anthropicBrain) chat(ctx context.Context, system, userMsg string) (string, []fact, error) {
	tools := []anthropic.ToolUnionParam{{OfTool: &anthropic.ToolParam{
		Name:        toolName,
		Description: anthropic.String(toolDesc),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"subject":      map[string]any{"type": "string", "description": toolSubjDesc},
				"relationship": map[string]any{"type": "string", "description": toolRelDesc},
				"object":       map[string]any{"type": "string", "description": toolObjDesc},
			},
			// subject is optional: omitted means the user, which keeps the common
			// case ("my name is Hajime") a two-field call exactly as before.
			Required: []string{"relationship", "object"},
		},
	}}}

	b.history = append(b.history, anthropic.NewUserMessage(anthropic.NewTextBlock(userMsg)))

	var facts []fact
	var reply strings.Builder
	for {
		resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropicModel,
			MaxTokens: 2048,
			System:    []anthropic.TextBlockParam{{Text: system}},
			Messages:  b.history,
			Tools:     tools,
			Thinking:  anthropic.ThinkingConfigParamUnion{OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{}},
		})
		if err != nil {
			return "", nil, err
		}
		b.history = append(b.history, resp.ToParam())

		var toolResults []anthropic.ContentBlockParamUnion
		for _, blk := range resp.Content {
			switch v := blk.AsAny().(type) {
			case anthropic.TextBlock:
				reply.WriteString(v.Text)
			case anthropic.ToolUseBlock:
				var in struct {
					Subject      string `json:"subject"`
					Relationship string `json:"relationship"`
					Object       string `json:"object"`
				}
				_ = json.Unmarshal([]byte(v.JSON.Input.Raw()), &in)
				if strings.TrimSpace(in.Relationship) != "" && strings.TrimSpace(in.Object) != "" {
					facts = append(facts, fact{subj: in.Subject, rel: in.Relationship, obj: in.Object})
				}
				toolResults = append(toolResults, anthropic.NewToolResultBlock(blk.ID, "noted", false))
			}
		}
		if resp.StopReason != anthropic.StopReasonToolUse {
			break
		}
		b.history = append(b.history, anthropic.NewUserMessage(toolResults...))
	}
	return strings.TrimSpace(reply.String()), facts, nil
}
