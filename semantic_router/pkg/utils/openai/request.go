package openai

import (
	"encoding/json"
	"fmt"
)

// OpenAIRequest represents an OpenAI API request
type OpenAIRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      *bool         `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
}

// ChatMessage represents a message in the OpenAI chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIResponse represents an OpenAI API response
type OpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ParseRequest parses the OpenAI request JSON
func ParseRequest(data []byte) (*OpenAIRequest, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// ParseTokensFromResponse extracts detailed token counts from the OpenAI schema based response JSON
func ParseTokensFromResponse(responseBody []byte) (promptTokens, completionTokens, totalTokens int, err error) {
	if responseBody == nil {
		return 0, 0, 0, fmt.Errorf("empty response body")
	}

	var response OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to parse response JSON: %w", err)
	}

	// Extract token counts from the usage field
	promptTokens = response.Usage.PromptTokens
	completionTokens = response.Usage.CompletionTokens
	totalTokens = response.Usage.TotalTokens

	return promptTokens, completionTokens, totalTokens, nil
}

// ExtractUserAndNonUserContent extracts user content and non-user messages from OpenAI request
func ExtractUserAndNonUserContent(request *OpenAIRequest) (userContent string, nonUserMessages []string) {
	for _, msg := range request.Messages {
		if msg.Role == "user" {
			userContent = msg.Content
		} else if msg.Role != "" {
			nonUserMessages = append(nonUserMessages, msg.Content)
		}
	}
	return userContent, nonUserMessages
}

// SerializeRequest serializes an OpenAI request to JSON
func SerializeRequest(request *OpenAIRequest) ([]byte, error) {
	return json.Marshal(request)
}
