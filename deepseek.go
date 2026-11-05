package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Usage struct {
	PromptTokens            int `json:"prompt_tokens"`
	CompletionTokens        int `json:"completion_tokens"`
	TotalTokens             int `json:"total_tokens"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func (u Usage) Reasoning() int {
	return u.CompletionTokensDetails.ReasoningTokens
}

type Thinking struct {
	Type string `json:"type"`
}

type apiRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	Thinking    *Thinking     `json:"thinking,omitempty"`
}

type apiResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type CallResult struct {
	Content      string
	Reasoning    string
	FinishReason string
	Usage        Usage
}

type DeepSeek struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

func NewDeepSeek(baseURL, apiKey, model string) *DeepSeek {
	return &DeepSeek{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{},
	}
}

func (d *DeepSeek) Model() string { return d.model }

func (d *DeepSeek) Call(ctx context.Context, prompt string, temperature float64) (CallResult, error) {
	temp := temperature
	req := apiRequest{
		Model: d.model,
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Temperature: &temp,
		Thinking:    &Thinking{Type: "disabled"},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return CallResult{}, fmt.Errorf("сборка запроса: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, d.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return CallResult{}, fmt.Errorf("создание запроса: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+d.apiKey)

	resp, err := d.client.Do(httpReq)
	if err != nil {
		return CallResult{}, fmt.Errorf("запрос к DeepSeek: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return CallResult{}, fmt.Errorf("чтение ответа: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(data, &apiErr) == nil && apiErr.Error.Message != "" {
			return CallResult{}, fmt.Errorf("DeepSeek: %s", apiErr.Error.Message)
		}
		return CallResult{}, fmt.Errorf("DeepSeek вернул статус %d", resp.StatusCode)
	}

	var out apiResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return CallResult{}, fmt.Errorf("разбор ответа DeepSeek: %w", err)
	}
	if len(out.Choices) == 0 {
		return CallResult{}, errors.New("DeepSeek вернул пустой choices")
	}

	choice := out.Choices[0]
	return CallResult{
		Content:      choice.Message.Content,
		Reasoning:    choice.Message.ReasoningContent,
		FinishReason: choice.FinishReason,
		Usage:        out.Usage,
	}, nil
}
