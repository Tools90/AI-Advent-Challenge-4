package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func newTestServer(t *testing.T, upstream http.HandlerFunc) *server {
	t.Helper()

	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)

	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	return &server{
		cfg: config{
			apiKey:  "test-key",
			baseURL: up.URL,
			model:   "deepseek-v4-flash",
			addr:    ":0",
		},
		llm:         NewDeepSeek(up.URL, "test-key", "deepseek-v4-flash"),
		store:       store,
		promptsPath: filepath.Join(t.TempDir(), "prompts.json"),
	}
}

func mockCompletion(w http.ResponseWriter, content string) {
	_ = json.NewEncoder(w).Encode(map[string]any{
		"choices": []any{map[string]any{
			"message":       map[string]string{"content": content},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	})
}

func TestAnalyzeLogic(t *testing.T) {
	prompts := defaultPrompts

	cases := []struct {
		content string
		format  bool
		correct bool
		extract string
	}{
		{"7", true, true, "7"},
		{"7.", false, true, "7"},
		{"Ответ: 7", false, true, "7"},
		{"[7]", false, true, "7"},
		{"8", true, false, "8"},
		{"нет ответа", false, false, ""},
	}

	for _, c := range cases {
		extracted, format, correct := analyzeRun("logic", c.content, prompts)
		if extracted != c.extract || format != c.format || correct != c.correct {
			t.Errorf("%q: extracted=%q format=%v correct=%v", c.content, extracted, format, correct)
		}
	}
}

func TestParseRecipe(t *testing.T) {
	valid := `{"recipe":"хлеб, авокадо, яйцо"}`
	recipe, ok := parseRecipe(valid)
	if !ok || recipe != "хлеб, авокадо, яйцо" {
		t.Fatalf("valid: recipe=%q ok=%v", recipe, ok)
	}

	withIntro := "Вот рецепт: " + valid
	if _, ok := parseRecipe(withIntro); ok {
		t.Error("текст вокруг JSON должен ломать формат")
	}

	wrongKey := `{"text":"рецепт"}`
	if _, ok := parseRecipe(wrongKey); ok {
		t.Error("неверный ключ должен ломать формат")
	}

	extraKey := `{"recipe":"рецепт","extra":1}`
	if _, ok := parseRecipe(extraKey); ok {
		t.Error("лишний ключ должен ломать формат")
	}

	nonString := `{"recipe":1}`
	if _, ok := parseRecipe(nonString); ok {
		t.Error("нестроковый recipe должен ломать формат")
	}

	empty := `{"recipe":"   "}`
	if _, ok := parseRecipe(empty); ok {
		t.Error("пустой recipe должен ломать формат")
	}
}

func TestSimilarityAndUnique(t *testing.T) {
	if got := avgSimilarity([]string{"идея про солнце", "идея про солнце"}); got != 1 {
		t.Errorf("same similarity = %v", got)
	}
	if got := avgSimilarity([]string{"солнце вода", "робот лес"}); got != 0 {
		t.Errorf("different similarity = %v", got)
	}
	if got := uniqueCount([]string{"a", "a", "b"}); got != 2 {
		t.Errorf("unique = %d", got)
	}
}

func TestCallSendsTemperatureAndDisablesThinking(t *testing.T) {
	var gotTemp float64
	var gotThinking string

	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req apiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		if req.Temperature == nil {
			t.Errorf("temperature не передан")
		} else {
			gotTemp = *req.Temperature
		}
		if req.Thinking != nil {
			gotThinking = req.Thinking.Type
		}
		mockCompletion(w, "44")
	})

	if _, err := s.llm.Call(context.Background(), "test", 0); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if gotTemp != 0 {
		t.Errorf("temperature = %v, ожидался 0", gotTemp)
	}
	if gotThinking != "disabled" {
		t.Errorf("thinking = %q", gotThinking)
	}
}

func TestExperimentFlow(t *testing.T) {
	var mu sync.Mutex
	temps := map[float64]bool{}
	thinkingDisabled := true

	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req apiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		mu.Lock()
		if req.Temperature != nil {
			temps[*req.Temperature] = true
		}
		if req.Thinking == nil || req.Thinking.Type != "disabled" {
			thinkingDisabled = false
		}
		mu.Unlock()

		prompt := ""
		if len(req.Messages) > 0 {
			prompt = req.Messages[len(req.Messages)-1].Content
		}
		if strings.Contains(prompt, "лошад") {
			mockCompletion(w, "7")
			return
		}
		mockCompletion(w, `{"recipe":"хлеб, авокадо, яйцо"}`)
	})

	e, err := s.startExperiment(experimentRequest{Runs: 3, Prompts: defaultPrompts})
	if err != nil {
		t.Fatalf("startExperiment: %v", err)
	}

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		got, ok := s.store.GetExperiment(e.ID)
		if !ok {
			t.Fatal("эксперимент не найден")
		}
		if got.Status == "done" {
			if len(got.Results) != 18 {
				t.Fatalf("results = %d, ожидалось 18", len(got.Results))
			}
			if len(got.Cells) != 6 {
				t.Fatalf("cells = %d, ожидалось 6", len(got.Cells))
			}
			if len(got.Findings) == 0 {
				t.Error("нет выводов")
			}

			for _, cell := range got.Cells {
				if cell.Runs != 3 {
					t.Errorf("%s/%v: runs = %d", cell.TaskType, cell.Temperature, cell.Runs)
				}
				if cell.FormatValid != 3 {
					t.Errorf("%s/%v: format = %d", cell.TaskType, cell.Temperature, cell.FormatValid)
				}
				if cell.TaskType == "logic" && cell.Correct != 3 {
					t.Errorf("logic/%v: correct = %d", cell.Temperature, cell.Correct)
				}
			}

			mu.Lock()
			for _, temp := range temperatures {
				if !temps[temp] {
					t.Errorf("temperature %v не отправлялась", temp)
				}
			}
			if !thinkingDisabled {
				t.Error("thinking не был disabled")
			}
			mu.Unlock()
			return
		}
		if got.Status == "error" {
			t.Fatalf("experiment error: %s", got.Error)
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatal("эксперимент не завершился за 10 секунд")
}

func TestPromptsFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.json")
	if err := savePrompts(path, defaultPrompts); err != nil {
		t.Fatalf("savePrompts: %v", err)
	}
	got, err := loadPrompts(path)
	if err != nil {
		t.Fatalf("loadPrompts: %v", err)
	}
	if got.LogicAnswer != 7 || got.GenerationPrompt == "" {
		t.Errorf("prompts = %+v", got)
	}

	bad := defaultPrompts
	bad.LogicPrompt = ""
	if err := validatePrompts(&bad); err == nil {
		t.Error("ожидалась ошибка для пустого промпта")
	}
}

func TestSandbox(t *testing.T) {
	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		mockCompletion(w, "ответ")
	})

	results := s.runSandbox(context.Background(), "test", 0.7, 2)
	if len(results) != 2 {
		t.Fatalf("results = %d", len(results))
	}
	for _, r := range results {
		if r.Content != "ответ" || r.Usage.TotalTokens != 15 {
			t.Errorf("result = %+v", r)
		}
	}
}

func TestStoreExperiments(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := &Experiment{ID: "x", Status: "done", CreatedAt: time.Now(), Runs: 3}
	if err := store.SaveExperiment(e); err != nil {
		t.Fatal(err)
	}
	got, ok := store.GetExperiment("x")
	if !ok || got.Runs != 3 {
		t.Fatalf("got = %+v", got)
	}
	if len(store.ListExperiments()) != 1 {
		t.Error("list experiments")
	}
	_ = os.Remove("nonexistent")
}
