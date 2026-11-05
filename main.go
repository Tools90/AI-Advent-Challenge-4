package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	apiKey  string
	baseURL string
	model   string
	addr    string
}

type server struct {
	cfg         config
	llm         *DeepSeek
	store       *Store
	promptsPath string
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}

	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	store, err := NewStore("data")
	if err != nil {
		log.Fatalf("store: %v", err)
	}

	if _, err := loadPrompts("prompts.json"); err != nil {
		log.Printf("prompts: %v", err)
	}

	s := &server{
		cfg:         cfg,
		llm:         NewDeepSeek(cfg.baseURL, cfg.apiKey, cfg.model),
		store:       store,
		promptsPath: "prompts.json",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /api/presets", s.presets)
	mux.HandleFunc("GET /api/prompts", s.getPrompts)
	mux.HandleFunc("PUT /api/prompts", s.putPrompts)
	mux.HandleFunc("GET /api/experiments", s.listExperiments)
	mux.HandleFunc("POST /api/experiments", s.createExperiment)
	mux.HandleFunc("GET /api/experiments/{id}", s.getExperiment)
	mux.HandleFunc("POST /api/sandbox", s.sandbox)

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("останавливаюсь...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": s.cfg.model})
}

func (s *server) presets(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"temperatures": temperatures,
		"defaultRuns":  defaultRuns,
		"model":        s.cfg.model,
		"thinking":     "disabled",
	})
}

func (s *server) getPrompts(w http.ResponseWriter, r *http.Request) {
	p, err := loadPrompts(s.promptsPath)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, p)
}

func (s *server) putPrompts(w http.ResponseWriter, r *http.Request) {
	var p PromptPreset
	if err := decodeJSON(w, r, &p); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validatePrompts(&p); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := savePrompts(s.promptsPath, p); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, p)
}

func (s *server) listExperiments(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.store.ListExperiments())
}

func (s *server) createExperiment(w http.ResponseWriter, r *http.Request) {
	var req experimentRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	e, err := s.startExperiment(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, e)
}

func (s *server) getExperiment(w http.ResponseWriter, r *http.Request) {
	e, ok := s.store.GetExperiment(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "эксперимент не найден")
		return
	}
	writeJSON(w, http.StatusOK, e)
}

func (s *server) sandbox(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt      string  `json:"prompt"`
		Temperature float64 `json:"temperature"`
		Runs        int     `json:"runs"`
	}
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	req.Prompt = strings.TrimSpace(req.Prompt)
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "промпт не может быть пустым")
		return
	}
	if len([]rune(req.Prompt)) > 4000 {
		writeError(w, http.StatusBadRequest, "промпт слишком длинный (максимум 4000 символов)")
		return
	}
	if req.Temperature < 0 || req.Temperature > 2 {
		writeError(w, http.StatusBadRequest, "temperature: от 0 до 2")
		return
	}
	if req.Runs == 0 {
		req.Runs = 1
	}
	if req.Runs < 1 || req.Runs > 5 {
		writeError(w, http.StatusBadRequest, "число прогонов: от 1 до 5")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	writeJSON(w, http.StatusOK, map[string]any{
		"temperature": req.Temperature,
		"runs":        req.Runs,
		"results":     s.runSandbox(ctx, req.Prompt, req.Temperature, req.Runs),
	})
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return fmt.Errorf("некорректный JSON: %w", err)
	}
	return nil
}

func loadConfig() (config, error) {
	apiKey, err := loadAPIKey()
	if err != nil {
		return config{}, err
	}

	baseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("DEEPSEEK_BASE_URL")), "/")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}

	model := strings.TrimSpace(os.Getenv("DEEPSEEK_MODEL"))
	if model == "" {
		model = "deepseek-v4-flash"
	}

	addr := strings.TrimSpace(os.Getenv("ADDR"))
	if port := strings.TrimSpace(os.Getenv("PORT")); addr == "" && port != "" {
		addr = ":" + port
	}
	if addr == "" {
		addr = ":8083"
	}

	return config{apiKey: apiKey, baseURL: baseURL, model: model, addr: addr}, nil
}

const placeholderKey = "sk-your-key-here"

func loadAPIKey() (string, error) {
	if key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY")); key != "" && key != placeholderKey {
		return key, nil
	}

	path := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY_FILE"))
	if path == "" {
		path = "api_key.txt"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("ключ API не найден: создайте файл %s и вставьте в него ключ DeepSeek", path)
		}
		return "", fmt.Errorf("чтение %s: %w", path, err)
	}

	key := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if key == "" || key == placeholderKey {
		return "", fmt.Errorf("файл %s пуст: вставьте в него ключ DeepSeek", path)
	}
	return key, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, использую переменные окружения", path)
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
