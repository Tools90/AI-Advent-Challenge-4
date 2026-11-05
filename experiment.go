package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

type PromptPreset struct {
	LogicPrompt      string `json:"logicPrompt"`
	LogicAnswer      int    `json:"logicAnswer"`
	GenerationPrompt string `json:"generationPrompt"`
}

var defaultPrompts = PromptPreset{
	LogicPrompt:      "Есть 25 лошадей и ипподром с 5 дорожками. За один забег можно проверить 5 лошадей; секундомера нет, известен только порядок финиша. Какое минимальное число забегов нужно, чтобы гарантированно определить трёх самых быстрых лошадей в правильном порядке? Ответь строго одним числом, без пояснений.",
	LogicAnswer:      7,
	GenerationPrompt: "Придумай рецепт для нового бутерброда. Ответь строго JSON-объектом вида {\"recipe\":\"рецепт\"}. Без пояснений, markdown и текста вокруг.",
}

var temperatures = []float64{0, 0.7, 1.2}
var taskTypes = []string{"logic", "generation"}

const defaultRuns = 5

type RunResult struct {
	ID          string  `json:"id"`
	TaskType    string  `json:"taskType"`
	Temperature float64 `json:"temperature"`
	Index       int     `json:"index"`
	Content     string  `json:"content"`
	Extracted   string  `json:"extracted,omitempty"`
	FormatValid bool    `json:"formatValid"`
	Correct     bool    `json:"correct"`
	Usage       Usage   `json:"usage"`
	LatencyMs   int64   `json:"latencyMs"`
	Error       string  `json:"error,omitempty"`
}

type CellSummary struct {
	TaskType      string  `json:"taskType"`
	Temperature   float64 `json:"temperature"`
	Runs          int     `json:"runs"`
	Correct       int     `json:"correct"`
	FormatValid   int     `json:"formatValid"`
	Unique        int     `json:"unique"`
	AvgSimilarity float64 `json:"avgSimilarity"`
	AvgTokens     int     `json:"avgTokens"`
	AvgLatencyMs  int64   `json:"avgLatencyMs"`
	Errors        int     `json:"errors"`
}

type ExperimentProgress struct {
	Total     int    `json:"total"`
	Completed int    `json:"completed"`
	Phase     string `json:"phase"`
}

type Experiment struct {
	ID         string             `json:"id"`
	Status     string             `json:"status"`
	CreatedAt  time.Time          `json:"createdAt"`
	FinishedAt *time.Time         `json:"finishedAt,omitempty"`
	Runs       int                `json:"runs"`
	Prompts    PromptPreset       `json:"prompts"`
	Results    []RunResult        `json:"results"`
	Cells      []CellSummary      `json:"cells"`
	Findings   []string           `json:"findings"`
	Progress   ExperimentProgress `json:"progress"`
	Error      string             `json:"error,omitempty"`
}

type experimentRequest struct {
	Runs    int          `json:"runs"`
	Prompts PromptPreset `json:"prompts"`
}

func loadPrompts(path string) (PromptPreset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return defaultPrompts, nil
		}
		return PromptPreset{}, fmt.Errorf("чтение %s: %w", path, err)
	}

	var p PromptPreset
	if err := json.Unmarshal(data, &p); err != nil {
		return PromptPreset{}, fmt.Errorf("разбор %s: %w", path, err)
	}
	if err := validatePrompts(&p); err != nil {
		return PromptPreset{}, err
	}
	return p, nil
}

func validatePrompts(p *PromptPreset) error {
	p.LogicPrompt = strings.TrimSpace(p.LogicPrompt)
	p.GenerationPrompt = strings.TrimSpace(p.GenerationPrompt)

	if p.LogicPrompt == "" {
		return errors.New("промпт логической задачи не может быть пустым")
	}
	if p.GenerationPrompt == "" {
		return errors.New("промпт генерации не может быть пустым")
	}
	if len([]rune(p.LogicPrompt)) > 4000 || len([]rune(p.GenerationPrompt)) > 4000 {
		return errors.New("промпт слишком длинный (максимум 4000 символов)")
	}
	return nil
}

func savePrompts(path string, p PromptPreset) error {
	if err := validatePrompts(&p); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(path)
	return os.Rename(tmp, path)
}

func (s *server) startExperiment(req experimentRequest) (*Experiment, error) {
	prompts := req.Prompts
	if prompts.LogicPrompt == "" && prompts.GenerationPrompt == "" {
		loaded, err := loadPrompts(s.promptsPath)
		if err != nil {
			return nil, err
		}
		prompts = loaded
	}
	if err := validatePrompts(&prompts); err != nil {
		return nil, err
	}

	runs := req.Runs
	if runs == 0 {
		runs = defaultRuns
	}
	if runs < 3 || runs > 10 {
		return nil, errors.New("число прогонов: от 3 до 10")
	}

	total := len(taskTypes) * len(temperatures) * runs
	e := &Experiment{
		ID:        newID(),
		Status:    "running",
		CreatedAt: time.Now(),
		Runs:      runs,
		Prompts:   prompts,
		Results:   []RunResult{},
		Progress: ExperimentProgress{
			Total: total,
			Phase: "Запросы",
		},
	}

	if err := s.store.SaveExperiment(e); err != nil {
		return nil, err
	}

	started := clone(e)
	go s.runExperiment(e)
	return started, nil
}

func (s *server) runExperiment(e *Experiment) {
	var mu sync.Mutex

	defer func() {
		if r := recover(); r != nil {
			mu.Lock()
			e.Status = "error"
			e.Error = fmt.Sprintf("внутренняя ошибка: %v", r)
			now := time.Now()
			e.FinishedAt = &now
			_ = s.store.SaveExperiment(e)
			mu.Unlock()
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	var wg sync.WaitGroup
	for _, taskType := range taskTypes {
		for _, temperature := range temperatures {
			wg.Add(1)
			go func(taskType string, temperature float64) {
				defer wg.Done()

				prompt := e.Prompts.LogicPrompt
				if taskType == "generation" {
					prompt = e.Prompts.GenerationPrompt
				}

				for i := 0; i < e.Runs; i++ {
					if ctx.Err() != nil {
						return
					}

					start := time.Now()
					call, err := s.llm.Call(ctx, prompt, temperature)
					latency := time.Since(start).Milliseconds()

					run := RunResult{
						ID:          newID(),
						TaskType:    taskType,
						Temperature: temperature,
						Index:       i + 1,
						LatencyMs:   latency,
					}

					if err != nil {
						run.Error = err.Error()
					} else {
						run.Content = call.Content
						run.Usage = call.Usage
						run.Extracted, run.FormatValid, run.Correct = analyzeRun(taskType, call.Content, e.Prompts)
					}

					mu.Lock()
					e.Results = append(e.Results, run)
					e.Progress.Completed++
					_ = s.store.SaveExperiment(e)
					mu.Unlock()
				}
			}(taskType, temperature)
		}
	}
	wg.Wait()

	mu.Lock()
	e.Cells = aggregate(e)
	e.Findings = buildFindings(e)
	e.Status = "done"
	e.Progress.Phase = "Готово"
	now := time.Now()
	e.FinishedAt = &now
	_ = s.store.SaveExperiment(e)
	mu.Unlock()
}

var (
	logicFormatRe = regexp.MustCompile(`^\s*-?\d+\s*$`)
	lastIntRe     = regexp.MustCompile(`-?\d+`)
)

func lastInt(s string) string {
	matches := lastIntRe.FindAllString(s, -1)
	if len(matches) == 0 {
		return ""
	}
	return matches[len(matches)-1]
}

func analyzeLogic(content string, expected int) (extracted string, formatValid, correct bool) {
	extracted = lastInt(content)
	formatValid = logicFormatRe.MatchString(content)
	if extracted != "" {
		if n, err := strconv.Atoi(extracted); err == nil {
			correct = n == expected
		}
	}
	return
}

func analyzeRun(taskType, content string, prompts PromptPreset) (extracted string, formatValid, correct bool) {
	switch taskType {
	case "logic":
		extracted, formatValid, correct = analyzeLogic(content, prompts.LogicAnswer)
	case "generation":
		recipe, ok := parseRecipe(content)
		extracted = recipe
		formatValid = ok
	}
	return
}

func parseRecipe(content string) (string, bool) {
	trimmed := strings.TrimSpace(content)

	var obj map[string]any
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		return "", false
	}
	if len(obj) != 1 {
		return "", false
	}
	raw, ok := obj["recipe"]
	if !ok {
		return "", false
	}
	recipe, ok := raw.(string)
	if !ok || strings.TrimSpace(recipe) == "" {
		return "", false
	}
	return strings.TrimSpace(recipe), true
}

func aggregate(e *Experiment) []CellSummary {
	cells := make([]CellSummary, 0, len(taskTypes)*len(temperatures))

	for _, taskType := range taskTypes {
		for _, temperature := range temperatures {
			cell := CellSummary{TaskType: taskType, Temperature: temperature}
			texts := make([]string, 0, e.Runs)

			for _, r := range e.Results {
				if r.TaskType != taskType || r.Temperature != temperature {
					continue
				}
				cell.Runs++
				if r.Error != "" {
					cell.Errors++
					continue
				}
				if r.FormatValid {
					cell.FormatValid++
				}
				if taskType == "logic" {
					if r.Correct {
						cell.Correct++
					}
					text := r.Extracted
					if text == "" {
						text = normalizeText(r.Content)
					}
					texts = append(texts, text)
				} else {
					texts = append(texts, normalizeText(r.Content))
				}
				cell.AvgTokens += r.Usage.TotalTokens
				cell.AvgLatencyMs += r.LatencyMs
			}

			success := len(texts)
			if success > 0 {
				cell.AvgTokens /= success
				cell.AvgLatencyMs /= int64(success)
			}
			cell.Unique = uniqueCount(texts)
			cell.AvgSimilarity = avgSimilarity(texts)

			cells = append(cells, cell)
		}
	}

	return cells
}

func normalizeText(s string) string {
	var b strings.Builder
	lastSpace := true

	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastSpace = false
			continue
		}
		if !lastSpace {
			b.WriteByte(' ')
			lastSpace = true
		}
	}

	return strings.TrimSpace(b.String())
}

func uniqueCount(texts []string) int {
	seen := map[string]bool{}
	for _, t := range texts {
		if t != "" {
			seen[t] = true
		}
	}
	return len(seen)
}

func wordSet(text string) map[string]bool {
	set := map[string]bool{}
	for _, word := range strings.Fields(text) {
		if word != "" {
			set[word] = true
		}
	}
	return set
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for word := range a {
		if b[word] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 1
	}
	return float64(inter) / float64(union)
}

func avgSimilarity(texts []string) float64 {
	if len(texts) < 2 {
		return 0
	}
	sets := make([]map[string]bool, len(texts))
	for i, text := range texts {
		sets[i] = wordSet(text)
	}

	var sum float64
	var pairs int
	for i := 0; i < len(sets); i++ {
		for j := i + 1; j < len(sets); j++ {
			sum += jaccard(sets[i], sets[j])
			pairs++
		}
	}
	if pairs == 0 {
		return 0
	}
	return sum / float64(pairs)
}

func cellFor(cells []CellSummary, taskType string, temperature float64) (CellSummary, bool) {
	for _, c := range cells {
		if c.TaskType == taskType && c.Temperature == temperature {
			return c, true
		}
	}
	return CellSummary{}, false
}

func buildFindings(e *Experiment) []string {
	out := make([]string, 0, 12)

	for _, taskType := range taskTypes {
		for _, temperature := range temperatures {
			c, ok := cellFor(e.Cells, taskType, temperature)
			if !ok || c.Runs == 0 {
				continue
			}
			if taskType == "logic" {
				out = append(out, fmt.Sprintf(
					"temperature=%s: логика — верно %d/%d, формат соблюдён %d/%d, уникальных ответов %d/%d.",
					fmtTemp(temperature), c.Correct, c.Runs, c.FormatValid, c.Runs, c.Unique, c.Runs))
			} else {
				out = append(out, fmt.Sprintf(
					"temperature=%s: генерация — формат %d/%d, уникальных рецептов %d/%d, средняя похожесть %.2f.",
					fmtTemp(temperature), c.FormatValid, c.Runs, c.Unique, c.Runs, c.AvgSimilarity))
			}
		}
	}

	for _, taskType := range taskTypes {
		c, ok := cellFor(e.Cells, taskType, 0)
		if !ok || c.Runs == 0 {
			continue
		}
		name := "логики"
		if taskType == "generation" {
			name = "генерации"
		}
		if c.Unique == 1 {
			out = append(out, fmt.Sprintf("temperature=0: все %d ответов в %s совпали.", c.Runs, name))
		} else {
			out = append(out, fmt.Sprintf("temperature=0: в %s %d уникальных из %d — полного детерминизма нет.", name, c.Unique, c.Runs))
		}
	}

	if gen0, ok0 := cellFor(e.Cells, "generation", 0); ok0 {
		if gen12, ok12 := cellFor(e.Cells, "generation", 1.2); ok12 {
			switch {
			case gen12.AvgSimilarity < gen0.AvgSimilarity-0.02:
				out = append(out, fmt.Sprintf(
					"Разнообразие полезно для генерации: с ростом температуры похожесть падает %.2f → %.2f, уникальных рецептов %d → %d.",
					gen0.AvgSimilarity, gen12.AvgSimilarity, gen0.Unique, gen12.Unique))
			case gen12.AvgSimilarity > gen0.AvgSimilarity+0.02:
				out = append(out, fmt.Sprintf(
					"Разнообразие для генерации не растёт: похожесть даже выросла %.2f → %.2f.",
					gen0.AvgSimilarity, gen12.AvgSimilarity))
			default:
				out = append(out, fmt.Sprintf(
					"Разнообразие для генерации почти не меняется: похожесть %.2f → %.2f.",
					gen0.AvgSimilarity, gen12.AvgSimilarity))
			}
		}
	}

	if logic0, ok0 := cellFor(e.Cells, "logic", 0); ok0 {
		if logic12, ok12 := cellFor(e.Cells, "logic", 1.2); ok12 {
			if logic12.FormatValid < logic0.FormatValid || logic12.Correct < logic0.Correct {
				out = append(out, fmt.Sprintf(
					"Для логики разнообразие вредит: формат %d/%d против %d/%d, верно %d/%d против %d/%d при temperature=1.2 и 0.",
					logic12.FormatValid, logic12.Runs, logic0.FormatValid, logic0.Runs,
					logic12.Correct, logic12.Runs, logic0.Correct, logic0.Runs))
			} else {
				out = append(out, "Для логики разнообразие не ухудшило формат и правильность.")
			}
		}
	}

	if gen12, ok := cellFor(e.Cells, "generation", 1.2); ok && gen12.Runs > 0 {
		if gen12.FormatValid < gen12.Runs {
			out = append(out, fmt.Sprintf(
				"Генерация: высокая температура ломает формат — %d из %d ответов не являются JSON-объектом {\"recipe\":\"...\"}.",
				gen12.Runs-gen12.FormatValid, gen12.Runs))
		}
	}

	for _, temperature := range temperatures {
		logic, hasLogic := cellFor(e.Cells, "logic", temperature)
		generation, hasGeneration := cellFor(e.Cells, "generation", temperature)

		var parts []string
		if hasLogic && logic.Runs > 0 {
			if logic.Correct == logic.Runs && logic.FormatValid == logic.Runs {
				parts = append(parts, fmt.Sprintf("для логики подходит (верно %d/%d, формат %d/%d)", logic.Correct, logic.Runs, logic.FormatValid, logic.Runs))
			} else {
				parts = append(parts, fmt.Sprintf("для логики рискованна (верно %d/%d, формат %d/%d)", logic.Correct, logic.Runs, logic.FormatValid, logic.Runs))
			}
		}
		if hasGeneration && generation.Runs > 0 {
			parts = append(parts, fmt.Sprintf("для генерации — %d уникальных рецептов из %d, формат %d/%d",
				generation.Unique, generation.Runs, generation.FormatValid, generation.Runs))
		}
		if len(parts) > 0 {
			out = append(out, fmt.Sprintf("temperature=%s: %s.", fmtTemp(temperature), strings.Join(parts, "; ")))
		}
	}

	out = append(out, recommendation(e))
	return out
}

func recommendation(e *Experiment) string {
	logicTemp := 0.0
	logicScore := -1
	for _, temperature := range temperatures {
		c, ok := cellFor(e.Cells, "logic", temperature)
		if !ok {
			continue
		}
		score := c.Correct*100 + c.FormatValid
		if score > logicScore {
			logicScore = score
			logicTemp = temperature
		}
	}

	genTemp := 0.0
	genScore := -1
	for _, temperature := range temperatures {
		c, ok := cellFor(e.Cells, "generation", temperature)
		if !ok || c.Runs == 0 {
			continue
		}
		formatRate := float64(c.FormatValid) / float64(c.Runs)
		score := c.Unique*10 + int(formatRate*100)
		if score > genScore {
			genScore = score
			genTemp = temperature
		}
	}

	return fmt.Sprintf(
		"Рекомендация: для задачи с одним ответом — temperature=%s (важнее стабильность и формат); для генерации — temperature=%s (больше уникальных рецептов при приемлемом формате).",
		fmtTemp(logicTemp), fmtTemp(genTemp))
}

func fmtTemp(t float64) string {
	if t == 0 {
		return "0"
	}
	return strconv.FormatFloat(t, 'f', -1, 64)
}

type SandboxResult struct {
	Index     int    `json:"index"`
	Content   string `json:"content"`
	Usage     Usage  `json:"usage"`
	LatencyMs int64  `json:"latencyMs"`
	Error     string `json:"error,omitempty"`
}

func (s *server) runSandbox(ctx context.Context, prompt string, temperature float64, runs int) []SandboxResult {
	results := make([]SandboxResult, 0, runs)

	for i := 0; i < runs; i++ {
		start := time.Now()
		call, err := s.llm.Call(ctx, prompt, temperature)
		latency := time.Since(start).Milliseconds()

		result := SandboxResult{Index: i + 1, LatencyMs: latency}
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Content = call.Content
			result.Usage = call.Usage
		}
		results = append(results, result)
	}

	return results
}
