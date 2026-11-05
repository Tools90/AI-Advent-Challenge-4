package main

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type Store struct {
	mu          sync.RWMutex
	path        string
	experiments map[string]*Experiment
}

func NewStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создание %s: %w", dir, err)
	}

	s := &Store{
		path:        filepath.Join(dir, "experiments.json"),
		experiments: map[string]*Experiment{},
	}

	data, err := os.ReadFile(s.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("чтение %s: %w", s.path, err)
		}
	} else if len(data) > 0 {
		if err := json.Unmarshal(data, &s.experiments); err != nil {
			return nil, fmt.Errorf("разбор %s: %w", s.path, err)
		}
	}
	if s.experiments == nil {
		s.experiments = map[string]*Experiment{}
	}
	return s, nil
}

func (s *Store) ListExperiments() []*Experiment {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]*Experiment, 0, len(s.experiments))
	for _, e := range s.experiments {
		out = append(out, clone(e))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

func (s *Store) GetExperiment(id string) (*Experiment, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	e, ok := s.experiments[id]
	if !ok {
		return nil, false
	}
	return clone(e), true
}

func (s *Store) SaveExperiment(e *Experiment) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.experiments[e.ID] = clone(e)
	return s.saveLocked()
}

func (s *Store) saveLocked() error {
	data, err := json.MarshalIndent(s.experiments, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(s.path)
	return os.Rename(tmp, s.path)
}

func newID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", buf)
}

func clone[T any](v T) T {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		return v
	}
	return out
}
