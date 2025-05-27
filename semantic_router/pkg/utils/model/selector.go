package model

import (
	"log"
	"strings"
	"sync"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
)

// Selector handles model selection logic
type Selector struct {
	Config        *config.RouterConfig
	ModelLoad     map[string]int
	ModelLoadLock sync.Mutex
	ModelTTFT     map[string]float64
}

// NewSelector creates a new model selector
func NewSelector(cfg *config.RouterConfig, modelTTFT map[string]float64) *Selector {
	return &Selector{
		Config:    cfg,
		ModelLoad: make(map[string]int),
		ModelTTFT: modelTTFT,
	}
}

// SelectBestModelForCategory selects the best model from a category based on score and TTFT
func (s *Selector) SelectBestModelForCategory(categoryName string) string {
	var cat *config.Category
	for i, category := range s.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &s.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		log.Printf("Could not find matching category %s in config, using default model", categoryName)
		return s.Config.DefaultModel
	}

	s.ModelLoadLock.Lock()
	defer s.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	if s.Config.Classifier.LoadAware {
		// Load-aware: combine accuracy and TTFT
		for _, modelScore := range cat.ModelScores {
			quality := modelScore.Score
			model := modelScore.Model

			baseTTFT := s.ModelTTFT[model]
			load := s.ModelLoad[model]
			estTTFT := baseTTFT * (1 + float64(load))
			if estTTFT == 0 {
				estTTFT = 1 // avoid div by zero
			}
			score := quality / estTTFT
			if score > bestScore {
				bestScore = score
				bestModel = model
				bestQuality = quality
			}
		}
	} else {
		// Not load-aware: pick the model with the highest accuracy only
		for _, modelScore := range cat.ModelScores {
			quality := modelScore.Score
			model := modelScore.Model
			if quality > bestScore {
				bestScore = quality
				bestModel = model
				bestQuality = quality
			}
		}
	}

	if bestModel == "" {
		log.Printf("No models found for category %s, using default model", categoryName)
		return s.Config.DefaultModel
	}

	log.Printf("Selected model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// SelectBestModelFromList selects the best model from a list of candidate models for a given category
func (s *Selector) SelectBestModelFromList(candidateModels []string, categoryName string) string {
	if len(candidateModels) == 0 {
		return s.Config.DefaultModel
	}

	// Find the category configuration
	var cat *config.Category
	for i, category := range s.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &s.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		// Return first candidate if category not found
		return candidateModels[0]
	}

	s.ModelLoadLock.Lock()
	defer s.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	if s.Config.Classifier.LoadAware {
		// Load-aware: combine accuracy and TTFT
		for _, modelScore := range cat.ModelScores {
			model := modelScore.Model

			// Check if this model is in the candidate list
			if !contains(candidateModels, model) {
				continue
			}

			quality := modelScore.Score
			baseTTFT := s.ModelTTFT[model]
			load := s.ModelLoad[model]
			estTTFT := baseTTFT * (1 + float64(load))
			if estTTFT == 0 {
				estTTFT = 1 // avoid div by zero
			}
			score := quality / estTTFT
			if score > bestScore {
				bestScore = score
				bestModel = model
				bestQuality = quality
			}
		}
	} else {
		// Not load-aware: pick the model with the highest accuracy only
		for _, modelScore := range cat.ModelScores {
			model := modelScore.Model

			// Check if this model is in the candidate list
			if !contains(candidateModels, model) {
				continue
			}

			quality := modelScore.Score
			if quality > bestScore {
				bestScore = quality
				bestModel = model
				bestQuality = quality
			}
		}
	}

	if bestModel == "" {
		log.Printf("No suitable model found from candidates for category %s, using first candidate", categoryName)
		return candidateModels[0]
	}

	log.Printf("Selected best model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// GetModelsForCategory returns all models that are configured for the given category
func (s *Selector) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, category := range s.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			for _, modelScore := range category.ModelScores {
				models = append(models, modelScore.Model)
			}
			break
		}
	}

	return models
}

// IncrementModelLoad increments the load counter for a model
func (s *Selector) IncrementModelLoad(model string) {
	s.ModelLoadLock.Lock()
	defer s.ModelLoadLock.Unlock()
	s.ModelLoad[model]++
}

// DecrementModelLoad decrements the load counter for a model
func (s *Selector) DecrementModelLoad(model string) {
	s.ModelLoadLock.Lock()
	defer s.ModelLoadLock.Unlock()
	if s.ModelLoad[model] > 0 {
		s.ModelLoad[model]--
	}
}

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
