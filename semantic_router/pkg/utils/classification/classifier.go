package classification

import (
	"fmt"
	"log"
	"strings"
	"sync"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/metrics"
)

// JailbreakDetection represents the result of jailbreak analysis for a piece of content
type JailbreakDetection struct {
	Content       string  `json:"content"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	ContentIndex  int     `json:"content_index"`
}

// Classifier handles text classification, model selection, and jailbreak detection functionality
type Classifier struct {
	Config           *config.RouterConfig
	CategoryMapping  *CategoryMapping
	PIIMapping       *PIIMapping
	JailbreakMapping *JailbreakMapping
	// Model selection fields
	ModelLoad     map[string]int
	ModelLoadLock sync.Mutex
	ModelTTFT     map[string]float64
	// Jailbreak detection state
	JailbreakInitialized bool
}

// NewClassifier creates a new classifier with model selection and jailbreak detection capabilities
func NewClassifier(cfg *config.RouterConfig, categoryMapping *CategoryMapping, piiMapping *PIIMapping, jailbreakMapping *JailbreakMapping, modelTTFT map[string]float64) *Classifier {
	return &Classifier{
		Config:               cfg,
		CategoryMapping:      categoryMapping,
		PIIMapping:           piiMapping,
		JailbreakMapping:     jailbreakMapping,
		ModelLoad:            make(map[string]int),
		ModelTTFT:            modelTTFT,
		JailbreakInitialized: false,
	}
}

// InitializeJailbreakClassifier initializes the jailbreak classification model
func (c *Classifier) InitializeJailbreakClassifier() error {
	if !c.IsJailbreakEnabled() {
		return nil
	}

	numClasses := c.JailbreakMapping.GetJailbreakTypeCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough jailbreak types for classification, need at least 2, got %d", numClasses)
	}

	err := candle_binding.InitJailbreakClassifier(c.Config.PromptGuard.ModelID, numClasses, c.Config.PromptGuard.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
	}

	c.JailbreakInitialized = true
	log.Printf("Initialized jailbreak classifier with %d classes", numClasses)
	return nil
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured
func (c *Classifier) IsJailbreakEnabled() bool {
	return c.Config.PromptGuard.Enabled && c.Config.PromptGuard.ModelID != "" && c.Config.PromptGuard.JailbreakMappingPath != "" && c.JailbreakMapping != nil
}

// CheckForJailbreak analyzes the given text for jailbreak attempts
func (c *Classifier) CheckForJailbreak(text string) (bool, string, float32, error) {
	if !c.IsJailbreakEnabled() || !c.JailbreakInitialized {
		return false, "", 0.0, nil
	}

	if text == "" {
		return false, "", 0.0, nil
	}

	// Classify the text for jailbreak detection
	result, err := candle_binding.ClassifyJailbreakText(text)
	if err != nil {
		return false, "", 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}
	log.Printf("Jailbreak classification result: %v", result)

	// Get the jailbreak type name from the class index
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	// Check if confidence meets threshold and indicates jailbreak
	isJailbreak := result.Confidence >= c.Config.PromptGuard.Threshold && jailbreakType == "jailbreak"

	if isJailbreak {
		log.Printf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, c.Config.PromptGuard.Threshold)
	} else {
		log.Printf("BENIGN: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, c.Config.PromptGuard.Threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, nil
}

// AnalyzeContentForJailbreak analyzes multiple content pieces for jailbreak attempts
func (c *Classifier) AnalyzeContentForJailbreak(contentList []string) (bool, []JailbreakDetection, error) {
	if !c.IsJailbreakEnabled() || !c.JailbreakInitialized {
		return false, nil, nil
	}

	var detections []JailbreakDetection
	hasJailbreak := false

	for i, content := range contentList {
		if content == "" {
			continue
		}

		isJailbreak, jailbreakType, confidence, err := c.CheckForJailbreak(content)
		if err != nil {
			log.Printf("Error analyzing content %d: %v", i, err)
			continue
		}

		detection := JailbreakDetection{
			Content:       content,
			IsJailbreak:   isJailbreak,
			JailbreakType: jailbreakType,
			Confidence:    confidence,
			ContentIndex:  i,
		}

		detections = append(detections, detection)

		if isJailbreak {
			hasJailbreak = true
		}
	}

	return hasJailbreak, detections, nil
}

// ClassifyCategory performs category classification on the given text
func (c *Classifier) ClassifyCategory(text string) (string, float64, error) {
	if c.CategoryMapping == nil {
		return "", 0.0, fmt.Errorf("category mapping not initialized")
	}

	// Use BERT classifier to get the category index and confidence
	result, err := candle_binding.ClassifyText(text)
	if err != nil {
		return "", 0.0, fmt.Errorf("classification error: %w", err)
	}

	log.Printf("Classification result: class=%d, confidence=%.4f", result.Class, result.Confidence)

	// Check confidence threshold
	if result.Confidence < c.Config.Classifier.CategoryModel.Threshold {
		log.Printf("Classification confidence (%.4f) below threshold (%.4f)",
			result.Confidence, c.Config.Classifier.CategoryModel.Threshold)
		return "", float64(result.Confidence), nil
	}

	// Convert class index to category name
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		log.Printf("Class index %d not found in category mapping", result.Class)
		return "", float64(result.Confidence), nil
	}

	// Record the category classification metric
	metrics.RecordCategoryClassification(categoryName)

	log.Printf("Classified as category: %s", categoryName)
	return categoryName, float64(result.Confidence), nil
}

// ClassifyPII performs PII classification on the given text
func (c *Classifier) ClassifyPII(text string) (string, float64, error) {
	if c.PIIMapping == nil {
		return "NO_PII", 1.0, nil // No PII classifier enabled
	}

	// Use BERT PII classifier to get the PII type index and confidence
	result, err := candle_binding.ClassifyPIIText(text)
	if err != nil {
		return "", 0.0, fmt.Errorf("PII classification error: %w", err)
	}

	log.Printf("PII classification result: class=%d, confidence=%.4f", result.Class, result.Confidence)

	// Check confidence threshold
	if result.Confidence < c.Config.Classifier.PIIModel.Threshold {
		log.Printf("PII classification confidence (%.4f) below threshold (%.4f), assuming no PII",
			result.Confidence, c.Config.Classifier.PIIModel.Threshold)
		return "NO_PII", float64(result.Confidence), nil
	}

	// Convert class index to PII type name
	piiType, ok := c.PIIMapping.GetPIITypeFromIndex(result.Class)
	if !ok {
		log.Printf("PII class index %d not found in mapping, assuming no PII", result.Class)
		return "NO_PII", float64(result.Confidence), nil
	}

	log.Printf("Classified PII type: %s", piiType)
	return piiType, float64(result.Confidence), nil
}

// DetectPIIInContent performs PII classification on all provided content
func (c *Classifier) DetectPIIInContent(allContent []string) []string {
	var detectedPII []string

	for _, content := range allContent {
		if content != "" {
			//TODO: classifier may not handle the entire content, so we need to split the content into smaller chunks
			piiType, confidence, err := c.ClassifyPII(content)
			if err != nil {
				log.Printf("PII classification error: %v", err)
				// Continue without PII enforcement on error
			} else if piiType != "NO_PII" {
				log.Printf("Detected PII type '%s' with confidence %.4f in content", piiType, confidence)
				// Avoid duplicates
				found := false
				for _, existing := range detectedPII {
					if existing == piiType {
						found = true
						break
					}
				}
				if !found {
					detectedPII = append(detectedPII, piiType)
				}
			}
		}
	}

	return detectedPII
}

// ClassifyAndSelectBestModel performs classification and selects the best model for the query
func (c *Classifier) ClassifyAndSelectBestModel(query string) string {
	// If no categories defined, return default model
	if len(c.Config.Categories) == 0 {
		return c.Config.DefaultModel
	}

	// First, classify the text to determine the category
	categoryName, confidence, err := c.ClassifyCategory(query)
	if err != nil {
		log.Printf("Classification error: %v, falling back to default model", err)
		return c.Config.DefaultModel
	}

	if categoryName == "" {
		log.Printf("Classification confidence (%.4f) below threshold, using default model", confidence)
		return c.Config.DefaultModel
	}

	// Then select the best model from the determined category based on score and TTFT
	return c.SelectBestModelForCategory(categoryName)
}

// SelectBestModelForCategory selects the best model from a category based on score and TTFT
func (c *Classifier) SelectBestModelForCategory(categoryName string) string {
	var cat *config.Category
	for i, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &c.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		log.Printf("Could not find matching category %s in config, using default model", categoryName)
		return c.Config.DefaultModel
	}

	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	if c.Config.Classifier.LoadAware {
		// Load-aware: combine accuracy and TTFT
		for _, modelScore := range cat.ModelScores {
			quality := modelScore.Score
			model := modelScore.Model

			baseTTFT := c.ModelTTFT[model]
			load := c.ModelLoad[model]
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
		return c.Config.DefaultModel
	}

	log.Printf("Selected model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// SelectBestModelFromList selects the best model from a list of candidate models for a given category
func (c *Classifier) SelectBestModelFromList(candidateModels []string, categoryName string) string {
	if len(candidateModels) == 0 {
		return c.Config.DefaultModel
	}

	// Find the category configuration
	var cat *config.Category
	for i, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &c.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		// Return first candidate if category not found
		return candidateModels[0]
	}

	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	if c.Config.Classifier.LoadAware {
		// Load-aware: combine accuracy and TTFT
		for _, modelScore := range cat.ModelScores {
			model := modelScore.Model

			// Check if this model is in the candidate list
			if !c.contains(candidateModels, model) {
				continue
			}

			quality := modelScore.Score
			baseTTFT := c.ModelTTFT[model]
			load := c.ModelLoad[model]
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
			if !c.contains(candidateModels, model) {
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
func (c *Classifier) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, category := range c.Config.Categories {
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
func (c *Classifier) IncrementModelLoad(model string) {
	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()
	c.ModelLoad[model]++
}

// DecrementModelLoad decrements the load counter for a model
func (c *Classifier) DecrementModelLoad(model string) {
	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()
	if c.ModelLoad[model] > 0 {
		c.ModelLoad[model]--
	}
}

// contains checks if a slice contains a string
func (c *Classifier) contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
