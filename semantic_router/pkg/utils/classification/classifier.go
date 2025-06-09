package classification

import (
	"fmt"
	"log"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/metrics"
)

// Classifier handles text classification functionality
type Classifier struct {
	Config          *config.RouterConfig
	CategoryMapping *CategoryMapping
	PIIMapping      *PIIMapping
}

// NewClassifier creates a new classifier
func NewClassifier(cfg *config.RouterConfig, categoryMapping *CategoryMapping, piiMapping *PIIMapping) *Classifier {
	return &Classifier{
		Config:          cfg,
		CategoryMapping: categoryMapping,
		PIIMapping:      piiMapping,
	}
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

// ClassifyPII performs PII detection on the given text
func (c *Classifier) ClassifyPII(text string) (string, float64, error) {
	if c.PIIMapping == nil {
		return "NO_PII", 1.0, nil // No PII detector enabled
	}

	// Use BERT PII detector to detect PII in the text
	result, err := candle_binding.DetectPII(text)
	if err != nil {
		return "", 0.0, fmt.Errorf("PII detection error: %w", err)
	}

	// If no PII types were detected, return NO_PII
	if len(result.DetectedPIITypes) == 0 {
		log.Printf("No PII detected in text")
		return "NO_PII", 1.0, nil
	}

	// Return the first detected PII type with highest confidence
	// For now, we'll use a simple approach and return the first detected type
	piiType := result.DetectedPIITypes[0]

	// Calculate average confidence for the detected PII type
	var totalConfidence float64
	var count int
	for i, prediction := range result.TokenPredictions {
		if i < len(result.ConfidenceScores) && prediction > 0 { // prediction > 0 means it's not "O" (Other/No PII)
			totalConfidence += float64(result.ConfidenceScores[i])
			count++
		}
	}

	confidence := 1.0
	if count > 0 {
		confidence = totalConfidence / float64(count)
	}

	// Check confidence threshold
	if confidence < float64(c.Config.Classifier.PIIModel.Threshold) {
		log.Printf("PII detection confidence (%.4f) below threshold (%.4f), assuming no PII",
			confidence, c.Config.Classifier.PIIModel.Threshold)
		return "NO_PII", confidence, nil
	}

	log.Printf("Detected PII type: %s with confidence %.4f", piiType, confidence)
	return piiType, confidence, nil
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
