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
