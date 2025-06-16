package extproc

import (
	"log"
)

// classifyAndSelectBestModel chooses best models based on category classification and model quality and expected TTFT
func (r *OpenAIRouter) classifyAndSelectBestModel(query string) string {
	// If no categories defined, return default model
	if len(r.CategoryDescriptions) == 0 {
		return r.Config.DefaultModel
	}

	// First, classify the text to determine the category
	categoryName, confidence, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		log.Printf("Classification error: %v, falling back to default model", err)
		return r.Config.DefaultModel
	}

	if categoryName == "" {
		log.Printf("Classification confidence (%.4f) below threshold, using default model", confidence)
		return r.Config.DefaultModel
	}

	// Then select the best model from the determined category based on score and TTFT
	return r.ModelSelector.SelectBestModelForCategory(categoryName)
}

// findCategoryForClassification determines the category for the given text using classification
func (r *OpenAIRouter) findCategoryForClassification(query string) string {
	if len(r.CategoryDescriptions) == 0 {
		return ""
	}

	categoryName, _, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		log.Printf("Category classification error: %v", err)
		return ""
	}

	return categoryName
}
