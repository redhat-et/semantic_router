//go:build linear
// +build linear

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	candle "github.com/redhat-et/semantic_route/candle-binding"
)

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

// loadCategoryMapping loads the category mapping from a JSON file
func loadCategoryMapping(path string) (*CategoryMapping, error) {
	// Read the mapping file
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read mapping file: %v", err)
	}

	// Parse the JSON data
	var mapping CategoryMapping
	err = json.Unmarshal(data, &mapping)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mapping JSON: %v", err)
	}

	return &mapping, nil
}

func main() {
	fmt.Println("Domain Classifier Test")
	fmt.Println("==================================================")
	fmt.Println("This demonstrates the new two-step initialization:")
	fmt.Println("1. Initialize base BERT model")
	fmt.Println("2. Initialize classification head")
	fmt.Println()

	// Try to load the category mapping
	mappingPath := "category_classifier_linear_model/category_mapping.json"
	mapping, err := loadCategoryMapping(mappingPath)

	if err != nil {
		fmt.Printf("Failed to load category mapping: %v\n", err)
		os.Exit(1)
	}

	// Get the model path
	modelPath, err := filepath.Abs("category_classifier_linear_model")
	if err != nil {
		fmt.Printf("Failed to get absolute path for model: %v\n", err)
		os.Exit(1)
	}

	numClasses := len(mapping.CategoryToIdx)

	// Initialize the system using the new two-step interface
	fmt.Printf("Step 1: Initializing base BERT model from: %s\n", modelPath)
	err = candle.InitBaseBertModel(modelPath, true) // Use the model path and CPU
	if err != nil {
		fmt.Printf("Failed to initialize base BERT model: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✓ Base BERT model initialized successfully!")

	fmt.Printf("Step 2: Initializing classification head with %d classes...\n", numClasses)
	err = candle.InitClassificationHead(modelPath, numClasses, int(candle.General))
	if err != nil {
		fmt.Printf("Failed to initialize domain classifier head: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✓ Category classifier head initialized with %d classes!\n", numClasses)

	// Test queries
	queries := []string{
		"What is the derivative of e^x?",
		"Explain the concept of supply and demand in economics.",
		"How does DNA replication work in eukaryotic cells?",
		"What is the difference between a civil law and common law system?",
		"Explain how transistors work in computer processors.",
		"Why do stars twinkle?",
		"How do I create a balanced portfolio for retirement?",
		"What causes mental illnesses?",
		"How do computer algorithms work?",
		"Explain the historical significance of the Roman Empire.",
		"What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?",
	}

	// Process each query
	fmt.Println("\nClassifying queries:")
	fmt.Println("==================")

	for i, query := range queries {
		result, err := candle.ClassifyText(query)
		if err != nil {
			fmt.Printf("Query %d: Classification failed: %v\n", i+1, err)
			continue
		}

		// Get the category name
		categoryID := fmt.Sprintf("%d", result.Class)
		categoryName := mapping.IdxToCategory[categoryID]

		// Print the result
		fmt.Printf("%d. Query: %s\n", i+1, query)
		fmt.Printf("   Classified as: %s (Class ID: %d, Confidence: %.4f)\n\n",
			categoryName, result.Class, result.Confidence)
	}

	fmt.Println("✓ Test complete!")
}
