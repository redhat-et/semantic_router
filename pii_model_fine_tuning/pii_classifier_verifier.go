package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	candle "github.com/redhat-et/semantic_route/candle-binding"
)

// Structs to match the JSON structure for mappings
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

type PIIMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
}

// Global variables for label mappings
var categoryLabels map[int]string
var piiLabels map[int]string

// loadCategoryMapping loads category labels from JSON file
func loadCategoryMapping(modelPath string) error {
	mappingPath := fmt.Sprintf("%s/category_mapping.json", modelPath)

	data, err := os.ReadFile(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to read category mapping file %s: %v", mappingPath, err)
	}

	var mapping CategoryMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return fmt.Errorf("failed to parse category mapping JSON: %v", err)
	}

	// Convert string keys to int keys for easier lookup
	categoryLabels = make(map[int]string)
	for idxStr, category := range mapping.IdxToCategory {
		var idx int
		if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
			return fmt.Errorf("failed to parse category index %s: %v", idxStr, err)
		}
		categoryLabels[idx] = category
	}

	fmt.Printf("Loaded %d category labels from %s\n", len(categoryLabels), mappingPath)
	return nil
}

// loadPIIMapping loads PII labels from JSON file
func loadPIIMapping(modelPath string) error {
	mappingPath := fmt.Sprintf("%s/pii_type_mapping.json", modelPath)

	data, err := os.ReadFile(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to read PII mapping file %s: %v", mappingPath, err)
	}

	var mapping PIIMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return fmt.Errorf("failed to parse PII mapping JSON: %v", err)
	}

	// Convert string keys to int keys for easier lookup
	piiLabels = make(map[int]string)
	for idxStr, label := range mapping.IdxToLabel {
		var idx int
		if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
			return fmt.Errorf("failed to parse PII index %s: %v", idxStr, err)
		}
		piiLabels[idx] = label
	}

	fmt.Printf("Loaded %d PII labels from %s\n", len(piiLabels), mappingPath)
	return nil
}

func main() {
	fmt.Println("PII Classifier Verifier")
	fmt.Println("========================")
	err := candle.InitModel("sentence-transformers/all-MiniLM-L6-v2", false)
	if err != nil {
		log.Printf("Failed to initialize model: %v", err)
	}

	categoryModelPath := "../classifier_model_fine_tuning/category_classifier_linear_model"
	err = loadCategoryMapping(categoryModelPath)
	if err != nil {
		log.Printf("Failed to load category mappings: %v", err)
	}

	piiModelPath := "./pii_classifier_linear_model"
	err = loadPIIMapping(piiModelPath)
	if err != nil {
		log.Printf("Failed to load PII mappings: %v", err)
	}

	if categoryLabels != nil {
		err = candle.InitClassifier(categoryModelPath, len(categoryLabels), false)
		if err != nil {
			log.Printf("Failed to initialize category classifier: %v", err)
		} else {
			fmt.Printf("Category classifier initialized with %d classes!\n", len(categoryLabels))
		}
	}

	if piiLabels != nil {
		err = candle.InitPIIClassifier(piiModelPath, len(piiLabels), false)
		if err != nil {
			log.Printf("Failed to initialize PII classifier: %v", err)
		} else {
			fmt.Printf("PII classifier initialized with %d classes!\n", len(piiLabels))
		}
	}

	fmt.Println("===================================")

	testTexts := []struct {
		text        string
		description string
	}{
		{"What is the derivative of x^2 with respect to x?", "Math Question"},
		{"My email address is john.smith@example.com", "Email PII"},
		{"Explain the concept of supply and demand in economics", "Economics Question"},
		{"Please call me at (555) 123-4567 for more information", "Phone PII"},
		{"How does DNA replication work in eukaryotic cells?", "Biology Question"},
		{"My social security number is 123-45-6789", "SSN PII"},
		{"What are the fundamental principles of computer algorithms?", "CS Question"},
		{"This is just a normal sentence without any personal information", "Clean Text"},
		{"What is the difference between civil law and common law?", "Law Question"},
		{"I live at 123 Main Street, New York, NY 10001", "Address Info"},
		{"My credit card number is 4532-1234-5678-9012", "Credit Card PII"},
		{"Visit our website at https://example.com for details", "URL Reference"},
	}

	for i, test := range testTexts {
		fmt.Printf("\nTest %d: %s\n", i+1, test.description)
		fmt.Printf("   Text: \"%s\"\n", test.text)

		// Category classification
		if categoryLabels != nil {
			categoryResult, err := candle.ClassifyText(test.text)
			if err != nil {
				fmt.Printf("Category: Error - %v\n", err)
			} else {
				categoryName := categoryLabels[categoryResult.Class]
				if categoryName == "" {
					categoryName = fmt.Sprintf("Class_%d", categoryResult.Class)
				}
				fmt.Printf("Category: %s (confidence: %.3f)\n", categoryName, categoryResult.Confidence)
			}
		}

		// PII classification
		if piiLabels != nil {
			piiResult, err := candle.ClassifyPIIText(test.text)
			if err != nil {
				fmt.Printf("PII: Error - %v\n", err)
			} else {
				piiName := piiLabels[piiResult.Class]
				if piiName == "" {
					piiName = fmt.Sprintf("Class_%d", piiResult.Class)
				}
				fmt.Printf("PII: %s (confidence: %.3f)", piiName, piiResult.Confidence)

				// Check if PII detected (assuming NO_PII is at index 0 or has "NO" in name)
				if piiName == "NO_PII" || piiResult.Class == 0 {
					fmt.Printf(" Clean")
				} else {
					fmt.Printf(" ALERT: PII detected!")
				}
				fmt.Println()
			}
		}

	}
}
