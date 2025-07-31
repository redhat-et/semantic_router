package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	candle "github.com/redhat-et/semantic_route/candle-binding"
)

// Struct to match the JSON structure for PII mappings
type PIIMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
}

// Global variable for PII label mappings
var piiLabels map[int]string

// Configuration for model type
type ModelConfig struct {
	UseModernBERT       bool
	PIIModelPath        string
	SimilarityModelPath string
	UseCPU              bool
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

// initializeModels initializes the similarity model and PII classifier based on config
func initializeModels(config ModelConfig) error {
	// Initialize similarity model (always use BERT for similarity for now)
	fmt.Printf("Initializing similarity model: %s\n", config.SimilarityModelPath)
	err := candle.InitModel(config.SimilarityModelPath, config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize similarity model: %v", err)
	}

	// Initialize PII classifier
	if piiLabels != nil {
		fmt.Printf("\nInitializing PII classifier (%s): %s\n", 
			map[bool]string{true: "ModernBERT", false: "BERT"}[config.UseModernBERT], 
			config.PIIModelPath)
		
		if config.UseModernBERT {
			err = candle.InitModernBertPIIClassifier(config.PIIModelPath, config.UseCPU)
		} else {
			err = candle.InitPIIClassifier(config.PIIModelPath, len(piiLabels), config.UseCPU)
		}
		
		if err != nil {
			return fmt.Errorf("failed to initialize PII classifier: %v", err)
		}
		fmt.Printf("✅ PII classifier initialized successfully!\n")
		if config.UseModernBERT {
			fmt.Println("   📝 Note: Number of classes auto-detected from model weights")
		} else {
			fmt.Printf("   📝 Note: Using %d classes from mapping file\n", len(piiLabels))
		}
	}

	return nil
}

// classifyPIIText performs PII classification using the appropriate model type
func classifyPIIText(text string, config ModelConfig) (candle.ClassResult, error) {
	if config.UseModernBERT {
		return candle.ClassifyModernBertPIIText(text)
	}
	return candle.ClassifyPIIText(text)
}

func main() {
	// Parse command line flags
	var (
		useModernBERT = flag.Bool("modernbert", false, "Use ModernBERT models instead of BERT")
		piiPath       = flag.String("pii-model", "./pii_classifier_linear_model", "Path to PII classifier model")
		simPath       = flag.String("similarity-model", "sentence-transformers/all-MiniLM-L6-v2", "Path/ID for similarity model")
		useCPU        = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	config := ModelConfig{
		UseModernBERT:       *useModernBERT,
		PIIModelPath:        *piiPath,
		SimilarityModelPath: *simPath,
		UseCPU:              *useCPU,
	}
	fmt.Println("PII Classifier Verifier")
	fmt.Println("========================")
	
	err := loadPIIMapping(config.PIIModelPath)
	if err != nil {
		log.Printf("Failed to load PII mappings: %v", err)
	}

	// Initialize models
	err = initializeModels(config)
	if err != nil {
		log.Fatalf("Failed to initialize models: %v", err)
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

		// PII classification
		if piiLabels != nil {
			piiResult, err := classifyPIIText(test.text, config)
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
