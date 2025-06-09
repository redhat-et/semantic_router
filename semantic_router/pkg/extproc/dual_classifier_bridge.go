package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
)

// DualClassifierResult represents the result from the Python dual classifier
type DualClassifierResult struct {
	Results []struct {
		Text     string `json:"text"`
		Category struct {
			PredictedCategory string             `json:"predicted_category"`
			Confidence        float64            `json:"confidence"`
			Probabilities     map[string]float64 `json:"probabilities"`
		} `json:"category"`
		PII struct {
			HasPII        bool `json:"has_pii"`
			PIITokenCount int  `json:"pii_token_count"`
			TotalTokens   int  `json:"total_tokens"`
			Tokens        []struct {
				Token      string  `json:"token"`
				Position   int     `json:"position"`
				IsPII      bool    `json:"is_pii"`
				Confidence float64 `json:"confidence"`
			} `json:"tokens"`
		} `json:"pii"`
	} `json:"results"`
}

// DualClassifierBridge handles communication with the Python dual classifier
type DualClassifierBridge struct {
	pythonPath string
	scriptPath string
	modelPath  string
	enabled    bool
	useCPU     bool
}

// NewDualClassifierBridge creates a new bridge to the Python dual classifier
func NewDualClassifierBridge(enabled bool, modelPath string, useCPU bool) (*DualClassifierBridge, error) {
	if !enabled {
		return &DualClassifierBridge{enabled: false}, nil
	}

	// Find Python executable
	pythonPath, err := findPythonExecutable()
	if err != nil {
		log.Printf("Warning: Could not find Python executable, disabling dual classifier: %v", err)
		return &DualClassifierBridge{enabled: false}, nil
	}

	// Construct script path - use enhanced bridge for trained model
	scriptPath := "dual_classifier/enhanced_bridge.py"

	// Fall back to simple bridge if enhanced bridge not found
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		log.Printf("Enhanced bridge not found, trying simple bridge...")
		scriptPath = "dual_classifier/simple_bridge.py"
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			log.Printf("Warning: No dual classifier script found, disabling: %v", err)
			return &DualClassifierBridge{enabled: false}, nil
		}
	}

	bridge := &DualClassifierBridge{
		pythonPath: pythonPath,
		scriptPath: scriptPath,
		modelPath:  modelPath,
		enabled:    true,
		useCPU:     useCPU,
	}

	// Test the dual classifier
	if err := bridge.testConnection(); err != nil {
		log.Printf("Warning: Dual classifier test failed, disabling: %v", err)
		return &DualClassifierBridge{enabled: false}, nil
	}

	log.Printf("Dual classifier bridge initialized successfully")
	return bridge, nil
}

// findPythonExecutable finds a suitable Python executable
func findPythonExecutable() (string, error) {
	// First, try the virtual environment Python if it exists
	venvPython := ".venv/bin/python"
	if _, err := os.Stat(venvPython); err == nil {
		// Test if this Python has the required packages
		cmd := exec.Command(venvPython, "-c", "import torch, transformers; print('OK')")
		if err := cmd.Run(); err == nil {
			return venvPython, nil
		}
	}

	// Try different Python executables in order of preference
	candidates := []string{"python3", "python", "python3.11", "python3.10", "python3.9"}

	for _, candidate := range candidates {
		if path, err := exec.LookPath(candidate); err == nil {
			// Test if this Python has the required packages
			cmd := exec.Command(path, "-c", "import torch, transformers; print('OK')")
			if err := cmd.Run(); err == nil {
				return path, nil
			}
		}
	}

	return "", fmt.Errorf("no suitable Python executable found with required packages")
}

// testConnection tests if the dual classifier is working
func (dcb *DualClassifierBridge) testConnection() error {
	if !dcb.enabled {
		return fmt.Errorf("dual classifier bridge is disabled")
	}

	// Simple test classification
	_, err := dcb.Classify("test text", "dual")
	return err
}

// Classify performs classification using the Python dual classifier
func (dcb *DualClassifierBridge) Classify(text string, mode string) (*DualClassifierResult, error) {
	if !dcb.enabled {
		return nil, fmt.Errorf("dual classifier bridge is disabled")
	}

	// Build command arguments
	args := []string{dcb.scriptPath, "--text", text, "--mode", mode}

	// Add model path if specified (for compatibility, but ignored by simple bridge)
	if dcb.modelPath != "" {
		args = append(args, "--model-path", dcb.modelPath)
	}

	// Add device specification (for compatibility, but ignored by simple bridge)
	if dcb.useCPU {
		args = append(args, "--device", "cpu")
	}

	// Execute the Python script
	cmd := exec.Command(dcb.pythonPath, args...)
	cmd.Dir = "." // Set working directory

	// Capture only stdout, ignore stderr to avoid parsing issues with warnings/info messages
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run dual classifier: %v", err)
	}

	// Find the JSON part of the output (starts with '{')
	outputStr := string(output)
	jsonStart := -1
	for i, char := range outputStr {
		if char == '{' {
			jsonStart = i
			break
		}
	}

	if jsonStart == -1 {
		return nil, fmt.Errorf("no JSON found in dual classifier output")
	}

	jsonOutput := outputStr[jsonStart:]

	// Parse the JSON result
	var result DualClassifierResult
	if err := json.Unmarshal([]byte(jsonOutput), &result); err != nil {
		return nil, fmt.Errorf("failed to parse dual classifier result: %v", err)
	}

	return &result, nil
}

// ClassifyCategory performs only category classification
func (dcb *DualClassifierBridge) ClassifyCategory(text string) (string, float64, error) {
	result, err := dcb.Classify(text, "category")
	if err != nil {
		return "", 0, err
	}

	if len(result.Results) == 0 {
		return "", 0, fmt.Errorf("no classification results returned")
	}

	firstResult := result.Results[0]
	return firstResult.Category.PredictedCategory, firstResult.Category.Confidence, nil
}

// DetectPII performs only PII detection
func (dcb *DualClassifierBridge) DetectPII(text string) (bool, []string, error) {
	result, err := dcb.Classify(text, "pii")
	if err != nil {
		return false, nil, err
	}

	if len(result.Results) == 0 {
		return false, nil, fmt.Errorf("no PII detection results returned")
	}

	firstResult := result.Results[0]

	// Extract PII types from tokens (simplified approach)
	var piiTypes []string
	if firstResult.PII.HasPII {
		// For now, we'll use a generic PII type since the token-level classification
		// from the untrained head isn't reliable. This will be improved with
		// proper PII training or by using regex detection instead.
		piiTypes = append(piiTypes, "DETECTED_PII")
	}

	return firstResult.PII.HasPII, piiTypes, nil
}

// ClassifyDual performs both category classification and PII detection
func (dcb *DualClassifierBridge) ClassifyDual(text string) (string, float64, bool, []string, error) {
	result, err := dcb.Classify(text, "dual")
	if err != nil {
		return "", 0, false, nil, err
	}

	if len(result.Results) == 0 {
		return "", 0, false, nil, fmt.Errorf("no classification results returned")
	}

	firstResult := result.Results[0]

	// Extract category information
	category := firstResult.Category.PredictedCategory
	confidence := firstResult.Category.Confidence

	// Extract PII information
	hasPII := firstResult.PII.HasPII
	var piiTypes []string
	if hasPII {
		piiTypes = append(piiTypes, "DETECTED_PII")
	}

	return category, confidence, hasPII, piiTypes, nil
}

// IsEnabled returns whether the dual classifier bridge is enabled
func (dcb *DualClassifierBridge) IsEnabled() bool {
	return dcb.enabled
}

// GetCategoryMapping returns the category mapping for model selection
func (dcb *DualClassifierBridge) GetCategoryMapping() map[string]int {
	// This should be populated from the model's training config
	// For now, return the known categories from your trained model
	return map[string]int{
		"business":      0,
		"entertainment": 1,
		"politics":      2,
		"sport":         3,
		"tech":          4,
	}
}

// GetCategoryDescriptions returns descriptions for the categories
func (dcb *DualClassifierBridge) GetCategoryDescriptions() []string {
	return []string{
		"Business and finance related content",
		"Entertainment, movies, music, and leisure content",
		"Political news, government, and policy content",
		"Sports, games, and athletic content",
		"Technology, computers, and technical content",
	}
}
