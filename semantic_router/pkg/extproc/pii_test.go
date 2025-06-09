package extproc

import (
	"testing"
	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/config"
)

func TestPIIDetectionResultStructure(t *testing.T) {
	// Test that PIIDetectionResult struct is properly defined
	result := PIIDetectionResult{
		HasPII:           true,
		DetectedTypes:    []string{"EMAIL_ADDRESS", "PHONE_NUMBER"},
		ConfidenceScores: []float32{0.9, 0.8, 0.95, 0.7},
		TokenPredictions: []int{0, 1, 0, 2},
		SanitizedText:    "Contact me at [EMAIL_ADDRESS] or [PHONE_NUMBER]",
	}

	if !result.HasPII {
		t.Error("Expected HasPII to be true")
	}

	if len(result.DetectedTypes) != 2 {
		t.Errorf("Expected 2 detected types, got %d", len(result.DetectedTypes))
	}

	if result.SanitizedText == "" {
		t.Error("Expected sanitized text to be non-empty")
	}
}

func TestPIIConfigurationLoading(t *testing.T) {
	// Test that PII configuration can be loaded properly
	cfg := &config.RouterConfig{
		PIIDetection: struct {
			Enabled         bool     `yaml:"enabled"`
			ModelID         string   `yaml:"model_id"`
			Threshold       float32  `yaml:"threshold"`
			UseCPU          bool     `yaml:"use_cpu"`
			PIITypes        []string `yaml:"pii_types"`
			BlockOnPII      bool     `yaml:"block_on_pii"`
			SanitizeEnabled bool     `yaml:"sanitize_enabled"`
		}{
			Enabled:         true,
			ModelID:         "bert-base-cased",
			Threshold:       0.5,
			UseCPU:          true,
			PIITypes:        []string{"O", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"},
			BlockOnPII:      false,
			SanitizeEnabled: true,
		},
	}

	if !cfg.PIIDetection.Enabled {
		t.Error("Expected PII detection to be enabled")
	}

	if len(cfg.PIIDetection.PIITypes) < 2 {
		t.Errorf("Expected at least 2 PII types, got %d", len(cfg.PIIDetection.PIITypes))
	}

	if cfg.PIIDetection.PIITypes[0] != "O" {
		t.Errorf("Expected first PII type to be 'O', got '%s'", cfg.PIIDetection.PIITypes[0])
	}
}

func TestSanitizeText(t *testing.T) {
	// Create a mock router for testing
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			PIIDetection: struct {
				Enabled         bool     `yaml:"enabled"`
				ModelID         string   `yaml:"model_id"`
				Threshold       float32  `yaml:"threshold"`
				UseCPU          bool     `yaml:"use_cpu"`
				PIITypes        []string `yaml:"pii_types"`
				BlockOnPII      bool     `yaml:"block_on_pii"`
				SanitizeEnabled bool     `yaml:"sanitize_enabled"`
			}{
				SanitizeEnabled: true,
			},
		},
	}

	// Test sanitization logic
	originalText := "Contact me at john@example.com or call my phone"
	predictions := []int{0, 0, 0, 1, 0, 0, 0, 2} // Mock predictions
	detectedTypes := []string{"EMAIL_ADDRESS", "PHONE_NUMBER"}

	sanitized := router.sanitizeText(originalText, predictions, detectedTypes)

	// The sanitized text should be different from the original
	if sanitized == originalText {
		t.Error("Expected sanitized text to be different from original")
	}

	// Should contain placeholders
	if !contains(sanitized, "[EMAIL_ADDRESS]") && !contains(sanitized, "[PHONE_NUMBER]") {
		t.Error("Expected sanitized text to contain PII placeholders")
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsInMiddle(s, substr)))
}

func containsInMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
} 