package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// HTTPClassificationRequest represents the request to the classification service
type HTTPClassificationRequest struct {
	Text string `json:"text"`
	Mode string `json:"mode"`
}

// HTTPClassificationResponse represents the response from the classification service
type HTTPClassificationResponse struct {
	Success         bool    `json:"success"`
	Error           string  `json:"error,omitempty"`
	InferenceTimeMs float64 `json:"inference_time_ms,omitempty"`
	ModelLoadTimeS  float64 `json:"model_load_time_s,omitempty"`
	Results         []struct {
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

// HTTPDualClassifierBridge handles HTTP communication with the persistent Python classification service
type HTTPDualClassifierBridge struct {
	serviceURL  string
	httpClient  *http.Client
	enabled     bool
	serviceCmd  *exec.Cmd
	servicePID  int
	modelPath   string
	useCPU      bool
	servicePort int
}

// NewHTTPDualClassifierBridge creates a new HTTP-based bridge to the Python dual classifier
func NewHTTPDualClassifierBridge(enabled bool, modelPath string, useCPU bool, servicePort int) (*HTTPDualClassifierBridge, error) {
	if !enabled {
		return &HTTPDualClassifierBridge{enabled: false}, nil
	}

	if servicePort == 0 {
		servicePort = 8888 // Default port
	}

	bridge := &HTTPDualClassifierBridge{
		serviceURL:  fmt.Sprintf("http://localhost:%d", servicePort),
		httpClient:  &http.Client{Timeout: 30 * time.Second},
		enabled:     true,
		modelPath:   modelPath,
		useCPU:      useCPU,
		servicePort: servicePort,
	}

	// Start the classification service
	if err := bridge.startClassificationService(); err != nil {
		log.Printf("Warning: Failed to start classification service, disabling: %v", err)
		return &HTTPDualClassifierBridge{enabled: false}, nil
	}

	// Wait for service to be ready
	if err := bridge.waitForService(30 * time.Second); err != nil {
		log.Printf("Warning: Classification service failed to start, disabling: %v", err)
		bridge.stopClassificationService()
		return &HTTPDualClassifierBridge{enabled: false}, nil
	}

	// Test the service
	if err := bridge.testConnection(); err != nil {
		log.Printf("Warning: Classification service test failed, disabling: %v", err)
		bridge.stopClassificationService()
		return &HTTPDualClassifierBridge{enabled: false}, nil
	}

	log.Printf("HTTP Dual classifier bridge initialized successfully on port %d", servicePort)
	return bridge, nil
}

// startClassificationService starts the persistent Python classification service
func (hcb *HTTPDualClassifierBridge) startClassificationService() error {
	// Find Python executable
	pythonPath, err := findPythonExecutable()
	if err != nil {
		return fmt.Errorf("could not find Python executable: %v", err)
	}

	// Check if service script exists
	scriptPath := "dual_classifier/classification_service.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("classification service script not found: %s", scriptPath)
	}

	// Build command arguments
	args := []string{scriptPath, "--port", fmt.Sprintf("%d", hcb.servicePort)}

	if hcb.modelPath != "" {
		args = append(args, "--model-path", hcb.modelPath)
	}

	if hcb.useCPU {
		args = append(args, "--device", "cpu")
	}

	// Start the service as a background process
	hcb.serviceCmd = exec.Command(pythonPath, args...)
	hcb.serviceCmd.Dir = "." // Set working directory

	// Set up process group to allow clean shutdown
	hcb.serviceCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	// Start the service
	if err := hcb.serviceCmd.Start(); err != nil {
		return fmt.Errorf("failed to start classification service: %v", err)
	}

	hcb.servicePID = hcb.serviceCmd.Process.Pid
	log.Printf("Started classification service with PID %d on port %d", hcb.servicePID, hcb.servicePort)

	return nil
}

// waitForService waits for the classification service to be ready
func (hcb *HTTPDualClassifierBridge) waitForService(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for classification service to start")
		case <-ticker.C:
			if hcb.isServiceHealthy() {
				return nil
			}
		}
	}
}

// isServiceHealthy checks if the classification service is responding
func (hcb *HTTPDualClassifierBridge) isServiceHealthy() bool {
	resp, err := hcb.httpClient.Get(hcb.serviceURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// stopClassificationService stops the persistent Python classification service
func (hcb *HTTPDualClassifierBridge) stopClassificationService() {
	if hcb.serviceCmd != nil && hcb.serviceCmd.Process != nil {
		log.Printf("Stopping classification service with PID %d", hcb.servicePID)

		// Try graceful shutdown first
		if err := hcb.serviceCmd.Process.Signal(syscall.SIGTERM); err != nil {
			log.Printf("Failed to send SIGTERM: %v", err)
		}

		// Wait briefly for graceful shutdown
		done := make(chan error, 1)
		go func() {
			done <- hcb.serviceCmd.Wait()
		}()

		select {
		case <-time.After(5 * time.Second):
			// Force kill if it doesn't stop gracefully
			log.Printf("Force killing classification service")
			hcb.serviceCmd.Process.Kill()
		case <-done:
			// Process stopped gracefully
		}

		hcb.serviceCmd = nil
		hcb.servicePID = 0
	}
}

// testConnection tests if the classification service is working
func (hcb *HTTPDualClassifierBridge) testConnection() error {
	if !hcb.enabled {
		return fmt.Errorf("HTTP dual classifier bridge is disabled")
	}

	// Simple test classification
	_, err := hcb.Classify("test text", "dual")
	return err
}

// Classify performs classification using the HTTP classification service
func (hcb *HTTPDualClassifierBridge) Classify(text string, mode string) (*DualClassifierResult, error) {
	if !hcb.enabled {
		return nil, fmt.Errorf("HTTP dual classifier bridge is disabled")
	}

	// Prepare request
	reqData := HTTPClassificationRequest{
		Text: text,
		Mode: mode,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Make HTTP request
	resp, err := hcb.httpClient.Post(hcb.serviceURL+"/classify", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to make classification request: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("classification service returned error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var httpResp HTTPClassificationResponse
	if err := json.Unmarshal(body, &httpResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	if !httpResp.Success {
		return nil, fmt.Errorf("classification failed: %s", httpResp.Error)
	}

	// Convert to expected format
	result := &DualClassifierResult{
		Results: httpResp.Results,
	}

	return result, nil
}

// ClassifyCategory performs only category classification
func (hcb *HTTPDualClassifierBridge) ClassifyCategory(text string) (string, float64, error) {
	result, err := hcb.Classify(text, "category")
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
func (hcb *HTTPDualClassifierBridge) DetectPII(text string) (bool, []string, error) {
	result, err := hcb.Classify(text, "pii")
	if err != nil {
		return false, nil, err
	}

	if len(result.Results) == 0 {
		return false, nil, fmt.Errorf("no PII detection results returned")
	}

	firstResult := result.Results[0]
	hasPII := firstResult.PII.HasPII

	var piiTokens []string
	for _, token := range firstResult.PII.Tokens {
		if token.IsPII {
			piiTokens = append(piiTokens, token.Token)
		}
	}

	return hasPII, piiTokens, nil
}

// ClassifyDual performs both category classification and PII detection
func (hcb *HTTPDualClassifierBridge) ClassifyDual(text string) (string, float64, bool, []string, error) {
	result, err := hcb.Classify(text, "dual")
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
	var piiTokens []string
	for _, token := range firstResult.PII.Tokens {
		if token.IsPII {
			piiTokens = append(piiTokens, token.Token)
		}
	}

	return category, confidence, hasPII, piiTokens, nil
}

// IsEnabled returns whether the HTTP dual classifier bridge is enabled
func (hcb *HTTPDualClassifierBridge) IsEnabled() bool {
	return hcb.enabled
}

// Shutdown cleanly shuts down the HTTP dual classifier bridge
func (hcb *HTTPDualClassifierBridge) Shutdown() {
	if hcb.enabled {
		log.Printf("Shutting down HTTP dual classifier bridge")
		hcb.stopClassificationService()
	}
}

// GetServiceStats returns statistics about the classification service
func (hcb *HTTPDualClassifierBridge) GetServiceStats() map[string]interface{} {
	if !hcb.enabled {
		return map[string]interface{}{
			"enabled": false,
		}
	}

	// Try to get health info
	resp, err := hcb.httpClient.Get(hcb.serviceURL + "/health")
	if err != nil {
		return map[string]interface{}{
			"enabled": true,
			"healthy": false,
			"error":   err.Error(),
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return map[string]interface{}{
			"enabled": true,
			"healthy": false,
			"error":   fmt.Sprintf("Health check returned %d", resp.StatusCode),
		}
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return map[string]interface{}{
			"enabled": true,
			"healthy": false,
			"error":   fmt.Sprintf("Failed to read health response: %v", err),
		}
	}

	var healthData map[string]interface{}
	if err := json.Unmarshal(body, &healthData); err != nil {
		return map[string]interface{}{
			"enabled": true,
			"healthy": false,
			"error":   fmt.Sprintf("Failed to parse health response: %v", err),
		}
	}

	return map[string]interface{}{
		"enabled":     true,
		"healthy":     true,
		"service_url": hcb.serviceURL,
		"service_pid": hcb.servicePID,
		"health_data": healthData,
	}
}
