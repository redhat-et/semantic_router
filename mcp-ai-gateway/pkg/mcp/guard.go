package mcp

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/mark3labs/mcp-go/mcp"
)

// JailbreakMapping holds the mapping between indices and jailbreak types
type JailbreakMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
}

// ToolJailbreakResult represents the result of analyzing a tool for jailbreaks
type ToolJailbreakResult struct {
	ToolName      string  `json:"tool_name"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	Description   string  `json:"description"`
}

// Guard handles jailbreak detection and policy enforcement for MCP clients
type Guard struct {
	Config           PromptGuard
	JailbreakMapping *JailbreakMapping
	Initialized      bool
	// Track tools that have been flagged as potentially malicious
	flaggedTools map[string]ToolJailbreakResult
	mutex        sync.RWMutex
}

// NewGuard creates a new prompt guard instance for MCP clients
func NewGuard(config PromptGuard) (*Guard, error) {
	if !config.Enabled {
		log.Println("Prompt guard is disabled")
		return &Guard{
			Config:       config,
			Initialized:  false,
			flaggedTools: make(map[string]ToolJailbreakResult),
		}, nil
	}

	// Load jailbreak mapping
	jailbreakMapping, err := LoadJailbreakMapping(config.JailbreakMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
	}

	log.Printf("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())

	guard := &Guard{
		Config:           config,
		JailbreakMapping: jailbreakMapping,
		Initialized:      false,
		flaggedTools:     make(map[string]ToolJailbreakResult),
	}

	// Initialize the jailbreak classifier
	err = guard.initializeClassifier()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
	}

	return guard, nil
}

// LoadJailbreakMapping loads the jailbreak mapping from a JSON file
func LoadJailbreakMapping(path string) (*JailbreakMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read jailbreak mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping JailbreakMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse jailbreak mapping JSON: %w", err)
	}

	return &mapping, nil
}

// GetJailbreakTypeFromIndex converts a class index to jailbreak type name using the mapping
func (jm *JailbreakMapping) GetJailbreakTypeFromIndex(classIndex int) (string, bool) {
	jailbreakType, ok := jm.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return jailbreakType, ok
}

// GetJailbreakTypeCount returns the number of jailbreak types in the mapping
func (jm *JailbreakMapping) GetJailbreakTypeCount() int {
	return len(jm.LabelToIdx)
}

// initializeClassifier initializes the jailbreak classification model
func (g *Guard) initializeClassifier() error {
	if !g.IsEnabled() {
		return nil
	}

	numClasses := g.JailbreakMapping.GetJailbreakTypeCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough jailbreak types for classification, need at least 2, got %d", numClasses)
	}

	err := candle_binding.InitJailbreakClassifier(g.Config.ModelID, numClasses, g.Config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
	}

	g.Initialized = true
	log.Printf("Initialized jailbreak classifier with %d classes", numClasses)
	return nil
}

// IsEnabled checks if prompt guard is enabled and properly configured
func (g *Guard) IsEnabled() bool {
	return g.Config.Enabled && g.Config.ModelID != "" && g.Config.JailbreakMappingPath != ""
}

// CheckForJailbreak analyzes the given text for jailbreak attempts
func (g *Guard) CheckForJailbreak(text string) (bool, string, float32, error) {
	if !g.IsEnabled() || !g.Initialized {
		return false, "", 0.0, nil
	}

	if text == "" {
		return false, "", 0.0, nil
	}

	// Classify the text for jailbreak detection using the ML model
	result, err := candle_binding.ClassifyJailbreakText(text)
	if err != nil {
		return false, "", 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}
	log.Printf("Jailbreak classification result: %v", result)
	
	// Get the jailbreak type name from the class index
	jailbreakType, ok := g.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	// Check if confidence meets threshold and indicates jailbreak
	isJailbreak := result.Confidence >= g.Config.Threshold && jailbreakType == "jailbreak"

	if isJailbreak {
		log.Printf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, g.Config.Threshold)
	} else {
		log.Printf("BENIGN: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, g.Config.Threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, nil
}

// AnalyzeToolDescription analyzes a tool description for potential jailbreak attempts
// This catches tool-level attacks like shadowing, poisoning, and other malicious instructions
func (g *Guard) AnalyzeToolDescription(tool mcp.Tool) (*ToolJailbreakResult, error) {
	if !g.IsEnabled() || !g.Initialized {
		return nil, nil
	}

	// Analyze the tool description using the BERT classifier
	isJailbreak, jailbreakType, confidence, err := g.CheckForJailbreak(tool.Description)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze tool description for '%s': %w", tool.Name, err)
	}

	result := &ToolJailbreakResult{
		ToolName:      tool.Name,
		IsJailbreak:   isJailbreak,
		JailbreakType: jailbreakType,
		Confidence:    confidence,
		Description:   tool.Description,
	}

	// Store the result for future reference
	g.mutex.Lock()
	g.flaggedTools[tool.Name] = *result
	g.mutex.Unlock()

	if isJailbreak {
		log.Printf("MALICIOUS TOOL DETECTED: '%s' - Type: %s (confidence: %.3f)", 
			tool.Name, jailbreakType, confidence)
	} else {
		log.Printf("TOOL ANALYSIS: '%s' - Type: %s (confidence: %.3f)", 
			tool.Name, jailbreakType, confidence)
	}

	return result, nil
}

// AnalyzeTools analyzes a list of tools for potential jailbreak attempts
func (g *Guard) AnalyzeTools(tools []mcp.Tool) ([]ToolJailbreakResult, error) {
	if !g.IsEnabled() || !g.Initialized {
		return nil, nil
	}

	var results []ToolJailbreakResult
	var maliciousTools []string

	for _, tool := range tools {
		result, err := g.AnalyzeToolDescription(tool)
		if err != nil {
			log.Printf("Error analyzing tool '%s': %v", tool.Name, err)
			continue
		}

		if result != nil {
			results = append(results, *result)
			if result.IsJailbreak {
				maliciousTools = append(maliciousTools, tool.Name)
			}
		}
	}

	// Log summary of analysis
	if len(maliciousTools) > 0 {
		log.Printf("SECURITY ALERT: Found %d potentially malicious tools: %v", 
			len(maliciousTools), maliciousTools)
	}

	return results, nil
}

// IsToolFlagged checks if a tool has been flagged as potentially malicious
func (g *Guard) IsToolFlagged(toolName string) (bool, ToolJailbreakResult) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	result, exists := g.flaggedTools[toolName]
	if !exists {
		return false, ToolJailbreakResult{}
	}
	
	return result.IsJailbreak, result
}

// GetFlaggedTools returns all tools that have been flagged as potentially malicious
func (g *Guard) GetFlaggedTools() map[string]ToolJailbreakResult {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	flagged := make(map[string]ToolJailbreakResult)
	for name, result := range g.flaggedTools {
		if result.IsJailbreak {
			flagged[name] = result
		}
	}
	
	return flagged
}

// ValidateToolCall checks if a tool call should be allowed based on previous analysis
func (g *Guard) ValidateToolCall(toolName string) error {
	if !g.IsEnabled() || !g.Initialized {
		return nil
	}

	isFlagged, result := g.IsToolFlagged(toolName)
	if isFlagged {
		return fmt.Errorf("tool call blocked: '%s' has been flagged as potentially malicious (type: %s, confidence: %.3f)", 
			toolName, result.JailbreakType, result.Confidence)
	}

	return nil
}

// AnalyzeContent analyzes multiple content pieces for jailbreak attempts
func (g *Guard) AnalyzeContent(contentList []string) (bool, []JailbreakDetection, error) {
	if !g.IsEnabled() || !g.Initialized {
		return false, nil, nil
	}

	var detections []JailbreakDetection
	hasJailbreak := false

	for i, content := range contentList {
		if content == "" {
			continue
		}

		isJailbreak, jailbreakType, confidence, err := g.CheckForJailbreak(content)
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

// ExtractTextFromArguments extracts text content from MCP tool arguments for analysis
func ExtractTextFromArguments(arguments map[string]interface{}) []string {
	var textContents []string
	
	for _, value := range arguments {
		switch v := value.(type) {
		case string:
			if v != "" {
				textContents = append(textContents, v)
			}
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok && str != "" {
					textContents = append(textContents, str)
				}
			}
		}
	}
	
	return textContents
} 