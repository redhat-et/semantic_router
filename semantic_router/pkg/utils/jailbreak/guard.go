package jailbreak

import (
	"fmt"
	"log"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/classification"
)

// Guard handles jailbreak detection and policy enforcement
type Guard struct {
	Config           *config.PromptGuardConfig
	JailbreakMapping *classification.JailbreakMapping
	Initialized      bool
}

// JailbreakDetection represents the result of jailbreak analysis for a piece of content
type JailbreakDetection struct {
	Content       string  `json:"content"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	ContentIndex  int     `json:"content_index"`
}

// NewGuard creates a new prompt guard instance
func NewGuard(cfg *config.RouterConfig) (*Guard, error) {
	if !cfg.IsPromptGuardEnabled() {
		log.Println("Prompt guard is disabled")
		return &Guard{
			Config:      &cfg.PromptGuard,
			Initialized: false,
		}, nil
	}

	// Load jailbreak mapping
	jailbreakMapping, err := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
	}

	log.Printf("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())

	guard := &Guard{
		Config:           &cfg.PromptGuard,
		JailbreakMapping: jailbreakMapping,
		Initialized:      false,
	}

	// Initialize the jailbreak classifier
	err = guard.initializeClassifier()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
	}

	return guard, nil
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

	err := candle_binding.InitClassificationHead(g.Config.ModelID, numClasses, int(candle_binding.Jailbreak))
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

	// Classify the text for jailbreak detection
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
