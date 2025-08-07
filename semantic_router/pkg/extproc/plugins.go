package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/cache"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/tools"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/classification"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/http"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/pii"
)

// =============================================================================
// SECURITY PLUGINS
// =============================================================================

// PIIDetectorPlugin implements SecurityPlugin for PII detection using the classifier
type PIIDetectorPlugin struct {
	name       string
	version    string
	priority   int
	enabled    bool
	threshold  float32
	classifier *classification.Classifier
	logger     Logger
}

// NewPIIDetectorPlugin creates a new PII detector plugin
func NewPIIDetectorPlugin(priority int, classifier *classification.Classifier, logger Logger) *PIIDetectorPlugin {
	return &PIIDetectorPlugin{
		name:       PluginNamePIIDetector,
		version:    "1.0.0",
		priority:   priority,
		threshold:  DefaultPIIThreshold,
		classifier: classifier,
		logger:     logger,
	}
}

func (p *PIIDetectorPlugin) Name() string    { return p.name }
func (p *PIIDetectorPlugin) Version() string { return p.version }
func (p *PIIDetectorPlugin) Priority() int   { return p.priority }

func (p *PIIDetectorPlugin) Initialize(config map[string]interface{}) error {
	if threshold, ok := config["threshold"].(float64); ok {
		p.threshold = float32(threshold)
	}
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	} else {
		p.enabled = true
	}
	return nil
}

func (p *PIIDetectorPlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	return nil
}

func (p *PIIDetectorPlugin) Shutdown() error {
	p.enabled = false
	return nil
}

func (p *PIIDetectorPlugin) CheckContent(content string, ctx *RequestContext) (*SecurityResult, error) {
	if !p.enabled {
		return &SecurityResult{Allowed: true}, nil
	}

	// Use thePII classifier
	allContent := []string{content}
	detectedPII := p.classifier.DetectPIIInContent(allContent)
	
	if len(detectedPII) > 0 {
		p.logger.Info("PII detected", 
			Field{Key: LogFieldRequestID, Value: ctx.RequestID},
			Field{Key: "pii_types", Value: detectedPII})
		
		response := http.CreatePIIViolationResponse(ctx.RequestModel, detectedPII)
		return &SecurityResult{
			Allowed:     false,
			BlockReason: fmt.Sprintf("PII detected: %v", detectedPII),
			Confidence:  p.threshold,
			Details: map[string]interface{}{
				"pii_types": detectedPII,
				"model":     ctx.RequestModel,
			},
			Response: response,
		}, nil
	}

	return &SecurityResult{Allowed: true}, nil
}

// JailbreakDetectorPlugin implements SecurityPlugin for jailbreak detection using the classifier
type JailbreakDetectorPlugin struct {
	name       string
	version    string
	priority   int
	enabled    bool
	threshold  float32
	classifier *classification.Classifier
	logger     Logger
}

func NewJailbreakDetectorPlugin(priority int, classifier *classification.Classifier, logger Logger) *JailbreakDetectorPlugin {
	return &JailbreakDetectorPlugin{
		name:       PluginNameJailbreakDetector,
		version:    "1.0.0",
		priority:   priority,
		threshold:  DefaultJailbreakThreshold,
		classifier: classifier,
		logger:     logger,
	}
}

func (p *JailbreakDetectorPlugin) Name() string    { return p.name }
func (p *JailbreakDetectorPlugin) Version() string { return p.version }
func (p *JailbreakDetectorPlugin) Priority() int   { return p.priority }

func (p *JailbreakDetectorPlugin) Initialize(config map[string]interface{}) error {
	if threshold, ok := config["threshold"].(float64); ok {
		p.threshold = float32(threshold)
	}
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	} else {
		p.enabled = true
	}
	return nil
}

func (p *JailbreakDetectorPlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	return nil
}

func (p *JailbreakDetectorPlugin) Shutdown() error {
	p.enabled = false
	return nil
}

func (p *JailbreakDetectorPlugin) CheckContent(content string, ctx *RequestContext) (*SecurityResult, error) {
	if !p.enabled {
		return &SecurityResult{Allowed: true}, nil
	}

	// Use thejailbreak classifier
	isJailbreak, jailbreakType, confidence, err := p.classifier.CheckForJailbreak(content)
	if err != nil {
		p.logger.Error("Jailbreak detection failed", err, Field{Key: LogFieldRequestID, Value: ctx.RequestID})
		return &SecurityResult{Allowed: true}, nil // Allow on error
	}
	
	if isJailbreak {
		p.logger.Warn("Jailbreak detected", 
			Field{Key: LogFieldRequestID, Value: ctx.RequestID},
			Field{Key: "jailbreak_type", Value: jailbreakType},
			Field{Key: LogFieldConfidence, Value: confidence})
		
		response := http.CreateJailbreakViolationResponse(jailbreakType, confidence)
		return &SecurityResult{
			Allowed:     false,
			BlockReason: fmt.Sprintf("Jailbreak detected: %s", jailbreakType),
			Confidence:  confidence,
			Details: map[string]interface{}{
				"jailbreak_type": jailbreakType,
				"confidence":     confidence,
			},
			Response: response,
		}, nil
	}

	return &SecurityResult{Allowed: true}, nil
}

// =============================================================================
// ROUTING PLUGINS
// =============================================================================

// ClassificationRouterPlugin implements RoutingPlugin for model selection using classification
type ClassificationRouterPlugin struct {
	name       string
	version    string
	priority   int
	enabled    bool
	classifier *classification.Classifier
	piiChecker *pii.PolicyChecker
	config     *config.RouterConfig
	logger     Logger
}

func NewClassificationRouterPlugin(priority int, classifier *classification.Classifier, piiChecker *pii.PolicyChecker, cfg *config.RouterConfig, logger Logger) *ClassificationRouterPlugin {
	return &ClassificationRouterPlugin{
		name:       PluginNameClassificationRouter,
		version:    "1.0.0",
		priority:   priority,
		enabled:    true,
		classifier: classifier,
		piiChecker: piiChecker,
		config:     cfg,
		logger:     logger,
	}
}

func (p *ClassificationRouterPlugin) Name() string    { return p.name }
func (p *ClassificationRouterPlugin) Version() string { return p.version }
func (p *ClassificationRouterPlugin) Priority() int   { return p.priority }

func (p *ClassificationRouterPlugin) Initialize(config map[string]interface{}) error {
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	}
	return nil
}

func (p *ClassificationRouterPlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	return nil
}

func (p *ClassificationRouterPlugin) Shutdown() error {
	p.enabled = false
	return nil
}

func (p *ClassificationRouterPlugin) SelectModel(request *openai.OpenAIRequest, content string, ctx *RequestContext) (*ModelSelection, error) {
	if !p.enabled {
		return &ModelSelection{
			Model:      request.Model,
			Confidence: 1.0,
			Reason:     "plugin disabled",
		}, nil
	}

	// Use the classification logic
	selectedModel := p.classifier.ClassifyAndSelectBestModel(content)
	
	// If load_aware is enabled, consider current model loads
	if p.config.Classifier.LoadAware {
		categoryName, _, classErr := p.classifier.ClassifyCategory(content)
		if classErr == nil && categoryName != "" {
			candidateModels := p.classifier.GetModelsForCategory(categoryName)
			if len(candidateModels) > 1 {
				// Select the best model considering current load
				selectedModel = p.classifier.SelectBestModelForCategory(categoryName)
			}
		}
	}
	
	// Check PII policy compliance for the selected model
	userContent, nonUserMessages := openai.ExtractUserAndNonUserContent(request)
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)
	detectedPII := p.classifier.DetectPIIInContent(allContent)
	
	allowed, deniedPII, err := p.piiChecker.CheckPolicy(selectedModel, detectedPII)
	if err != nil {
		p.logger.Error("PII policy check failed", err, Field{Key: LogFieldRequestID, Value: ctx.RequestID})
		// Continue with original model on error
		selectedModel = request.Model
	} else if !allowed {
		p.logger.Warn("Model violates PII policy, finding alternative",
			Field{Key: LogFieldRequestID, Value: ctx.RequestID},
			Field{Key: "selected_model", Value: selectedModel},
			Field{Key: "denied_pii", Value: deniedPII})
		
		// Try to find an alternative model that passes PII policy
		categoryName, _, classErr := p.classifier.ClassifyCategory(content)
		if classErr == nil && categoryName != "" {
			alternativeModels := p.classifier.GetModelsForCategory(categoryName)
			allowedModels := p.piiChecker.FilterModelsForPII(alternativeModels, detectedPII)
			if len(allowedModels) > 0 {
				selectedModel = p.classifier.SelectBestModelFromList(allowedModels, categoryName)
				p.logger.Info("Found alternative model", 
					Field{Key: LogFieldRequestID, Value: ctx.RequestID},
					Field{Key: "alternative_model", Value: selectedModel})
			} else {
				// Fall back to default model
				selectedModel = p.config.DefaultModel
				// Check if default also passes
				defaultAllowed, _, _ := p.piiChecker.CheckPolicy(selectedModel, detectedPII)
				if !defaultAllowed {
					return nil, PIIViolationError(selectedModel, deniedPII)
				}
			}
		} else {
			// Can't classify, return PII violation
			return nil, PIIViolationError(selectedModel, deniedPII)
		}
	}

	confidence := 0.9 // High confidence for classification-based selection
	reason := "classification-based selection"
	
	if selectedModel == p.config.DefaultModel {
		confidence = 0.5
		reason = "fallback to default model"
	}

	return &ModelSelection{
		Model:      selectedModel,
		Confidence: float32(confidence),
		Reason:     reason,
		Metadata: map[string]interface{}{
			"original_model": request.Model,
			"detected_pii":   detectedPII,
		},
	}, nil
}

func (p *ClassificationRouterPlugin) SupportsModel(model string) bool {
	return model == AutoModel
}

// LoadBalancerRouterPlugin implements a simple load balancing router
type LoadBalancerRouterPlugin struct {
	name     string
	version  string
	priority int
	enabled  bool
	models   []string
	current  int
}

func NewLoadBalancerRouterPlugin(priority int) *LoadBalancerRouterPlugin {
	return &LoadBalancerRouterPlugin{
		name:     PluginNameLoadBalancerRouter,
		version:  "1.0.0",
		priority: priority,
		enabled:  true,
		models:   []string{"gpt-3.5-turbo", "gpt-4", "claude-3-haiku"},
		current:  0,
	}
}

func (p *LoadBalancerRouterPlugin) Name() string    { return p.name }
func (p *LoadBalancerRouterPlugin) Version() string { return p.version }
func (p *LoadBalancerRouterPlugin) Priority() int   { return p.priority }

func (p *LoadBalancerRouterPlugin) Initialize(config map[string]interface{}) error {
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	}
	if models, ok := config["models"].([]interface{}); ok {
		p.models = make([]string, len(models))
		for i, model := range models {
			p.models[i] = fmt.Sprintf("%v", model)
		}
	}
	return nil
}

func (p *LoadBalancerRouterPlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	if len(p.models) == 0 {
		return fmt.Errorf("no models configured")
	}
	return nil
}

func (p *LoadBalancerRouterPlugin) Shutdown() error {
	p.enabled = false
	return nil
}

func (p *LoadBalancerRouterPlugin) SelectModel(request *openai.OpenAIRequest, content string, ctx *RequestContext) (*ModelSelection, error) {
	if !p.enabled || len(p.models) == 0 {
		return &ModelSelection{
			Model:      request.Model,
			Confidence: 1.0,
			Reason:     "plugin disabled or no models",
		}, nil
	}

	// Simple round-robin selection
	selectedModel := p.models[p.current]
	p.current = (p.current + 1) % len(p.models)

	return &ModelSelection{
		Model:      selectedModel,
		Confidence: 0.8,
		Reason:     "load-balanced selection",
		Metadata: map[string]interface{}{
			"original_model": request.Model,
			"selection_index": p.current,
		},
	}, nil
}

func (p *LoadBalancerRouterPlugin) SupportsModel(model string) bool {
	return model == AutoModel
}

// =============================================================================
// CACHE PLUGINS
// =============================================================================

// SemanticCachePlugin implements CachePlugin usingsemantic cache
type SemanticCachePlugin struct {
	name    string
	version string
	cache   *cache.SemanticCache
	logger  Logger
}

func NewSemanticCachePlugin(semanticCache *cache.SemanticCache, logger Logger) *SemanticCachePlugin {
	return &SemanticCachePlugin{
		name:    PluginNameSemanticCache,
		version: "1.0.0",
		cache:   semanticCache,
		logger:  logger,
	}
}

func (p *SemanticCachePlugin) Name() string    { return p.name }
func (p *SemanticCachePlugin) Version() string { return p.version }

func (p *SemanticCachePlugin) Initialize(config map[string]interface{}) error {
	// Cache is already initialized with proper configuration
	return nil
}

func (p *SemanticCachePlugin) Health() error {
	if !p.cache.IsEnabled() {
		return fmt.Errorf("cache is disabled")
	}
	return nil
}

func (p *SemanticCachePlugin) Shutdown() error {
	// Cache doesn't need explicit shutdown
	return nil
}

func (p *SemanticCachePlugin) Get(key string) ([]byte, bool, error) {
	if !p.cache.IsEnabled() {
		return nil, false, nil
	}

	// Extract model and query from key (format: "model:query")
	parts := strings.SplitN(key, ":", 2)
	if len(parts) != 2 {
		return nil, false, fmt.Errorf("invalid cache key format")
	}
	
	model, query := parts[0], parts[1]
	response, found, err := p.cache.FindSimilar(model, query)
	if err != nil {
		return nil, false, err
	}
	
	if found {
		return response, true, nil
	}
	
	return nil, false, nil
}

func (p *SemanticCachePlugin) Set(key string, value []byte, ttl time.Duration) error {
	if !p.cache.IsEnabled() {
		return nil
	}

	// Extract model and query from key
	parts := strings.SplitN(key, ":", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid cache key format")
	}
	
	model, query := parts[0], parts[1]
	
	// For setting, we need to add as pending first, then update
	cacheID, err := p.cache.AddPendingRequest(model, query, value)
	if err != nil {
		return err
	}
	
	return p.cache.UpdateWithResponse(cacheID, value)
}

func (p *SemanticCachePlugin) Invalidate(pattern string) error {
	// Semantic cache doesn't support pattern invalidation
	// This would need to be implemented if needed
	return nil
}

func (p *SemanticCachePlugin) IsEnabled() bool {
	return p.cache.IsEnabled()
}

// RedisCachePlugin implements CachePlugin with Redis backend
type RedisCachePlugin struct {
	name    string
	version string
	enabled bool
	// Redis client would go here
}

func NewRedisCachePlugin() *RedisCachePlugin {
	return &RedisCachePlugin{
		name:    PluginNameRedisCache,
		version: "1.0.0",
		enabled: false, // Disabled by default since it requires Redis
	}
}

func (p *RedisCachePlugin) Name() string    { return p.name }
func (p *RedisCachePlugin) Version() string { return p.version }

func (p *RedisCachePlugin) Initialize(config map[string]interface{}) error {
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	}
	// Initialize Redis client here
	return nil
}

func (p *RedisCachePlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	// Check Redis connection
	return nil
}

func (p *RedisCachePlugin) Shutdown() error {
	p.enabled = false
	// Close Redis connection
	return nil
}

func (p *RedisCachePlugin) Get(key string) ([]byte, bool, error) {
	if !p.enabled {
		return nil, false, nil
	}
	// Redis GET operation
	return nil, false, nil
}

func (p *RedisCachePlugin) Set(key string, value []byte, ttl time.Duration) error {
	if !p.enabled {
		return nil
	}
	// Redis SET operation with TTL
	return nil
}

func (p *RedisCachePlugin) Invalidate(pattern string) error {
	if !p.enabled {
		return nil
	}
	// Redis pattern deletion
	return nil
}

func (p *RedisCachePlugin) IsEnabled() bool {
	return p.enabled
}

// =============================================================================
// TOOLS PLUGINS
// =============================================================================

// SimilarityToolsPlugin implements ToolsPlugin usingtools database
type SimilarityToolsPlugin struct {
	name          string
	version       string
	toolsDatabase *tools.ToolsDatabase
	config        config.ToolsConfig
	logger        Logger
}

func NewSimilarityToolsPlugin(toolsDatabase *tools.ToolsDatabase, toolsConfig config.ToolsConfig, logger Logger) *SimilarityToolsPlugin {
	return &SimilarityToolsPlugin{
		name:          PluginNameSimilarityTools,
		version:       "1.0.0",
		toolsDatabase: toolsDatabase,
		config:        toolsConfig,
		logger:        logger,
	}
}

func (p *SimilarityToolsPlugin) Name() string    { return p.name }
func (p *SimilarityToolsPlugin) Version() string { return p.version }

func (p *SimilarityToolsPlugin) Initialize(config map[string]interface{}) error {
	// Tools database is already initialized and loaded
	return nil
}

func (p *SimilarityToolsPlugin) Health() error {
	if !p.toolsDatabase.IsEnabled() {
		return fmt.Errorf("tools database is disabled")
	}
	return nil
}

func (p *SimilarityToolsPlugin) Shutdown() error {
	// Tools database doesn't need explicit shutdown
	return nil
}

func (p *SimilarityToolsPlugin) SelectTools(content string, ctx *RequestContext) ([]openai.Tool, error) {
	if !p.toolsDatabase.IsEnabled() {
		return []openai.Tool{}, nil
	}

	topK := p.config.TopK
	if topK <= 0 {
		topK = DefaultTopK
	}

	selectedTools, err := p.toolsDatabase.FindSimilarTools(content, topK)
	if err != nil {
		if p.config.FallbackToEmpty {
			p.logger.Warn("Tools selection failed, falling back to empty",
				Field{Key: LogFieldRequestID, Value: ctx.RequestID},
				Field{Key: "error", Value: err.Error()})
			return []openai.Tool{}, nil
		}
		return nil, err
	}

	p.logger.Info("Tools selected",
		Field{Key: LogFieldRequestID, Value: ctx.RequestID},
		Field{Key: "tool_count", Value: len(selectedTools)})

	return selectedTools, nil
}

func (p *SimilarityToolsPlugin) SupportsAutoSelection() bool {
	return p.toolsDatabase.IsEnabled()
}

// CategoryToolsPlugin implements category-based tool selection
type CategoryToolsPlugin struct {
	name     string
	version  string
	enabled  bool
	toolsMap map[string][]openai.Tool
}

func NewCategoryToolsPlugin() *CategoryToolsPlugin {
	return &CategoryToolsPlugin{
		name:     PluginNameCategoryTools,
		version:  "1.0.0",
		enabled:  true,
		toolsMap: make(map[string][]openai.Tool),
	}
}

func (p *CategoryToolsPlugin) Name() string    { return p.name }
func (p *CategoryToolsPlugin) Version() string { return p.version }

func (p *CategoryToolsPlugin) Initialize(config map[string]interface{}) error {
	if enabled, ok := config["enabled"].(bool); ok {
		p.enabled = enabled
	}
	
	// Load category-to-tools mapping from config
	if toolsConfig, ok := config["category_tools"].(map[string]interface{}); ok {
		for category, toolsData := range toolsConfig {
			if toolsBytes, err := json.Marshal(toolsData); err == nil {
				var tools []openai.Tool
				if err := json.Unmarshal(toolsBytes, &tools); err == nil {
					p.toolsMap[category] = tools
				}
			}
		}
	}
	
	return nil
}

func (p *CategoryToolsPlugin) Health() error {
	if !p.enabled {
		return fmt.Errorf("plugin is disabled")
	}
	return nil
}

func (p *CategoryToolsPlugin) Shutdown() error {
	p.enabled = false
	return nil
}

func (p *CategoryToolsPlugin) SelectTools(content string, ctx *RequestContext) ([]openai.Tool, error) {
	if !p.enabled {
		return []openai.Tool{}, nil
	}

	// Get category from context if available
	if category, exists := ctx.GetMetadata("category"); exists {
		if categoryStr, ok := category.(string); ok {
			if tools, exists := p.toolsMap[categoryStr]; exists {
				return tools, nil
			}
		}
	}

	return []openai.Tool{}, nil
}

func (p *CategoryToolsPlugin) SupportsAutoSelection() bool {
	return p.enabled
}