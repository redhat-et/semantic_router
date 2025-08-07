package extproc

import "time"

// Processing constants
const (
	// Default values
	DefaultTopK         = 3
	DefaultCacheEntries = 1000
	DefaultCacheTTL     = 3600 // seconds
	DefaultTimeout      = 30 * time.Second
	
	// Model constants
	AutoModel = "auto"
	AutoTools = "auto"
	
	// Header names
	HeaderRequestID     = "x-request-id"
	HeaderContentType   = "content-type"
	HeaderContentLength = "content-length"
	HeaderAuthorization = "authorization"
	
	// Content types
	ContentTypeJSON = "application/json"
	
	// Cache constants
	CacheKeyModelPrefix = "model:"
	CacheKeyToolsPrefix = "tools:"
	
	// Metrics constants
	MetricRequestTotal      = "extproc_requests_total"
	MetricRequestDuration   = "extproc_request_duration_seconds"
	MetricModelRouting      = "extproc_model_routing_total"
	MetricCacheHits         = "extproc_cache_hits_total"
	MetricCacheMisses       = "extproc_cache_misses_total"
	MetricErrors            = "extproc_errors_total"
	MetricSecurityBlocks    = "extproc_security_blocks_total"
	
	// Plugin types
	PluginTypeSecurity  = "security"
	PluginTypeRouting   = "routing"
	PluginTypeCache     = "cache"
	PluginTypeTools     = "tools"
	PluginTypeProcessor = "processor"
	
	// Security thresholds
	DefaultPIIThreshold        = 0.8
	DefaultJailbreakThreshold  = 0.7
	DefaultSimilarityThreshold = 0.85
	
	// Circuit breaker constants
	DefaultFailureThreshold = 5
	DefaultResetTimeout     = 60 * time.Second
	
	// Retry constants
	DefaultMaxRetries    = 3
	DefaultRetryDelay    = 100 * time.Millisecond
	DefaultMaxRetryDelay = 5 * time.Second
)

// Error codes
const (
	ErrorCodeValidationFailed    = "VALIDATION_FAILED"
	ErrorCodeInvalidRequest      = "INVALID_REQUEST"
	ErrorCodeMissingHeaders      = "MISSING_HEADERS"
	ErrorCodeEmptyBody          = "EMPTY_BODY"
	ErrorCodeInvalidJSON        = "INVALID_JSON"
	ErrorCodeModelNotFound      = "MODEL_NOT_FOUND"
	ErrorCodePIIDetected        = "PII_DETECTED"
	ErrorCodeJailbreakDetected  = "JAILBREAK_DETECTED"
	ErrorCodePluginFailure      = "PLUGIN_FAILURE"
	ErrorCodeTimeout            = "TIMEOUT"
	ErrorCodeCircuitBreakerOpen = "CIRCUIT_BREAKER_OPEN"
	ErrorCodeCacheFailure       = "CACHE_FAILURE"
	ErrorCodeConfigurationError = "CONFIGURATION_ERROR"
)

// Plugin priorities (lower number = higher priority)
const (
	PriorityHigh   = 10
	PriorityMedium = 50
	PriorityLow    = 100
)

// Built-in plugin names
const (
	PluginNamePIIDetector           = "pii_detector"
	PluginNameJailbreakDetector     = "jailbreak_detector"
	PluginNameClassificationRouter  = "classification_router"
	PluginNameLoadBalancerRouter    = "load_balancer_router"
	PluginNameSemanticCache         = "semantic_cache"
	PluginNameRedisCache           = "redis_cache"
	PluginNameSimilarityTools      = "similarity_tools"
	PluginNameCategoryTools        = "category_tools"
)

// Processing stages
const (
	StageInit               = "init"
	StageHeadersProcessing  = "headers_processing"
	StageBodyProcessing     = "body_processing"
	StageSecurityChecks     = "security_checks"
	StageCacheLookup        = "cache_lookup"
	StageModelSelection     = "model_selection"
	StageToolsSelection     = "tools_selection"
	StageResponseProcessing = "response_processing"
	StageCacheUpdate        = "cache_update"
	StageComplete          = "complete"
)

// Log levels and contexts
const (
	LogFieldRequestID   = "request_id"
	LogFieldStage       = "stage"
	LogFieldDuration    = "duration_ms"
	LogFieldModel       = "model"
	LogFieldPlugin      = "plugin"
	LogFieldError       = "error"
	LogFieldComponent   = "component"
	LogFieldOperation   = "operation"
	LogFieldBodySize    = "body_size"
	LogFieldCacheHit    = "cache_hit"
	LogFieldConfidence  = "confidence"
	LogFieldPIITypes    = "pii_types"
	LogFieldJailbreakType = "jailbreak_type"
)

// Configuration defaults
var (
	DefaultPluginConfig = map[string]interface{}{
		"enabled": true,
		"timeout": DefaultTimeout,
	}
	
	DefaultSecurityConfig = map[string]interface{}{
		"pii_threshold":       DefaultPIIThreshold,
		"jailbreak_threshold": DefaultJailbreakThreshold,
		"enabled":            true,
	}
	
	DefaultCacheConfig = map[string]interface{}{
		"enabled":              true,
		"max_entries":          DefaultCacheEntries,
		"ttl_seconds":          DefaultCacheTTL,
		"similarity_threshold": DefaultSimilarityThreshold,
	}
	
	DefaultToolsConfig = map[string]interface{}{
		"enabled":              true,
		"top_k":               DefaultTopK,
		"similarity_threshold": DefaultSimilarityThreshold,
		"fallback_to_empty":   true,
	}
)