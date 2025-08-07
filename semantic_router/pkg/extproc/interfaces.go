package extproc

import (
	"context"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
)

// Core processing interfaces for pluggable architecture

// RequestProcessor defines the main request processing interface
type RequestProcessor interface {
	ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error)
	ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error)
	ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error)
}

// SecurityChecker interface for pluggable security modules
type SecurityChecker interface {
	Name() string
	CheckContent(content string, ctx *RequestContext) (*SecurityResult, error)
	Priority() int // Lower numbers = higher priority
}

// SecurityResult represents the result of a security check
type SecurityResult struct {
	Allowed     bool
	BlockReason string
	Confidence  float32
	Details     map[string]interface{}
	Response    *ext_proc.ProcessingResponse // If set, immediately return this response
}

// ModelRouter interface for pluggable model selection
type ModelRouter interface {
	Name() string
	SelectModel(request *openai.OpenAIRequest, content string, ctx *RequestContext) (*ModelSelection, error)
	SupportsModel(model string) bool
	Priority() int
}

// ModelSelection represents the result of model selection
type ModelSelection struct {
	Model       string
	Confidence  float32
	Reason      string
	Metadata    map[string]interface{}
}

// CacheManager interface for pluggable caching strategies
type CacheManager interface {
	Name() string
	Get(key string) ([]byte, bool, error)
	Set(key string, value []byte, ttl time.Duration) error
	Invalidate(pattern string) error
	IsEnabled() bool
}

// ToolsSelector interface for pluggable tools selection
type ToolsSelector interface {
	Name() string
	SelectTools(content string, ctx *RequestContext) ([]openai.Tool, error)
	SupportsAutoSelection() bool
}

// Logger interface for structured logging
type Logger interface {
	Info(msg string, fields ...Field)
	Error(msg string, err error, fields ...Field)
	Debug(msg string, fields ...Field)
	Warn(msg string, fields ...Field)
}

type Field struct {
	Key   string
	Value interface{}
}

// MetricsCollector interface for observability
type MetricsCollector interface {
	RecordRequest(model string, latency time.Duration)
	RecordRouting(from, to string)
	RecordTokens(model string, promptTokens, completionTokens float64)
	RecordCacheHit(model string)
	RecordCacheMiss(model string)
	RecordError(component string, errorType string)
}

// Plugin represents a pluggable component
type Plugin interface {
	Name() string
	Version() string
	Initialize(config map[string]interface{}) error
	Health() error
	Shutdown() error
}

// ProcessorPlugin extends Plugin for request processors
type ProcessorPlugin interface {
	Plugin
	RequestProcessor
}

// SecurityPlugin extends Plugin for security checkers
type SecurityPlugin interface {
	Plugin
	SecurityChecker
}

// RoutingPlugin extends Plugin for model routers
type RoutingPlugin interface {
	Plugin
	ModelRouter
}

// CachePlugin extends Plugin for cache managers
type CachePlugin interface {
	Plugin
	CacheManager
}

// ToolsPlugin extends Plugin for tools selectors
type ToolsPlugin interface {
	Plugin
	ToolsSelector
}

// Request/Response wrapper types for cleaner interfaces
type HeadersRequest struct {
	Headers   map[string]string
	RequestID string
}

type BodyRequest struct {
	Body      []byte
	Headers   map[string]string
	RequestID string
}

type ProcessingResponse struct {
	Status         ResponseStatus
	HeaderMutation *HeaderMutation
	BodyMutation   *BodyMutation
	ImmediateResponse *ext_proc.ProcessingResponse // For immediate responses (errors, cache hits, etc.)
}

type ResponseStatus int

const (
	StatusContinue ResponseStatus = iota
	StatusBlock
	StatusImmediate
)

type HeaderMutation struct {
	Add    map[string]string
	Remove []string
}

type BodyMutation struct {
	Body []byte
}

// Middleware represents a processing middleware
type Middleware func(RequestProcessor) RequestProcessor

// PluginManager manages all plugins
type PluginManager interface {
	RegisterPlugin(plugin Plugin) error
	GetPlugin(name string) (Plugin, bool)
	GetPluginsByType(pluginType string) []Plugin
	UnregisterPlugin(name string) error
	ListPlugins() []Plugin
	LoadPluginsFromConfig(config []PluginConfig) error
	HealthCheck() error
	Shutdown() error
}

// PluginConfig represents plugin configuration
type PluginConfig struct {
	Name     string                 `yaml:"name"`
	Type     string                 `yaml:"type"`
	Enabled  bool                   `yaml:"enabled"`
	Priority int                    `yaml:"priority"`
	Config   map[string]interface{} `yaml:"config"`
}

// ProcessingStage represents different stages in request processing
type ProcessingStage string

const (
	StageHeaders      ProcessingStage = "headers"
	StageRequestBody  ProcessingStage = "request_body"
	StageResponseBody ProcessingStage = "response_body"
)

// RequestContext extended with plugin support
type RequestContext struct {
	Headers             map[string]string
	RequestID           string
	OriginalRequestBody []byte
	RequestModel        string
	RequestQuery        string
	StartTime           time.Time
	ProcessingStartTime time.Time
	Stage               ProcessingStage
	Metadata            map[string]interface{} // For plugins to store data
	Context             context.Context
}

// Add helper methods for RequestContext
func (ctx *RequestContext) SetMetadata(key string, value interface{}) {
	if ctx.Metadata == nil {
		ctx.Metadata = make(map[string]interface{})
	}
	ctx.Metadata[key] = value
}

func (ctx *RequestContext) GetMetadata(key string) (interface{}, bool) {
	if ctx.Metadata == nil {
		return nil, false
	}
	value, exists := ctx.Metadata[key]
	return value, exists
}