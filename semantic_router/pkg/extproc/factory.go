package extproc

import (
	"fmt"
	"time"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
)

// RouterFactory creates configured router instances
type RouterFactory struct {
	logger Logger
}

// NewRouterFactory creates a new router factory
func NewRouterFactory(logger Logger) *RouterFactory {
	return &RouterFactory{logger: logger}
}

// CreateRouter creates a fully configured plugin-based router
func (f *RouterFactory) CreateRouter(configPath string) (*OpenAIRouter, error) {
	// Load configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Create logger if not provided
	logger := f.logger
	if logger == nil {
		logger = NewStructuredLogger("extproc", LogLevelInfo, false)
	}

	// Create plugin manager
	pluginManager := NewPluginManager(logger)

	// Register built-in plugins
	if err := f.registerBuiltinPlugins(pluginManager, cfg); err != nil {
		return nil, fmt.Errorf("failed to register plugins: %w", err)
	}

	// Create error handler
	errorHandler := NewDefaultErrorHandler(logger)

	// Create metrics collector (placeholder implementation)
	metrics := NewDefaultMetricsCollector()

	// Create dependencies
	deps := RouterDependencies{
		Config:        cfg,
		Logger:        logger,
		PluginManager: pluginManager,
		ErrorHandler:  errorHandler,
		Metrics:       metrics,
	}

	// Create router
	router, err := NewOpenAIRouterWithDeps(deps)
	if err != nil {
		return nil, fmt.Errorf("failed to create router: %w", err)
	}

	logger.Info("Plugin-based router created successfully",
		Field{Key: "plugin_count", Value: len(pluginManager.ListPlugins())})

	return router, nil
}

// registerBuiltinPlugins registers and configures built-in plugins based on config
func (f *RouterFactory) registerBuiltinPlugins(pm *DefaultPluginManager, cfg *config.RouterConfig) error {
	// Use the same registration logic as in router.go
	return registerBuiltinPlugins(pm, cfg, f.logger)
}
// DefaultMetricsCollector is a simple metrics collector implementation
type DefaultMetricsCollector struct {
	logger Logger
}

// NewDefaultMetricsCollector creates a new default metrics collector
func NewDefaultMetricsCollector() *DefaultMetricsCollector {
	return &DefaultMetricsCollector{
		logger: NewStructuredLogger("metrics", LogLevelInfo, false),
	}
}

func (m *DefaultMetricsCollector) RecordRequest(model string, latency time.Duration) {
	m.logger.Info("Request recorded",
		Field{Key: LogFieldModel, Value: model},
		Field{Key: LogFieldDuration, Value: latency.Milliseconds()})
}

func (m *DefaultMetricsCollector) RecordRouting(from, to string) {
	m.logger.Info("Routing recorded",
		Field{Key: "from_model", Value: from},
		Field{Key: "to_model", Value: to})
}

func (m *DefaultMetricsCollector) RecordTokens(model string, promptTokens, completionTokens float64) {
	m.logger.Info("Tokens recorded",
		Field{Key: LogFieldModel, Value: model},
		Field{Key: "prompt_tokens", Value: promptTokens},
		Field{Key: "completion_tokens", Value: completionTokens})
}

func (m *DefaultMetricsCollector) RecordCacheHit(model string) {
	m.logger.Info("Cache hit recorded", Field{Key: LogFieldModel, Value: model})
}

func (m *DefaultMetricsCollector) RecordCacheMiss(model string) {
	m.logger.Debug("Cache miss recorded", Field{Key: LogFieldModel, Value: model})
}

func (m *DefaultMetricsCollector) RecordError(component string, errorType string) {
	m.logger.Error("Error recorded", nil,
		Field{Key: LogFieldComponent, Value: component},
		Field{Key: "error_type", Value: errorType})
}

// PluginConfig represents configuration for loading plugins
type PluginConfigList struct {
	Plugins []PluginConfig `yaml:"plugins"`
}

// LoadPluginsFromConfig loads plugins from a configuration struct
func LoadPluginsFromConfig(pm PluginManager, configs []PluginConfig) error {
	return pm.LoadPluginsFromConfig(configs)
}

// Simple router creation function  
func CreateRouter(configPath string, logger Logger, customPlugins []Plugin) (*OpenAIRouter, error) {
	factory := NewRouterFactory(logger)
	
	router, err := factory.CreateRouter(configPath)
	if err != nil {
		return nil, err
	}

	// Register any custom plugins
	for _, plugin := range customPlugins {
		if err := router.pluginManager.RegisterPlugin(plugin); err != nil {
			return nil, fmt.Errorf("failed to register custom plugin %s: %w", plugin.Name(), err)
		}
	}

	return router, nil
}

// Health check for the router
func (f *RouterFactory) HealthCheck(router *OpenAIRouter) error {
	return router.pluginManager.HealthCheck()
}