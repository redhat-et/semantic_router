package extproc

import (
	"fmt"
	"sort"
	"sync"
)

// DefaultPluginManager implements the PluginManager interface
type DefaultPluginManager struct {
	plugins map[string]Plugin
	mutex   sync.RWMutex
	logger  Logger
}

// NewPluginManager creates a new plugin manager
func NewPluginManager(logger Logger) *DefaultPluginManager {
	return &DefaultPluginManager{
		plugins: make(map[string]Plugin),
		logger:  logger,
	}
}

// RegisterPlugin registers a new plugin
func (pm *DefaultPluginManager) RegisterPlugin(plugin Plugin) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	name := plugin.Name()
	if _, exists := pm.plugins[name]; exists {
		return fmt.Errorf("plugin %s already registered", name)
	}

	pm.plugins[name] = plugin
	pm.logger.Info("Plugin registered", Field{Key: "name", Value: name}, Field{Key: "version", Value: plugin.Version()})
	return nil
}

// GetPlugin retrieves a plugin by name
func (pm *DefaultPluginManager) GetPlugin(name string) (Plugin, bool) {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	plugin, exists := pm.plugins[name]
	return plugin, exists
}

// GetPluginsByType returns all plugins of a specific type, sorted by priority
func (pm *DefaultPluginManager) GetPluginsByType(pluginType string) []Plugin {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	var plugins []Plugin
	for _, plugin := range pm.plugins {
		switch pluginType {
		case "security":
			if _, ok := plugin.(SecurityPlugin); ok {
				plugins = append(plugins, plugin)
			}
		case "routing":
			if _, ok := plugin.(RoutingPlugin); ok {
				plugins = append(plugins, plugin)
			}
		case "cache":
			if _, ok := plugin.(CachePlugin); ok {
				plugins = append(plugins, plugin)
			}
		case "tools":
			if _, ok := plugin.(ToolsPlugin); ok {
				plugins = append(plugins, plugin)
			}
		case "processor":
			if _, ok := plugin.(ProcessorPlugin); ok {
				plugins = append(plugins, plugin)
			}
		}
	}

	// Sort by priority (lower number = higher priority)
	sort.Slice(plugins, func(i, j int) bool {
		var priorityI, priorityJ int
		
		switch p := plugins[i].(type) {
		case SecurityPlugin:
			priorityI = p.Priority()
		case RoutingPlugin:
			priorityI = p.Priority()
		default:
			priorityI = 100 // Default priority
		}

		switch p := plugins[j].(type) {
		case SecurityPlugin:
			priorityJ = p.Priority()
		case RoutingPlugin:
			priorityJ = p.Priority()
		default:
			priorityJ = 100 // Default priority
		}

		return priorityI < priorityJ
	})

	return plugins
}

// UnregisterPlugin removes a plugin
func (pm *DefaultPluginManager) UnregisterPlugin(name string) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	plugin, exists := pm.plugins[name]
	if !exists {
		return fmt.Errorf("plugin %s not found", name)
	}

	// Shutdown the plugin before removing
	if err := plugin.Shutdown(); err != nil {
		pm.logger.Error("Error shutting down plugin", err, Field{Key: "name", Value: name})
	}

	delete(pm.plugins, name)
	pm.logger.Info("Plugin unregistered", Field{Key: "name", Value: name})
	return nil
}

// ListPlugins returns all registered plugins
func (pm *DefaultPluginManager) ListPlugins() []Plugin {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	plugins := make([]Plugin, 0, len(pm.plugins))
	for _, plugin := range pm.plugins {
		plugins = append(plugins, plugin)
	}
	return plugins
}

// LoadPluginsFromConfig loads and initializes plugins from configuration
func (pm *DefaultPluginManager) LoadPluginsFromConfig(configs []PluginConfig) error {
	for _, config := range configs {
		if !config.Enabled {
			pm.logger.Info("Skipping disabled plugin", Field{Key: "name", Value: config.Name})
			continue
		}

		// For now, we'll register built-in plugins based on type
		plugin, err := pm.createBuiltinPlugin(config)
		if err != nil {
			pm.logger.Error("Failed to create plugin", err, Field{Key: "name", Value: config.Name})
			continue
		}

		if err := plugin.Initialize(config.Config); err != nil {
			pm.logger.Error("Failed to initialize plugin", err, Field{Key: "name", Value: config.Name})
			continue
		}

		if err := pm.RegisterPlugin(plugin); err != nil {
			pm.logger.Error("Failed to register plugin", err, Field{Key: "name", Value: config.Name})
			continue
		}
	}

	return nil
}

// createBuiltinPlugin creates built-in plugins based on configuration
func (pm *DefaultPluginManager) createBuiltinPlugin(config PluginConfig) (Plugin, error) {
	switch config.Type {
	case "security":
		return pm.createSecurityPlugin(config)
	case "routing":
		return pm.createRoutingPlugin(config)
	case "cache":
		return pm.createCachePlugin(config)
	case "tools":
		return pm.createToolsPlugin(config)
	default:
		return nil, fmt.Errorf("unknown plugin type: %s", config.Type)
	}
}

// Helper methods to create specific plugin types
func (pm *DefaultPluginManager) createSecurityPlugin(config PluginConfig) (Plugin, error) {
	return nil, fmt.Errorf("dynamic plugin creation not supported - plugins should be registered directly")
}

func (pm *DefaultPluginManager) createRoutingPlugin(config PluginConfig) (Plugin, error) {
	return nil, fmt.Errorf("dynamic plugin creation not supported - plugins should be registered directly")
}

func (pm *DefaultPluginManager) createCachePlugin(config PluginConfig) (Plugin, error) {
	return nil, fmt.Errorf("dynamic plugin creation not supported - plugins should be registered directly")
}

func (pm *DefaultPluginManager) createToolsPlugin(config PluginConfig) (Plugin, error) {
	return nil, fmt.Errorf("dynamic plugin creation not supported - plugins should be registered directly")
}

// Health checks all plugins
func (pm *DefaultPluginManager) HealthCheck() error {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	for name, plugin := range pm.plugins {
		if err := plugin.Health(); err != nil {
			return fmt.Errorf("plugin %s health check failed: %w", name, err)
		}
	}
	return nil
}

// Shutdown gracefully shuts down all plugins
func (pm *DefaultPluginManager) Shutdown() error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	var lastErr error
	for name, plugin := range pm.plugins {
		if err := plugin.Shutdown(); err != nil {
			pm.logger.Error("Error shutting down plugin", err, Field{Key: "name", Value: name})
			lastErr = err
		}
	}

	pm.plugins = make(map[string]Plugin)
	return lastErr
}