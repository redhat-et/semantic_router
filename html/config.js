/**
 * Configuration Management System for Semantic Router Frontend
 * Supports dual-mode operation (mock vs live), environment settings, and API configuration
 */

class ConfigurationManager {
  constructor() {
    // Default configuration values
    this.defaults = {
      // API Configuration
      apiEndpoint: 'http://localhost:8801/v1/chat/completions',
      environment: 'development',
      
      // Request Settings
      timeout: 30000,          // 30 seconds (based on testing)
      retryAttempts: 3,
      retryDelay: 1000,        // 1 second base delay
      retryBackoffMultiplier: 2,
      
      // Mode Configuration (key feature for dual-mode)
      mode: 'mock',            // 'mock' or 'live'
      
      // Performance Settings
      debounceDelay: 300,      // Milliseconds to debounce user input
      loadingMinDuration: 500, // Minimum loading time for better UX
      
      // UI Configuration
      animationSpeed: 'normal', // 'fast', 'normal', 'slow'
      theme: 'default',
      showDebugInfo: false,
      
      // Backend Service Configuration
      enablePiiDetection: true,
      enableClassification: true,
      enableModelSelection: true,
      
      // Development Configuration
      logLevel: 'info',        // 'debug', 'info', 'warn', 'error'
      enableTelemetry: false,
    };

    // Load and merge configuration
    this.config = this.loadConfiguration();
    
    // Validate the final configuration
    this.validateConfiguration();
    
    // Set up event listeners for config changes
    this.setupEventListeners();
  }

  /**
   * Load configuration from various sources in priority order:
   * 1. URL parameters (highest priority)
   * 2. Local storage
   * 3. Default values (lowest priority)
   */
  loadConfiguration() {
    let config = { ...this.defaults };
    
    // Load from localStorage if available
    const storedConfig = this.loadFromLocalStorage();
    if (storedConfig) {
      config = { ...config, ...storedConfig };
    }
    
    // Override with URL parameters
    const urlConfig = this.loadFromUrlParameters();
    if (urlConfig) {
      config = { ...config, ...urlConfig };
    }
    
    return config;
  }

  /**
   * Load configuration from localStorage
   */
  loadFromLocalStorage() {
    try {
      const stored = localStorage.getItem('semantic-router-config');
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.warn('Failed to load configuration from localStorage:', error);
    }
    return null;
  }

  /**
   * Load configuration from URL parameters
   */
  loadFromUrlParameters() {
    const params = new URLSearchParams(window.location.search);
    const config = {};
    
    // Map URL parameters to configuration keys
    const paramMappings = {
      'mode': 'mode',
      'api-endpoint': 'apiEndpoint',
      'environment': 'environment',
      'timeout': 'timeout',
      'debug': 'showDebugInfo',
      'log-level': 'logLevel',
      'theme': 'theme'
    };
    
    for (const [param, configKey] of Object.entries(paramMappings)) {
      if (params.has(param)) {
        let value = params.get(param);
        
        // Type conversion for specific parameters
        if (configKey === 'timeout' || configKey === 'retryAttempts') {
          value = parseInt(value, 10);
        } else if (configKey === 'showDebugInfo' || configKey === 'enableTelemetry') {
          value = value.toLowerCase() === 'true';
        }
        
        config[configKey] = value;
      }
    }
    
    return Object.keys(config).length > 0 ? config : null;
  }

  /**
   * Validate configuration values
   */
  validateConfiguration() {
    const errors = [];
    
    // Validate mode
    if (!['mock', 'live'].includes(this.config.mode)) {
      errors.push(`Invalid mode: ${this.config.mode}. Must be 'mock' or 'live'.`);
      this.config.mode = 'mock'; // Fallback to safe default
    }
    
    // Validate API endpoint
    if (this.config.mode === 'live' && !this.isValidUrl(this.config.apiEndpoint)) {
      errors.push(`Invalid API endpoint: ${this.config.apiEndpoint}`);
    }
    
    // Validate timeout
    if (this.config.timeout < 1000 || this.config.timeout > 300000) {
      errors.push(`Invalid timeout: ${this.config.timeout}. Must be between 1000 and 300000ms.`);
      this.config.timeout = this.defaults.timeout;
    }
    
    // Validate retry attempts
    if (this.config.retryAttempts < 0 || this.config.retryAttempts > 10) {
      errors.push(`Invalid retryAttempts: ${this.config.retryAttempts}. Must be between 0 and 10.`);
      this.config.retryAttempts = this.defaults.retryAttempts;
    }
    
    // Validate log level
    if (!['debug', 'info', 'warn', 'error'].includes(this.config.logLevel)) {
      errors.push(`Invalid logLevel: ${this.config.logLevel}`);
      this.config.logLevel = this.defaults.logLevel;
    }
    
    // Log validation errors
    if (errors.length > 0) {
      console.warn('Configuration validation errors:', errors);
    }
    
    return errors.length === 0;
  }

  /**
   * Helper function to validate URLs
   */
  isValidUrl(string) {
    try {
      new URL(string);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get a configuration value
   */
  get(key, defaultValue = undefined) {
    return this.config.hasOwnProperty(key) ? this.config[key] : defaultValue;
  }

  /**
   * Set a configuration value and optionally persist it
   */
  set(key, value, persist = true) {
    const oldValue = this.config[key];
    this.config[key] = value;
    
    // Validate the new configuration
    this.validateConfiguration();
    
    // Persist to localStorage if requested
    if (persist) {
      this.saveToLocalStorage();
    }
    
    // Emit change event
    this.emitConfigChange(key, value, oldValue);
    
    return this;
  }

  /**
   * Set multiple configuration values at once
   */
  setMultiple(updates, persist = true) {
    const oldConfig = { ...this.config };
    
    Object.assign(this.config, updates);
    this.validateConfiguration();
    
    if (persist) {
      this.saveToLocalStorage();
    }
    
    // Emit change events for each changed key
    for (const [key, value] of Object.entries(updates)) {
      this.emitConfigChange(key, value, oldConfig[key]);
    }
    
    return this;
  }

  /**
   * Save configuration to localStorage
   */
  saveToLocalStorage() {
    try {
      const configToStore = {
        ...this.config,
        // Don't store sensitive or temporary data
        lastUpdated: Date.now()
      };
      
      localStorage.setItem('semantic-router-config', JSON.stringify(configToStore));
    } catch (error) {
      console.warn('Failed to save configuration to localStorage:', error);
    }
  }

  /**
   * Reset configuration to defaults
   */
  reset(persist = true) {
    const oldConfig = { ...this.config };
    this.config = { ...this.defaults };
    
    if (persist) {
      localStorage.removeItem('semantic-router-config');
    }
    
    // Emit reset event
    this.emitConfigChange('__reset__', this.config, oldConfig);
    
    return this;
  }

  /**
   * Get the current operating mode
   */
  isLiveMode() {
    return this.config.mode === 'live';
  }

  /**
   * Get the current operating mode
   */
  isMockMode() {
    return this.config.mode === 'mock';
  }

  /**
   * Toggle between mock and live modes
   */
  toggleMode(persist = true) {
    const newMode = this.config.mode === 'mock' ? 'live' : 'mock';
    this.set('mode', newMode, persist);
    return newMode;
  }

  /**
   * Set up event listeners for configuration changes
   */
  setupEventListeners() {
    this.eventListeners = new Map();
    
    // Listen for storage events (when config changes in another tab)
    window.addEventListener('storage', (event) => {
      if (event.key === 'semantic-router-config' && event.newValue) {
        try {
          const newConfig = JSON.parse(event.newValue);
          this.config = { ...this.defaults, ...newConfig };
          this.validateConfiguration();
          this.emitConfigChange('__external_update__', this.config);
        } catch (error) {
          console.warn('Failed to parse updated configuration from storage:', error);
        }
      }
    });
  }

  /**
   * Add event listener for configuration changes
   */
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
    return this;
  }

  /**
   * Remove event listener
   */
  off(event, callback) {
    if (this.eventListeners.has(event)) {
      const listeners = this.eventListeners.get(event);
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
    return this;
  }

  /**
   * Emit configuration change event
   */
  emitConfigChange(key, newValue, oldValue = undefined) {
    const event = { key, newValue, oldValue, config: this.config };
    
    // Emit specific key event
    if (this.eventListeners.has(key)) {
      this.eventListeners.get(key).forEach(callback => {
        try {
          callback(event);
        } catch (error) {
          console.error('Error in config change callback:', error);
        }
      });
    }
    
    // Emit general change event
    if (this.eventListeners.has('change')) {
      this.eventListeners.get('change').forEach(callback => {
        try {
          callback(event);
        } catch (error) {
          console.error('Error in config change callback:', error);
        }
      });
    }
  }

  /**
   * Get configuration summary for debugging
   */
  getDebugInfo() {
    return {
      config: this.config,
      defaults: this.defaults,
      isValid: this.validateConfiguration(),
      source: {
        localStorage: !!this.loadFromLocalStorage(),
        urlParams: !!this.loadFromUrlParameters()
      }
    };
  }

  /**
   * Export configuration as JSON
   */
  export() {
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Import configuration from JSON
   */
  import(jsonString, persist = true) {
    try {
      const importedConfig = JSON.parse(jsonString);
      this.setMultiple(importedConfig, persist);
      return true;
    } catch (error) {
      console.error('Failed to import configuration:', error);
      return false;
    }
  }
}

// Create and export the global configuration instance
const configManager = new ConfigurationManager();

// Export for use in other files
window.SemanticRouterConfig = configManager;

// Also export the class for advanced usage
window.ConfigurationManager = ConfigurationManager; 