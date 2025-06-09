package config

import (
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// BERT model configuration for Candle BERT similarity comparison
	BertModel struct {
		ModelID   string  `yaml:"model_id"`
		Threshold float32 `yaml:"threshold"`
		UseCPU    bool    `yaml:"use_cpu"`
	} `yaml:"bert_model"`

	// Classifier configuration for text classification
	Classifier struct {
		ModelID             string  `yaml:"model_id"`
		Threshold           float32 `yaml:"threshold"`
		UseCPU              bool    `yaml:"use_cpu"`
		CategoryMappingPath string  `yaml:"category_mapping_path"`
		LoadAware           bool    `yaml:"load_aware"`
	} `yaml:"classifier"`

	// PII detection configuration
	PIIDetection struct {
		Enabled         bool     `yaml:"enabled"`
		ModelID         string   `yaml:"model_id"`
		Threshold       float32  `yaml:"threshold"`
		UseCPU          bool     `yaml:"use_cpu"`
		PIITypes        []string `yaml:"pii_types"`
		BlockOnPII      bool     `yaml:"block_on_pii"`
		SanitizeEnabled bool     `yaml:"sanitize_enabled"`
	} `yaml:"pii_detection"`

	// Dual Classifier configuration (preferred over separate classifier/PII detection)
	DualClassifier struct {
		Enabled   bool   `yaml:"enabled"`
		ModelPath string `yaml:"model_path"`
		UseCPU    bool   `yaml:"use_cpu"`
	} `yaml:"dual_classifier"`

	// Categories for routing queries
	Categories []Category `yaml:"categories"`

	// Default LLM model to use if no match is found
	DefaultModel string `yaml:"default_model"`

	// Semantic cache configuration
	SemanticCache SemanticCacheConfig `yaml:"semantic_cache"`

	// Model parameters configuration
	ModelConfig map[string]ModelParams `yaml:"model_config"`

	// GPU configuration for TTFT calculation
	GPUConfig GPUConfig `yaml:"gpu_config"`
}

// SemanticCacheConfig represents configuration for the semantic cache
type SemanticCacheConfig struct {
	// Enable semantic caching
	Enabled bool `yaml:"enabled"`

	// Similarity threshold for cache hits (0.0-1.0)
	// If not specified, will use the BertModel.Threshold
	SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`

	// Maximum number of cache entries to keep
	MaxEntries int `yaml:"max_entries,omitempty"`

	// Time-to-live for cache entries in seconds (0 means no expiration)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`
}

// ModelParams represents configuration for model-specific parameters
type ModelParams struct {
	// Number of parameters in the model
	ParamCount float64 `yaml:"param_count"`

	// Default batch size for this model
	BatchSize float64 `yaml:"batch_size"`

	// Default context size for this model
	ContextSize float64 `yaml:"context_size"`
}

// GPUConfig represents configuration for GPU parameters used in TTFT calculation
type GPUConfig struct {
	// FLOPs performance in operations per second
	FLOPS float64 `yaml:"flops"`

	// HBM memory bandwidth in bytes per second
	HBM float64 `yaml:"hbm"`

	// Description of the GPU configuration (e.g., "A100-80G")
	Description string `yaml:"description"`
}

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SemanticCache.SimilarityThreshold != nil {
		return *c.SemanticCache.SimilarityThreshold
	}
	return c.BertModel.Threshold
}

// Category represents a category for routing queries
type ModelScore struct {
	Model string  `yaml:"model"`
	Score float64 `yaml:"score"`
}

type Category struct {
	Name        string       `yaml:"name"`
	Description string       `yaml:"description,omitempty"`
	ModelScores []ModelScore `yaml:"model_scores"`
}

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
)

// LoadConfig loads the configuration from the specified YAML file
func LoadConfig(configPath string) (*RouterConfig, error) {
	configOnce.Do(func() {
		data, err := os.ReadFile(configPath)
		if err != nil {
			configErr = fmt.Errorf("failed to read config file: %w", err)
			return
		}

		config = &RouterConfig{}
		if err := yaml.Unmarshal(data, config); err != nil {
			configErr = fmt.Errorf("failed to parse config file: %w", err)
			return
		}
	})

	if configErr != nil {
		return nil, configErr
	}
	return config, nil
}

// GetConfig returns the current configuration
func GetConfig() *RouterConfig {
	return config
}

// GetCategoryDescriptions returns all category descriptions for similarity matching
func (c *RouterConfig) GetCategoryDescriptions() []string {
	var descriptions []string
	for _, category := range c.Categories {
		if category.Description != "" {
			descriptions = append(descriptions, category.Description)
		} else {
			// Use category name if no description is available
			descriptions = append(descriptions, category.Name)
		}
	}
	return descriptions
}

// GetModelForCategoryIndex returns the best LLM model name for the category at the given index
func (c *RouterConfig) GetModelForCategoryIndex(index int) string {
	if index < 0 || index >= len(c.Categories) {
		return c.DefaultModel
	}

	category := c.Categories[index]
	if len(category.ModelScores) > 0 {
		return category.ModelScores[0].Model
	}

	// Fall back to default model if category has no models
	return c.DefaultModel
}

// GetModelParamCount returns the parameter count for a given model
// If the model is not found in the config, returns the default value
func (c *RouterConfig) GetModelParamCount(modelName string, defaultValue float64) float64 {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.ParamCount
	}
	return defaultValue
}

// GetModelBatchSize returns the batch size for a given model
// If the model is not found in the config, returns the default value
func (c *RouterConfig) GetModelBatchSize(modelName string, defaultValue float64) float64 {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.BatchSize
	}
	return defaultValue
}

// GetModelContextSize returns the context size for a given model
// If the model is not found in the config, returns the default value
func (c *RouterConfig) GetModelContextSize(modelName string, defaultValue float64) float64 {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.ContextSize
	}
	return defaultValue
}
