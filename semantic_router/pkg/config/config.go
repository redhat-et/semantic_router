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
		CategoryModel struct {
			ModelID             string  `yaml:"model_id"`
			Threshold           float32 `yaml:"threshold"`
			UseCPU              bool    `yaml:"use_cpu"`
			CategoryMappingPath string  `yaml:"category_mapping_path"`
		} `yaml:"category_model"`
		PIIModel struct {
			ModelID        string  `yaml:"model_id"`
			Threshold      float32 `yaml:"threshold"`
			UseCPU         bool    `yaml:"use_cpu"`
			PIIMappingPath string  `yaml:"pii_mapping_path"`
		} `yaml:"pii_model"`
		LoadAware bool `yaml:"load_aware"`
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

	// PII policy configuration for this model
	PIIPolicy PIIPolicy `yaml:"pii_policy,omitempty"`
}

// PIIPolicy represents the PII (Personally Identifiable Information) policy for a model
type PIIPolicy struct {
	// Allow all PII by default (true) or deny all by default (false)
	AllowByDefault bool `yaml:"allow_by_default"`

	// List of specific PII types to allow when AllowByDefault is false
	// This field explicitly lists the PII types that are allowed for this model
	PIITypes []string `yaml:"pii_types_allowed,omitempty"`
}

// PIIType constants for common PII types (matching pii_type_mapping.json)
const (
	PIITypeAge             = "AGE"               // Age information
	PIITypeCreditCard      = "CREDIT_CARD"       // Credit Card Number
	PIITypeDateTime        = "DATE_TIME"         // Date/Time information
	PIITypeDomainName      = "DOMAIN_NAME"       // Domain/Website names
	PIITypeEmailAddress    = "EMAIL_ADDRESS"     // Email Address
	PIITypeGPE             = "GPE"               // Geopolitical Entity
	PIITypeIBANCode        = "IBAN_CODE"         // International Bank Account Number
	PIITypeIPAddress       = "IP_ADDRESS"        // IP Address
	PIITypeNoPII           = "NO_PII"            // No PII detected
	PIITypeNRP             = "NRP"               // Nationality/Religious/Political group
	PIITypeOrganization    = "ORGANIZATION"      // Organization names
	PIITypePerson          = "PERSON"            // Person names
	PIITypePhoneNumber     = "PHONE_NUMBER"      // Phone Number
	PIITypeStreetAddress   = "STREET_ADDRESS"    // Physical Address
	PIITypeUSDriverLicense = "US_DRIVER_LICENSE" // US Driver's License Number
	PIITypeUSSSN           = "US_SSN"            // US Social Security Number
	PIITypeZipCode         = "ZIP_CODE"          // ZIP/Postal codes
)

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

// GetModelPIIPolicy returns the PII policy for a given model
// If the model is not found in the config, returns a default policy that allows all PII
func (c *RouterConfig) GetModelPIIPolicy(modelName string) PIIPolicy {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.PIIPolicy
	}
	// Default policy allows all PII
	return PIIPolicy{
		AllowByDefault: true,
		PIITypes:       []string{},
	}
}

// IsModelAllowedForPIIType checks if a model is allowed to process a specific PII type
func (c *RouterConfig) IsModelAllowedForPIIType(modelName string, piiType string) bool {
	policy := c.GetModelPIIPolicy(modelName)

	// If allow_by_default is true, all PII types are allowed unless explicitly denied
	if policy.AllowByDefault {
		return true
	}

	// If allow_by_default is false, only explicitly allowed PII types are permitted
	for _, allowedPII := range policy.PIITypes {
		if allowedPII == piiType {
			return true
		}
	}

	// PII type not found in allowed list and allow_by_default is false
	return false
}

// IsModelAllowedForPIITypes checks if a model is allowed to process any of the given PII types
func (c *RouterConfig) IsModelAllowedForPIITypes(modelName string, piiTypes []string) bool {
	for _, piiType := range piiTypes {
		if !c.IsModelAllowedForPIIType(modelName, piiType) {
			return false
		}
	}
	return true
}

// GetPIIClassifierConfig returns the PII classifier configuration
func (c *RouterConfig) GetPIIClassifierConfig() PIIClassifierConfig {
	return c.Classifier.PIIModel
}

// GetCategoryClassifierConfig returns the category classifier configuration
func (c *RouterConfig) GetCategoryClassifierConfig() CategoryClassifierConfig {
	return c.Classifier.CategoryModel
}

// IsPIIClassifierEnabled checks if PII classification is enabled
func (c *RouterConfig) IsPIIClassifierEnabled() bool {
	return c.Classifier.PIIModel.ModelID != "" && c.Classifier.PIIModel.PIIMappingPath != ""
}

// IsCategoryClassifierEnabled checks if category classification is enabled
func (c *RouterConfig) IsCategoryClassifierEnabled() bool {
	return c.Classifier.CategoryModel.ModelID != "" && c.Classifier.CategoryModel.CategoryMappingPath != ""
}
