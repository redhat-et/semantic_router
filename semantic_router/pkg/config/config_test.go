package config_test

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
)

func TestConfig(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Config Suite")
}

var _ = Describe("Config Package", func() {
	var (
		tempDir    string
		configFile string
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "config_test")
		Expect(err).NotTo(HaveOccurred())
		configFile = filepath.Join(tempDir, "config.yaml")
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
		// Reset the singleton config for next test
		config.ResetConfig()
	})

	Describe("LoadConfig", func() {
		Context("with valid YAML configuration", func() {
			BeforeEach(func() {
				validConfig := `
bert_model:
  model_id: "test-bert-model"
  threshold: 0.8
  use_cpu: true

classifier:
  category_model:
    model_id: "test-category-model"
    threshold: 0.7
    use_cpu: false
    use_modernbert: true
    category_mapping_path: "/path/to/category.json"
  pii_model:
    model_id: "test-pii-model"
    threshold: 0.6
    use_cpu: true
    use_modernbert: false
    pii_mapping_path: "/path/to/pii.json"
  load_aware: true

categories:
  - name: "general"
    description: "General purpose tasks"
    model_scores:
      - model: "gpt-4"
        score: 0.9
      - model: "gpt-3.5-turbo"
        score: 0.8

default_model: "gpt-3.5-turbo"

semantic_cache:
  enabled: true
  similarity_threshold: 0.9
  max_entries: 1000
  ttl_seconds: 3600

prompt_guard:
  enabled: true
  model_id: "test-jailbreak-model"
  threshold: 0.5
  use_cpu: false
  use_modernbert: true
  jailbreak_mapping_path: "/path/to/jailbreak.json"

model_config:
  "gpt-4":
    param_count: 1000000000
    batch_size: 32
    context_size: 8192
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["NO_PII", "ORGANIZATION"]
  "gpt-3.5-turbo":
    param_count: 175000000
    batch_size: 64
    context_size: 4096
    pii_policy:
      allow_by_default: true

gpu_config:
  flops: 312000000000000
  hbm: 2000000000000
  description: "A100-80G"

tools:
  enabled: true
  top_k: 5
  similarity_threshold: 0.8
  tools_db_path: "/path/to/tools.json"
  fallback_to_empty: true
`
				err := os.WriteFile(configFile, []byte(validConfig), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load configuration successfully", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())

				// Verify BERT model config
				Expect(cfg.BertModel.ModelID).To(Equal("test-bert-model"))
				Expect(cfg.BertModel.Threshold).To(Equal(float32(0.8)))
				Expect(cfg.BertModel.UseCPU).To(BeTrue())

				// Verify classifier config
				Expect(cfg.Classifier.CategoryModel.ModelID).To(Equal("test-category-model"))
				Expect(cfg.Classifier.CategoryModel.UseModernBERT).To(BeTrue())
				Expect(cfg.Classifier.PIIModel.UseModernBERT).To(BeFalse())
				Expect(cfg.Classifier.LoadAware).To(BeTrue())

				// Verify categories
				Expect(cfg.Categories).To(HaveLen(1))
				Expect(cfg.Categories[0].Name).To(Equal("general"))
				Expect(cfg.Categories[0].ModelScores).To(HaveLen(2))

				// Verify default model
				Expect(cfg.DefaultModel).To(Equal("gpt-3.5-turbo"))

				// Verify semantic cache
				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.9)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(1000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(3600))

				// Verify prompt guard
				Expect(cfg.PromptGuard.Enabled).To(BeTrue())
				Expect(cfg.PromptGuard.ModelID).To(Equal("test-jailbreak-model"))
				Expect(cfg.PromptGuard.UseModernBERT).To(BeTrue())

				// Verify model config
				Expect(cfg.ModelConfig).To(HaveKey("gpt-4"))
				Expect(cfg.ModelConfig["gpt-4"].ParamCount).To(Equal(float64(1000000000)))
				Expect(cfg.ModelConfig["gpt-4"].PIIPolicy.AllowByDefault).To(BeFalse())
				Expect(cfg.ModelConfig["gpt-4"].PIIPolicy.PIITypes).To(ContainElements("NO_PII", "ORGANIZATION"))

				// Verify GPU config
				Expect(cfg.GPUConfig.FLOPS).To(Equal(float64(312000000000000)))
				Expect(cfg.GPUConfig.Description).To(Equal("A100-80G"))

				// Verify tools config
				Expect(cfg.Tools.Enabled).To(BeTrue())
				Expect(cfg.Tools.TopK).To(Equal(5))
				Expect(*cfg.Tools.SimilarityThreshold).To(Equal(float32(0.8)))
			})

			It("should return the same config instance on subsequent calls (singleton)", func() {
				cfg1, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				cfg2, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg1).To(BeIdenticalTo(cfg2))
			})
		})

		Context("with missing config file", func() {
			It("should return an error", func() {
				cfg, err := config.LoadConfig("/nonexistent/config.yaml")
				Expect(err).To(HaveOccurred())
				Expect(cfg).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to read config file"))
			})
		})

		Context("with invalid YAML syntax", func() {
			BeforeEach(func() {
				invalidYAML := `
bert_model:
  model_id: "test-model"
  invalid: [ unclosed array
`
				err := os.WriteFile(configFile, []byte(invalidYAML), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return a parsing error", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).To(HaveOccurred())
				Expect(cfg).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to parse config file"))
			})
		})

		Context("with empty config file", func() {
			BeforeEach(func() {
				err := os.WriteFile(configFile, []byte(""), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load successfully with zero values", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())
				Expect(cfg.BertModel.ModelID).To(BeEmpty())
				Expect(cfg.DefaultModel).To(BeEmpty())
			})
		})

		Context("concurrent access", func() {
			BeforeEach(func() {
				validConfig := `
bert_model:
  model_id: "test-model"
  threshold: 0.8
default_model: "gpt-3.5-turbo"
`
				err := os.WriteFile(configFile, []byte(validConfig), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle concurrent LoadConfig calls safely", func() {
				const numGoroutines = 10
				var wg sync.WaitGroup
				results := make([]*config.RouterConfig, numGoroutines)
				errors := make([]error, numGoroutines)

				wg.Add(numGoroutines)
				for i := 0; i < numGoroutines; i++ {
					go func(index int) {
						defer wg.Done()
						cfg, err := config.LoadConfig(configFile)
						results[index] = cfg
						errors[index] = err
					}(i)
				}

				wg.Wait()

				// All calls should succeed
				for i := 0; i < numGoroutines; i++ {
					Expect(errors[i]).NotTo(HaveOccurred())
					Expect(results[i]).NotTo(BeNil())
				}

				// All should return the same instance
				for i := 1; i < numGoroutines; i++ {
					Expect(results[i]).To(BeIdenticalTo(results[0]))
				}
			})
		})
	})

	Describe("GetCacheSimilarityThreshold", func() {
		Context("when semantic cache has explicit threshold", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.8
semantic_cache:
  similarity_threshold: 0.9
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the semantic cache threshold", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.9)))
			})
		})

		Context("when semantic cache has no explicit threshold", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.8
semantic_cache:
  enabled: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the BERT model threshold", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.8)))
			})
		})
	})

	Describe("GetModelForCategoryIndex", func() {
		BeforeEach(func() {
			configContent := `
categories:
  - name: "category1"
    model_scores:
      - model: "model1"
        score: 0.9
      - model: "model2"
        score: 0.8
  - name: "category2"
    model_scores:
      - model: "model3"
        score: 0.95
default_model: "default-model"
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())
		})

		Context("with valid category index", func() {
			It("should return the best model for the category", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(0)
				Expect(model).To(Equal("model1"))

				model = cfg.GetModelForCategoryIndex(1)
				Expect(model).To(Equal("model3"))
			})
		})

		Context("with invalid category index", func() {
			It("should return the default model for negative index", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(-1)
				Expect(model).To(Equal("default-model"))
			})

			It("should return the default model for index beyond range", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(10)
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("with category having no models", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "empty_category"
    model_scores: []
default_model: "fallback-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the default model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(0)
				Expect(model).To(Equal("fallback-model"))
			})
		})
	})

	Describe("PII Policy Functions", func() {
		BeforeEach(func() {
			configContent := `
model_config:
  "strict-model":
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["NO_PII", "ORGANIZATION"]
  "permissive-model":
    pii_policy:
      allow_by_default: true
  "unconfigured-model":
    param_count: 1000000
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())
		})

		Describe("GetModelPIIPolicy", func() {
			It("should return configured PII policy for existing model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				policy := cfg.GetModelPIIPolicy("strict-model")
				Expect(policy.AllowByDefault).To(BeFalse())
				Expect(policy.PIITypes).To(ContainElements("NO_PII", "ORGANIZATION"))

				policy = cfg.GetModelPIIPolicy("permissive-model")
				Expect(policy.AllowByDefault).To(BeTrue())
			})

			It("should return default allow-all policy for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				policy := cfg.GetModelPIIPolicy("non-existent-model")
				Expect(policy.AllowByDefault).To(BeTrue())
				Expect(policy.PIITypes).To(BeEmpty())
			})
		})

		Describe("IsModelAllowedForPIIType", func() {
			It("should allow all PII types when allow_by_default is true", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypePerson)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypeCreditCard)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypeEmailAddress)).To(BeTrue())
			})

			It("should only allow explicitly permitted PII types when allow_by_default is false", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Should allow explicitly listed PII types
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeNoPII)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeOrganization)).To(BeTrue())

				// Should deny non-listed PII types
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypePerson)).To(BeFalse())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeCreditCard)).To(BeFalse())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeEmailAddress)).To(BeFalse())
			})

			It("should handle unknown models with default allow-all policy", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIIType("unknown-model", config.PIITypePerson)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("unknown-model", config.PIITypeCreditCard)).To(BeTrue())
			})
		})

		Describe("IsModelAllowedForPIITypes", func() {
			It("should return true when all PII types are allowed", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				piiTypes := []string{config.PIITypeNoPII, config.PIITypeOrganization}
				Expect(cfg.IsModelAllowedForPIITypes("strict-model", piiTypes)).To(BeTrue())
			})

			It("should return false when any PII type is not allowed", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				piiTypes := []string{config.PIITypeNoPII, config.PIITypePerson}
				Expect(cfg.IsModelAllowedForPIITypes("strict-model", piiTypes)).To(BeFalse())
			})

			It("should return true for empty PII types list", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIITypes("strict-model", []string{})).To(BeTrue())
			})
		})
	})

	Describe("Feature Enablement Checks", func() {
		Context("PII Classifier", func() {
			It("should return true when properly configured", func() {
				configContent := `
classifier:
  pii_model:
    model_id: "pii-model"
    pii_mapping_path: "/path/to/pii.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeTrue())
			})

			It("should return false when model_id is missing", func() {
				configContent := `
classifier:
  pii_model:
    pii_mapping_path: "/path/to/pii.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeFalse())
			})

			It("should return false when mapping path is missing", func() {
				configContent := `
classifier:
  pii_model:
    model_id: "pii-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeFalse())
			})
		})

		Context("Category Classifier", func() {
			It("should return true when properly configured", func() {
				configContent := `
classifier:
  category_model:
    model_id: "category-model"
    category_mapping_path: "/path/to/category.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsCategoryClassifierEnabled()).To(BeTrue())
			})

			It("should return false when not configured", func() {
				// Create an empty config file
				err := os.WriteFile(configFile, []byte(""), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("Prompt Guard", func() {
			It("should return true when fully enabled and configured", func() {
				configContent := `
prompt_guard:
  enabled: true
  model_id: "jailbreak-model"
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeTrue())
			})

			It("should return false when disabled", func() {
				configContent := `
prompt_guard:
  enabled: false
  model_id: "jailbreak-model"
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeFalse())
			})

			It("should return false when model_id is missing", func() {
				configContent := `
prompt_guard:
  enabled: true
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeFalse())
			})
		})
	})

	Describe("Model Parameter Functions", func() {
		BeforeEach(func() {
			configContent := `
model_config:
  "configured-model":
    param_count: 175000000
    batch_size: 32
    context_size: 4096
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())
		})

		Describe("GetModelParamCount", func() {
			It("should return configured value for existing model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				count := cfg.GetModelParamCount("configured-model", 1000000)
				Expect(count).To(Equal(float64(175000000)))
			})

			It("should return default value for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				count := cfg.GetModelParamCount("unknown-model", 999999)
				Expect(count).To(Equal(float64(999999)))
			})
		})

		Describe("GetModelBatchSize", func() {
			It("should return configured value for existing model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				batchSize := cfg.GetModelBatchSize("configured-model", 16)
				Expect(batchSize).To(Equal(float64(32)))
			})

			It("should return default value for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				batchSize := cfg.GetModelBatchSize("unknown-model", 64)
				Expect(batchSize).To(Equal(float64(64)))
			})
		})

		Describe("GetModelContextSize", func() {
			It("should return configured value for existing model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				contextSize := cfg.GetModelContextSize("configured-model", 2048)
				Expect(contextSize).To(Equal(float64(4096)))
			})

			It("should return default value for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				contextSize := cfg.GetModelContextSize("unknown-model", 8192)
				Expect(contextSize).To(Equal(float64(8192)))
			})
		})
	})

	Describe("GetCategoryDescriptions", func() {
		Context("with categories having descriptions", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "category1"
    description: "Description for category 1"
  - name: "category2"
    description: "Description for category 2"
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return all category descriptions", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(HaveLen(2))
				Expect(descriptions).To(ContainElements(
					"Description for category 1",
					"Description for category 2",
				))
			})
		})

		Context("with categories missing descriptions", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "category1"
    description: "Has description"
  - name: "category2"
    # No description field
`
				err := os.WriteFile(configFile, []byte(configContent), 0644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should use category name as fallback for missing descriptions", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(HaveLen(2))
				Expect(descriptions).To(ContainElements(
					"Has description",
					"category2",
				))
			})
		})

		Context("with no categories", func() {
			It("should return empty slice", func() {
				// Create an empty config file
				err := os.WriteFile(configFile, []byte(""), 0644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(BeEmpty())
			})
		})
	})

	Describe("Edge Cases and Error Conditions", func() {
		It("should handle configuration with all fields as zero values", func() {
			configContent := `
bert_model:
  threshold: 0
semantic_cache:
  max_entries: 0
  ttl_seconds: 0
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := config.LoadConfig(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.BertModel.Threshold).To(Equal(float32(0)))
			Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
			Expect(cfg.SemanticCache.TTLSeconds).To(Equal(0))
		})

		It("should handle very large numeric values", func() {
			configContent := `
model_config:
  "large-model":
    param_count: 1.7976931348623157e+308
gpu_config:
  flops: 1e20
  hbm: 1e15
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := config.LoadConfig(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.ModelConfig["large-model"].ParamCount).To(Equal(1.7976931348623157e+308))
			Expect(cfg.GPUConfig.FLOPS).To(Equal(1e20))
		})

		It("should handle special string values", func() {
			configContent := `
bert_model:
  model_id: "model/with/slashes"
default_model: "model-with-hyphens_and_underscores"
categories:
  - name: "category with spaces"
    description: "Description with special chars: @#$%^&*()"
`
			err := os.WriteFile(configFile, []byte(configContent), 0644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := config.LoadConfig(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.BertModel.ModelID).To(Equal("model/with/slashes"))
			Expect(cfg.DefaultModel).To(Equal("model-with-hyphens_and_underscores"))
			Expect(cfg.Categories[0].Name).To(Equal("category with spaces"))
		})
	})

	Describe("PII Constants", func() {
		It("should have all expected PII type constants defined", func() {
			expectedPIITypes := []string{
				config.PIITypeAge,
				config.PIITypeCreditCard,
				config.PIITypeDateTime,
				config.PIITypeDomainName,
				config.PIITypeEmailAddress,
				config.PIITypeGPE,
				config.PIITypeIBANCode,
				config.PIITypeIPAddress,
				config.PIITypeNoPII,
				config.PIITypeNRP,
				config.PIITypeOrganization,
				config.PIITypePerson,
				config.PIITypePhoneNumber,
				config.PIITypeStreetAddress,
				config.PIITypeUSDriverLicense,
				config.PIITypeUSSSN,
				config.PIITypeZipCode,
			}

			// Verify all constants are non-empty strings
			for _, piiType := range expectedPIITypes {
				Expect(piiType).NotTo(BeEmpty())
			}

			// Verify specific values
			Expect(config.PIITypeNoPII).To(Equal("NO_PII"))
			Expect(config.PIITypePerson).To(Equal("PERSON"))
			Expect(config.PIITypeEmailAddress).To(Equal("EMAIL_ADDRESS"))
		})
	})
})