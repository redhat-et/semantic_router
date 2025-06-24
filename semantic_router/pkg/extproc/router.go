package extproc

import (
	"fmt"
	"log"
	"sync"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/cache"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/classification"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/pii"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/ttft"
)

var (
	initialized bool
	initMutex   sync.Mutex
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	PIIChecker           *pii.PolicyChecker
	Cache                *cache.SemanticCache

	// Map to track pending requests and their unique IDs
	pendingRequests     map[string][]byte
	pendingRequestsLock sync.Mutex
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = &OpenAIRouter{}

// NewOpenAIRouter creates a new OpenAI API router instance
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	initMutex.Lock()
	defer initMutex.Unlock()

	// Load category mapping if classifier is enabled
	var categoryMapping *classification.CategoryMapping
	if cfg.Classifier.CategoryModel.CategoryMappingPath != "" {
		categoryMapping, err = classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		log.Printf("Loaded category mapping with %d categories", categoryMapping.GetCategoryCount())
	}

	// Load PII mapping if PII classifier is enabled
	var piiMapping *classification.PIIMapping
	if cfg.Classifier.PIIModel.PIIMappingPath != "" {
		piiMapping, err = classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		log.Printf("Loaded PII mapping with %d PII types", piiMapping.GetPIITypeCount())
	}

	// Load jailbreak mapping if prompt guard is enabled
	var jailbreakMapping *classification.JailbreakMapping
	if cfg.IsPromptGuardEnabled() {
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		log.Printf("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())
	}

	if !initialized {
		if err := initializeModels(cfg, categoryMapping, piiMapping, jailbreakMapping); err != nil {
			return nil, err
		}
		initialized = true
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	log.Printf("Category descriptions: %v", categoryDescriptions)

	// Create semantic cache with config options
	cacheOptions := cache.SemanticCacheOptions{
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		Enabled:             cfg.SemanticCache.Enabled,
	}
	semanticCache := cache.NewSemanticCache(cacheOptions)

	if semanticCache.IsEnabled() {
		log.Printf("Semantic cache enabled with threshold: %.4f, max entries: %d, TTL: %d seconds",
			cacheOptions.SimilarityThreshold, cacheOptions.MaxEntries, cacheOptions.TTLSeconds)
	} else {
		log.Println("Semantic cache is disabled")
	}

	// Create utility components
	piiChecker := pii.NewPolicyChecker(cfg.ModelConfig)
	ttftCalculator := ttft.NewCalculator(cfg.GPUConfig)
	modelTTFT := ttftCalculator.InitializeModelTTFT(cfg)
	classifier := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping, modelTTFT)

	// Initialize jailbreak classifier if enabled
	if jailbreakMapping != nil {
		err = classifier.InitializeJailbreakClassifier()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
		}
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		pendingRequests:      make(map[string][]byte),
	}

	return router, nil
}

// initializeModels initializes the BERT and classifier models
func initializeModels(cfg *config.RouterConfig, categoryMapping *classification.CategoryMapping, piiMapping *classification.PIIMapping, jailbreakMapping *classification.JailbreakMapping) error {
	// Initialize the BERT model for similarity search
	err := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	// Initialize the classifier model if enabled
	if categoryMapping != nil {
		// Get the number of categories from the mapping
		numClasses := categoryMapping.GetCategoryCount()
		if numClasses < 2 {
			log.Printf("Warning: Not enough categories for classification, need at least 2, got %d", numClasses)
		} else {
			// Use the category classifier model
			classifierModelID := cfg.Classifier.CategoryModel.ModelID
			if classifierModelID == "" {
				classifierModelID = cfg.BertModel.ModelID
			}

			err = candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.CategoryModel.UseCPU)
			if err != nil {
				return fmt.Errorf("failed to initialize classifier model: %w", err)
			}
			log.Printf("Initialized category classifier with %d categories", numClasses)
		}
	}

	// Initialize PII classifier if enabled
	if piiMapping != nil {
		// Get the number of PII types from the mapping
		numPIIClasses := piiMapping.GetPIITypeCount()
		if numPIIClasses < 2 {
			log.Printf("Warning: Not enough PII types for classification, need at least 2, got %d", numPIIClasses)
		} else {
			// Use the PII classifier model
			piiClassifierModelID := cfg.Classifier.PIIModel.ModelID
			if piiClassifierModelID == "" {
				piiClassifierModelID = cfg.BertModel.ModelID
			}

			err = candle_binding.InitPIIClassifier(piiClassifierModelID, numPIIClasses, cfg.Classifier.PIIModel.UseCPU)
			if err != nil {
				return fmt.Errorf("failed to initialize PII classifier model: %w", err)
			}
			log.Printf("Initialized PII classifier with %d PII types", numPIIClasses)
		}
	}

	// Initialize jailbreak classifier if enabled
	if jailbreakMapping != nil {
		// Get the number of jailbreak types from the mapping
		numJailbreakClasses := jailbreakMapping.GetJailbreakTypeCount()
		if numJailbreakClasses < 2 {
			log.Printf("Warning: Not enough jailbreak types for classification, need at least 2, got %d", numJailbreakClasses)
		} else {
			// Use the jailbreak classifier model
			jailbreakClassifierModelID := cfg.PromptGuard.ModelID
			if jailbreakClassifierModelID == "" {
				jailbreakClassifierModelID = cfg.BertModel.ModelID
			}

			err = candle_binding.InitJailbreakClassifier(jailbreakClassifierModelID, numJailbreakClasses, cfg.PromptGuard.UseCPU)
			if err != nil {
				return fmt.Errorf("failed to initialize jailbreak classifier model: %w", err)
			}
			log.Printf("Initialized jailbreak classifier with %d jailbreak types", numJailbreakClasses)
		}
	}

	return nil
}
