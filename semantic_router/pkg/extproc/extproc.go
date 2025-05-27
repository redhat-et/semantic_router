package extproc

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/cache"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/metrics"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/classification"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/http"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/model"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/pii"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/ttft"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
	ModelSelector        *model.Selector
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

	if !initialized {
		// Initialize the BERT model for similarity search
		err = candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize BERT model: %w", err)
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
					return nil, fmt.Errorf("failed to initialize classifier model: %w", err)
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
					return nil, fmt.Errorf("failed to initialize PII classifier model: %w", err)
				}
				log.Printf("Initialized PII classifier with %d PII types", numPIIClasses)
			}
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
	classifier := classification.NewClassifier(cfg, categoryMapping, piiMapping)
	piiChecker := pii.NewPolicyChecker(cfg.ModelConfig)
	ttftCalculator := ttft.NewCalculator(cfg.GPUConfig)
	modelTTFT := ttftCalculator.InitializeModelTTFT(cfg)
	modelSelector := model.NewSelector(cfg, modelTTFT)

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		ModelSelector:        modelSelector,
		Cache:                semanticCache,
		pendingRequests:      make(map[string][]byte),
	}

	return router, nil
}

// Send a response with proper error handling and logging
func sendResponse(stream ext_proc.ExternalProcessor_ProcessServer, response *ext_proc.ProcessingResponse, msgType string) error {
	// log.Printf("Sending %s response: %+v", msgType, response)
	if err := stream.Send(response); err != nil {
		log.Printf("Error sending %s response: %v", msgType, err)
		return err
	}
	log.Printf("Successfully sent %s response", msgType)
	return nil
}

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	log.Println("Started processing a new request")
	requestHeaders := make(map[string]string)
	var requestID string
	var originalRequestBody []byte
	var requestModel string
	var requestQuery string
	var startTime time.Time
	var processingStartTime time.Time

	for {
		req, err := stream.Recv()
		if err != nil {
			log.Printf("Error receiving request: %v", err)
			return err
		}

		log.Printf("Processing message type: %T", req.Request)

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			// Record start time for overall request processing
			startTime = time.Now()
			log.Println("Received request headers")

			// Store headers for later use
			headers := v.RequestHeaders.Headers
			for _, h := range headers.Headers {
				requestHeaders[h.Key] = h.Value
				// Store request ID if present
				if strings.ToLower(h.Key) == "x-request-id" {
					requestID = h.Value
				}
			}

			// Allow the request to continue
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestHeaders{
					RequestHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			log.Println("Received request body")
			// Record start time for model routing
			processingStartTime = time.Now()
			// Save the original request body
			originalRequestBody = v.RequestBody.Body

			// Parse the OpenAI request
			openAIRequest, err := openai.ParseRequest(originalRequestBody)
			if err != nil {
				log.Printf("Error parsing OpenAI request: %v", err)
				return status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
			}

			// Store the original model
			originalModel := openAIRequest.Model
			log.Printf("Original model: %s", originalModel)

			// Record the initial request to this model
			metrics.RecordModelRequest(originalModel)

			// Get content from messages
			userContent, nonUserMessages := openai.ExtractUserAndNonUserContent(openAIRequest)

			// Perform PII classification on all message content
			allContent := pii.ExtractAllContent(userContent, nonUserMessages)
			detectedPII := r.Classifier.DetectPIIInContent(allContent)

			if len(detectedPII) > 0 {
				log.Printf("Total detected PII types: %v", detectedPII)
			} else {
				log.Printf("No PII detected in request content")
			}

			// Extract the model and query for cache lookup
			requestModel, requestQuery, err = cache.ExtractQueryFromOpenAIRequest(originalRequestBody)
			if err != nil {
				log.Printf("Error extracting query from request: %v", err)
				// Continue without caching
			} else if requestQuery != "" && r.Cache.IsEnabled() {
				// Try to find a similar cached response
				cachedResponse, found, err := r.Cache.FindSimilar(requestModel, requestQuery)
				if err != nil {
					log.Printf("Error searching cache: %v", err)
				} else if found {
					// Return immediate response from cache
					response := http.CreateCacheHitResponse(cachedResponse)
					if err := sendResponse(stream, response, "immediate response from cache"); err != nil {
						return err
					}
					return nil
				}

				// Cache miss, store the request for later
				cacheID, err := r.Cache.AddPendingRequest(requestModel, requestQuery, originalRequestBody)
				if err != nil {
					log.Printf("Error adding pending request to cache: %v", err)
				} else {
					r.pendingRequestsLock.Lock()
					r.pendingRequests[requestID] = []byte(cacheID)
					r.pendingRequestsLock.Unlock()
				}
			}

			// Create default response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			// Only change the model if the original model is "auto"
			actualModel := originalModel
			if originalModel == "auto" && (len(nonUserMessages) > 0 || userContent != "") {
				// Determine text to use for classification/similarity
				var classificationText string
				if len(userContent) > 0 {
					classificationText = userContent
				} else if len(nonUserMessages) > 0 {
					// Fall back to user content if no system/assistant messages
					classificationText = strings.Join(nonUserMessages, " ")
				}

				if classificationText != "" {
					// Find the most similar task description or classify, then select best model
					matchedModel := r.classifyAndSelectBestModel(classificationText)
					if matchedModel != originalModel && matchedModel != "" {
						// Check if the initially selected model passes PII policy
						allowed, deniedPII, err := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
						if err != nil {
							log.Printf("Error checking PII policy for model %s: %v", matchedModel, err)
							// Continue with original selection on error
						} else if !allowed {
							log.Printf("Initially selected model %s violates PII policy, finding alternative", matchedModel)
							// Find alternative models from the same category that pass PII policy
							categoryName := r.findCategoryForClassification(classificationText)
							if categoryName != "" {
								alternativeModels := r.ModelSelector.GetModelsForCategory(categoryName)
								allowedModels := r.PIIChecker.FilterModelsForPII(alternativeModels, detectedPII)
								if len(allowedModels) > 0 {
									// Select the best allowed model from this category
									matchedModel = r.ModelSelector.SelectBestModelFromList(allowedModels, categoryName)
									log.Printf("Selected alternative model %s that passes PII policy", matchedModel)
								} else {
									log.Printf("No models in category %s pass PII policy, using default", categoryName)
									matchedModel = r.Config.DefaultModel
									// Check if default model passes policy
									defaultAllowed, defaultDeniedPII, _ := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
									if !defaultAllowed {
										log.Printf("Default model also violates PII policy, returning error")
										piiResponse := http.CreatePIIViolationResponse(matchedModel, defaultDeniedPII)
										if err := sendResponse(stream, piiResponse, "PII violation"); err != nil {
											return err
										}
										return nil
									}
								}
							} else {
								log.Printf("Could not determine category, returning PII violation for model %s", matchedModel)
								piiResponse := http.CreatePIIViolationResponse(matchedModel, deniedPII)
								if err := sendResponse(stream, piiResponse, "PII violation"); err != nil {
									return err
								}
								return nil
							}
						}

						log.Printf("Routing to model: %s", matchedModel)

						// Track the model load for the selected model
						r.ModelSelector.IncrementModelLoad(matchedModel)

						// Track the model routing change
						metrics.RecordModelRouting(originalModel, matchedModel)

						// Update the actual model that will be used
						actualModel = matchedModel

						// Modify the model in the request
						openAIRequest.Model = matchedModel

						// Serialize the modified request
						modifiedBody, err := openai.SerializeRequest(openAIRequest)
						if err != nil {
							log.Printf("Error serializing modified request: %v", err)
							return status.Errorf(codes.Internal, "error serializing modified request: %v", err)
						}

						// Create body mutation with the modified body
						bodyMutation := &ext_proc.BodyMutation{
							Mutation: &ext_proc.BodyMutation_Body{
								Body: modifiedBody,
							},
						}

						// Also create a header mutation to remove the original content-length
						headerMutation := &ext_proc.HeaderMutation{
							RemoveHeaders: []string{"content-length"},
						}

						// Set the response with both mutations
						response = &ext_proc.ProcessingResponse{
							Response: &ext_proc.ProcessingResponse_RequestBody{
								RequestBody: &ext_proc.BodyResponse{
									Response: &ext_proc.CommonResponse{
										Status:         ext_proc.CommonResponse_CONTINUE,
										HeaderMutation: headerMutation,
										BodyMutation:   bodyMutation,
									},
								},
							},
						}

						log.Printf("Use new model: %s", matchedModel)
					}
				}
			} else if originalModel != "auto" {
				// For non-auto models, check PII policy compliance
				allowed, deniedPII, err := r.PIIChecker.CheckPolicy(originalModel, detectedPII)
				if err != nil {
					log.Printf("Error checking PII policy for model %s: %v", originalModel, err)
					// Continue with request on error
				} else if !allowed {
					log.Printf("Model %s violates PII policy, returning error", originalModel)
					piiResponse := http.CreatePIIViolationResponse(originalModel, deniedPII)
					if err := sendResponse(stream, piiResponse, "PII violation"); err != nil {
						return err
					}
					return nil
				}
			}

			// Save the actual model that will be used for token tracking
			requestModel = actualModel

			// Record the routing latency
			routingLatency := time.Since(processingStartTime)
			metrics.RecordModelRoutingLatency(routingLatency.Seconds())

			if err := sendResponse(stream, response, "body"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			log.Println("Received response headers")

			// Allow the response to continue without modification
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "response header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseBody:
			completionLatency := time.Since(startTime)
			log.Println("Received response body")

			// Process the response for caching
			responseBody := v.ResponseBody.Body

			// Parse tokens from the response JSON
			promptTokens, completionTokens, _, err := openai.ParseTokensFromResponse(responseBody)
			if err != nil {
				log.Printf("Error parsing tokens from response: %v", err)
			}

			// Record tokens used with the model that was used
			if requestModel != "" {
				metrics.RecordModelTokensDetailed(
					requestModel,
					float64(promptTokens),
					float64(completionTokens),
				)
				metrics.RecordModelCompletionLatency(requestModel, completionLatency.Seconds())
				r.ModelSelector.DecrementModelLoad(requestModel)
			}

			// Check if this request has a pending cache entry
			r.pendingRequestsLock.Lock()
			cacheID, exists := r.pendingRequests[requestID]
			if exists {
				delete(r.pendingRequests, requestID)
			}
			r.pendingRequestsLock.Unlock()

			// If we have a pending request, update the cache
			if exists && requestQuery != "" && responseBody != nil {
				err := r.Cache.UpdateWithResponse(string(cacheID), responseBody)
				if err != nil {
					log.Printf("Error updating cache: %v", err)
					// Continue even if cache update fails
				} else {
					log.Printf("Cache updated for request ID: %s", requestID)
				}
			}

			// Allow the response to continue without modification
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseBody{
					ResponseBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "response body"); err != nil {
				return err
			}

		default:
			log.Printf("Unknown request type: %v", v)

			// For unknown message types, create a body response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "unknown"); err != nil {
				return err
			}
		}
	}
}

// Choose best models based on category classification and model quality and expected TTFT
func (r *OpenAIRouter) classifyAndSelectBestModel(query string) string {
	// If no categories defined, return default model
	if len(r.CategoryDescriptions) == 0 {
		return r.Config.DefaultModel
	}

	// First, classify the text to determine the category
	categoryName, confidence, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		log.Printf("Classification error: %v, falling back to default model", err)
		return r.Config.DefaultModel
	}

	if categoryName == "" {
		log.Printf("Classification confidence (%.4f) below threshold, using default model", confidence)
		return r.Config.DefaultModel
	}

	// Then select the best model from the determined category based on score and TTFT
	return r.ModelSelector.SelectBestModelForCategory(categoryName)
}

// findCategoryForClassification determines the category for the given text using classification
func (r *OpenAIRouter) findCategoryForClassification(query string) string {
	if len(r.CategoryDescriptions) == 0 {
		return ""
	}

	categoryName, _, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		log.Printf("Category classification error: %v", err)
		return ""
	}

	return categoryName
}

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	router *OpenAIRouter
	server *grpc.Server
	port   int
}

// NewServer creates a new ExtProc gRPC server
func NewServer(configPath string, port int) (*Server, error) {
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		return nil, err
	}

	return &Server{
		router: router,
		port:   port,
	}, nil
}

// Start starts the gRPC server
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}

	s.server = grpc.NewServer()
	ext_proc.RegisterExternalProcessorServer(s.server, s.router)

	log.Printf("Starting LLM Router ExtProc server on port %d...", s.port)

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := s.server.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			log.Printf("Server error: %v", err)
			serverErrCh <- err
		} else {
			serverErrCh <- nil
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for either server error or shutdown signal
	select {
	case err := <-serverErrCh:
		if err != nil {
			log.Printf("Server exited with error: %v", err)
			return err
		}
	case <-signalChan:
		log.Println("Received shutdown signal, gracefully stopping server...")
	}

	s.Stop()
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
		log.Println("Server stopped")
	}
}
