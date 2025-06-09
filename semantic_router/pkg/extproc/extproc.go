package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	candle_binding "github.com/neuralmagic/semantic_router_poc/candle-binding"
	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/cache"
	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/config"
	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/metrics"
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
	CategoryMapping      *CategoryMapping
	Cache                *cache.SemanticCache
	// Map to track pending requests and their unique IDs
	pendingRequests     map[string][]byte
	pendingRequestsLock sync.Mutex

	// Model load tracking: model name -> active request count
	modelLoad     map[string]int
	modelLoadLock sync.Mutex

	// Model TTFT info: model name -> base TTFT (ms)
	modelTTFT map[string]float64

	// PII detection state
	piiDetectionEnabled bool

	// Dual classifier bridge
	dualClassifierBridge *DualClassifierBridge
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
	var categoryMapping *CategoryMapping
	if cfg.Classifier.CategoryMappingPath != "" {
		categoryMapping, err = LoadCategoryMapping(cfg.Classifier.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		log.Printf("Loaded category mapping with %d categories", len(categoryMapping.CategoryToIdx))
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
			numClasses := len(categoryMapping.CategoryToIdx)
			if numClasses < 2 {
				log.Printf("Warning: Not enough categories for classification, need at least 2, got %d", numClasses)
			} else {
				// Try to use local finetuned model first, fall back to HuggingFace
				var classifierModelID string
				localModelPath := "finetune-model"

				// Check if local model exists
				if _, err := os.Stat(localModelPath); err == nil {
					classifierModelID = localModelPath
					log.Printf("Using local finetuned model: %s", localModelPath)
				} else {
					// Fall back to configured HuggingFace model
					classifierModelID = cfg.Classifier.ModelID
					if classifierModelID == "" {
						classifierModelID = cfg.BertModel.ModelID
					}
					log.Printf("Local model not found, using HuggingFace model: %s", classifierModelID)
				}

				err = candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.UseCPU)
				if err != nil {
					return nil, fmt.Errorf("failed to initialize classifier model: %w", err)
				}
				log.Printf("Initialized classifier with %d categories", numClasses)
			}
		}

		// Initialize the PII detector if enabled
		if cfg.PIIDetection.Enabled {
			if len(cfg.PIIDetection.PIITypes) < 2 {
				log.Printf("Warning: Not enough PII types for detection, need at least 2, got %d", len(cfg.PIIDetection.PIITypes))
			} else {
				piiModelID := cfg.PIIDetection.ModelID
				if piiModelID == "" {
					piiModelID = cfg.BertModel.ModelID
				}

				err = candle_binding.InitPIIDetector(piiModelID, cfg.PIIDetection.PIITypes, cfg.PIIDetection.UseCPU)
				if err != nil {
					return nil, fmt.Errorf("failed to initialize PII detector: %w", err)
				}
				log.Printf("Initialized PII detector with %d types: %v", len(cfg.PIIDetection.PIITypes), cfg.PIIDetection.PIITypes)
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

	// Initialize dual classifier bridge if enabled
	var dualClassifierBridge *DualClassifierBridge
	if cfg.DualClassifier.Enabled {
		bridge, err := NewDualClassifierBridge(
			cfg.DualClassifier.Enabled,
			cfg.DualClassifier.ModelPath,
			cfg.DualClassifier.UseCPU,
		)
		if err != nil {
			log.Printf("Warning: Failed to initialize dual classifier bridge: %v", err)
		} else {
			dualClassifierBridge = bridge
		}
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		CategoryMapping:      categoryMapping,
		Cache:                semanticCache,
		pendingRequests:      make(map[string][]byte),
		modelLoad:            make(map[string]int),
		modelTTFT:            make(map[string]float64),
		piiDetectionEnabled:  cfg.PIIDetection.Enabled,
		dualClassifierBridge: dualClassifierBridge,
	}
	router.initModelTTFT()
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
			openAIRequest, err := parseOpenAIRequest(originalRequestBody)
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
			var userContent string
			var nonUserMessages []string

			for _, msg := range openAIRequest.Messages {
				if msg.Role == "user" {
					userContent = msg.Content
				} else if msg.Role != "" {
					nonUserMessages = append(nonUserMessages, msg.Content)
				}
			}

			// Perform PII detection on user content
			var piiDetectionResult *PIIDetectionResult
			if userContent != "" && r.piiDetectionEnabled {
				var err error
				piiDetectionResult, err = r.detectPII(userContent)
				if err != nil {
					log.Printf("PII detection failed: %v", err)
					// Continue processing even if PII detection fails
					piiDetectionResult = &PIIDetectionResult{HasPII: false}
				}

				// Check if we should block the request based on PII detection
				if piiDetectionResult.HasPII && r.Config.PIIDetection.BlockOnPII {
					log.Printf("Blocking request due to PII detection: %v", piiDetectionResult.DetectedTypes)

					// Return an error response
					immediateResponse := &ext_proc.ImmediateResponse{
						Status: &typev3.HttpStatus{
							Code: typev3.StatusCode_BadRequest,
						},
						Headers: &ext_proc.HeaderMutation{
							SetHeaders: []*core.HeaderValueOption{
								{
									Header: &core.HeaderValue{
										Key:   "content-type",
										Value: "application/json",
									},
								},
								{
									Header: &core.HeaderValue{
										Key:   "x-pii-blocked",
										Value: "true",
									},
								},
							},
						},
						Body: []byte(`{"error": {"message": "Request blocked due to PII detection", "type": "pii_violation", "detected_types": "` + strings.Join(piiDetectionResult.DetectedTypes, ",") + `"}}`),
					}

					response := &ext_proc.ProcessingResponse{
						Response: &ext_proc.ProcessingResponse_ImmediateResponse{
							ImmediateResponse: immediateResponse,
						},
					}

					if err := sendResponse(stream, response, "PII blocked response"); err != nil {
						return err
					}
					return nil
				}

				// If sanitization is enabled and PII was detected, replace user content with sanitized version
				if piiDetectionResult.HasPII && r.Config.PIIDetection.SanitizeEnabled && piiDetectionResult.SanitizedText != "" {
					log.Printf("Sanitizing user content due to PII detection")
					for i, msg := range openAIRequest.Messages {
						if msg.Role == "user" {
							openAIRequest.Messages[i].Content = piiDetectionResult.SanitizedText
							userContent = piiDetectionResult.SanitizedText
							break
						}
					}
				}
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
					// log.Printf("Cache hit! Returning cached response for query: %s", requestQuery)

					// Return immediate response from cache
					immediateResponse := &ext_proc.ImmediateResponse{
						Status: &typev3.HttpStatus{
							Code: typev3.StatusCode_OK,
						},
						Headers: &ext_proc.HeaderMutation{
							SetHeaders: []*core.HeaderValueOption{
								{
									Header: &core.HeaderValue{
										Key:   "content-type",
										Value: "application/json",
									},
								},
								{
									Header: &core.HeaderValue{
										Key:   "x-cache-hit",
										Value: "true",
									},
								},
							},
						},
						Body: cachedResponse,
					}

					response := &ext_proc.ProcessingResponse{
						Response: &ext_proc.ProcessingResponse_ImmediateResponse{
							ImmediateResponse: immediateResponse,
						},
					}

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
					// log.Printf("Added pending request with ID: %s, cacheID: %s", requestID, cacheID)
				}
			}

			// Create default response with CONTINUE status and PII context headers
			var defaultHeaderMutation *ext_proc.HeaderMutation
			if piiDetectionResult != nil {
				defaultHeaderMutation = &ext_proc.HeaderMutation{}
				if piiDetectionResult.HasPII {
					defaultHeaderMutation.SetHeaders = append(defaultHeaderMutation.SetHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:   "x-pii-detected",
							Value: "true",
						},
					})
					defaultHeaderMutation.SetHeaders = append(defaultHeaderMutation.SetHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:   "x-pii-types",
							Value: strings.Join(piiDetectionResult.DetectedTypes, ","),
						},
					})
				} else {
					defaultHeaderMutation.SetHeaders = append(defaultHeaderMutation.SetHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:   "x-pii-detected",
							Value: "false",
						},
					})
				}
			}

			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status:         ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: defaultHeaderMutation,
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
						log.Printf("Routing to model: %s", matchedModel)

						// Track the model load for the selected model
						r.modelLoadLock.Lock()
						r.modelLoad[matchedModel]++
						r.modelLoadLock.Unlock()

						// Track the model routing change
						metrics.RecordModelRouting(originalModel, matchedModel)

						// Update the actual model that will be used
						actualModel = matchedModel

						// Modify the model in the request
						openAIRequest.Model = matchedModel

						// Serialize the modified request
						modifiedBody, err := json.Marshal(openAIRequest)
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

						// Also create a header mutation to remove the original content-length and add PII context
						headerMutation := &ext_proc.HeaderMutation{
							RemoveHeaders: []string{"content-length"},
						}

						// Add PII detection results to headers if available
						if piiDetectionResult != nil {
							if piiDetectionResult.HasPII {
								headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
									Header: &core.HeaderValue{
										Key:   "x-pii-detected",
										Value: "true",
									},
								})
								headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
									Header: &core.HeaderValue{
										Key:   "x-pii-types",
										Value: strings.Join(piiDetectionResult.DetectedTypes, ","),
									},
								})
							} else {
								headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
									Header: &core.HeaderValue{
										Key:   "x-pii-detected",
										Value: "false",
									},
								})
							}
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
			promptTokens, completionTokens, _, err := parseTokensFromResponse(responseBody)
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
				r.modelLoadLock.Lock()
				if r.modelLoad[requestModel] > 0 {
					r.modelLoad[requestModel]--
				}
				r.modelLoadLock.Unlock()
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
	var categoryName string
	var confidence float64

	// Try dual classifier first if available
	if r.dualClassifierBridge != nil && r.dualClassifierBridge.IsEnabled() {
		category, conf, err := r.dualClassifierBridge.ClassifyCategory(query)
		if err != nil {
			log.Printf("Dual classifier error: %v, falling back to BERT classifier", err)
		} else {
			categoryName = category
			confidence = conf
			log.Printf("Dual classifier result: category=%s, confidence=%.4f", categoryName, confidence)
		}
	}

	// Fall back to BERT classifier if dual classifier failed or not available
	if categoryName == "" && r.CategoryMapping != nil {
		// Use BERT classifier to get the category index and confidence
		result, err := candle_binding.ClassifyText(query)
		if err != nil {
			log.Printf("Classification error: %v, falling back to default model", err)
			return r.Config.DefaultModel
		}

		log.Printf("BERT classification result: class=%d, confidence=%.4f", result.Class, result.Confidence)
		confidence = float64(result.Confidence)

		// Convert class index to category name
		var ok bool
		categoryName, ok = r.CategoryMapping.IdxToCategory[fmt.Sprintf("%d", result.Class)]
		if !ok {
			log.Printf("Class index %d not found in category mapping, using default model", result.Class)
			return r.Config.DefaultModel
		}
	}

	// If we still don't have a category, use default
	if categoryName == "" {
		return r.Config.DefaultModel
	}

	// Check confidence threshold
	threshold := r.Config.Classifier.Threshold
	if confidence < float64(threshold) {
		log.Printf("Classification confidence (%.4f) below threshold (%.4f), using default model",
			confidence, threshold)
		return r.Config.DefaultModel
	}

	// Record the category classification metric
	metrics.RecordCategoryClassification(categoryName)
	log.Printf("Classified as category: %s", categoryName)

	var cat *config.Category
	for i, category := range r.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &r.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		log.Printf("Could not find matching category %s in config, using default model", categoryName)
		return r.Config.DefaultModel
	}
	// Then select the best model from the determined category based on score and TTFT
	r.modelLoadLock.Lock()
	defer r.modelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0 // initialize to a low score
	bestQuality := 0.0

	if r.Config.Classifier.LoadAware {
		// Load-aware: combine accuracy and TTFT
		for _, modelScore := range cat.ModelScores {
			quality := modelScore.Score
			model := modelScore.Model

			baseTTFT := r.modelTTFT[model]
			load := r.modelLoad[model]
			estTTFT := baseTTFT * (1 + float64(load))
			if estTTFT == 0 {
				estTTFT = 1 // avoid div by zero
			}
			score := quality / estTTFT
			if score > bestScore {
				bestScore = score
				bestModel = model
				bestQuality = quality
			}
		}
	} else {
		// Not load-aware: pick the model with the highest accuracy only
		for _, modelScore := range cat.ModelScores {
			quality := modelScore.Score
			model := modelScore.Model
			if quality > bestScore {
				bestScore = quality
				bestModel = model
				bestQuality = quality
			}
		}
	}

	if bestModel == "" {
		log.Printf("No models found for category %s, using default model", categoryName)
		return r.Config.DefaultModel
	}

	log.Printf("Selected model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// OpenAIRequest represents an OpenAI API request
type OpenAIRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ChatMessage represents a message in the OpenAI chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Parse the OpenAI request JSON
func parseOpenAIRequest(data []byte) (*OpenAIRequest, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// OpenAIResponse represents an OpenAI API response
type OpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// parseTokensFromResponse extracts detailed token counts from the OpenAI schema based response JSON
func parseTokensFromResponse(responseBody []byte) (promptTokens, completionTokens, totalTokens int, err error) {
	if responseBody == nil {
		return 0, 0, 0, fmt.Errorf("empty response body")
	}

	var response OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to parse response JSON: %w", err)
	}

	// Extract token counts from the usage field
	promptTokens = response.Usage.PromptTokens
	completionTokens = response.Usage.CompletionTokens
	totalTokens = response.Usage.TotalTokens

	log.Printf("Parsed token usage from response: total=%d (prompt=%d, completion=%d)",
		totalTokens, promptTokens, completionTokens)

	return promptTokens, completionTokens, totalTokens, nil
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

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

// LoadCategoryMapping loads the category mapping from a JSON file
func LoadCategoryMapping(path string) (*CategoryMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping CategoryMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse mapping JSON: %w", err)
	}

	return &mapping, nil
}

// Compute base TTFT for a model using the formula based on https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/
// TTFT = (2*N*b*s)/(FLOPs) + (2*N)/(HBM)
// Parameters are loaded from config: model-specific (N, b, s) and GPU-specific (FLOPs, HBM)
func (r *OpenAIRouter) computeBaseTTFT(modelName string) float64 {
	// Get model-specific parameters from config
	defaultParamCount := 7e9    // Default to 7B if unknown
	defaultBatchSize := 512.0   // Default batch size
	defaultContextSize := 256.0 // Default context size

	// Get model parameters
	N := r.Config.GetModelParamCount(modelName, defaultParamCount)
	b := r.Config.GetModelBatchSize(modelName, defaultBatchSize)
	s := r.Config.GetModelContextSize(modelName, defaultContextSize)

	// Get GPU parameters from config
	FLOPs := r.Config.GPUConfig.FLOPS
	HBM := r.Config.GPUConfig.HBM

	prefillCompute := 2 * N * b * s
	prefillMemory := 2 * N

	TTFT := (prefillCompute/FLOPs + prefillMemory/HBM) * 1000 // ms
	return TTFT
}

// Initialize modelTTFT map for all models in config
func (r *OpenAIRouter) initModelTTFT() {
	if r.modelTTFT == nil {
		r.modelTTFT = make(map[string]float64)
	}
	for _, cat := range r.Config.Categories {
		for _, modelScore := range cat.ModelScores {
			if _, ok := r.modelTTFT[modelScore.Model]; !ok {
				r.modelTTFT[modelScore.Model] = r.computeBaseTTFT(modelScore.Model)
			}
		}
	}
	if r.Config.DefaultModel != "" {
		if _, ok := r.modelTTFT[r.Config.DefaultModel]; !ok {
			r.modelTTFT[r.Config.DefaultModel] = r.computeBaseTTFT(r.Config.DefaultModel)
		}
	}
}

// PIIDetectionResult represents the result of PII detection
type PIIDetectionResult struct {
	HasPII           bool      `json:"has_pii"`
	DetectedTypes    []string  `json:"detected_types"`
	ConfidenceScores []float32 `json:"confidence_scores"`
	TokenPredictions []int     `json:"token_predictions"`
	SanitizedText    string    `json:"sanitized_text,omitempty"`
}

// detectPII performs PII detection on the given text
func (r *OpenAIRouter) detectPII(text string) (*PIIDetectionResult, error) {
	if !r.piiDetectionEnabled {
		return &PIIDetectionResult{HasPII: false}, nil
	}

	var hasPII bool
	var detectedTypes []string
	var confidenceScores []float32
	var tokenPredictions []int

	// Try dual classifier first if available (but use regex fallback since PII head isn't well-trained)
	if r.dualClassifierBridge != nil && r.dualClassifierBridge.IsEnabled() {
		// For now, we'll use regex-based detection since the dual classifier's PII head
		// isn't properly trained. In the future, this could be replaced with the
		// dual classifier's PII detection once it's properly trained.
		hasPII, detectedTypes = r.detectPIIWithRegex(text)
		log.Printf("Using regex-based PII detection (dual classifier available but PII head untrained)")
	} else {
		// Fall back to candle-binding PII detector
		piiResult, err := candle_binding.DetectPII(text)
		if err != nil {
			log.Printf("PII detection failed: %v", err)
			return nil, err
		}

		if piiResult.Error {
			return nil, fmt.Errorf("PII detection returned error")
		}

		// Check if any PII was detected (any prediction != 0, which should be "O" for Other/No PII)
		hasPII = false
		for _, pred := range piiResult.TokenPredictions {
			if pred > 0 { // Any non-zero prediction indicates PII
				hasPII = true
				break
			}
		}

		detectedTypes = piiResult.DetectedPIITypes
		confidenceScores = piiResult.ConfidenceScores
		tokenPredictions = piiResult.TokenPredictions
	}

	// Create sanitized version if PII is detected and sanitization is enabled
	sanitizedText := ""
	if hasPII && r.Config.PIIDetection.SanitizeEnabled {
		sanitizedText = r.sanitizeText(text, tokenPredictions, detectedTypes)
	}

	result := &PIIDetectionResult{
		HasPII:           hasPII,
		DetectedTypes:    detectedTypes,
		ConfidenceScores: confidenceScores,
		TokenPredictions: tokenPredictions,
		SanitizedText:    sanitizedText,
	}

	// Log PII detection results
	if hasPII {
		log.Printf("PII detected in request: types=%v", detectedTypes)
	} else {
		log.Printf("No PII detected in request")
	}

	return result, nil
}

// detectPIIWithRegex performs regex-based PII detection as a fallback
func (r *OpenAIRouter) detectPIIWithRegex(text string) (bool, []string) {
	var detectedTypes []string

	// Email detection
	emailRegex := regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`)
	if emailRegex.MatchString(text) {
		detectedTypes = append(detectedTypes, "EMAIL_ADDRESS")
	}

	// Phone number detection
	phonePatterns := []*regexp.Regexp{
		regexp.MustCompile(`\b\d{3}-\d{3}-\d{4}\b`),       // 555-123-4567
		regexp.MustCompile(`\b\(\d{3}\)\s?\d{3}-\d{4}\b`), // (555) 123-4567
		regexp.MustCompile(`\b\d{3}\.\d{3}\.\d{4}\b`),     // 555.123.4567
		regexp.MustCompile(`\b\d{3}\s\d{3}\s\d{4}\b`),     // 555 123 4567
		regexp.MustCompile(`\b\d{10}\b`),                  // 5551234567
	}
	for _, pattern := range phonePatterns {
		if pattern.MatchString(text) {
			detectedTypes = append(detectedTypes, "PHONE_NUMBER")
			break
		}
	}

	// SSN detection
	ssnPatterns := []*regexp.Regexp{
		regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`), // 123-45-6789
		regexp.MustCompile(`\b\d{9}\b`),             // 123456789
	}
	for _, pattern := range ssnPatterns {
		if pattern.MatchString(text) {
			detectedTypes = append(detectedTypes, "SSN")
			break
		}
	}

	// Credit card detection
	ccPatterns := []*regexp.Regexp{
		regexp.MustCompile(`\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`), // 1234-5678-9012-3456
		regexp.MustCompile(`\b\d{16}\b`),                                 // 1234567890123456
	}
	for _, pattern := range ccPatterns {
		if pattern.MatchString(text) {
			detectedTypes = append(detectedTypes, "CREDIT_CARD")
			break
		}
	}

	// Person name detection (conservative)
	personRegex := regexp.MustCompile(`\b[A-Z][a-z]+\s+[A-Z][a-z]+\b`)
	if personRegex.MatchString(text) {
		detectedTypes = append(detectedTypes, "PERSON")
	}

	return len(detectedTypes) > 0, detectedTypes
}

// sanitizeText replaces detected PII with masked placeholders using regex patterns
func (r *OpenAIRouter) sanitizeText(text string, predictions []int, detectedTypes []string) string {
	if len(predictions) == 0 {
		return text
	}

	sanitized := text

	// Use regex patterns to properly identify and replace PII for detected types
	for _, piiType := range detectedTypes {
		placeholder := fmt.Sprintf("[REDACTED_%s]", piiType)

		switch piiType {
		case "EMAIL_ADDRESS":
			// Match email patterns: user@domain.com
			emailRegex := `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
			re := regexp.MustCompile(emailRegex)
			sanitized = re.ReplaceAllString(sanitized, placeholder)

		case "PHONE_NUMBER":
			// Match various phone number patterns
			phonePatterns := []string{
				`\b\d{3}-\d{3}-\d{4}\b`,       // 555-123-4567
				`\b\(\d{3}\)\s?\d{3}-\d{4}\b`, // (555) 123-4567 or (555)123-4567
				`\b\d{3}\.\d{3}\.\d{4}\b`,     // 555.123.4567
				`\b\d{3}\s\d{3}\s\d{4}\b`,     // 555 123 4567
				`\b\d{10}\b`,                  // 5551234567
			}
			for _, pattern := range phonePatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}

		case "SSN":
			// Match SSN patterns: 123-45-6789 or 123456789
			ssnPatterns := []string{
				`\b\d{3}-\d{2}-\d{4}\b`, // 123-45-6789
				`\b\d{9}\b`,             // 123456789
			}
			for _, pattern := range ssnPatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}

		case "CREDIT_CARD":
			// Match credit card patterns (various formats)
			ccPatterns := []string{
				`\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`, // 1234-5678-9012-3456 or similar
				`\b\d{16}\b`, // 1234567890123456
			}
			for _, pattern := range ccPatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}

		case "PERSON":
			// For person names, we'll use a more conservative approach
			// Only replace if it looks like a full name (First Last pattern)
			personRegex := `\b[A-Z][a-z]+\s+[A-Z][a-z]+\b`
			re := regexp.MustCompile(personRegex)
			sanitized = re.ReplaceAllString(sanitized, placeholder)

		case "ADDRESS":
			// Match address-like patterns (simplified)
			addressPatterns := []string{
				`\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b`,
			}
			for _, pattern := range addressPatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}

		case "DATE":
			// Match various date patterns
			datePatterns := []string{
				`\b\d{1,2}/\d{1,2}/\d{4}\b`, // MM/DD/YYYY or M/D/YYYY
				`\b\d{1,2}-\d{1,2}-\d{4}\b`, // MM-DD-YYYY or M-D-YYYY
				`\b\d{4}-\d{1,2}-\d{1,2}\b`, // YYYY-MM-DD or YYYY-M-D
			}
			for _, pattern := range datePatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}

		case "ORGANIZATION":
			// This is harder to detect with regex, so we'll be conservative
			// and only replace obvious organization patterns
			orgPatterns := []string{
				`\b[A-Z][A-Za-z\s]*(?:Inc|LLC|Corp|Corporation|Company|Co)\b`,
			}
			for _, pattern := range orgPatterns {
				re := regexp.MustCompile(pattern)
				sanitized = re.ReplaceAllString(sanitized, placeholder)
			}
		}
	}

	return sanitized
}
