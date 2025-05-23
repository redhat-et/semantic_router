package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
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
				// Use the same model or a specific classifier model
				classifierModelID := cfg.Classifier.ModelID
				if classifierModelID == "" {
					classifierModelID = cfg.BertModel.ModelID
				}

				err = candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.UseCPU)
				if err != nil {
					return nil, fmt.Errorf("failed to initialize classifier model: %w", err)
				}
				log.Printf("Initialized classifier with %d categories", numClasses)
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

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		CategoryMapping:      categoryMapping,
		Cache:                semanticCache,
		pendingRequests:      make(map[string][]byte),
		modelLoad:            make(map[string]int),
		modelTTFT:            make(map[string]float64),
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
	if r.CategoryMapping != nil {
		// Use BERT classifier to get the category index and confidence
		result, err := candle_binding.ClassifyText(query)
		if err != nil {
			log.Printf("Classification error: %v, falling back to default model", err)
			return r.Config.DefaultModel
		}

		log.Printf("Classification result: class=%d, confidence=%.4f", result.Class, result.Confidence)

		// Check confidence threshold
		if result.Confidence < r.Config.Classifier.Threshold {
			log.Printf("Classification confidence (%.4f) below threshold (%.4f), using default model",
				result.Confidence, r.Config.Classifier.Threshold)
			return r.Config.DefaultModel
		}

		// Convert class index to category name
		var ok bool
		categoryName, ok = r.CategoryMapping.IdxToCategory[fmt.Sprintf("%d", result.Class)]
		if !ok {
			log.Printf("Class index %d not found in category mapping, using default model", result.Class)
			return r.Config.DefaultModel
		}

		// Record the category classification metric
		metrics.RecordCategoryClassification(categoryName)

		log.Printf("Classified as category: %s", categoryName)
	} else {
		return r.Config.DefaultModel
	}

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
