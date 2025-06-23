package extproc

import (
	"log"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/cache"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/metrics"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/http"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/pii"
)

// RequestContext holds the context for processing a request
type RequestContext struct {
	Headers             map[string]string
	RequestID           string
	OriginalRequestBody []byte
	RequestModel        string
	RequestQuery        string
	StartTime           time.Time
	ProcessingStartTime time.Time
}

// handleRequestHeaders processes the request headers
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Record start time for overall request processing
	ctx.StartTime = time.Now()
	log.Println("Received request headers")

	// Store headers for later use
	headers := v.RequestHeaders.Headers
	for _, h := range headers.Headers {
		ctx.Headers[h.Key] = h.Value
		// Store request ID if present
		if strings.ToLower(h.Key) == "x-request-id" {
			ctx.RequestID = h.Value
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

	return response, nil
}

// handleRequestBody processes the request body
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	log.Println("Received request body")
	// Record start time for model routing
	ctx.ProcessingStartTime = time.Now()
	// Save the original request body
	ctx.OriginalRequestBody = v.RequestBody.Body

	// Parse the OpenAI request
	openAIRequest, err := openai.ParseRequest(ctx.OriginalRequestBody)
	if err != nil {
		log.Printf("Error parsing OpenAI request: %v", err)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}

	// Store the original model
	originalModel := openAIRequest.Model
	log.Printf("Original model: %s", originalModel)

	// Record the initial request to this model
	metrics.RecordModelRequest(originalModel)

	// Get content from messages
	userContent, nonUserMessages := openai.ExtractUserAndNonUserContent(openAIRequest)

	// Perform security checks
	if response, shouldReturn := r.performSecurityChecks(userContent, nonUserMessages); shouldReturn {
		return response, nil
	}

	// Handle caching
	if response, shouldReturn := r.handleCaching(ctx); shouldReturn {
		return response, nil
	}

	// Handle model selection and routing
	return r.handleModelRouting(openAIRequest, originalModel, userContent, nonUserMessages, ctx)
}

// performSecurityChecks performs PII and jailbreak detection
func (r *OpenAIRouter) performSecurityChecks(userContent string, nonUserMessages []string) (*ext_proc.ProcessingResponse, bool) {
	// Perform PII classification on all message content
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)
	detectedPII := r.Classifier.DetectPIIInContent(allContent)

	if len(detectedPII) > 0 {
		log.Printf("Total detected PII types: %v", detectedPII)
	} else {
		log.Printf("No PII detected in request content")
	}

	// Perform jailbreak detection on all message content
	if r.PromptGuard.IsEnabled() {
		hasJailbreak, jailbreakDetections, err := r.PromptGuard.AnalyzeContent(allContent)
		if err != nil {
			log.Printf("Error performing jailbreak analysis: %v", err)
			// Continue processing despite jailbreak analysis error
		} else if hasJailbreak {
			// Find the first jailbreak detection for response
			var jailbreakType string
			var confidence float32
			for _, detection := range jailbreakDetections {
				if detection.IsJailbreak {
					jailbreakType = detection.JailbreakType
					confidence = detection.Confidence
					break
				}
			}

			log.Printf("JAILBREAK ATTEMPT BLOCKED: %s (confidence: %.3f)", jailbreakType, confidence)

			// Return immediate jailbreak violation response
			jailbreakResponse := http.CreateJailbreakViolationResponse(jailbreakType, confidence)
			return jailbreakResponse, true
		} else {
			log.Printf("No jailbreak detected in request content")
		}
	}

	return nil, false
}

// handleCaching handles cache lookup and storage
func (r *OpenAIRouter) handleCaching(ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		log.Printf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	if requestQuery != "" && r.Cache.IsEnabled() {
		// Try to find a similar cached response
		cachedResponse, found, err := r.Cache.FindSimilar(requestModel, requestQuery)
		if err != nil {
			log.Printf("Error searching cache: %v", err)
		} else if found {
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse)
			return response, true
		}

		// Cache miss, store the request for later
		cacheID, err := r.Cache.AddPendingRequest(requestModel, requestQuery, ctx.OriginalRequestBody)
		if err != nil {
			log.Printf("Error adding pending request to cache: %v", err)
		} else {
			r.pendingRequestsLock.Lock()
			r.pendingRequests[ctx.RequestID] = []byte(cacheID)
			r.pendingRequestsLock.Unlock()
		}
	}

	return nil, false
}

// handleModelRouting handles model selection and routing logic
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.OpenAIRequest, originalModel, userContent string, nonUserMessages []string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
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
			matchedModel := r.Classifier.ClassifyAndSelectBestModel(classificationText)
			if matchedModel != originalModel && matchedModel != "" {
				// Get detected PII for policy checking
				allContent := pii.ExtractAllContent(userContent, nonUserMessages)
				detectedPII := r.Classifier.DetectPIIInContent(allContent)

				// Check if the initially selected model passes PII policy
				allowed, deniedPII, err := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
				if err != nil {
					log.Printf("Error checking PII policy for model %s: %v", matchedModel, err)
					// Continue with original selection on error
				} else if !allowed {
					log.Printf("Initially selected model %s violates PII policy, finding alternative", matchedModel)
					// Find alternative models from the same category that pass PII policy
					categoryName := r.Classifier.FindCategoryForClassification(classificationText)
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
								return piiResponse, nil
							}
						}
					} else {
						log.Printf("Could not determine category, returning PII violation for model %s", matchedModel)
						piiResponse := http.CreatePIIViolationResponse(matchedModel, deniedPII)
						return piiResponse, nil
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
					return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
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
		allContent := pii.ExtractAllContent(userContent, nonUserMessages)
		detectedPII := r.Classifier.DetectPIIInContent(allContent)

		allowed, deniedPII, err := r.PIIChecker.CheckPolicy(originalModel, detectedPII)
		if err != nil {
			log.Printf("Error checking PII policy for model %s: %v", originalModel, err)
			// Continue with request on error
		} else if !allowed {
			log.Printf("Model %s violates PII policy, returning error", originalModel)
			piiResponse := http.CreatePIIViolationResponse(originalModel, deniedPII)
			return piiResponse, nil
		}
	}

	// Save the actual model that will be used for token tracking
	ctx.RequestModel = actualModel

	// Record the routing latency
	routingLatency := time.Since(ctx.ProcessingStartTime)
	metrics.RecordModelRoutingLatency(routingLatency.Seconds())

	return response, nil
}
