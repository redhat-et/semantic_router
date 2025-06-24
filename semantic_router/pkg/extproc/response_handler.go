package extproc

import (
	"log"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/metrics"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
)

// handleResponseHeaders processes the response headers
func (r *OpenAIRouter) handleResponseHeaders(_ *ext_proc.ProcessingRequest_ResponseHeaders) (*ext_proc.ProcessingResponse, error) {
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

	return response, nil
}

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)
	log.Println("Received response body")

	// Process the response for caching
	responseBody := v.ResponseBody.Body

	// Parse tokens from the response JSON
	promptTokens, completionTokens, _, err := openai.ParseTokensFromResponse(responseBody)
	if err != nil {
		log.Printf("Error parsing tokens from response: %v", err)
	}

	// Record tokens used with the model that was used
	if ctx.RequestModel != "" {
		metrics.RecordModelTokensDetailed(
			ctx.RequestModel,
			float64(promptTokens),
			float64(completionTokens),
		)
		metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())
		r.Classifier.DecrementModelLoad(ctx.RequestModel)
	}

	// Check if this request has a pending cache entry
	r.pendingRequestsLock.Lock()
	cacheID, exists := r.pendingRequests[ctx.RequestID]
	if exists {
		delete(r.pendingRequests, ctx.RequestID)
	}
	r.pendingRequestsLock.Unlock()

	// If we have a pending request, update the cache
	if exists && ctx.RequestQuery != "" && responseBody != nil {
		err := r.Cache.UpdateWithResponse(string(cacheID), responseBody)
		if err != nil {
			log.Printf("Error updating cache: %v", err)
			// Continue even if cache update fails
		} else {
			log.Printf("Cache updated for request ID: %s", ctx.RequestID)
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

	return response, nil
}
