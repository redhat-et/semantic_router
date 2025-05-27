package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ModelRequests tracks the number of requests made to each model
	ModelRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_requests_total",
			Help: "The total number of requests made to each LLM model",
		},
		[]string{"model"},
	)

	// ModelTokens tracks the number of tokens used by each model
	ModelTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_tokens_total",
			Help: "The total number of tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// ModelPromptTokens tracks the number of prompt tokens used by each model
	ModelPromptTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_prompt_tokens_total",
			Help: "The total number of prompt tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// ModelCompletionTokens tracks the number of completion tokens used by each model
	ModelCompletionTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_completion_tokens_total",
			Help: "The total number of completion tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// ModelRoutingModifications tracks when a model is changed from one to another
	ModelRoutingModifications = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_routing_modifications_total",
			Help: "The total number of times a request was rerouted from source model to target model",
		},
		[]string{"source_model", "target_model"},
	)

	// ModelCompletionLatency tracks the latency of completions by model
	ModelCompletionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_model_completion_latency_seconds",
			Help:    "The latency of LLM model completions in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// ModelRoutingLatency tracks the latency of model routing
	ModelRoutingLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_model_routing_latency_seconds",
			Help:    "The latency of model routing operations in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// CacheHits tracks cache hits and misses
	CacheHits = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_cache_hits_total",
			Help: "The total number of cache hits",
		},
	)

	// CategoryClassifications tracks the number of times each category is classified
	CategoryClassifications = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_category_classifications_total",
			Help: "The total number of times each category is classified",
		},
		[]string{"category"},
	)

	// PIIViolations tracks PII policy violations by model and PII data type
	PIIViolations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_pii_violations_total",
			Help: "The total number of PII policy violations by model and PII data type",
		},
		[]string{"model", "pii_type"},
	)
)

// RecordModelRequest increments the counter for requests to a specific model
func RecordModelRequest(model string) {
	ModelRequests.WithLabelValues(model).Inc()
}

// RecordModelRouting records that a request was routed from one model to another
func RecordModelRouting(sourceModel, targetModel string) {
	if sourceModel != targetModel {
		ModelRoutingModifications.WithLabelValues(sourceModel, targetModel).Inc()
	}
}

// RecordModelTokens adds the number of tokens used by a specific model
func RecordModelTokens(model string, tokens float64) {
	ModelTokens.WithLabelValues(model).Add(tokens)
}

// RecordModelTokensDetailed records detailed token usage (prompt and completion)
func RecordModelTokensDetailed(model string, promptTokens, completionTokens float64) {
	// Record in both the aggregated and detailed metrics
	totalTokens := promptTokens + completionTokens
	ModelTokens.WithLabelValues(model).Add(totalTokens)
	ModelPromptTokens.WithLabelValues(model).Add(promptTokens)
	ModelCompletionTokens.WithLabelValues(model).Add(completionTokens)
}

// RecordModelCompletionLatency records the latency of a model completion
func RecordModelCompletionLatency(model string, seconds float64) {
	ModelCompletionLatency.WithLabelValues(model).Observe(seconds)
}

// RecordModelRoutingLatency records the latency of model routing
func RecordModelRoutingLatency(seconds float64) {
	ModelRoutingLatency.Observe(seconds)
}

// RecordCacheHit records a cache hit
func RecordCacheHit() {
	CacheHits.Inc()
}

// RecordCategoryClassification increments the gauge for a specific category classification
func RecordCategoryClassification(category string) {
	CategoryClassifications.WithLabelValues(category).Inc()
}

// RecordPIIViolation records a PII policy violation for a specific model and PII data type
func RecordPIIViolation(model string, piiType string) {
	PIIViolations.WithLabelValues(model, piiType).Inc()
}

// RecordPIIViolations records multiple PII policy violations for a specific model
func RecordPIIViolations(model string, piiTypes []string) {
	for _, piiType := range piiTypes {
		PIIViolations.WithLabelValues(model, piiType).Inc()
	}
}
