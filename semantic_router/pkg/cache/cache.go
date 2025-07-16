package cache

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
)

// CacheEntry represents a cached request-response pair
type CacheEntry struct {
	RequestBody  []byte
	ResponseBody []byte
	Model        string
	Query        string
	Embedding    []float32
	Timestamp    time.Time
}

// SemanticCache implements a semantic cache using BERT embeddings
type SemanticCache struct {
	entries             []CacheEntry
	mu                  sync.RWMutex
	similarityThreshold float32
	maxEntries          int
	ttlSeconds          int
	enabled             bool
}

// SemanticCacheOptions holds options for creating a new semantic cache
type SemanticCacheOptions struct {
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	Enabled             bool
}

// NewSemanticCache creates a new semantic cache with the given options
func NewSemanticCache(options SemanticCacheOptions) *SemanticCache {
	return &SemanticCache{
		entries:             []CacheEntry{},
		similarityThreshold: options.SimilarityThreshold,
		maxEntries:          options.MaxEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}
}

// IsEnabled returns whether the cache is enabled
func (c *SemanticCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest adds a pending request to the cache (without response yet)
func (c *SemanticCache) AddPendingRequest(model string, query string, requestBody []byte) (string, error) {
	if !c.enabled {
		log.Printf("Cache disabled, skipping AddPendingRequest")
		return query, nil
	}

	log.Printf("AddPendingRequest: model=%s, query=%s", model, query)

	// Generate embedding for the query
	log.Printf("Calling candle_binding.GetEmbedding for query: %s", query)
	embedding, err := candle_binding.GetEmbedding(query, 512)
	if err != nil {
		log.Printf("CRITICAL: candle_binding.GetEmbedding failed: %v", err)
		return "", fmt.Errorf("failed to generate embedding: %w", err)
	}
	log.Printf("Embedding generated successfully, length: %d", len(embedding))

	c.mu.Lock()
	defer c.mu.Unlock()

	// Cleanup expired entries if TTL is set
	c.cleanupExpiredEntries()

	// Create a new entry with the pending request
	entry := CacheEntry{
		RequestBody: requestBody,
		Model:       model,
		Query:       query,
		Embedding:   embedding,
		Timestamp:   time.Now(),
	}

	c.entries = append(c.entries, entry)
	// log.Printf("Added pending cache entry for: %s", query)

	// Enforce max entries limit if set
	if c.maxEntries > 0 && len(c.entries) > c.maxEntries {
		// Sort by timestamp (oldest first)
		sort.Slice(c.entries, func(i, j int) bool {
			return c.entries[i].Timestamp.Before(c.entries[j].Timestamp)
		})
		// Remove oldest entries
		c.entries = c.entries[len(c.entries)-c.maxEntries:]
		log.Printf("Trimmed cache to %d entries", c.maxEntries)
	}

	return query, nil
}

// UpdateWithResponse updates a pending request with its response
func (c *SemanticCache) UpdateWithResponse(query string, responseBody []byte) error {
	if !c.enabled {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Find the pending request by query
	for i, entry := range c.entries {
		if entry.Query == query && entry.ResponseBody == nil {
			// Update with response
			c.entries[i].ResponseBody = responseBody
			c.entries[i].Timestamp = time.Now()
			// log.Printf("Cache entry updated: %s", query)
			return nil
		}
	}

	return fmt.Errorf("no pending request found for query: %s", query)
}

// AddEntry adds a complete entry to the cache
func (c *SemanticCache) AddEntry(model string, query string, requestBody, responseBody []byte) error {
	if !c.enabled {
		return nil
	}

	// Generate embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 512)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	entry := CacheEntry{
		RequestBody:  requestBody,
		ResponseBody: responseBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    time.Now(),
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Cleanup expired entries
	c.cleanupExpiredEntries()

	c.entries = append(c.entries, entry)
	log.Printf("Added cache entry: %s", query)

	// Enforce max entries limit
	if c.maxEntries > 0 && len(c.entries) > c.maxEntries {
		// Sort by timestamp (oldest first)
		sort.Slice(c.entries, func(i, j int) bool {
			return c.entries[i].Timestamp.Before(c.entries[j].Timestamp)
		})
		// Remove oldest entries
		c.entries = c.entries[len(c.entries)-c.maxEntries:]
	}

	return nil
}

// FindSimilar looks for a similar request in the cache
func (c *SemanticCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	if !c.enabled {
		log.Printf("Cache disabled, skipping cache lookup")
		return nil, false, nil
	}

	log.Printf("Cache lookup: model=%s, query=%s", model, query)

	// Generate embedding for the query
	log.Printf("Generating embedding for query: %s", query)
	queryEmbedding, err := candle_binding.GetEmbedding(query, 512)
	if err != nil {
		log.Printf("Failed to generate embedding: %v", err)
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}
	log.Printf("Generated embedding successfully, length: %d", len(queryEmbedding))

	c.mu.RLock()
	defer c.mu.RUnlock()

	// Cleanup expired entries
	c.cleanupExpiredEntriesReadOnly()

	type SimilarityResult struct {
		Entry      CacheEntry
		Similarity float32
	}

	// Only compare with entries that have responses
	results := make([]SimilarityResult, 0, len(c.entries))
	for _, entry := range c.entries {
		if entry.ResponseBody == nil {
			continue // Skip entries without responses
		}

		// Only compare with entries with the same model
		if entry.Model != model {
			continue
		}

		// Calculate cosine similarity
		var dotProduct, queryMagnitude, entryMagnitude float32
		for i := 0; i < len(queryEmbedding) && i < len(entry.Embedding); i++ {
			dotProduct += queryEmbedding[i] * entry.Embedding[i]
			queryMagnitude += queryEmbedding[i] * queryEmbedding[i]
			entryMagnitude += entry.Embedding[i] * entry.Embedding[i]
		}

		// Calculate cosine similarity (dot product / (magnitude1 * magnitude2))
		var similarity float32
		if queryMagnitude > 0 && entryMagnitude > 0 {
			similarity = dotProduct / (float32(math.Sqrt(float64(queryMagnitude))) * float32(math.Sqrt(float64(entryMagnitude))))
		}

		results = append(results, SimilarityResult{
			Entry:      entry,
			Similarity: similarity,
		})
	}

	// No results found
	if len(results) == 0 {
		log.Printf("Cache miss: no entries in cache")
		return nil, false, nil
	}

	log.Printf("Cache check: found %d entries to compare", len(results))

	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Log similarity scores for debugging
	log.Printf("Cache check results for query '%s':", query)
	for i, result := range results {
		if i < 3 { // Only log top 3 results
			log.Printf("  Entry %d: similarity=%.4f, query='%s'", i+1, result.Similarity, result.Entry.Query)
		}
	}

	// Check if the best match exceeds the threshold
	if results[0].Similarity >= c.similarityThreshold {
		log.Printf("Cache hit: similarity=%.4f, threshold=%.4f",
			results[0].Similarity, c.similarityThreshold)
		return results[0].Entry.ResponseBody, true, nil
	}

	log.Printf("Cache miss: best similarity=%.4f, threshold=%.4f",
		results[0].Similarity, c.similarityThreshold)
	return nil, false, nil
}

// cleanupExpiredEntries removes expired entries from the cache
// Assumes the caller holds a write lock
func (c *SemanticCache) cleanupExpiredEntries() {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()
	validEntries := make([]CacheEntry, 0, len(c.entries))

	for _, entry := range c.entries {
		// Keep entries that haven't expired
		if now.Sub(entry.Timestamp).Seconds() < float64(c.ttlSeconds) {
			validEntries = append(validEntries, entry)
		}
	}

	if len(validEntries) < len(c.entries) {
		log.Printf("Removed %d expired cache entries", len(c.entries)-len(validEntries))
		c.entries = validEntries
	}
}

// cleanupExpiredEntriesReadOnly checks for expired entries but doesn't modify the cache
// Used during read operations where we only have a read lock
func (c *SemanticCache) cleanupExpiredEntriesReadOnly() {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()
	expiredCount := 0

	for _, entry := range c.entries {
		if now.Sub(entry.Timestamp).Seconds() >= float64(c.ttlSeconds) {
			expiredCount++
		}
	}

	if expiredCount > 0 {
		log.Printf("Found %d expired cache entries during read operation", expiredCount)
	}
}

// ChatMessage represents a message in the OpenAI chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIRequest represents an OpenAI API request
type OpenAIRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      *bool         `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
}

// ExtractQueryFromOpenAIRequest extracts the user query from an OpenAI request
func ExtractQueryFromOpenAIRequest(requestBody []byte) (string, string, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return "", "", fmt.Errorf("invalid request body: %w", err)
	}

	// Extract user messages
	var userMessages []string
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			userMessages = append(userMessages, msg.Content)
		}
	}

	// Join all user messages
	query := ""
	if len(userMessages) > 0 {
		query = userMessages[len(userMessages)-1] // Use the last user message
	}

	return req.Model, query, nil
}
