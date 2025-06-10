/**
 * HTTP API Client for Semantic Router Backend
 * Provides robust communication with the semantic router service via Envoy proxy
 * Supports dual-mode operation (mock vs live) with comprehensive error handling
 */

class ApiClient {
  constructor(configManager) {
    this.config = configManager;
    this.requestDefaults = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    };
    
    // Performance metrics
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      lastError: null,
      // Retry-specific metrics
      totalRetries: 0,
      retriedRequests: 0,
      successAfterRetry: 0
    };
    
    // Request interceptors for debugging and logging
    this.requestInterceptors = [];
    this.responseInterceptors = [];
    
    // Active request tracking for cancellation
    this.activeRequests = new Map();
    this.requestCounter = 0;
    
    // Error handling and circuit breaker
    this.errorHandler = new ErrorHandler(this.config);
    this.circuitBreaker = new CircuitBreaker(this.config);
  }

  /**
   * Core HTTP request method with timeout and abort controller support
   * @param {string} url - The endpoint URL
   * @param {Object} options - Request options
   * @returns {Promise<ApiResponse>} - Standardized API response
   */
  async request(url, options = {}) {
    const startTime = Date.now();
    this.metrics.totalRequests++;
    
    // Generate unique request ID for tracking
    const requestId = ++this.requestCounter;
    
    // Check circuit breaker
    if (!this.circuitBreaker.allowRequest()) {
      const circuitError = new ApiError(
        'Circuit breaker is open - too many recent failures',
        503,
        null,
        0,
        'CIRCUIT_BREAKER'
      );
      circuitError.requestId = requestId;
      circuitError.circuitBreakerStatus = this.circuitBreaker.getStatus();
      
      // Handle error through error handler
      const handledError = this.errorHandler.handleError(circuitError, {
        url,
        requestId,
        circuitBreakerOpen: true
      });
      
      throw circuitError;
    }
    
    // Build request configuration
    const requestConfig = this.buildRequestConfig(url, options);
    
    // Create abort controller for timeout/cancellation
    const controller = new AbortController();
    
    // Set up timeout with configurable behavior
    const timeoutMs = requestConfig.timeout;
    const timeoutId = setTimeout(() => {
      controller.abort();
      this.logTimeout(requestId, timeoutMs);
    }, timeoutMs);
    
    // Track active request for manual cancellation
    const requestInfo = {
      id: requestId,
      url: requestConfig.url,
      method: requestConfig.method,
      startTime,
      controller,
      timeoutId
    };
    this.activeRequests.set(requestId, requestInfo);
    
    try {
      // Apply request interceptors
      const interceptedConfig = await this.applyRequestInterceptors(requestConfig);
      
      // Log request in debug mode
      this.logRequest(interceptedConfig, requestId);
      
      // Make the HTTP request
      const response = await fetch(interceptedConfig.url, {
        ...interceptedConfig,
        signal: controller.signal
      });
      
      // Clean up request tracking
      this.cleanupRequest(requestId);
      
      // Process response
      const apiResponse = await this.processResponse(response, startTime, requestId);
      
      // Apply response interceptors
      const interceptedResponse = await this.applyResponseInterceptors(apiResponse);
      
      // Record success in circuit breaker
      this.circuitBreaker.recordSuccess();
      
      // Update metrics
      this.updateMetrics(true, Date.now() - startTime);
      
      return interceptedResponse;
      
    } catch (error) {
      // Clean up request tracking
      this.cleanupRequest(requestId);
      
      // Process error
      const processedError = this.processError(error, startTime, requestId);
      
      // Record failure in circuit breaker if it should trigger it
      if (processedError.shouldTriggerCircuitBreaker()) {
        this.circuitBreaker.recordFailure(processedError);
      }
      
      // Handle error through error handler
      const handledError = this.errorHandler.handleError(processedError, {
        url,
        requestId,
        requestConfig,
        circuitBreakerStatus: this.circuitBreaker.getStatus()
      });
      
      // Update metrics
      this.updateMetrics(false, Date.now() - startTime, processedError);
      
      throw processedError;
    }
  }

  /**
   * Build complete request configuration
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @returns {Object} - Complete request configuration
   */
  buildRequestConfig(url, options) {
    const config = {
      url: url,
      method: options.method || 'GET',
      headers: {
        ...this.requestDefaults.headers,
        ...(options.headers || {})
      },
      timeout: options.timeout || this.config.get('timeout'),
      ...options
    };
    
    // Add body for POST/PUT requests
    if (config.method !== 'GET' && config.method !== 'HEAD' && options.body) {
      if (typeof options.body === 'object') {
        config.body = JSON.stringify(options.body);
      } else {
        config.body = options.body;
      }
    }
    
    return config;
  }

  /**
   * Process HTTP response into standardized format
   * @param {Response} response - Fetch API response
   * @param {number} startTime - Request start timestamp
   * @param {number} requestId - Request ID for tracking
   * @returns {Promise<ApiResponse>} - Standardized response object
   */
  async processResponse(response, startTime, requestId) {
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    // Extract headers
    const headers = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });
    
    // Parse response body
    let data = null;
    const contentType = response.headers.get('content-type') || '';
    
    try {
      if (contentType.includes('application/json')) {
        data = await response.json();
      } else if (contentType.includes('text/')) {
        data = await response.text();
      } else {
        data = await response.blob();
      }
    } catch (parseError) {
      console.warn('Failed to parse response body:', parseError);
      data = null;
    }
    
    const apiResponse = {
      data,
      status: response.status,
      statusText: response.statusText,
      headers,
      ok: response.ok,
      responseTime,
      timestamp: endTime,
      url: response.url
    };
    
    // Log response in debug mode
    this.logResponse(apiResponse, requestId);
    
    // Throw error for non-2xx responses
    if (!response.ok) {
      throw new ApiError(
        `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        apiResponse
      );
    }
    
    return apiResponse;
  }

  /**
   * Process and standardize errors
   * @param {Error} error - Original error
   * @param {number} startTime - Request start timestamp
   * @param {number} requestId - Request ID for tracking
   * @returns {ApiError} - Standardized API error
   */
  processError(error, startTime, requestId) {
    const responseTime = Date.now() - startTime;
    
    // Handle different error types
    if (error.name === 'AbortError') {
      const apiError = new ApiError('Request timeout', 408, null, responseTime, 'TIMEOUT');
      apiError.requestId = requestId;
      return apiError;
    }
    
    if (error instanceof TypeError && error.message.includes('fetch')) {
      const apiError = new ApiError('Network error', 0, null, responseTime, 'NETWORK');
      apiError.requestId = requestId;
      return apiError;
    }
    
    if (error instanceof ApiError) {
      error.requestId = requestId;
      return error;
    }
    
    // Generic error
    const apiError = new ApiError(
      error.message || 'Unknown error',
      0,
      null,
      responseTime,
      'UNKNOWN'
    );
    apiError.requestId = requestId;
    return apiError;
  }

  /**
   * HTTP GET request
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @returns {Promise<ApiResponse>}
   */
  async get(url, options = {}) {
    return this.request(url, { ...options, method: 'GET' });
  }

  /**
   * HTTP POST request
   * @param {string} url - Request URL
   * @param {Object} body - Request body
   * @param {Object} options - Request options
   * @returns {Promise<ApiResponse>}
   */
  async post(url, body = null, options = {}) {
    return this.request(url, { ...options, method: 'POST', body });
  }

  /**
   * HTTP PUT request
   * @param {string} url - Request URL
   * @param {Object} body - Request body
   * @param {Object} options - Request options
   * @returns {Promise<ApiResponse>}
   */
  async put(url, body = null, options = {}) {
    return this.request(url, { ...options, method: 'PUT', body });
  }

  /**
   * HTTP DELETE request
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @returns {Promise<ApiResponse>}
   */
  async delete(url, options = {}) {
    return this.request(url, { ...options, method: 'DELETE' });
  }

  /**
   * Make a request with automatic retry logic for transient failures
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @param {Object} retryOptions - Retry configuration
   * @returns {Promise<ApiResponse>}
   */
  async requestWithRetry(url, options = {}, retryOptions = {}) {
    const retryConfig = {
      maxAttempts: retryOptions.maxAttempts || this.config.get('retryAttempts'),
      baseDelay: retryOptions.baseDelay || this.config.get('retryDelay'),
      backoffMultiplier: retryOptions.backoffMultiplier || this.config.get('retryBackoffMultiplier'),
      maxDelay: retryOptions.maxDelay || 30000, // 30 seconds max delay
      jitter: retryOptions.jitter !== false, // Add jitter by default
      retryCondition: retryOptions.retryCondition || this.defaultRetryCondition.bind(this),
      onRetry: retryOptions.onRetry || null
    };

    let lastError;
    let attempt = 0;
    let hasRetried = false;

    while (attempt <= retryConfig.maxAttempts) {
      try {
        // Add attempt info to request options for logging
        const requestOptions = {
          ...options,
          _retryAttempt: attempt,
          _maxAttempts: retryConfig.maxAttempts
        };

        const response = await this.request(url, requestOptions);
        
        // Success - update metrics and log if this was a retry
        if (attempt > 0) {
          this.metrics.successAfterRetry++;
          this.logRetrySuccess(url, attempt, retryConfig.maxAttempts);
        }
        
        return response;

      } catch (error) {
        lastError = error;
        attempt++;

        // Check if we should retry this error
        if (attempt > retryConfig.maxAttempts || !retryConfig.retryCondition(error, attempt)) {
          break;
        }

        // Update retry metrics
        if (!hasRetried) {
          this.metrics.retriedRequests++;
          hasRetried = true;
        }
        this.metrics.totalRetries++;

        // Calculate delay with exponential backoff and jitter
        const delay = this.calculateRetryDelay(
          retryConfig.baseDelay,
          attempt,
          retryConfig.backoffMultiplier,
          retryConfig.maxDelay,
          retryConfig.jitter
        );

        // Log retry attempt
        this.logRetryAttempt(url, attempt, retryConfig.maxAttempts, delay, error);

        // Call retry callback if provided
        if (retryConfig.onRetry) {
          try {
            await retryConfig.onRetry(error, attempt, delay);
          } catch (callbackError) {
            console.warn('Retry callback failed:', callbackError);
          }
        }

        // Wait before retry
        await this.sleep(delay);
      }
    }

    // All retries exhausted
    this.logRetryExhausted(url, retryConfig.maxAttempts, lastError);
    throw lastError;
  }

  /**
   * Default retry condition - determines if an error should trigger a retry
   * @param {ApiError} error - The error that occurred
   * @param {number} attempt - Current attempt number (1-based)
   * @returns {boolean} - Whether to retry
   */
  defaultRetryCondition(error, attempt) {
    // Don't retry if it's an ApiError and it's not retryable
    if (error instanceof ApiError) {
      return error.isRetryable();
    }

    // For non-ApiError instances, be conservative
    // Only retry timeouts and network errors
    if (error.name === 'AbortError') {
      return true; // Timeout
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return true; // Network error
    }

    return false;
  }

  /**
   * Calculate retry delay with exponential backoff and optional jitter
   * @param {number} baseDelay - Base delay in milliseconds
   * @param {number} attempt - Current attempt number (1-based)
   * @param {number} backoffMultiplier - Backoff multiplier
   * @param {number} maxDelay - Maximum delay in milliseconds
   * @param {boolean} jitter - Whether to add jitter
   * @returns {number} - Delay in milliseconds
   */
  calculateRetryDelay(baseDelay, attempt, backoffMultiplier, maxDelay, jitter) {
    // Exponential backoff: delay = baseDelay * (backoffMultiplier ^ (attempt - 1))
    let delay = baseDelay * Math.pow(backoffMultiplier, attempt - 1);

    // Cap at max delay
    delay = Math.min(delay, maxDelay);

    // Add jitter to prevent thundering herd
    if (jitter) {
      // Add random jitter up to 25% of the delay
      const jitterAmount = delay * 0.25 * Math.random();
      delay += jitterAmount;
    }

    return Math.round(delay);
  }

  /**
   * Sleep for specified milliseconds
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise} - Promise that resolves after delay
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Convenience method for POST with retry
   * @param {string} url - Request URL
   * @param {Object} body - Request body
   * @param {Object} options - Request options
   * @param {Object} retryOptions - Retry configuration
   * @returns {Promise<ApiResponse>}
   */
  async postWithRetry(url, body = null, options = {}, retryOptions = {}) {
    return this.requestWithRetry(url, { ...options, method: 'POST', body }, retryOptions);
  }

  /**
   * Convenience method for GET with retry
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @param {Object} retryOptions - Retry configuration
   * @returns {Promise<ApiResponse>}
   */
  async getWithRetry(url, options = {}, retryOptions = {}) {
    return this.requestWithRetry(url, { ...options, method: 'GET' }, retryOptions);
  }

  /**
   * Send a message to the semantic router and extract structured data
   * @param {string} message - The message to send
   * @param {Object} options - Request options
   * @returns {Promise<SemanticRouterResponse>} - Structured response with extracted data
   */
  async sendSemanticRouterMessage(message, options = {}) {
    const endpoint = this.config.get('apiEndpoint');
    
    const requestBody = {
      model: 'mistral-small3.1', // Default model from your config, will be overridden by router
      messages: [
        { role: 'user', content: message }
      ],
      stream: false
    };

    // Use retry by default for semantic router requests
    const useRetry = options.useRetry !== false;
    const requestMethod = useRetry ? this.postWithRetry.bind(this) : this.post.bind(this);

    try {
      const response = await requestMethod(endpoint, requestBody, options);
      
      // Extract structured data from response
      const extractedData = this.extractSemanticRouterData(response);
      
      // Process markdown content if present
      let formattedContent = null;
      if (response.data && response.data.choices && response.data.choices[0] && response.data.choices[0].message) {
        const messageContent = response.data.choices[0].message.content;
        if (messageContent) {
          formattedContent = this.processMarkdownResponse(messageContent);
        }
      }
      
      // Enhance with message analysis if no metadata found
      if (!extractedData.semanticCategory || extractedData.semanticCategory === 'General') {
        extractedData.semanticCategory = this.analyzeMessageForCategory(message);
        extractedData.confidence = this.calculateInferredConfidence(message, extractedData.semanticCategory);
      }
      
      // Enhance PII detection if not found
      if (!extractedData.piiDetection || extractedData.piiDetection.length === 0) {
        extractedData.piiDetection = this.analyzeMessageForPii(message);
      }
      
      return {
        ...response,
        semanticData: extractedData,
        formattedContent,
        originalMessage: message
      };
      
    } catch (error) {
      // Add context to error
      error.originalMessage = message;
      error.endpoint = endpoint;
      throw error;
    }
  }

  /**
   * Analyze message content to infer semantic category
   * @param {string} message - The message to analyze
   * @returns {string} - Inferred category
   */
  analyzeMessageForCategory(message) {
    const messageLower = message.toLowerCase();
    
    // Enhanced keyword matching for better inference with word boundaries
    const categoryKeywords = {
      'Mathematics': [
        // Math terms
        '\\bmath\\b', '\\bcalculate\\b', '\\bderivative\\b', '\\bintegral\\b', '\\bequation\\b', 
        '\\bsolve\\b', '\\balgebra\\b', '\\bgeometry\\b', '\\bstatistics\\b', '\\bprobability\\b',
        '\\bformula\\b', '\\bcalculus\\b', '\\bnumber\\b', '\\bsum\\b', '\\broot\\b',
        // Math symbols (keep as simple contains for symbols)
        '\\+', '\\-', '\\*', '/', '=', '\\^'
      ],
      'Programming': [
        '\\bcode\\b', '\\bfunction\\b', '\\bvariable\\b', '\\bloop\\b', '\\barray\\b', 
        '\\bjavascript\\b', '\\bpython\\b', '\\breact\\b', '\\bapi\\b', '\\bdebug\\b', 
        '\\bsyntax\\b', '\\bprogramming\\b', '\\bsoftware\\b', '\\balgorithm\\b', 
        '\\bclass\\b', '\\bobject\\b', '\\bmethod\\b', '\\bhtml\\b', '\\bcss\\b'
      ],
      'Health': [
        '\\bdoctor\\b', '\\bmedicine\\b', '\\bsymptom\\b', '\\btreatment\\b', '\\bhealth\\b', 
        '\\bmedical\\b', '\\bhospital\\b', '\\bdisease\\b', '\\bdrug\\b', '\\btherapy\\b', 
        '\\bdiagnosis\\b', '\\bpain\\b', '\\bsick\\b', '\\bmedication\\b', '\\bpatient\\b'
      ],
      'History': [
        '\\bhistory\\b', '\\bhistorical\\b', '\\bwar\\b', '\\bcentury\\b', '\\bancient\\b', 
        '\\bcivilization\\b', '\\bempire\\b', '\\brevolution\\b', '\\bbattle\\b', '\\btimeline\\b',
        '\\bking\\b', '\\bqueen\\b', '\\bpast\\b', '\\bold\\b', '\\byears ago\\b'
      ],
      'General': [
        '\\bhello\\b', '\\bhelp\\b', '\\bwhat\\b', '\\bhow\\b', '\\bwhen\\b', '\\bwhere\\b', 
        '\\bwhy\\b', '\\bquestion\\b', '\\banswer\\b', '\\bdog\\b', '\\bcat\\b', '\\btrain\\b',
        '\\bsit\\b', '\\blearn\\b', '\\bteach\\b', '\\bweather\\b', '\\btoday\\b', '\\badvice\\b'
      ]
    };
    
    let bestCategory = 'General';
    let bestScore = 0;
    let detectedKeywords = [];
    
    for (const [category, keywords] of Object.entries(categoryKeywords)) {
      let score = 0;
      let categoryKeywords = [];
      
      keywords.forEach(keyword => {
        if (keyword.startsWith('\\') && keyword.endsWith('\\b')) {
          // Use regex for word boundary matching
          const regex = new RegExp(keyword, 'i');
          if (regex.test(messageLower)) {
            score += 1;
            // Extract the actual word without regex markers
            categoryKeywords.push(keyword.replace(/\\b/g, '').replace(/\\/g, ''));
          }
        } else {
          // Use simple includes for symbols
          if (messageLower.includes(keyword)) {
            score += 1;
            categoryKeywords.push(keyword);
          }
        }
      });
      
      if (score > bestScore) {
        bestScore = score;
        bestCategory = category;
        detectedKeywords = categoryKeywords;
      }
    }
    
    // Debug logging
    console.log('Classification Debug:', {
      message: message.substring(0, 50) + '...',
      bestCategory,
      bestScore,
      detectedKeywords
    });
    
    return bestCategory;
  }

  /**
   * Calculate confidence for inferred classification
   * @param {string} message - The message
   * @param {string} category - The inferred category
   * @returns {number} - Confidence between 0 and 1
   */
  calculateInferredConfidence(message, category) {
    if (category === 'General') {
      return 0.7; // Lower confidence for general classification
    }
    return 0.85; // Higher confidence for specific category matches
  }

  /**
   * Analyze message for PII content
   * @param {string} message - The message to analyze
   * @returns {Array} - Detected PII items
   */
  analyzeMessageForPii(message) {
    const piiPatterns = [
      {
        type: 'EMAIL_ADDRESS',
        pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
        risk: 'medium'
      },
      {
        type: 'PHONE_NUMBER', 
        pattern: /\b(?:\+?1[-\s]?)?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b/g,
        risk: 'medium'
      },
      {
        type: 'CREDIT_CARD',
        pattern: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
        risk: 'high'
      },
      {
        type: 'SSN',
        pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
        risk: 'high'
      }
    ];
    
    const detectedPii = [];
    
    piiPatterns.forEach(pattern => {
      const matches = message.match(pattern.pattern);
      if (matches) {
        matches.forEach(match => {
          detectedPii.push({
            type: pattern.type,
            value: match,
            risk: pattern.risk
          });
        });
      }
    });
    
    return detectedPii;
  }

  /**
   * Extract semantic router data from API response
   * @param {ApiResponse} response - Raw API response
   * @returns {Object} - Extracted semantic data
   */
  extractSemanticRouterData(response) {
    const data = response.data;
    const headers = response.headers;
    
    return {
      semanticCategory: this.extractSemanticCategory(data, headers),
      piiDetection: this.extractPiiDetection(data, headers),
      selectedModel: this.extractSelectedModel(data, headers),
      processingTime: this.extractProcessingTime(data, headers),
      confidence: this.extractConfidence(data, headers),
      metadata: this.extractMetadata(data, headers)
    };
  }

  /**
   * Extract semantic category from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {string} - Semantic category
   */
  extractSemanticCategory(data, headers) {
    // Try multiple possible locations for semantic category
    
    // Check headers first (common in proxy setups)
    if (headers['x-semantic-category']) {
      return headers['x-semantic-category'];
    }
    
    if (headers['semantic-category']) {
      return headers['semantic-category'];
    }
    
    // Check response data
    if (data && typeof data === 'object') {
      // Direct property
      if (data.semantic_category) {
        return data.semantic_category;
      }
      
      if (data.semanticCategory) {
        return data.semanticCategory;
      }
      
      if (data.category) {
        return data.category;
      }
      
      // Nested in metadata
      if (data.metadata && data.metadata.semantic_category) {
        return data.metadata.semantic_category;
      }
      
      // Check if it's in choices (OpenAI format)
      if (data.choices && data.choices.length > 0) {
        const choice = data.choices[0];
        if (choice.metadata && choice.metadata.semantic_category) {
          return choice.metadata.semantic_category;
        }
      }
      
      // Infer category from model selection (fallback strategy)
      if (data.model) {
        return this.inferCategoryFromModel(data.model);
      }
    }
    
    return 'General'; // Default fallback
  }

  /**
   * Infer semantic category from the selected model
   * @param {string} model - Model name
   * @returns {string} - Inferred category
   */
  inferCategoryFromModel(model) {
    const modelCategoryMap = {
      'phi4': 'Mathematics',        // Best for Math category
      'mistral-small3.1': 'General', // Best for General, good for History  
      'gemma3:27b': 'Programming',   // Best for Programming, good for Health
      'default': 'General'
    };
    
    const normalizedModel = model.toLowerCase();
    
    // Check exact matches first
    for (const [modelName, category] of Object.entries(modelCategoryMap)) {
      if (normalizedModel.includes(modelName.toLowerCase())) {
        return category;
      }
    }
    
    return modelCategoryMap['default'];
  }

  /**
   * Extract PII detection results from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {Array|Object} - PII detection results
   */
  extractPiiDetection(data, headers) {
    // Check headers
    if (headers['x-pii-detected']) {
      try {
        return JSON.parse(headers['x-pii-detected']);
      } catch (e) {
        return headers['x-pii-detected'] === 'true' ? ['PII detected'] : [];
      }
    }
    
    // Check response data
    if (data && typeof data === 'object') {
      // Direct properties
      if (data.pii_detected !== undefined) {
        return Array.isArray(data.pii_detected) ? data.pii_detected : [data.pii_detected];
      }
      
      if (data.piiDetected !== undefined) {
        return Array.isArray(data.piiDetected) ? data.piiDetected : [data.piiDetected];
      }
      
      if (data.pii !== undefined) {
        return Array.isArray(data.pii) ? data.pii : [data.pii];
      }
      
      // Nested in metadata
      if (data.metadata && data.metadata.pii_detected !== undefined) {
        return Array.isArray(data.metadata.pii_detected) ? data.metadata.pii_detected : [data.metadata.pii_detected];
      }
      
      // Check choices
      if (data.choices && data.choices.length > 0) {
        const choice = data.choices[0];
        if (choice.metadata && choice.metadata.pii_detected !== undefined) {
          return Array.isArray(choice.metadata.pii_detected) ? choice.metadata.pii_detected : [choice.metadata.pii_detected];
        }
      }
    }
    
    return []; // Default to no PII detected
  }

  /**
   * Extract selected model from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {string} - Selected model name
   */
  extractSelectedModel(data, headers) {
    // Check headers
    if (headers['x-selected-model']) {
      return headers['x-selected-model'];
    }
    
    if (headers['model']) {
      return headers['model'];
    }
    
    // Check response data
    if (data && typeof data === 'object') {
      // Direct property
      if (data.model) {
        return data.model;
      }
      
      if (data.selected_model) {
        return data.selected_model;
      }
      
      if (data.selectedModel) {
        return data.selectedModel;
      }
      
      // Nested in metadata
      if (data.metadata && data.metadata.model) {
        return data.metadata.model;
      }
      
      // Check choices
      if (data.choices && data.choices.length > 0) {
        const choice = data.choices[0];
        if (choice.metadata && choice.metadata.model) {
          return choice.metadata.model;
        }
      }
    }
    
    return 'default-model'; // Default fallback
  }

  /**
   * Extract processing time from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {number|null} - Processing time in milliseconds
   */
  extractProcessingTime(data, headers) {
    // Check headers
    if (headers['x-processing-time']) {
      return parseFloat(headers['x-processing-time']);
    }
    
    if (headers['processing-time']) {
      return parseFloat(headers['processing-time']);
    }
    
    // Check response data
    if (data && typeof data === 'object') {
      if (data.processing_time !== undefined) {
        return parseFloat(data.processing_time);
      }
      
      if (data.processingTime !== undefined) {
        return parseFloat(data.processingTime);
      }
      
      if (data.metadata && data.metadata.processing_time !== undefined) {
        return parseFloat(data.metadata.processing_time);
      }
    }
    
    return null;
  }

  /**
   * Extract confidence score from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {number|null} - Confidence score (0-1)
   */
  extractConfidence(data, headers) {
    // Check headers
    if (headers['x-confidence']) {
      return parseFloat(headers['x-confidence']);
    }
    
    // Check response data
    if (data && typeof data === 'object') {
      if (data.confidence !== undefined) {
        return parseFloat(data.confidence);
      }
      
      if (data.metadata && data.metadata.confidence !== undefined) {
        return parseFloat(data.metadata.confidence);
      }
    }
    
    return null;
  }

  /**
   * Extract additional metadata from response
   * @param {Object} data - Response data
   * @param {Object} headers - Response headers
   * @returns {Object} - Additional metadata
   */
  extractMetadata(data, headers) {
    const metadata = {};
    
    // Extract custom headers
    Object.keys(headers).forEach(key => {
      if (key.startsWith('x-') || key.startsWith('semantic-')) {
        metadata[key] = headers[key];
      }
    });
    
    // Extract metadata from response data
    if (data && typeof data === 'object') {
      if (data.metadata) {
        Object.assign(metadata, data.metadata);
      }
      
      // Extract usage information (common in OpenAI-style responses)
      if (data.usage) {
        metadata.usage = data.usage;
      }
      
      // Extract timing information
      if (data.created) {
        metadata.created = data.created;
      }
      
      if (data.id) {
        metadata.responseId = data.id;
      }
    }
    
    return metadata;
  }

  /**
   * Validate and normalize semantic category
   * @param {string} category - Raw category string
   * @returns {string} - Normalized category
   */
  normalizeSemanticCategory(category) {
    if (!category || typeof category !== 'string') {
      return 'General';
    }
    
    // Common category mappings
    const categoryMappings = {
      'math': 'Mathematics',
      'mathematics': 'Mathematics',
      'history': 'History',
      'historical': 'History',
      'health': 'Health',
      'medical': 'Health',
      'programming': 'Programming',
      'code': 'Programming',
      'coding': 'Programming',
      'tech': 'Technology',
      'technology': 'Technology',
      'general': 'General',
      'other': 'General'
    };
    
    const normalized = category.toLowerCase().trim();
    return categoryMappings[normalized] || category;
  }

  /**
   * Validate and normalize PII detection results
   * @param {Array|Object|string} piiData - Raw PII data
   * @returns {Array} - Normalized PII array
   */
  normalizePiiDetection(piiData) {
    if (!piiData) {
      return [];
    }
    
    // Handle different input types
    if (typeof piiData === 'string') {
      if (piiData.toLowerCase() === 'true' || piiData.toLowerCase() === 'detected') {
        return ['PII detected'];
      }
      if (piiData.toLowerCase() === 'false' || piiData.toLowerCase() === 'none') {
        return [];
      }
      return [piiData];
    }
    
    if (Array.isArray(piiData)) {
      return piiData.filter(item => item && typeof item === 'string');
    }
    
    if (typeof piiData === 'object') {
      // Handle object format like { detected: true, types: ['email', 'phone'] }
      if (piiData.detected === false) {
        return [];
      }
      if (piiData.types && Array.isArray(piiData.types)) {
        return piiData.types;
      }
      if (piiData.entities && Array.isArray(piiData.entities)) {
        return piiData.entities;
      }
    }
    
    return [];
  }

  /**
   * Validate and normalize model name
   * @param {string} model - Raw model name
   * @returns {string} - Normalized model name
   */
  normalizeModelName(model) {
    if (!model || typeof model !== 'string') {
      return 'default-model';
    }
    
    // Common model name mappings
    const modelMappings = {
      'gpt-3.5-turbo': 'GPT-3.5 Turbo',
      'gpt-4': 'GPT-4',
      'claude-3-sonnet': 'Claude 3 Sonnet',
      'gemma3:27b': 'Gemma 3 27B',
      'phi4': 'Phi-4',
      'mistral-small3.1': 'Mistral Small 3.1'
    };
    
    return modelMappings[model] || model;
  }

  /**
   * Create a standardized semantic router response object
   * @param {string} message - Original message
   * @param {Object} rawResponse - Raw API response
   * @returns {Object} - Standardized response
   */
  createStandardizedResponse(message, rawResponse) {
    const extractedData = this.extractSemanticRouterData(rawResponse);
    
    return {
      // Original data
      originalMessage: message,
      rawResponse: rawResponse,
      
      // Extracted and normalized data
      semanticCategory: this.normalizeSemanticCategory(extractedData.semanticCategory),
      piiDetection: this.normalizePiiDetection(extractedData.piiDetection),
      selectedModel: this.normalizeModelName(extractedData.selectedModel),
      
      // Additional metadata
      processingTime: extractedData.processingTime,
      confidence: extractedData.confidence,
      metadata: extractedData.metadata,
      
      // Response metadata
      responseTime: rawResponse.responseTime,
      timestamp: rawResponse.timestamp,
      status: rawResponse.status,
      
      // Utility methods
      hasPii: () => extractedData.piiDetection.length > 0,
      isHighConfidence: () => extractedData.confidence && extractedData.confidence > 0.8,
      getDisplayCategory: () => this.normalizeSemanticCategory(extractedData.semanticCategory),
      getDisplayModel: () => this.normalizeModelName(extractedData.selectedModel)
    };
  }

  /**
   * Batch process multiple messages
   * @param {Array<string>} messages - Array of messages to process
   * @param {Object} options - Request options
   * @returns {Promise<Array>} - Array of standardized responses
   */
  async batchProcessMessages(messages, options = {}) {
    const batchOptions = {
      concurrent: options.concurrent || 3, // Max concurrent requests
      delay: options.delay || 100, // Delay between requests
      ...options
    };
    
    const results = [];
    const errors = [];
    
    // Process in batches to avoid overwhelming the server
    for (let i = 0; i < messages.length; i += batchOptions.concurrent) {
      const batch = messages.slice(i, i + batchOptions.concurrent);
      
      const batchPromises = batch.map(async (message, index) => {
        try {
          const response = await this.sendSemanticRouterMessage(message, batchOptions);
          return this.createStandardizedResponse(message, response);
        } catch (error) {
          errors.push({ message, error, index: i + index });
          return null;
        }
      });
      
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults.filter(result => result !== null));
      
      // Add delay between batches
      if (i + batchOptions.concurrent < messages.length && batchOptions.delay > 0) {
        await this.sleep(batchOptions.delay);
      }
    }
    
    return {
      results,
      errors,
      totalProcessed: results.length,
      totalErrors: errors.length
    };
  }

  /**
   * Get comprehensive client health status
   * @returns {Object} - Health status including metrics, errors, and circuit breaker
   */
  getHealthStatus() {
    return {
      metrics: this.getMetrics(),
      errorStats: this.errorHandler.getErrorStats(),
      circuitBreaker: this.circuitBreaker.getStatus(),
      activeRequests: this.getActiveRequests(),
      timestamp: Date.now()
    };
  }

  /**
   * Register error callback for specific error types
   * @param {string} errorType - Error type to listen for ('*' for all)
   * @param {Function} callback - Callback function
   */
  onError(errorType, callback) {
    this.errorHandler.onError(errorType, callback);
  }

  /**
   * Reset all error tracking and circuit breaker
   */
  resetErrorTracking() {
    this.errorHandler.clearHistory();
    this.circuitBreaker.reset();
    this.resetMetrics();
  }

  /**
   * Check if the client is in a healthy state
   * @returns {boolean} - Whether the client is healthy
   */
  isHealthy() {
    const circuitStatus = this.circuitBreaker.getStatus();
    const metrics = this.getMetrics();
    
    return !circuitStatus.isOpen && 
           metrics.successRate > 50 && 
           this.activeRequests.size < 10;
  }

  /**
   * Get recovery recommendations based on current state
   * @returns {Array<string>} - Array of recovery recommendations
   */
  getRecoveryRecommendations() {
    const recommendations = [];
    const health = this.getHealthStatus();
    
    if (health.circuitBreaker.isOpen) {
      recommendations.push(`Circuit breaker is open. Wait ${Math.ceil(health.circuitBreaker.timeUntilRetry / 1000)} seconds before retrying.`);
    }
    
    if (health.metrics.successRate < 50) {
      recommendations.push('Low success rate detected. Check network connectivity and API endpoint.');
    }
    
    if (health.activeRequests.length > 5) {
      recommendations.push('High number of active requests. Consider reducing request frequency.');
    }
    
    if (health.errorStats.bySeverity.critical > 0) {
      recommendations.push('Critical errors detected. Check network connectivity and API configuration.');
    }
    
    if (health.metrics.averageResponseTime > 10000) {
      recommendations.push('High response times detected. Consider increasing timeout or checking server performance.');
    }
    
    return recommendations;
  }

  /**
   * Add request interceptor
   * @param {Function} interceptor - Interceptor function
   */
  addRequestInterceptor(interceptor) {
    this.requestInterceptors.push(interceptor);
  }

  /**
   * Add response interceptor
   * @param {Function} interceptor - Interceptor function
   */
  addResponseInterceptor(interceptor) {
    this.responseInterceptors.push(interceptor);
  }

  /**
   * Apply request interceptors
   * @param {Object} config - Request configuration
   * @returns {Promise<Object>} - Modified configuration
   */
  async applyRequestInterceptors(config) {
    let modifiedConfig = { ...config };
    
    for (const interceptor of this.requestInterceptors) {
      try {
        modifiedConfig = await interceptor(modifiedConfig);
      } catch (error) {
        console.warn('Request interceptor failed:', error);
      }
    }
    
    return modifiedConfig;
  }

  /**
   * Apply response interceptors
   * @param {ApiResponse} response - API response
   * @returns {Promise<ApiResponse>} - Modified response
   */
  async applyResponseInterceptors(response) {
    let modifiedResponse = { ...response };
    
    for (const interceptor of this.responseInterceptors) {
      try {
        modifiedResponse = await interceptor(modifiedResponse);
      } catch (error) {
        console.warn('Response interceptor failed:', error);
      }
    }
    
    return modifiedResponse;
  }

  /**
   * Log request details
   * @param {Object} config - Request configuration
   * @param {number} requestId - Request ID for tracking
   */
  logRequest(config, requestId) {
    if (this.config.get('logLevel') === 'debug') {
      console.group(`🚀 API Request #${requestId}: ${config.method} ${config.url}`);
      console.log('Headers:', config.headers);
      console.log('Body:', config.body);
      console.log('Timeout:', config.timeout);
      console.log('Active Requests:', this.activeRequests.size);
      console.groupEnd();
    }
  }

  /**
   * Log response details
   * @param {ApiResponse} response - API response
   * @param {number} requestId - Request ID for tracking
   */
  logResponse(response, requestId) {
    if (this.config.get('logLevel') === 'debug') {
      console.group(`📥 API Response #${requestId}: ${response.status} ${response.statusText}`);
      console.log('Response Time:', `${response.responseTime}ms`);
      console.log('Headers:', response.headers);
      console.log('Data:', response.data);
      console.groupEnd();
    }
  }

  /**
   * Log timeout event
   * @param {number} requestId - Request ID that timed out
   * @param {number} timeoutMs - Timeout duration in milliseconds
   */
  logTimeout(requestId, timeoutMs) {
    if (this.config.get('logLevel') === 'debug' || this.config.get('logLevel') === 'info') {
      console.warn(`⏰ Request #${requestId} timed out after ${timeoutMs}ms`);
    }
  }

  /**
   * Log request cancellation
   * @param {number} requestId - Request ID that was cancelled
   */
  logRequestCancelled(requestId) {
    if (this.config.get('logLevel') === 'debug' || this.config.get('logLevel') === 'info') {
      console.info(`❌ Request #${requestId} was manually cancelled`);
    }
  }

  /**
   * Log bulk request cancellation
   * @param {number} count - Number of requests cancelled
   */
  logAllRequestsCancelled(count) {
    if (this.config.get('logLevel') === 'debug' || this.config.get('logLevel') === 'info') {
      console.info(`❌ Cancelled ${count} active requests`);
    }
  }

  /**
   * Log retry attempt
   * @param {string} url - Request URL
   * @param {number} attempt - Current attempt number
   * @param {number} maxAttempts - Maximum attempts
   * @param {number} delay - Delay before retry
   * @param {Error} error - Error that triggered retry
   */
  logRetryAttempt(url, attempt, maxAttempts, delay, error) {
    const logLevel = this.config.get('logLevel');
    if (logLevel === 'debug' || logLevel === 'info') {
      console.warn(`🔄 Retry ${attempt}/${maxAttempts} for ${url} in ${delay}ms (${error.message})`);
    }
  }

  /**
   * Log successful retry
   * @param {string} url - Request URL
   * @param {number} attempts - Number of attempts made
   * @param {number} maxAttempts - Maximum attempts
   */
  logRetrySuccess(url, attempts, maxAttempts) {
    const logLevel = this.config.get('logLevel');
    if (logLevel === 'debug' || logLevel === 'info') {
      console.info(`✅ Request succeeded after ${attempts}/${maxAttempts} attempts: ${url}`);
    }
  }

  /**
   * Log retry exhaustion
   * @param {string} url - Request URL
   * @param {number} maxAttempts - Maximum attempts
   * @param {Error} lastError - Final error
   */
  logRetryExhausted(url, maxAttempts, lastError) {
    const logLevel = this.config.get('logLevel');
    if (logLevel === 'debug' || logLevel === 'info' || logLevel === 'warn') {
      console.error(`💥 All ${maxAttempts} retry attempts exhausted for ${url}: ${lastError.message}`);
    }
  }

  /**
   * Update performance metrics
   * @param {boolean} success - Whether request was successful
   * @param {number} responseTime - Response time in milliseconds
   * @param {Error} error - Error object if request failed
   */
  updateMetrics(success, responseTime, error = null) {
    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
      this.metrics.lastError = error;
    }
    
    // Update average response time
    const totalTime = this.metrics.averageResponseTime * (this.metrics.totalRequests - 1) + responseTime;
    this.metrics.averageResponseTime = Math.round(totalTime / this.metrics.totalRequests);
  }

  /**
   * Get performance metrics
   * @returns {Object} - Current metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.metrics.totalRequests > 0 
        ? Math.round((this.metrics.successfulRequests / this.metrics.totalRequests) * 100) 
        : 0,
      retryRate: this.metrics.totalRequests > 0
        ? Math.round((this.metrics.retriedRequests / this.metrics.totalRequests) * 100)
        : 0,
      averageRetriesPerRequest: this.metrics.retriedRequests > 0
        ? Math.round((this.metrics.totalRetries / this.metrics.retriedRequests) * 100) / 100
        : 0,
      retrySuccessRate: this.metrics.retriedRequests > 0
        ? Math.round((this.metrics.successAfterRetry / this.metrics.retriedRequests) * 100)
        : 0
    };
  }

  /**
   * Reset performance metrics
   */
  resetMetrics() {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      lastError: null,
      // Retry-specific metrics
      totalRetries: 0,
      retriedRequests: 0,
      successAfterRetry: 0
    };
  }

  /**
   * Clean up request tracking and timeout
   * @param {number} requestId - Request ID to clean up
   */
  cleanupRequest(requestId) {
    const requestInfo = this.activeRequests.get(requestId);
    if (requestInfo) {
      clearTimeout(requestInfo.timeoutId);
      this.activeRequests.delete(requestId);
    }
  }

  /**
   * Cancel a specific active request
   * @param {number} requestId - Request ID to cancel
   * @returns {boolean} - Whether the request was successfully cancelled
   */
  cancelRequest(requestId) {
    const requestInfo = this.activeRequests.get(requestId);
    if (requestInfo) {
      requestInfo.controller.abort();
      this.cleanupRequest(requestId);
      this.logRequestCancelled(requestId);
      return true;
    }
    return false;
  }

  /**
   * Cancel all active requests
   * @returns {number} - Number of requests cancelled
   */
  cancelAllRequests() {
    let cancelledCount = 0;
    
    for (const [requestId, requestInfo] of this.activeRequests) {
      requestInfo.controller.abort();
      this.cleanupRequest(requestId);
      cancelledCount++;
    }
    
    if (cancelledCount > 0) {
      this.logAllRequestsCancelled(cancelledCount);
    }
    
    return cancelledCount;
  }

  /**
   * Get information about active requests
   * @returns {Array} - Array of active request information
   */
  getActiveRequests() {
    return Array.from(this.activeRequests.values()).map(request => ({
      id: request.id,
      url: request.url,
      method: request.method,
      duration: Date.now() - request.startTime
    }));
  }

  /**
   * Check if there are any active requests
   * @returns {boolean}
   */
  hasActiveRequests() {
    return this.activeRequests.size > 0;
  }

  /**
   * Create a request with custom timeout
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @param {number} customTimeout - Custom timeout in milliseconds
   * @returns {Promise<ApiResponse>}
   */
  async requestWithTimeout(url, options = {}, customTimeout) {
    return this.request(url, {
      ...options,
      timeout: customTimeout
    });
  }

  /**
   * Create a request that can be manually cancelled
   * @param {string} url - Request URL
   * @param {Object} options - Request options
   * @returns {Object} - Object with promise and cancel function
   */
  createCancellableRequest(url, options = {}) {
    // This will be populated with the request ID after the request starts
    let requestId = null;
    
    const promise = this.request(url, options).catch(error => {
      // If this is a manual cancellation, provide more context
      if (error.name === 'AbortError' && requestId) {
        error.requestId = requestId;
        error.message = `Request ${requestId} was cancelled`;
      }
      throw error;
    });
    
    // Monkey-patch to capture the request ID
    const originalRequest = this.request.bind(this);
    this.request = async (u, o) => {
      const result = await originalRequest(u, o);
      requestId = this.requestCounter; // Get the most recent request ID
      return result;
    };
    
    // Restore original request method after call
    setTimeout(() => {
      this.request = originalRequest;
    }, 0);
    
    return {
      promise,
      cancel: () => {
        if (requestId) {
          return this.cancelRequest(requestId);
        }
        return false;
      },
      getRequestId: () => requestId
    };
  }

  /**
   * Process markdown content from API response into formatted HTML
   * @param {string} markdownText - Raw markdown text from API
   * @returns {Object} - Processed content with HTML and metadata
   */
  processMarkdownResponse(markdownText) {
    if (!markdownText || typeof markdownText !== 'string') {
      return {
        htmlContent: '',
        plainText: '',
        hasMarkdown: false
      };
    }

    // Simple markdown to HTML conversion
    let htmlContent = markdownText
      // Headers
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      
      // Bold text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      
      // Italic text
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      
      // Code blocks
      .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      
      // Line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>')
      
      // Wrap in paragraphs
      .replace(/^(.+)/, '<p>$1')
      .replace(/(.+)$/, '$1</p>')
      
      // Fix double paragraph tags
      .replace(/<\/p><p><\/p><p>/g, '</p><p>');

    // Process numbered lists
    htmlContent = this.processNumberedLists(htmlContent);
    
    // Process bullet lists
    htmlContent = this.processBulletLists(htmlContent);
    
    // Extract and format footer
    const footerMatch = htmlContent.match(/---\s*<br>\s*\*(.+?)\*/);
    let footer = '';
    if (footerMatch) {
      footer = `<div class="response-footer">${footerMatch[1]}</div>`;
      htmlContent = htmlContent.replace(/---\s*<br>\s*\*(.+?)\*/, '');
    }

    // Clean up extra tags
    htmlContent = htmlContent
      .replace(/<p><\/p>/g, '')
      .replace(/<p><br>/g, '<p>')
      .replace(/<br><\/p>/g, '</p>');

    // Add footer if found
    if (footer) {
      htmlContent += footer;
    }

    // Wrap in container
    htmlContent = `<div class="response-content">${htmlContent}</div>`;

    // Check if original text had markdown syntax
    const hasMarkdown = /(\*\*.*?\*\*|\*.*?\*|^#+\s|^\d+\.\s|^-\s)/m.test(markdownText);

    return {
      htmlContent,
      plainText: markdownText.replace(/[*#`-]/g, '').replace(/\n+/g, ' ').trim(),
      hasMarkdown,
      originalMarkdown: markdownText
    };
  }

  /**
   * Process numbered lists in HTML content
   * @param {string} html - HTML content
   * @returns {string} - HTML with proper ordered lists
   */
  processNumberedLists(html) {
    // Match numbered list items (1. item, 2. item, etc.)
    const listRegex = /<p>(\d+)\.\s*<strong>(.*?)<\/strong>:(.*?)<\/p>/g;
    const items = [];
    let match;
    
    // Collect all numbered list items
    while ((match = listRegex.exec(html)) !== null) {
      items.push({
        number: parseInt(match[1]),
        title: match[2],
        content: match[3].trim(),
        fullMatch: match[0]
      });
    }
    
    if (items.length === 0) return html;
    
    // Sort by number to ensure proper order
    items.sort((a, b) => a.number - b.number);
    
    // Build ordered list
    const listItems = items.map(item => 
      `<li><strong>${item.title}</strong>:${item.content}</li>`
    ).join('');
    
    const orderedList = `<ol>${listItems}</ol>`;
    
    // Replace all numbered items with the ordered list
    let result = html;
    items.forEach(item => {
      result = result.replace(item.fullMatch, '');
    });
    
    // Insert the ordered list where the first item was
    if (items.length > 0) {
      const firstItemIndex = html.indexOf(items[0].fullMatch);
      result = html.substring(0, firstItemIndex) + orderedList + html.substring(firstItemIndex);
      
      // Remove all the original items
      items.forEach(item => {
        result = result.replace(item.fullMatch, '');
      });
    }
    
    return result;
  }

  /**
   * Process bullet lists in HTML content
   * @param {string} html - HTML content
   * @returns {string} - HTML with proper unordered lists
   */
  processBulletLists(html) {
    // Match bullet list items (- item, * item)
    const listRegex = /<p>[-*]\s+(.*?)<\/p>/g;
    const items = [];
    let match;
    
    while ((match = listRegex.exec(html)) !== null) {
      items.push({
        content: match[1],
        fullMatch: match[0]
      });
    }
    
    if (items.length === 0) return html;
    
    // Build unordered list
    const listItems = items.map(item => `<li>${item.content}</li>`).join('');
    const unorderedList = `<ul>${listItems}</ul>`;
    
    // Replace all bullet items with the unordered list
    let result = html;
    items.forEach(item => {
      result = result.replace(item.fullMatch, '');
    });
    
    // Insert the unordered list where the first item was
    if (items.length > 0) {
      const firstItemIndex = html.indexOf(items[0].fullMatch);
      result = html.substring(0, firstItemIndex) + unorderedList + html.substring(firstItemIndex);
      
      // Remove all the original items
      items.forEach(item => {
        result = result.replace(item.fullMatch, '');
      });
    }
    
    return result;
  }
}

/**
 * Custom API Error class for standardized error handling
 */
class ApiError extends Error {
  constructor(message, status = 0, response = null, responseTime = 0, type = 'API_ERROR') {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.response = response;
    this.responseTime = responseTime;
    this.type = type;
    this.timestamp = Date.now();
    this.requestId = null; // Will be set by the API client
    this.retryable = null; // Cache for retryable check
    this.severity = this.calculateSeverity();
  }

  /**
   * Check if error is a timeout
   * @returns {boolean}
   */
  isTimeout() {
    return this.type === 'TIMEOUT';
  }

  /**
   * Check if error is a network error
   * @returns {boolean}
   */
  isNetworkError() {
    return this.type === 'NETWORK';
  }

  /**
   * Check if error is a server error (5xx)
   * @returns {boolean}
   */
  isServerError() {
    return this.status >= 500 && this.status < 600;
  }

  /**
   * Check if error is a client error (4xx)
   * @returns {boolean}
   */
  isClientError() {
    return this.status >= 400 && this.status < 500;
  }

  /**
   * Check if error is retryable
   * @returns {boolean}
   */
  isRetryable() {
    return this.isTimeout() || 
           this.isNetworkError() || 
           this.isServerError() ||
           this.status === 429; // Too Many Requests
  }

  /**
   * Get user-friendly error message
   * @returns {string}
   */
  getUserMessage() {
    if (this.isTimeout()) {
      return 'Request timed out. Please try again.';
    }
    
    if (this.isNetworkError()) {
      return 'Network error. Please check your connection.';
    }
    
    if (this.status === 429) {
      return 'Too many requests. Please wait a moment and try again.';
    }
    
    if (this.isServerError()) {
      return 'Server error. Please try again later.';
    }
    
    if (this.isClientError()) {
      return 'Invalid request. Please check your input.';
    }
    
    return this.message || 'An unexpected error occurred.';
  }

  /**
   * Calculate error severity level
   * @returns {string} - 'low', 'medium', 'high', 'critical'
   */
  calculateSeverity() {
    // Critical errors - system failures
    if (this.isNetworkError() || this.status === 0) {
      return 'critical';
    }
    
    // High severity - server errors that might indicate backend issues
    if (this.status >= 500 && this.status < 600) {
      return 'high';
    }
    
    // Medium severity - timeouts and rate limiting
    if (this.isTimeout() || this.status === 429) {
      return 'medium';
    }
    
    // Low severity - client errors (user fixable)
    if (this.status >= 400 && this.status < 500) {
      return 'low';
    }
    
    return 'medium'; // Default
  }

  /**
   * Get detailed error information for debugging
   * @returns {Object} - Comprehensive error details
   */
  getDebugInfo() {
    return {
      message: this.message,
      status: this.status,
      type: this.type,
      severity: this.severity,
      responseTime: this.responseTime,
      timestamp: this.timestamp,
      requestId: this.requestId,
      isRetryable: this.isRetryable(),
      userMessage: this.getUserMessage(),
      stack: this.stack,
      response: this.response ? {
        status: this.response.status,
        statusText: this.response.statusText,
        headers: this.response.headers,
        url: this.response.url
      } : null
    };
  }

  /**
   * Get recovery suggestions based on error type
   * @returns {Array<string>} - Array of suggested recovery actions
   */
  getRecoverySuggestions() {
    const suggestions = [];
    
    if (this.isNetworkError()) {
      suggestions.push('Check your internet connection');
      suggestions.push('Verify the API endpoint is accessible');
      suggestions.push('Try again in a few moments');
      suggestions.push('Check if you\'re behind a firewall or proxy');
    } else if (this.isTimeout()) {
      suggestions.push('The request took too long to complete');
      suggestions.push('Try again with a longer timeout');
      suggestions.push('Check if the server is experiencing high load');
    } else if (this.status === 429) {
      suggestions.push('You\'re making requests too quickly');
      suggestions.push('Wait a moment before trying again');
      suggestions.push('Consider implementing request throttling');
    } else if (this.status === 401) {
      suggestions.push('Check your authentication credentials');
      suggestions.push('Verify your API key is valid');
    } else if (this.status === 403) {
      suggestions.push('You don\'t have permission for this operation');
      suggestions.push('Check your account permissions');
    } else if (this.status === 404) {
      suggestions.push('The requested resource was not found');
      suggestions.push('Verify the API endpoint URL');
    } else if (this.isServerError()) {
      suggestions.push('The server encountered an error');
      suggestions.push('Try again later');
      suggestions.push('Contact support if the problem persists');
    }
    
    return suggestions;
  }

  /**
   * Check if this error should trigger a circuit breaker
   * @returns {boolean}
   */
  shouldTriggerCircuitBreaker() {
    return this.severity === 'critical' || 
           this.severity === 'high' ||
           (this.isTimeout() && this.responseTime > 30000);
  }

  /**
   * Get estimated recovery time in milliseconds
   * @returns {number}
   */
  getEstimatedRecoveryTime() {
    if (this.isNetworkError()) {
      return 5000; // 5 seconds for network issues
    }
    
    if (this.status === 429) {
      return 60000; // 1 minute for rate limiting
    }
    
    if (this.isTimeout()) {
      return 10000; // 10 seconds for timeouts
    }
    
    if (this.isServerError()) {
      return 30000; // 30 seconds for server errors
    }
    
    return 1000; // 1 second default
  }
}

/**
 * Error Handler for comprehensive error management
 */
class ErrorHandler {
  constructor(configManager) {
    this.config = configManager;
    this.errorHistory = [];
    this.maxHistorySize = 100;
    this.errorCallbacks = new Map();
  }

  /**
   * Handle an error with comprehensive logging and recovery suggestions
   * @param {ApiError} error - The error to handle
   * @param {Object} context - Additional context
   * @returns {Object} - Handled error with suggestions
   */
  handleError(error, context = {}) {
    // Add to error history
    this.addToHistory(error, context);
    
    // Log error based on severity
    this.logError(error, context);
    
    // Trigger callbacks
    this.triggerCallbacks(error, context);
    
    // Return enhanced error information
    return {
      error,
      context,
      suggestions: error.getRecoverySuggestions(),
      debugInfo: error.getDebugInfo(),
      shouldRetry: error.isRetryable(),
      estimatedRecoveryTime: error.getEstimatedRecoveryTime(),
      similarErrors: this.findSimilarErrors(error),
      timestamp: Date.now()
    };
  }

  /**
   * Add error to history for pattern analysis
   * @param {ApiError} error - The error
   * @param {Object} context - Error context
   */
  addToHistory(error, context) {
    const historyEntry = {
      error: {
        message: error.message,
        status: error.status,
        type: error.type,
        severity: error.severity,
        timestamp: error.timestamp
      },
      context,
      timestamp: Date.now()
    };
    
    this.errorHistory.unshift(historyEntry);
    
    // Limit history size
    if (this.errorHistory.length > this.maxHistorySize) {
      this.errorHistory = this.errorHistory.slice(0, this.maxHistorySize);
    }
  }

  /**
   * Log error based on severity level
   * @param {ApiError} error - The error
   * @param {Object} context - Error context
   */
  logError(error, context) {
    const logLevel = this.config.get('logLevel');
    const logMessage = `API Error [${error.severity.toUpperCase()}]: ${error.message}`;
    
    if (error.severity === 'critical') {
      console.error(logMessage, error.getDebugInfo(), context);
    } else if (error.severity === 'high') {
      console.error(logMessage, error.getDebugInfo());
    } else if (error.severity === 'medium') {
      if (logLevel === 'debug' || logLevel === 'info') {
        console.warn(logMessage, error.getDebugInfo());
      }
    } else {
      if (logLevel === 'debug') {
        console.info(logMessage, error.getDebugInfo());
      }
    }
  }

  /**
   * Find similar errors in history for pattern detection
   * @param {ApiError} error - Current error
   * @returns {Array} - Similar errors from history
   */
  findSimilarErrors(error) {
    const recentHistory = this.errorHistory.slice(0, 20); // Last 20 errors
    
    return recentHistory.filter(entry => {
      const historyError = entry.error;
      return historyError.status === error.status ||
             historyError.type === error.type ||
             historyError.severity === error.severity;
    });
  }

  /**
   * Register error callback
   * @param {string} errorType - Error type to listen for
   * @param {Function} callback - Callback function
   */
  onError(errorType, callback) {
    if (!this.errorCallbacks.has(errorType)) {
      this.errorCallbacks.set(errorType, []);
    }
    this.errorCallbacks.get(errorType).push(callback);
  }

  /**
   * Trigger registered callbacks for error type
   * @param {ApiError} error - The error
   * @param {Object} context - Error context
   */
  triggerCallbacks(error, context) {
    const callbacks = this.errorCallbacks.get(error.type) || [];
    const allCallbacks = this.errorCallbacks.get('*') || [];
    
    [...callbacks, ...allCallbacks].forEach(callback => {
      try {
        callback(error, context);
      } catch (callbackError) {
        console.warn('Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Get error statistics
   * @returns {Object} - Error statistics
   */
  getErrorStats() {
    const stats = {
      total: this.errorHistory.length,
      bySeverity: { critical: 0, high: 0, medium: 0, low: 0 },
      byType: {},
      byStatus: {},
      recentErrors: this.errorHistory.slice(0, 5)
    };
    
    this.errorHistory.forEach(entry => {
      const error = entry.error;
      
      // Count by severity
      stats.bySeverity[error.severity]++;
      
      // Count by type
      stats.byType[error.type] = (stats.byType[error.type] || 0) + 1;
      
      // Count by status
      if (error.status) {
        stats.byStatus[error.status] = (stats.byStatus[error.status] || 0) + 1;
      }
    });
    
    return stats;
  }

  /**
   * Clear error history
   */
  clearHistory() {
    this.errorHistory = [];
  }
}

/**
 * Circuit Breaker for preventing cascading failures
 */
class CircuitBreaker {
  constructor(configManager) {
    this.config = configManager;
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.successCount = 0;
    
    // Configuration
    this.failureThreshold = 5; // Number of failures to open circuit
    this.recoveryTimeout = 30000; // 30 seconds
    this.successThreshold = 3; // Successes needed to close circuit from half-open
  }

  /**
   * Check if request should be allowed through circuit breaker
   * @returns {boolean} - Whether request is allowed
   */
  allowRequest() {
    if (this.state === 'CLOSED') {
      return true;
    }
    
    if (this.state === 'OPEN') {
      // Check if recovery timeout has passed
      if (Date.now() - this.lastFailureTime >= this.recoveryTimeout) {
        this.state = 'HALF_OPEN';
        this.successCount = 0;
        return true;
      }
      return false;
    }
    
    if (this.state === 'HALF_OPEN') {
      return true;
    }
    
    return false;
  }

  /**
   * Record a successful request
   */
  recordSuccess() {
    if (this.state === 'HALF_OPEN') {
      this.successCount++;
      if (this.successCount >= this.successThreshold) {
        this.state = 'CLOSED';
        this.failureCount = 0;
      }
    } else if (this.state === 'CLOSED') {
      this.failureCount = 0;
    }
  }

  /**
   * Record a failed request
   * @param {ApiError} error - The error that occurred
   */
  recordFailure(error) {
    this.lastFailureTime = Date.now();
    
    if (this.state === 'HALF_OPEN') {
      this.state = 'OPEN';
      this.successCount = 0;
    } else if (this.state === 'CLOSED') {
      this.failureCount++;
      if (this.failureCount >= this.failureThreshold) {
        this.state = 'OPEN';
      }
    }
  }

  /**
   * Get current circuit breaker status
   * @returns {Object} - Circuit breaker status
   */
  getStatus() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
      isOpen: this.state === 'OPEN',
      timeUntilRetry: this.state === 'OPEN' 
        ? Math.max(0, this.recoveryTimeout - (Date.now() - this.lastFailureTime))
        : 0
    };
  }

  /**
   * Manually reset circuit breaker
   */
  reset() {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = null;
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ApiClient, ApiError, ErrorHandler, CircuitBreaker };
} else {
  window.ApiClient = ApiClient;
  window.ApiError = ApiError;
  window.ErrorHandler = ErrorHandler;
  window.CircuitBreaker = CircuitBreaker;
} 