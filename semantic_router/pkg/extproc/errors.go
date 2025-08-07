package extproc

import (
	"fmt"
	"time"
)

// ErrorType represents the type of processing error
type ErrorType string

const (
	ErrorTypeValidation     ErrorType = "validation"
	ErrorTypeAuthentication ErrorType = "authentication"
	ErrorTypeAuthorization  ErrorType = "authorization"
	ErrorTypeRateLimit      ErrorType = "rate_limit"
	ErrorTypePII            ErrorType = "pii_violation"
	ErrorTypeJailbreak      ErrorType = "jailbreak"
	ErrorTypeInternal       ErrorType = "internal"
	ErrorTypePluginFailure  ErrorType = "plugin_failure"
	ErrorTypeTimeout        ErrorType = "timeout"
	ErrorTypeCache          ErrorType = "cache"
	ErrorTypeModelSelection ErrorType = "model_selection"
	ErrorTypeConfiguration  ErrorType = "configuration"
)

// ProcessingError represents a structured error in request processing
type ProcessingError struct {
	Type      ErrorType              `json:"type"`
	Code      string                 `json:"code"`
	Message   string                 `json:"message"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Cause     error                  `json:"-"`
	Timestamp time.Time              `json:"timestamp"`
	RequestID string                 `json:"request_id,omitempty"`
}

// Error implements the error interface
func (e *ProcessingError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s [%s]: %s (caused by: %v)", e.Type, e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s [%s]: %s", e.Type, e.Code, e.Message)
}

// Unwrap implements error unwrapping
func (e *ProcessingError) Unwrap() error {
	return e.Cause
}

// NewProcessingError creates a new processing error
func NewProcessingError(errorType ErrorType, code, message string) *ProcessingError {
	return &ProcessingError{
		Type:      errorType,
		Code:      code,
		Message:   message,
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}
}

// NewProcessingErrorWithCause creates a new processing error with a cause
func NewProcessingErrorWithCause(errorType ErrorType, code, message string, cause error) *ProcessingError {
	return &ProcessingError{
		Type:      errorType,
		Code:      code,
		Message:   message,
		Cause:     cause,
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}
}

// WithDetail adds a detail to the error
func (e *ProcessingError) WithDetail(key string, value interface{}) *ProcessingError {
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details[key] = value
	return e
}

// WithRequestID adds a request ID to the error
func (e *ProcessingError) WithRequestID(requestID string) *ProcessingError {
	e.RequestID = requestID
	return e
}

// IsRetryable returns whether this error type is retryable
func (e *ProcessingError) IsRetryable() bool {
	switch e.Type {
	case ErrorTypeTimeout, ErrorTypeInternal, ErrorTypeCache, ErrorTypePluginFailure:
		return true
	case ErrorTypeValidation, ErrorTypeAuthentication, ErrorTypeAuthorization, 
		 ErrorTypePII, ErrorTypeJailbreak, ErrorTypeConfiguration:
		return false
	default:
		return false
	}
}

// Common error constructors for convenience

// ValidationError creates a validation error
func ValidationError(code, message string) *ProcessingError {
	return NewProcessingError(ErrorTypeValidation, code, message)
}

// PIIViolationError creates a PII violation error
func PIIViolationError(model string, piiTypes []string) *ProcessingError {
	return NewProcessingError(ErrorTypePII, "PII_DETECTED", 
		fmt.Sprintf("PII detected in content for model %s", model)).
		WithDetail("model", model).
		WithDetail("pii_types", piiTypes)
}

// JailbreakError creates a jailbreak detection error
func JailbreakError(jailbreakType string, confidence float32) *ProcessingError {
	return NewProcessingError(ErrorTypeJailbreak, "JAILBREAK_DETECTED",
		fmt.Sprintf("Jailbreak attempt detected: %s", jailbreakType)).
		WithDetail("jailbreak_type", jailbreakType).
		WithDetail("confidence", confidence)
}

// InternalError creates an internal error
func InternalError(message string, cause error) *ProcessingError {
	return NewProcessingErrorWithCause(ErrorTypeInternal, "INTERNAL_ERROR", message, cause)
}

// PluginError creates a plugin failure error
func PluginError(pluginName, message string, cause error) *ProcessingError {
	return NewProcessingErrorWithCause(ErrorTypePluginFailure, "PLUGIN_FAILURE", message, cause).
		WithDetail("plugin", pluginName)
}

// TimeoutError creates a timeout error
func TimeoutError(operation string, timeout time.Duration) *ProcessingError {
	return NewProcessingError(ErrorTypeTimeout, "OPERATION_TIMEOUT",
		fmt.Sprintf("Operation %s timed out after %s", operation, timeout)).
		WithDetail("operation", operation).
		WithDetail("timeout", timeout.String())
}

// ModelSelectionError creates a model selection error
func ModelSelectionError(message string, cause error) *ProcessingError {
	return NewProcessingErrorWithCause(ErrorTypeModelSelection, "MODEL_SELECTION_FAILED", message, cause)
}

// ErrorHandler interface for handling different types of errors
type ErrorHandler interface {
	Handle(err error, ctx *RequestContext) (*ProcessingResponse, error)
	CanHandle(err error) bool
}

// DefaultErrorHandler provides default error handling
type DefaultErrorHandler struct {
	logger Logger
}

// NewDefaultErrorHandler creates a new default error handler
func NewDefaultErrorHandler(logger Logger) *DefaultErrorHandler {
	return &DefaultErrorHandler{logger: logger}
}

// Handle handles the error and returns an appropriate response
func (h *DefaultErrorHandler) Handle(err error, ctx *RequestContext) (*ProcessingResponse, error) {
	h.logger.Error("Processing error occurred", err, 
		Field{Key: "request_id", Value: ctx.RequestID})

	// For now, just continue processing on most errors
	return &ProcessingResponse{
		Status: StatusContinue,
	}, nil
}

// CanHandle returns whether this handler can handle the error
func (h *DefaultErrorHandler) CanHandle(err error) bool {
	// Default handler can handle any error
	return true
}

// CircuitBreakerState represents the state of a circuit breaker
type CircuitBreakerState int

const (
	CircuitStateClosed CircuitBreakerState = iota
	CircuitStateOpen
	CircuitStateHalfOpen
)

// CircuitBreaker interface for circuit breaker pattern
type CircuitBreaker interface {
	Execute(fn func() error) error
	State() CircuitBreakerState
	Reset()
}

// SimpleCircuitBreaker is a basic circuit breaker implementation
type SimpleCircuitBreaker struct {
	failureThreshold int
	resetTimeout     time.Duration
	failureCount     int
	lastFailureTime  time.Time
	state           CircuitBreakerState
}

// NewSimpleCircuitBreaker creates a new simple circuit breaker
func NewSimpleCircuitBreaker(failureThreshold int, resetTimeout time.Duration) *SimpleCircuitBreaker {
	return &SimpleCircuitBreaker{
		failureThreshold: failureThreshold,
		resetTimeout:     resetTimeout,
		state:           CircuitStateClosed,
	}
}

// Execute executes a function with circuit breaker protection
func (cb *SimpleCircuitBreaker) Execute(fn func() error) error {
	if cb.state == CircuitStateOpen {
		if time.Since(cb.lastFailureTime) > cb.resetTimeout {
			cb.state = CircuitStateHalfOpen
		} else {
			return NewProcessingError(ErrorTypeInternal, "CIRCUIT_BREAKER_OPEN", 
				"circuit breaker is open")
		}
	}

	err := fn()
	if err != nil {
		cb.onFailure()
		return err
	}

	cb.onSuccess()
	return nil
}

// State returns the current circuit breaker state
func (cb *SimpleCircuitBreaker) State() CircuitBreakerState {
	return cb.state
}

// Reset resets the circuit breaker to closed state
func (cb *SimpleCircuitBreaker) Reset() {
	cb.state = CircuitStateClosed
	cb.failureCount = 0
}

func (cb *SimpleCircuitBreaker) onSuccess() {
	cb.failureCount = 0
	cb.state = CircuitStateClosed
}

func (cb *SimpleCircuitBreaker) onFailure() {
	cb.failureCount++
	cb.lastFailureTime = time.Now()
	
	if cb.failureCount >= cb.failureThreshold {
		cb.state = CircuitStateOpen
	}
}

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	MaxAttempts     int
	BackoffFunc     func(attempt int) time.Duration
	RetryableErrors []ErrorType
}

// DefaultRetryPolicy creates a default retry policy
func DefaultRetryPolicy() *RetryPolicy {
	return &RetryPolicy{
		MaxAttempts: 3,
		BackoffFunc: func(attempt int) time.Duration {
			return time.Duration(attempt) * 100 * time.Millisecond
		},
		RetryableErrors: []ErrorType{
			ErrorTypeInternal,
			ErrorTypeTimeout,
			ErrorTypePluginFailure,
		},
	}
}

// ShouldRetry returns whether an error should be retried
func (rp *RetryPolicy) ShouldRetry(err error, attempt int) bool {
	if attempt >= rp.MaxAttempts {
		return false
	}

	processingErr, ok := err.(*ProcessingError)
	if !ok {
		// Retry unknown errors
		return true
	}

	for _, retryableType := range rp.RetryableErrors {
		if processingErr.Type == retryableType {
			return true
		}
	}

	return false
}

// Wait waits for the backoff duration
func (rp *RetryPolicy) Wait(attempt int) {
	if rp.BackoffFunc != nil {
		time.Sleep(rp.BackoffFunc(attempt))
	}
}