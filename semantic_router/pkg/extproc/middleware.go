package extproc

import (
	"time"
)

// MiddlewareChain manages a chain of middleware
type MiddlewareChain struct {
	middlewares []Middleware
	processor   RequestProcessor
	logger      Logger
}

// NewMiddlewareChain creates a new middleware chain
func NewMiddlewareChain(processor RequestProcessor, logger Logger) *MiddlewareChain {
	return &MiddlewareChain{
		processor: processor,
		logger:    logger,
	}
}

// Use adds a middleware to the chain
func (c *MiddlewareChain) Use(middleware Middleware) {
	c.middlewares = append(c.middlewares, middleware)
}

// Build builds the middleware chain and returns the final processor
func (c *MiddlewareChain) Build() RequestProcessor {
	if len(c.middlewares) == 0 {
		return c.processor
	}

	// Build the chain in reverse order
	final := c.processor
	for i := len(c.middlewares) - 1; i >= 0; i-- {
		final = c.middlewares[i](final)
	}
	return final
}

// Built-in middleware implementations

// LoggingMiddleware logs request processing
func LoggingMiddleware(logger Logger) Middleware {
	return func(next RequestProcessor) RequestProcessor {
		return &loggingProcessor{next: next, logger: logger}
	}
}

type loggingProcessor struct {
	next   RequestProcessor
	logger Logger
}

func (p *loggingProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	start := time.Now()
	p.logger.Debug("Processing headers", 
		Field{Key: "request_id", Value: req.RequestID},
		Field{Key: "stage", Value: "headers"})

	response, err := p.next.ProcessHeaders(req, ctx)
	
	duration := time.Since(start)
	if err != nil {
		p.logger.Error("Headers processing failed", err,
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	} else {
		p.logger.Debug("Headers processing completed",
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	}

	return response, err
}

func (p *loggingProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	start := time.Now()
	p.logger.Debug("Processing request body",
		Field{Key: "request_id", Value: req.RequestID},
		Field{Key: "body_size", Value: len(req.Body)})

	response, err := p.next.ProcessRequestBody(req, ctx)
	
	duration := time.Since(start)
	if err != nil {
		p.logger.Error("Request body processing failed", err,
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	} else {
		p.logger.Info("Request body processing completed",
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	}

	return response, err
}

func (p *loggingProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	start := time.Now()
	p.logger.Debug("Processing response body",
		Field{Key: "request_id", Value: req.RequestID},
		Field{Key: "body_size", Value: len(req.Body)})

	response, err := p.next.ProcessResponseBody(req, ctx)
	
	duration := time.Since(start)
	if err != nil {
		p.logger.Error("Response body processing failed", err,
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	} else {
		p.logger.Debug("Response body processing completed",
			Field{Key: "request_id", Value: req.RequestID},
			Field{Key: "duration_ms", Value: duration.Milliseconds()})
	}

	return response, err
}

// MetricsMiddleware collects metrics
func MetricsMiddleware(metrics MetricsCollector) Middleware {
	return func(next RequestProcessor) RequestProcessor {
		return &metricsProcessor{next: next, metrics: metrics}
	}
}

type metricsProcessor struct {
	next    RequestProcessor
	metrics MetricsCollector
}

func (p *metricsProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	response, err := p.next.ProcessHeaders(req, ctx)

	if err != nil {
		p.metrics.RecordError("headers_processor", "processing_error")
	}
	
	// Could add more specific metrics here
	return response, err
}

func (p *metricsProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	start := time.Now()
	response, err := p.next.ProcessRequestBody(req, ctx)
	duration := time.Since(start)

	if err != nil {
		p.metrics.RecordError("request_processor", "processing_error")
	} else {
		// Record processing latency
		p.metrics.RecordRequest(ctx.RequestModel, duration)
	}

	return response, err
}

func (p *metricsProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	response, err := p.next.ProcessResponseBody(req, ctx)

	if err != nil {
		p.metrics.RecordError("response_processor", "processing_error")
	}

	return response, err
}

// ValidationMiddleware validates requests
func ValidationMiddleware(logger Logger) Middleware {
	return func(next RequestProcessor) RequestProcessor {
		return &validationProcessor{next: next, logger: logger}
	}
}

type validationProcessor struct {
	next   RequestProcessor
	logger Logger
}

func (p *validationProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	// Validate headers request
	if req == nil {
		return nil, &ProcessingError{
			Type:    ErrorTypeValidation,
			Code:    "INVALID_HEADERS_REQUEST",
			Message: "headers request cannot be nil",
		}
	}
	
	if req.RequestID == "" {
		p.logger.Warn("Missing request ID in headers")
	}

	return p.next.ProcessHeaders(req, ctx)
}

func (p *validationProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	// Validate body request
	if req == nil {
		return nil, &ProcessingError{
			Type:    ErrorTypeValidation,
			Code:    "INVALID_BODY_REQUEST", 
			Message: "body request cannot be nil",
		}
	}

	if len(req.Body) == 0 {
		return nil, &ProcessingError{
			Type:    ErrorTypeValidation,
			Code:    "EMPTY_BODY",
			Message: "request body cannot be empty",
		}
	}

	if ctx == nil {
		return nil, &ProcessingError{
			Type:    ErrorTypeValidation,
			Code:    "INVALID_CONTEXT",
			Message: "request context cannot be nil",
		}
	}

	return p.next.ProcessRequestBody(req, ctx)
}

func (p *validationProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	// Validate response body request
	if req == nil {
		return nil, &ProcessingError{
			Type:    ErrorTypeValidation,
			Code:    "INVALID_RESPONSE_REQUEST",
			Message: "response request cannot be nil",
		}
	}

	return p.next.ProcessResponseBody(req, ctx)
}

// SecurityMiddleware runs security checks through plugins
func SecurityMiddleware(pluginManager PluginManager, logger Logger) Middleware {
	return func(next RequestProcessor) RequestProcessor {
		return &securityProcessor{
			next:          next,
			pluginManager: pluginManager,
			logger:        logger,
		}
	}
}

type securityProcessor struct {
	next          RequestProcessor
	pluginManager PluginManager
	logger        Logger
}

func (p *securityProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	// Headers don't typically need security checks, but plugins might want to inspect them
	return p.next.ProcessHeaders(req, ctx)
}

func (p *securityProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	// Run security checks through all security plugins
	securityPlugins := p.pluginManager.GetPluginsByType("security")
	
	for _, plugin := range securityPlugins {
		securityPlugin := plugin.(SecurityPlugin)
		
		// Extract content for security check (simplified for now)
		content := string(req.Body)
		
		result, err := securityPlugin.CheckContent(content, ctx)
		if err != nil {
			p.logger.Error("Security check failed", err,
				Field{Key: "plugin", Value: securityPlugin.Name()},
				Field{Key: "request_id", Value: req.RequestID})
			continue
		}

		if !result.Allowed {
			p.logger.Warn("Security check blocked request",
				Field{Key: "plugin", Value: securityPlugin.Name()},
				Field{Key: "reason", Value: result.BlockReason},
				Field{Key: "confidence", Value: result.Confidence})
			
			if result.Response != nil {
				return &ProcessingResponse{
					Status:            StatusImmediate,
					ImmediateResponse: result.Response,
				}, nil
			}
			
			// Return a generic block response
			return &ProcessingResponse{
				Status: StatusBlock,
			}, nil
		}
	}

	return p.next.ProcessRequestBody(req, ctx)
}

func (p *securityProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	return p.next.ProcessResponseBody(req, ctx)
}

// RecoveryMiddleware handles panics and errors gracefully
func RecoveryMiddleware(logger Logger) Middleware {
	return func(next RequestProcessor) RequestProcessor {
		return &recoveryProcessor{next: next, logger: logger}
	}
}

type recoveryProcessor struct {
	next   RequestProcessor
	logger Logger
}

func (p *recoveryProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (response *ProcessingResponse, err error) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.Error("Panic in headers processing", nil,
				Field{Key: "panic", Value: r},
				Field{Key: "request_id", Value: req.RequestID})
			
			response = &ProcessingResponse{Status: StatusContinue}
			err = &ProcessingError{
				Type:    ErrorTypeInternal,
				Code:    "PANIC_RECOVERY",
				Message: "internal panic recovered",
			}
		}
	}()

	return p.next.ProcessHeaders(req, ctx)
}

func (p *recoveryProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (response *ProcessingResponse, err error) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.Error("Panic in request body processing", nil,
				Field{Key: "panic", Value: r},
				Field{Key: "request_id", Value: req.RequestID})
			
			response = &ProcessingResponse{Status: StatusContinue}
			err = &ProcessingError{
				Type:    ErrorTypeInternal,
				Code:    "PANIC_RECOVERY",
				Message: "internal panic recovered",
			}
		}
	}()

	return p.next.ProcessRequestBody(req, ctx)
}

func (p *recoveryProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (response *ProcessingResponse, err error) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.Error("Panic in response body processing", nil,
				Field{Key: "panic", Value: r},
				Field{Key: "request_id", Value: req.RequestID})
			
			response = &ProcessingResponse{Status: StatusContinue}
			err = &ProcessingError{
				Type:    ErrorTypeInternal,
				Code:    "PANIC_RECOVERY",
				Message: "internal panic recovered",
			}
		}
	}()

	return p.next.ProcessResponseBody(req, ctx)
}