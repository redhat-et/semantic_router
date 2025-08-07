package extproc

import (
	"fmt"
	"strings"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	candle_binding "github.com/redhat-et/semantic_route/candle-binding"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/cache"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/tools"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/classification"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/openai"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/pii"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/utils/ttft"
)

// OpenAIRouter is the main plugin-based Envoy ExtProc server
type OpenAIRouter struct {
	config        *config.RouterConfig
	logger        Logger
	pluginManager PluginManager
	processor     RequestProcessor
	errorHandler  ErrorHandler
	metrics       MetricsCollector
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = &OpenAIRouter{}

// RouterDependencies holds all dependencies for the router
type RouterDependencies struct {
	Config        *config.RouterConfig
	Logger        Logger
	PluginManager PluginManager
	ErrorHandler  ErrorHandler
	Metrics       MetricsCollector
}

var (
	initialized bool
	initMutex   sync.Mutex
)

// NewOpenAIRouter creates a new plugin-based OpenAI router
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	// Load configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Initialize models (BERT, classifiers) if not already done
	initMutex.Lock()
	if !initialized {
		if err := initializeModels(cfg); err != nil {
			initMutex.Unlock()
			return nil, fmt.Errorf("failed to initialize models: %w", err)
		}
		initialized = true
	}
	initMutex.Unlock()

	// Create logger
	logger := NewStructuredLogger("extproc", LogLevelInfo, false)

	// Create plugin manager
	pluginManager := NewPluginManager(logger)

	// Register built-in plugins
	if err := registerBuiltinPlugins(pluginManager, cfg, logger); err != nil {
		return nil, fmt.Errorf("failed to register plugins: %w", err)
	}

	// Create error handler
	errorHandler := NewDefaultErrorHandler(logger)

	// Create metrics collector
	metrics := NewDefaultMetricsCollector()

	// Create dependencies
	deps := RouterDependencies{
		Config:        cfg,
		Logger:        logger,
		PluginManager: pluginManager,
		ErrorHandler:  errorHandler,
		Metrics:       metrics,
	}

	return NewOpenAIRouterWithDeps(deps)
}

// NewOpenAIRouterWithDeps creates a router with provided dependencies
func NewOpenAIRouterWithDeps(deps RouterDependencies) (*OpenAIRouter, error) {
	if err := validateDependencies(deps); err != nil {
		return nil, fmt.Errorf("invalid dependencies: %w", err)
	}

	router := &OpenAIRouter{
		config:        deps.Config,
		logger:        deps.Logger,
		pluginManager: deps.PluginManager,
		errorHandler:  deps.ErrorHandler,
		metrics:       deps.Metrics,
	}

	// Create the processing pipeline with middleware
	processor := NewCoreProcessor(router)
	chain := NewMiddlewareChain(processor, deps.Logger)
	
	// Add middleware in order of execution
	chain.Use(RecoveryMiddleware(deps.Logger))
	chain.Use(LoggingMiddleware(deps.Logger))
	chain.Use(MetricsMiddleware(deps.Metrics))
	chain.Use(ValidationMiddleware(deps.Logger))
	chain.Use(SecurityMiddleware(deps.PluginManager, deps.Logger))
	
	router.processor = chain.Build()

	deps.Logger.Info("Plugin-based router created successfully",
		Field{Key: "plugin_count", Value: len(deps.PluginManager.ListPlugins())})

	return router, nil
}

// Process implements the ext_proc calls using the plugin architecture
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	r.logger.Info("Started processing new request")
	
	// Initialize request context
	ctx := &RequestContext{
		Headers: make(map[string]string),
		Context: stream.Context(),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			r.logger.Error("Error receiving request", err, Field{Key: LogFieldRequestID, Value: ctx.RequestID})
			return err
		}

		r.logger.Debug("Processing message", Field{Key: "type", Value: fmt.Sprintf("%T", req.Request)})

		var response *ProcessingResponse
		var processingErr error

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			ctx.Stage = StageHeaders
			headerReq := &HeadersRequest{
				Headers:   extractHeaders(v.RequestHeaders.Headers),
				RequestID: extractRequestID(v.RequestHeaders.Headers),
			}
			ctx.RequestID = headerReq.RequestID
			response, processingErr = r.processor.ProcessHeaders(headerReq, ctx)

		case *ext_proc.ProcessingRequest_RequestBody:
			ctx.Stage = StageRequestBody
			bodyReq := &BodyRequest{
				Body:      v.RequestBody.Body,
				Headers:   ctx.Headers,
				RequestID: ctx.RequestID,
			}
			response, processingErr = r.processor.ProcessRequestBody(bodyReq, ctx)

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			ctx.Stage = StageResponseBody
			// For response headers, just continue without processing
			response = &ProcessingResponse{Status: StatusContinue}

		case *ext_proc.ProcessingRequest_ResponseBody:
			ctx.Stage = StageResponseBody
			bodyReq := &BodyRequest{
				Body:      v.ResponseBody.Body,
				Headers:   ctx.Headers,
				RequestID: ctx.RequestID,
			}
			response, processingErr = r.processor.ProcessResponseBody(bodyReq, ctx)

		default:
			r.logger.Warn("Unknown request type", Field{Key: "type", Value: fmt.Sprintf("%T", v)})
			response = &ProcessingResponse{Status: StatusContinue}
		}

		// Handle processing errors
		if processingErr != nil {
			errorResponse, handlerErr := r.errorHandler.Handle(processingErr, ctx)
			if handlerErr != nil {
				r.logger.Error("Error handler failed", handlerErr, Field{Key: LogFieldRequestID, Value: ctx.RequestID})
				return handlerErr
			}
			if errorResponse != nil {
				response = errorResponse
			}
		}

		// Convert and send response
		extProcResponse := r.convertToExtProcResponse(response, ctx)
		if err := stream.Send(extProcResponse); err != nil {
			r.logger.Error("Error sending response", err, Field{Key: LogFieldRequestID, Value: ctx.RequestID})
			return err
		}

		r.logger.Debug("Response sent successfully", Field{Key: LogFieldRequestID, Value: ctx.RequestID})
	}
}

// convertToExtProcResponse converts our internal response to ext_proc response
func (r *OpenAIRouter) convertToExtProcResponse(response *ProcessingResponse, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if response == nil {
		// Default continue response
		return createContinueResponse(ctx.Stage)
	}

	// If there's an immediate response, return it directly
	if response.ImmediateResponse != nil {
		return response.ImmediateResponse
	}

	// Handle different response statuses
	switch response.Status {
	case StatusImmediate:
		if response.ImmediateResponse != nil {
			return response.ImmediateResponse
		}
		fallthrough
	case StatusBlock:
		// Create a block response (usually a 403 or similar)
		return createBlockResponse(ctx.Stage)
	case StatusContinue:
		fallthrough
	default:
		// Create continue response with mutations if any
		return createContinueResponseWithMutations(ctx.Stage, response.HeaderMutation, response.BodyMutation)
	}
}

// Helper functions for creating responses
func createContinueResponse(stage ProcessingStage) *ext_proc.ProcessingResponse {
	switch stage {
	case StageHeaders:
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestHeaders{
				RequestHeaders: &ext_proc.HeadersResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
	case StageRequestBody:
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestBody{
				RequestBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
	case StageResponseBody:
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ResponseBody{
				ResponseBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
	default:
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestBody{
				RequestBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
	}
}

func createBlockResponse(stage ProcessingStage) *ext_proc.ProcessingResponse {
	// For blocking, we typically return an immediate response
	// This is a simplified implementation
	return createContinueResponse(stage)
}

func createContinueResponseWithMutations(stage ProcessingStage, headerMut *HeaderMutation, bodyMut *BodyMutation) *ext_proc.ProcessingResponse {
	response := createContinueResponse(stage)
	
	// Apply mutations based on stage
	switch stage {
	case StageHeaders:
		if headerMut != nil {
			if headers := response.GetRequestHeaders(); headers != nil {
				headers.Response.HeaderMutation = convertHeaderMutation(headerMut)
			}
		}
	case StageRequestBody:
		if body := response.GetRequestBody(); body != nil {
			if headerMut != nil {
				body.Response.HeaderMutation = convertHeaderMutation(headerMut)
			}
			if bodyMut != nil {
				body.Response.BodyMutation = convertBodyMutation(bodyMut)
			}
		}
	case StageResponseBody:
		if body := response.GetResponseBody(); body != nil {
			if bodyMut != nil {
				body.Response.BodyMutation = convertBodyMutation(bodyMut)
			}
		}
	}

	return response
}

func convertHeaderMutation(mut *HeaderMutation) *ext_proc.HeaderMutation {
	if mut == nil {
		return nil
	}

	extMut := &ext_proc.HeaderMutation{
		RemoveHeaders: mut.Remove,
	}

	if len(mut.Add) > 0 {
		for key, value := range mut.Add {
			extMut.SetHeaders = append(extMut.SetHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:   key,
					Value: value,
				},
			})
		}
	}

	return extMut
}

func convertBodyMutation(mut *BodyMutation) *ext_proc.BodyMutation {
	if mut == nil || len(mut.Body) == 0 {
		return nil
	}

	return &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: mut.Body,
		},
	}
}

// Helper functions for extracting data from ext_proc types
func extractHeaders(headers *core.HeaderMap) map[string]string {
	result := make(map[string]string)
	if headers != nil {
		for _, header := range headers.Headers {
			result[header.Key] = header.Value
		}
	}
	return result
}

func extractRequestID(headers *core.HeaderMap) string {
	if headers != nil {
		for _, header := range headers.Headers {
			if strings.ToLower(header.Key) == HeaderRequestID {
				return header.Value
			}
		}
	}
	return ""
}

// CoreProcessor implements the core processing logic using plugins
type CoreProcessor struct {
	router *OpenAIRouter
}

// NewCoreProcessor creates a new core processor
func NewCoreProcessor(router *OpenAIRouter) *CoreProcessor {
	return &CoreProcessor{router: router}
}

// ProcessHeaders processes request headers
func (p *CoreProcessor) ProcessHeaders(req *HeadersRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	ctx.Headers = req.Headers
	ctx.RequestID = req.RequestID
	ctx.StartTime = time.Now()

	p.router.logger.Debug("Processing headers",
		Field{Key: LogFieldRequestID, Value: req.RequestID},
		Field{Key: "header_count", Value: len(req.Headers)})

	return &ProcessingResponse{Status: StatusContinue}, nil
}

// ProcessRequestBody processes request body with plugin-based routing
func (p *CoreProcessor) ProcessRequestBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	ctx.ProcessingStartTime = time.Now()
	
	p.router.logger.Info("Processing request body",
		Field{Key: LogFieldRequestID, Value: req.RequestID},
		Field{Key: LogFieldBodySize, Value: len(req.Body)})

	// Parse OpenAI request
	openAIRequest, err := openai.ParseRequest(req.Body)
	if err != nil {
		return nil, NewProcessingErrorWithCause(ErrorTypeValidation, "INVALID_OPENAI_REQUEST", "failed to parse OpenAI request", err).
			WithRequestID(req.RequestID)
	}

	// Store original model
	originalModel := openAIRequest.Model
	ctx.RequestModel = originalModel

	// Extract content for processing
	userContent, nonUserMessages := openai.ExtractUserAndNonUserContent(openAIRequest)
	allContent := strings.Join(append([]string{userContent}, nonUserMessages...), " ")

	// Handle caching through plugins
	if response := p.handleCaching(req, ctx, allContent); response != nil {
		return response, nil
	}

	// Handle model routing through plugins
	if response, err := p.handleModelRouting(openAIRequest, allContent, ctx); err != nil {
		return nil, err
	} else if response != nil {
		return response, nil
	}

	// Handle tools selection through plugins
	if response, err := p.handleToolsSelection(openAIRequest, allContent, ctx); err != nil {
		return nil, err
	} else if response != nil {
		return response, nil
	}

	return &ProcessingResponse{Status: StatusContinue}, nil
}

// ProcessResponseBody processes response body
func (p *CoreProcessor) ProcessResponseBody(req *BodyRequest, ctx *RequestContext) (*ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)
	
	p.router.logger.Info("Processing response body",
		Field{Key: LogFieldRequestID, Value: req.RequestID},
		Field{Key: LogFieldDuration, Value: completionLatency.Milliseconds()})

	// Handle cache updates through plugins
	p.handleCacheUpdate(req, ctx)

	// Record metrics
	if ctx.RequestModel != "" {
		p.router.metrics.RecordRequest(ctx.RequestModel, completionLatency)
		// Decrease model load for completed request
		p.trackModelLoad(ctx.RequestModel, -1)
	}

	return &ProcessingResponse{Status: StatusContinue}, nil
}

// Plugin-based processing methods

func (p *CoreProcessor) handleCaching(req *BodyRequest, ctx *RequestContext, content string) *ProcessingResponse {
	cachePlugins := p.router.pluginManager.GetPluginsByType(PluginTypeCache)
	
	for _, plugin := range cachePlugins {
		cachePlugin := plugin.(CachePlugin)
		if !cachePlugin.IsEnabled() {
			continue
		}

		// Try to get from cache
		cacheKey := fmt.Sprintf("%s:%s", ctx.RequestModel, content)
		if _, found, err := cachePlugin.Get(cacheKey); err == nil && found {
			p.router.logger.Info("Cache hit", 
				Field{Key: LogFieldRequestID, Value: req.RequestID},
				Field{Key: LogFieldPlugin, Value: cachePlugin.Name()})
			
			p.router.metrics.RecordCacheHit(ctx.RequestModel)
			
			// Return immediate response with cached data
			// This would need to be converted to proper ext_proc response
			return &ProcessingResponse{
				Status: StatusImmediate,
				// ImmediateResponse: createCacheResponse(data),
			}
		}
		
		p.router.metrics.RecordCacheMiss(ctx.RequestModel)
	}

	return nil
}

func (p *CoreProcessor) handleModelRouting(request *openai.OpenAIRequest, content string, ctx *RequestContext) (*ProcessingResponse, error) {
	if request.Model != AutoModel {
		return nil, nil // No routing needed for non-auto models
	}

	routingPlugins := p.router.pluginManager.GetPluginsByType(PluginTypeRouting)
	
	for _, plugin := range routingPlugins {
		routingPlugin := plugin.(RoutingPlugin)
		
		if !routingPlugin.SupportsModel(request.Model) {
			continue
		}

		selection, err := routingPlugin.SelectModel(request, content, ctx)
		if err != nil {
			p.router.logger.Error("Model routing failed", err,
				Field{Key: LogFieldPlugin, Value: routingPlugin.Name()},
				Field{Key: LogFieldRequestID, Value: ctx.RequestID})
			continue
		}

		if selection.Model != request.Model {
			p.router.logger.Info("Model routed",
				Field{Key: LogFieldRequestID, Value: ctx.RequestID},
				Field{Key: "from_model", Value: request.Model},
				Field{Key: "to_model", Value: selection.Model},
				Field{Key: LogFieldPlugin, Value: routingPlugin.Name()},
				Field{Key: LogFieldConfidence, Value: selection.Confidence})

			// Update request
			request.Model = selection.Model
			ctx.RequestModel = selection.Model
			
			// Track model load increase for the selected model
			p.trackModelLoad(selection.Model, 1)
			
			// Serialize modified request
			modifiedBody, err := openai.SerializeRequest(request)
			if err != nil {
				return nil, InternalError("failed to serialize modified request", err)
			}

			p.router.metrics.RecordRouting(ctx.RequestModel, selection.Model)

			return &ProcessingResponse{
				Status: StatusContinue,
				HeaderMutation: &HeaderMutation{
					Remove: []string{HeaderContentLength},
				},
				BodyMutation: &BodyMutation{
					Body: modifiedBody,
				},
			}, nil
		}
	}

	return nil, nil
}

func (p *CoreProcessor) handleToolsSelection(request *openai.OpenAIRequest, content string, ctx *RequestContext) (*ProcessingResponse, error) {
	if request.Tools == nil {
		return nil, nil
	}

	// Check if tools is set to "auto"
	if toolsStr, ok := request.Tools.(string); !ok || toolsStr != AutoTools {
		return nil, nil
	}

	toolsPlugins := p.router.pluginManager.GetPluginsByType(PluginTypeTools)
	
	for _, plugin := range toolsPlugins {
		toolsPlugin := plugin.(ToolsPlugin)
		
		if !toolsPlugin.SupportsAutoSelection() {
			continue
		}

		tools, err := toolsPlugin.SelectTools(content, ctx)
		if err != nil {
			p.router.logger.Error("Tools selection failed", err,
				Field{Key: LogFieldPlugin, Value: toolsPlugin.Name()},
				Field{Key: LogFieldRequestID, Value: ctx.RequestID})
			continue
		}

		p.router.logger.Info("Tools selected",
			Field{Key: LogFieldRequestID, Value: ctx.RequestID},
			Field{Key: "tool_count", Value: len(tools)},
			Field{Key: LogFieldPlugin, Value: toolsPlugin.Name()})

		// Update request
		request.Tools = tools
		
		// Serialize modified request
		modifiedBody, err := openai.SerializeRequest(request)
		if err != nil {
			return nil, InternalError("failed to serialize modified request", err)
		}

		return &ProcessingResponse{
			Status: StatusContinue,
			HeaderMutation: &HeaderMutation{
				Remove: []string{HeaderContentLength},
			},
			BodyMutation: &BodyMutation{
				Body: modifiedBody,
			},
		}, nil
	}

	return nil, nil
}

func (p *CoreProcessor) handleCacheUpdate(req *BodyRequest, ctx *RequestContext) {
	if len(req.Body) == 0 {
		return
	}

	cachePlugins := p.router.pluginManager.GetPluginsByType(PluginTypeCache)
	
	for _, plugin := range cachePlugins {
		cachePlugin := plugin.(CachePlugin)
		if !cachePlugin.IsEnabled() {
			continue
		}

		// Store response in cache
		cacheKey := fmt.Sprintf("%s:%s", ctx.RequestModel, ctx.RequestQuery)
		if err := cachePlugin.Set(cacheKey, req.Body, time.Duration(DefaultCacheTTL)*time.Second); err != nil {
			p.router.logger.Error("Cache update failed", err,
				Field{Key: LogFieldPlugin, Value: cachePlugin.Name()},
				Field{Key: LogFieldRequestID, Value: req.RequestID})
		}
	}
}

// trackModelLoad tracks model load changes (increase/decrease)
func (p *CoreProcessor) trackModelLoad(model string, delta int) {
	// Get the classifier from routing plugins to track model load
	routingPlugins := p.router.pluginManager.GetPluginsByType(PluginTypeRouting)
	for _, plugin := range routingPlugins {
		if routingPlugin, ok := plugin.(*ClassificationRouterPlugin); ok {
			if delta > 0 {
				routingPlugin.classifier.IncrementModelLoad(model)
			} else {
				routingPlugin.classifier.DecrementModelLoad(model)
			}
			break
		}
	}
}

// initializeModels initializes the BERT and classifier models
func initializeModels(cfg *config.RouterConfig) error {
	// Initialize the BERT model for similarity search
	if err := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU); err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	// Load mappings and initialize classifiers
	var categoryMapping *classification.CategoryMapping
	if cfg.Classifier.CategoryModel.CategoryMappingPath != "" {
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load category mapping: %w", err)
		}
		
		// Initialize category classifier
		numClasses := categoryMapping.GetCategoryCount()
		if numClasses >= 2 {
			classifierModelID := cfg.Classifier.CategoryModel.ModelID
			if classifierModelID == "" {
				classifierModelID = cfg.BertModel.ModelID
			}

			if cfg.Classifier.CategoryModel.UseModernBERT {
				if err := candle_binding.InitModernBertClassifier(classifierModelID, cfg.Classifier.CategoryModel.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize ModernBERT classifier: %w", err)
				}
			} else {
				if err := candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.CategoryModel.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize linear classifier: %w", err)
				}
			}
		}
	}

	// Initialize PII classifier if enabled
	var piiMapping *classification.PIIMapping
	if cfg.Classifier.PIIModel.PIIMappingPath != "" {
		var err error
		piiMapping, err = classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load PII mapping: %w", err)
		}

		numPIIClasses := piiMapping.GetPIITypeCount()
		if numPIIClasses >= 2 {
			piiClassifierModelID := cfg.Classifier.PIIModel.ModelID
			if piiClassifierModelID == "" {
				piiClassifierModelID = cfg.BertModel.ModelID
			}

			if cfg.Classifier.PIIModel.UseModernBERT {
				if err := candle_binding.InitModernBertPIIClassifier(piiClassifierModelID, cfg.Classifier.PIIModel.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize ModernBERT PII classifier: %w", err)
				}
			} else {
				if err := candle_binding.InitPIIClassifier(piiClassifierModelID, numPIIClasses, cfg.Classifier.PIIModel.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize linear PII classifier: %w", err)
				}
			}
		}
	}

	// Initialize jailbreak classifier if enabled
	if cfg.IsPromptGuardEnabled() {
		jailbreakMapping, err := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}

		numJailbreakClasses := jailbreakMapping.GetJailbreakTypeCount()
		if numJailbreakClasses >= 2 {
			jailbreakClassifierModelID := cfg.PromptGuard.ModelID
			if jailbreakClassifierModelID == "" {
				jailbreakClassifierModelID = cfg.BertModel.ModelID
			}

			if cfg.PromptGuard.UseModernBERT {
				if err := candle_binding.InitModernBertJailbreakClassifier(jailbreakClassifierModelID, cfg.PromptGuard.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize ModernBERT jailbreak classifier: %w", err)
				}
			} else {
				if err := candle_binding.InitJailbreakClassifier(jailbreakClassifierModelID, numJailbreakClasses, cfg.PromptGuard.UseCPU); err != nil {
					return fmt.Errorf("failed to initialize linear jailbreak classifier: %w", err)
				}
			}
		}
	}

	return nil
}

// Helper functions

// validateDependencies validates that all required dependencies are provided
func validateDependencies(deps RouterDependencies) error {
	if deps.Config == nil {
		return ValidationError("MISSING_CONFIG", "config is required")
	}
	if deps.Logger == nil {
		return ValidationError("MISSING_LOGGER", "logger is required")
	}
	if deps.PluginManager == nil {
		return ValidationError("MISSING_PLUGIN_MANAGER", "plugin manager is required")
	}
	if deps.ErrorHandler == nil {
		return ValidationError("MISSING_ERROR_HANDLER", "error handler is required")
	}
	if deps.Metrics == nil {
		return ValidationError("MISSING_METRICS", "metrics collector is required")
	}
	return nil
}

// registerBuiltinPlugins registers and configures built-in plugins based on config
func registerBuiltinPlugins(pm *DefaultPluginManager, cfg *config.RouterConfig, logger Logger) error {
	// Load mappings
	var categoryMapping *classification.CategoryMapping
	var piiMapping *classification.PIIMapping
	var jailbreakMapping *classification.JailbreakMapping

	if cfg.Classifier.CategoryModel.CategoryMappingPath != "" {
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	if cfg.Classifier.PIIModel.PIIMappingPath != "" {
		var err error
		piiMapping, err = classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}

	if cfg.IsPromptGuardEnabled() {
		var err error
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}

	// Create classifier and utilities
	ttftCalculator := ttft.NewCalculator(cfg.GPUConfig)
	modelTTFT := ttftCalculator.InitializeModelTTFT(cfg)
	classifier := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping, modelTTFT)
	piiChecker := pii.NewPolicyChecker(cfg.ModelConfig)

	// Initialize jailbreak classifier if enabled
	if jailbreakMapping != nil {
		if err := classifier.InitializeJailbreakClassifier(); err != nil {
			return fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
		}
	}

	// Security plugins
	if piiMapping != nil {
		piiPlugin := NewPIIDetectorPlugin(PriorityHigh, classifier, logger)
		if err := piiPlugin.Initialize(map[string]interface{}{
			"enabled":   true,
			"threshold": cfg.Classifier.PIIModel.Threshold,
		}); err != nil {
			return fmt.Errorf("failed to initialize PII plugin: %w", err)
		}
		if err := pm.RegisterPlugin(piiPlugin); err != nil {
			return err
		}
		logger.Info("Registered PII detector plugin")
	}

	if jailbreakMapping != nil {
		jailbreakPlugin := NewJailbreakDetectorPlugin(PriorityHigh, classifier, logger)
		if err := jailbreakPlugin.Initialize(map[string]interface{}{
			"enabled":   true,
			"threshold": cfg.PromptGuard.Threshold,
		}); err != nil {
			return fmt.Errorf("failed to initialize jailbreak plugin: %w", err)
		}
		if err := pm.RegisterPlugin(jailbreakPlugin); err != nil {
			return err
		}
		logger.Info("Registered jailbreak detector plugin")
	}

	// Routing plugins
	if categoryMapping != nil && len(cfg.Categories) > 0 {
		routingPlugin := NewClassificationRouterPlugin(PriorityMedium, classifier, piiChecker, cfg, logger)
		if err := routingPlugin.Initialize(map[string]interface{}{
			"enabled": true,
		}); err != nil {
			return fmt.Errorf("failed to initialize classification router: %w", err)
		}
		if err := pm.RegisterPlugin(routingPlugin); err != nil {
			return err
		}
		logger.Info("Registered classification router plugin")
	}

	// Cache plugins
	if cfg.SemanticCache.Enabled {
		cacheOptions := cache.SemanticCacheOptions{
			SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
			MaxEntries:          cfg.SemanticCache.MaxEntries,
			TTLSeconds:          cfg.SemanticCache.TTLSeconds,
			Enabled:             cfg.SemanticCache.Enabled,
		}
		semanticCache := cache.NewSemanticCache(cacheOptions)
		
		cachePlugin := NewSemanticCachePlugin(semanticCache, logger)
		if err := cachePlugin.Initialize(map[string]interface{}{
			"enabled": cfg.SemanticCache.Enabled,
		}); err != nil {
			return fmt.Errorf("failed to initialize cache plugin: %w", err)
		}
		if err := pm.RegisterPlugin(cachePlugin); err != nil {
			return err
		}
		logger.Info("Registered semantic cache plugin")
	}

	// Tools plugins
	if cfg.Tools.Enabled {
		toolsThreshold := cfg.BertModel.Threshold
		if cfg.Tools.SimilarityThreshold != nil {
			toolsThreshold = *cfg.Tools.SimilarityThreshold
		}
		
		toolsOptions := tools.ToolsDatabaseOptions{
			SimilarityThreshold: toolsThreshold,
			Enabled:             cfg.Tools.Enabled,
		}
		toolsDatabase := tools.NewToolsDatabase(toolsOptions)

		// Load tools from file if path is provided
		if cfg.Tools.ToolsDBPath != "" {
			if err := toolsDatabase.LoadToolsFromFile(cfg.Tools.ToolsDBPath); err != nil {
				logger.Warn("Failed to load tools from file", Field{Key: "path", Value: cfg.Tools.ToolsDBPath}, Field{Key: "error", Value: err.Error()})
			}
		}

		toolsPlugin := NewSimilarityToolsPlugin(toolsDatabase, cfg.Tools, logger)
		if err := toolsPlugin.Initialize(map[string]interface{}{
			"enabled": cfg.Tools.Enabled,
			"top_k":   cfg.Tools.TopK,
		}); err != nil {
			return fmt.Errorf("failed to initialize tools plugin: %w", err)
		}
		if err := pm.RegisterPlugin(toolsPlugin); err != nil {
			return err
		}
		logger.Info("Registered similarity tools plugin")
	}

	return nil
}
