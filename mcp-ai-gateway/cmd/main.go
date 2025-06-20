package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/semantic_router/mcp-ai-gateway/pkg/mcp"
)

// Config represents the main configuration structure
type Config struct {
	MCPProxy   ProxyConfig            `json:"mcpProxy"`
	MCPServers map[string]ServerConfig `json:"mcpServers"`
}

// ProxyConfig represents the proxy server configuration
type ProxyConfig struct {
	BaseURL string        `json:"baseURL"`
	Addr    string        `json:"addr"`
	Name    string        `json:"name"`
	Version string        `json:"version"`
	Type    string        `json:"type"` // "streamable-http"
	Options ProxyOptions  `json:"options"`
}

// ServerConfig represents individual MCP server configuration
type ServerConfig struct {
	Command       string            `json:"command,omitempty"`
	Args          []string          `json:"args,omitempty"`
	Env           map[string]string `json:"env,omitempty"`
	URL           string            `json:"url,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	TransportType string            `json:"transportType,omitempty"`
	Timeout       int               `json:"timeout,omitempty"`
	Options       ServerOptions     `json:"options"`
}

// ProxyOptions represents proxy-level options
type ProxyOptions struct {
	PanicIfInvalid bool     `json:"panicIfInvalid"`
	LogEnabled     bool     `json:"logEnabled"`
	AuthTokens     []string `json:"authTokens"`
}

// ServerOptions represents server-level options
type ServerOptions struct {
	PanicIfInvalid bool       `json:"panicIfInvalid"`
	LogEnabled     bool       `json:"logEnabled"`
	AuthTokens     []string   `json:"authTokens"`
	ToolFilter     ToolFilter `json:"toolFilter"`
}

// ToolFilter represents tool filtering configuration
type ToolFilter struct {
	Mode string   `json:"mode"` // "allow" or "block"
	List []string `json:"list"`
}

// MCPGateway represents the main gateway server
type MCPGateway struct {
	config    *Config
	router    *mux.Router
	clients   map[string]*MCPClient
	server    *http.Server
}

// MCPClient represents a connection to an MCP server
type MCPClient struct {
	Name          string
	Config        ServerConfig
	TransportType string
	IsActive      bool
	Client        mcp.MCPClient
}

func main() {
	var configPath = flag.String("config", "config.json", "path to config file or HTTP(S) URL")
	var showHelp = flag.Bool("help", false, "print help and exit")
	var showVersion = flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showHelp {
		fmt.Println("Usage of mcp-ai-gateway:")
		flag.PrintDefaults()
		os.Exit(0)
	}

	if *showVersion {
		fmt.Println("mcp-ai-gateway version 1.0.0")
		os.Exit(0)
	}

	// Load configuration
	config, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create and start gateway
	gateway, err := NewMCPGateway(config)
	if err != nil {
		log.Fatalf("Failed to create gateway: %v", err)
	}

	// Start server
	if err := gateway.Start(); err != nil {
		log.Fatalf("Failed to start gateway: %v", err)
	}
}

// loadConfig loads configuration from file or URL
func loadConfig(configPath string) (*Config, error) {
	var data []byte
	var err error

	// Check if it's a URL
	if configPath[:4] == "http" {
		resp, err := http.Get(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch config from URL: %w", err)
		}
		defer resp.Body.Close()

		data = make([]byte, 0)
		buf := make([]byte, 1024)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				data = append(data, buf[:n]...)
			}
			if err != nil {
				break
			}
		}
	} else {
		// Load from file
		data, err = os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	// Set defaults
	if config.MCPProxy.Addr == "" {
		config.MCPProxy.Addr = ":8080"
	}
	if config.MCPProxy.Type == "" {
		config.MCPProxy.Type = "streamable-http"
	}
	if config.MCPProxy.Name == "" {
		config.MCPProxy.Name = "MCP AI Gateway"
	}
	if config.MCPProxy.Version == "" {
		config.MCPProxy.Version = "1.0.0"
	}

	return &config, nil
}

// NewMCPGateway creates a new MCP gateway instance
func NewMCPGateway(config *Config) (*MCPGateway, error) {
	gateway := &MCPGateway{
		config:  config,
		router:  mux.NewRouter(),
		clients: make(map[string]*MCPClient),
	}

	// Initialize MCP clients
	for name, serverConfig := range config.MCPServers {
		client := &MCPClient{
			Name:          name,
			Config:        serverConfig,
			TransportType: determineTransportType(serverConfig),
			IsActive:      false,
		}
		gateway.clients[name] = client
	}

	// Setup routes
	gateway.setupRoutes()

	return gateway, nil
}

// determineTransportType determines the transport type based on configuration
func determineTransportType(config ServerConfig) string {
	if config.TransportType != "" {
		return config.TransportType
	}
	
	if config.Command != "" {
		return "stdio"
	}
	
	if config.URL != "" {
		return "streamable-http"
	}
	
	return "stdio"
}

// setupRoutes configures the HTTP routes
func (g *MCPGateway) setupRoutes() {
	// Health check endpoint
	g.router.HandleFunc("/health", g.handleHealth).Methods("GET")
	
	// Server info endpoint
	g.router.HandleFunc("/info", g.handleInfo).Methods("GET")
	
	// List available servers
	g.router.HandleFunc("/servers", g.handleListServers).Methods("GET")
	
	// Server-specific endpoints
	for clientName := range g.clients {
		// Direct server endpoint for MCP calls
		g.router.HandleFunc(fmt.Sprintf("/%s", clientName), g.handleHTTP(clientName)).Methods("POST")
		
		// HTTP streaming endpoint (alternative)
		g.router.HandleFunc(fmt.Sprintf("/%s/mcp", clientName), g.handleHTTP(clientName)).Methods("POST")
		
		// Tools endpoint
		g.router.HandleFunc(fmt.Sprintf("/%s/tools", clientName), g.handleTools(clientName)).Methods("GET")
	}
}

// Start starts the gateway server
func (g *MCPGateway) Start() error {
	g.server = &http.Server{
		Addr:         g.config.MCPProxy.Addr,
		Handler:      g.router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start MCP clients
	for name, client := range g.clients {
		if err := g.startMCPClient(name, client); err != nil {
			log.Printf("Failed to start MCP client %s: %v", name, err)
			if client.Config.Options.PanicIfInvalid {
				return fmt.Errorf("failed to start required MCP client %s: %w", name, err)
			}
		}
	}

	// Handle graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down gateway...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := g.server.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
	}()

	log.Printf("Starting MCP AI Gateway on %s", g.config.MCPProxy.Addr)
	log.Printf("Base URL: %s", g.config.MCPProxy.BaseURL)
	log.Printf("Transport Type: %s", g.config.MCPProxy.Type)
	
	return g.server.ListenAndServe()
}

// startMCPClient starts an individual MCP client
func (g *MCPGateway) startMCPClient(name string, client *MCPClient) error {
	log.Printf("Starting MCP client: %s (transport: %s)", name, client.TransportType)
	
	// Set default timeout if not specified
	timeout := client.Config.Timeout
	if timeout == 0 {
		timeout = 30 // default 30 seconds
	}
	
	// Convert ServerConfig to ClientConfig
	clientConfig := mcp.ClientConfig{
		Command:       client.Config.Command,
		Args:          client.Config.Args,
		Env:           client.Config.Env,
		URL:           client.Config.URL,
		Headers:       client.Config.Headers,
		TransportType: client.Config.TransportType,
		Timeout:       time.Duration(timeout) * time.Second,
		Options: mcp.ClientOptions{
			PanicIfInvalid: client.Config.Options.PanicIfInvalid,
			LogEnabled:     client.Config.Options.LogEnabled,
			AuthTokens:     client.Config.Options.AuthTokens,
			ToolFilter: mcp.ToolFilter{
				Mode: client.Config.Options.ToolFilter.Mode,
				List: client.Config.Options.ToolFilter.List,
			},
		},
	}
	
	// Create MCP client
	mcpClient, err := mcp.NewClient(name, clientConfig)
	if err != nil {
		return fmt.Errorf("failed to create MCP client %s: %w", name, err)
	}
	
	// Set log handler
	mcpClient.SetLogHandler(func(level mcp.LoggingLevel, message string) {
		log.Printf("[%s] %s: %s", level, name, message)
	})
	
	// Connect to the MCP server with timeout handling
	log.Printf("Connecting to MCP client: %s with timeout: %vs", name, timeout)
	if err := mcpClient.Connect(); err != nil {
		return fmt.Errorf("failed to connect MCP client %s: %w", name, err)
	}
	
	client.Client = mcpClient
	client.IsActive = true
	
	log.Printf("Successfully started MCP client: %s", name)
	return nil
}

// HTTP handlers
func (g *MCPGateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "healthy",
		"version": g.config.MCPProxy.Version,
		"name":    g.config.MCPProxy.Name,
	})
}

func (g *MCPGateway) handleInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"name":       g.config.MCPProxy.Name,
		"version":    g.config.MCPProxy.Version,
		"baseURL":    g.config.MCPProxy.BaseURL,
		"type":       g.config.MCPProxy.Type,
		"serverCount": len(g.clients),
	})
}

func (g *MCPGateway) handleListServers(w http.ResponseWriter, r *http.Request) {
	servers := make(map[string]interface{})
	for name, client := range g.clients {
		servers[name] = map[string]interface{}{
			"name":          name,
			"transportType": client.TransportType,
			"isActive":      client.IsActive,
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(servers)
}

func (g *MCPGateway) handleHTTP(clientName string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Check authentication
		if !g.isAuthorized(r, clientName) {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		
		client, exists := g.clients[clientName]
		if !exists {
			http.Error(w, "Server not found", http.StatusNotFound)
			return
		}
		
		if !client.IsActive || client.Client == nil {
			http.Error(w, "Server not active", http.StatusServiceUnavailable)
			return
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}
		
		// Handle different request types
		method, ok := reqBody["method"].(string)
		if !ok {
			http.Error(w, "Missing method field", http.StatusBadRequest)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		
		switch method {
		case "tools/call":
			g.handleToolCall(w, r, client, reqBody)
		case "tools/list":
			g.handleToolsList(w, r, client)
		case "resources/list":
			g.handleResourcesList(w, r, client)
		case "prompts/list":
			g.handlePromptsList(w, r, client)
		default:
			http.Error(w, fmt.Sprintf("Unsupported method: %s", method), http.StatusBadRequest)
		}
	}
}

// handleToolCall handles tool call requests
func (g *MCPGateway) handleToolCall(w http.ResponseWriter, r *http.Request, client *MCPClient, reqBody map[string]interface{}) {
	// Extract tool call parameters
	params, ok := reqBody["params"].(map[string]interface{})
	if !ok {
		http.Error(w, "Missing params", http.StatusBadRequest)
		return
	}

	toolName, ok := params["name"].(string)
	if !ok {
		http.Error(w, "Missing tool name", http.StatusBadRequest)
		return
	}

	arguments, _ := params["arguments"].(map[string]interface{})

	// Call the tool
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	result, err := client.Client.CallTool(ctx, toolName, arguments)
	if err != nil {
		http.Error(w, fmt.Sprintf("Tool call failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Check if client wants OpenAI format
	format := r.URL.Query().Get("format")
	if format == "openai" {
		openAIResult := mcp.ConvertMCPResultToOpenAI(result)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"result": openAIResult,
			"format": "openai",
		})
	} else {
		// Return MCP format
		json.NewEncoder(w).Encode(map[string]interface{}{
			"result": result,
			"format": "mcp",
		})
	}
}

// handleToolsList handles tools/list requests
func (g *MCPGateway) handleToolsList(w http.ResponseWriter, r *http.Request, client *MCPClient) {
	tools := client.Client.GetTools()
	
	// Check if client wants OpenAI format
	format := r.URL.Query().Get("format")
	if format == "openai" {
		openAITools := make([]mcp.OpenAITool, len(tools))
		for i, tool := range tools {
			openAITools[i] = mcp.ConvertToolToOpenAI(tool)
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools":  openAITools,
			"format": "openai",
		})
	} else {
		// Return MCP format
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools":  tools,
			"format": "mcp",
		})
	}
}

// handleResourcesList handles resources/list requests
func (g *MCPGateway) handleResourcesList(w http.ResponseWriter, r *http.Request, client *MCPClient) {
	resources := client.Client.GetResources()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"resources": resources,
	})
}

// handlePromptsList handles prompts/list requests
func (g *MCPGateway) handlePromptsList(w http.ResponseWriter, r *http.Request, client *MCPClient) {
	prompts := client.Client.GetPrompts()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"prompts": prompts,
	})
}

// isAuthorized checks if the request is authorized
func (g *MCPGateway) isAuthorized(r *http.Request, clientName string) bool {
	client, exists := g.clients[clientName]
	if !exists {
		return false
	}

	// Check global auth tokens first
	authHeader := r.Header.Get("Authorization")
	token := mcp.ExtractAuthToken(authHeader)
	
	// Check global tokens
	if mcp.ValidateAuthToken(token, g.config.MCPProxy.Options.AuthTokens) {
		return true
	}
	
	// Check server-specific tokens
	return mcp.ValidateAuthToken(token, client.Config.Options.AuthTokens)
}

// handleTools handles the tools endpoint with OpenAI compatibility
func (g *MCPGateway) handleTools(clientName string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Check authentication
		if !g.isAuthorized(r, clientName) {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		
		client, exists := g.clients[clientName]
		if !exists {
			http.Error(w, "Server not found", http.StatusNotFound)
			return
		}
		
		if !client.IsActive || client.Client == nil {
			http.Error(w, "Server not active", http.StatusServiceUnavailable)
			return
		}

		// Get format from query parameter (default to "mcp", but support "openai")
		format := r.URL.Query().Get("format")
		if format == "" {
			format = "mcp"
		}
		
		tools := client.Client.GetTools()
		
		w.Header().Set("Content-Type", "application/json")
		
		if format == "openai" {
			// Convert to OpenAI format
			openAITools := make([]mcp.OpenAITool, len(tools))
			for i, tool := range tools {
				openAITools[i] = mcp.ConvertToolToOpenAI(tool)
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"tools":  openAITools,
				"server": clientName,
				"format": "openai",
			})
		} else {
			// Return MCP format
			json.NewEncoder(w).Encode(map[string]interface{}{
				"tools":  tools,
				"server": clientName,
				"format": "mcp",
			})
		}
	}
}
