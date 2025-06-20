module github.com/semantic_router/mcp-ai-gateway

go 1.24.1

replace github.com/redhat-et/semantic_route/candle-binding => ../candle-binding

require (
	github.com/gorilla/mux v1.8.1
	github.com/mark3labs/mcp-go v0.32.0
	github.com/redhat-et/semantic_route/candle-binding v0.0.0-00010101000000-000000000000
)

require (
	github.com/google/uuid v1.6.0 // indirect
	github.com/spf13/cast v1.7.1 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
)
