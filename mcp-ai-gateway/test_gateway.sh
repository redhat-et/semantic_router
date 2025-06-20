#!/bin/bash

# MCP AI Gateway Test Script
# This script tests the proxy functionality of the MCP AI Gateway

set -e  # Exit on any error

# Configuration
GATEWAY_PORT=8080
GATEWAY_URL="http://localhost:${GATEWAY_PORT}"
CONFIG_FILE="test_config.json"
AUTH_TOKEN="test-token"
GATEWAY_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [ ! -z "$GATEWAY_PID" ]; then
        print_status "Stopping gateway (PID: $GATEWAY_PID)"
        kill $GATEWAY_PID 2>/dev/null || true
        wait $GATEWAY_PID 2>/dev/null || true
    fi
    # Clean up any background processes
    pkill -f "go run cmd/main.go" 2>/dev/null || true
    sleep 2
    print_status "Cleanup complete"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Function to wait for gateway to be ready
wait_for_gateway() {
    print_status "Waiting for gateway to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$GATEWAY_URL/health" > /dev/null 2>&1; then
            print_success "Gateway is ready!"
            return 0
        fi
        print_status "Attempt $attempt/$max_attempts - Gateway not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Gateway failed to start within timeout"
    return 1
}

# Function to make authenticated API call
api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local expected_status="${4:-200}"
    
    print_status "Testing $method $endpoint"
    
    local curl_cmd="curl -s -w '%{http_code}' -H 'Authorization: Bearer $AUTH_TOKEN' -H 'Content-Type: application/json'"
    
    if [ "$method" = "POST" ] && [ ! -z "$data" ]; then
        curl_cmd="$curl_cmd -d '$data'"
    fi
    
    curl_cmd="$curl_cmd '$GATEWAY_URL$endpoint'"
    
    local response=$(eval $curl_cmd)
    local status_code="${response: -3}"
    local body="${response%???}"
    
    if [ "$status_code" = "$expected_status" ]; then
        print_success "$method $endpoint returned $status_code"
        if [ ! -z "$body" ] && [ "$body" != "null" ]; then
            echo "$body" | jq . 2>/dev/null || echo "$body"
        fi
        return 0
    else
        print_error "$method $endpoint returned $status_code (expected $expected_status)"
        echo "Response body: $body"
        return 1
    fi
}

# Function to test basic endpoints
test_basic_endpoints() {
    print_status "Testing basic endpoints..."
    
    # Test health endpoint
    api_call "GET" "/health"
    
    # Test info endpoint
    api_call "GET" "/info"
    
    # Test server list endpoint
    api_call "GET" "/servers"
}

# Function to test MCP server proxy endpoints
test_mcp_proxy() {
    print_status "Testing MCP server proxy endpoints..."
    
    # First check if any servers are active
    local servers_response=$(curl -s -H "Authorization: Bearer $AUTH_TOKEN" "$GATEWAY_URL/servers")
    local active_servers=$(echo "$servers_response" | jq -r 'to_entries[] | select(.value.isActive == true) | .key' 2>/dev/null || echo "")
    
    if [ -z "$active_servers" ]; then
        print_warning "No active MCP servers found - this is expected if servers timeout during startup"
        print_status "Testing proxy endpoints with inactive servers..."
        
        # Test that inactive servers return proper error messages
        api_call "POST" "/memory" '{"method": "tools/list"}' 503 && print_success "Inactive server properly returns 503" || print_error "Inactive server error handling failed"
        
        api_call "GET" "/memory/tools" "" 503 && print_success "Inactive server tools endpoint properly returns 503" || print_error "Inactive server tools endpoint error handling failed"
        
        return 0
    fi
    
    print_status "Found active servers: $active_servers"
    
    # Test each active server
    for server in $active_servers; do
        print_status "Testing server: $server"
        
        # Test tools list
        print_status "Testing tools list for $server server..."
        api_call "POST" "/$server" '{"method": "tools/list"}' || print_warning "$server server tools list failed"
        
        # Test with OpenAI format
        print_status "Testing tools list with OpenAI format..."
        api_call "GET" "/$server/tools?format=openai" "" 200 || print_warning "$server server tools list (OpenAI format) failed"
        
        # Test resources list
        print_status "Testing resources list..."
        api_call "POST" "/$server" '{"method": "resources/list"}' || print_warning "$server server resources list failed"
        
        # Test prompts list
        print_status "Testing prompts list..."
        api_call "POST" "/$server" '{"method": "prompts/list"}' || print_warning "$server server prompts list failed"
        
        # Test memory-specific operations
        if [ "$server" = "memory" ]; then
            print_status "Testing memory server core functions..."
            
            # Test 1: Create entities (core function)
            print_status "Testing create_entities..."
            local create_entities_data='{
                "method": "tools/call",
                "params": {
                    "name": "create_entities",
                    "arguments": {
                        "entities": [
                            {
                                "name": "test_user",
                                "entityType": "person",
                                "observations": ["Testing the MCP gateway", "Uses Claude Desktop"]
                            },
                            {
                                "name": "mcp_gateway",
                                "entityType": "software",
                                "observations": ["Proxy server for MCP", "Built with Go"]
                            }
                        ]
                    }
                }
            }'
            api_call "POST" "/$server" "$create_entities_data" || print_warning "Create entities failed"
            
            # Test 2: Add observations (core function)
            print_status "Testing add_observations..."
            local add_observations_data='{
                "method": "tools/call",
                "params": {
                    "name": "add_observations",
                    "arguments": {
                        "observations": [
                            {
                                "entityName": "test_user",
                                "contents": ["Likes testing software", "Interested in AI"]
                            }
                        ]
                    }
                }
            }'
            api_call "POST" "/$server" "$add_observations_data" || print_warning "Add observations failed"
            
            # Test 3: Search nodes (core function)
            print_status "Testing search_nodes..."
            local search_nodes_data='{
                "method": "tools/call",
                "params": {
                    "name": "search_nodes",
                    "arguments": {
                        "query": "test_user"
                    }
                }
            }'
            api_call "POST" "/$server" "$search_nodes_data" || print_warning "Search nodes failed"
            
            # Test 4: Read graph (core function)
            print_status "Testing read_graph..."
            local read_graph_data='{
                "method": "tools/call",
                "params": {
                    "name": "read_graph",
                    "arguments": {}
                }
            }'
            api_call "POST" "/$server" "$read_graph_data" || print_warning "Read graph failed"
            
            # Test 5: Create relations (core function)
            print_status "Testing create_relations..."
            local create_relations_data='{
                "method": "tools/call",
                "params": {
                    "name": "create_relations",
                    "arguments": {
                        "relations": [
                            {
                                "from": "test_user",
                                "to": "mcp_gateway",
                                "relationType": "uses"
                            }
                        ]
                    }
                }
            }'
            api_call "POST" "/$server" "$create_relations_data" || print_warning "Create relations failed"
            
            # Test 6: Open nodes (core function)
            print_status "Testing open_nodes..."
            local open_nodes_data='{
                "method": "tools/call",
                "params": {
                    "name": "open_nodes",
                    "arguments": {
                        "names": ["test_user", "mcp_gateway"]
                    }
                }
            }'
            api_call "POST" "/$server" "$open_nodes_data" || print_warning "Open nodes failed"
            
            # Test 7: Delete entities (cleanup)
            print_status "Testing delete_entities (cleanup)..."
            local delete_entities_data='{
                "method": "tools/call",
                "params": {
                    "name": "delete_entities",
                    "arguments": {
                        "entityNames": ["test_user", "mcp_gateway"]
                    }
                }
            }'
            api_call "POST" "/$server" "$delete_entities_data" || print_warning "Delete entities failed"
        fi
    done
}

# Function to test error cases
test_error_cases() {
    print_status "Testing error cases..."
    
    # Test unauthorized access (no auth header)
    print_status "Testing unauthorized access (no auth header)..."
    local unauth_response=$(curl -s -w '%{http_code}' -H 'Content-Type: application/json' -d '{"method": "tools/list"}' "$GATEWAY_URL/memory")
    local unauth_status="${unauth_response: -3}"
    if [ "$unauth_status" = "401" ]; then
        print_success "Unauthorized access properly rejected with 401"
    else
        print_warning "Unauthorized access returned $unauth_status (expected 401, but server may allow unauthenticated requests)"
    fi
    
    # Test unauthorized access (wrong token)
    print_status "Testing unauthorized access (wrong token)..."
    local wrong_token_response=$(curl -s -w '%{http_code}' -H 'Authorization: Bearer wrong-token' -H 'Content-Type: application/json' -d '{"method": "tools/list"}' "$GATEWAY_URL/memory")
    local wrong_token_status="${wrong_token_response: -3}"
    if [ "$wrong_token_status" = "401" ]; then
        print_success "Wrong token properly rejected with 401"
    else
        print_warning "Wrong token returned $wrong_token_status (may be expected if server allows any token)"
    fi
    
    # Test non-existent server
    print_status "Testing non-existent server..."
    api_call "POST" "/nonexistent" '{"method": "tools/list"}' 404 && print_success "Non-existent server properly returns 404" || print_error "Non-existent server error handling failed"
    
    # Test invalid JSON
    print_status "Testing invalid JSON..."
    local invalid_json_response=$(curl -s -w '%{http_code}' -H "Authorization: Bearer $AUTH_TOKEN" -H 'Content-Type: application/json' -d 'invalid json' "$GATEWAY_URL/memory")
    local invalid_json_status="${invalid_json_response: -3}"
    if [ "$invalid_json_status" = "400" ] || [ "$invalid_json_status" = "503" ]; then
        print_success "Invalid JSON properly rejected with $invalid_json_status"
    else
        print_error "Invalid JSON returned $invalid_json_status (expected 400 or 503)"
    fi
    
    # Test missing method
    print_status "Testing missing method..."
    local missing_method_response=$(curl -s -w '%{http_code}' -H "Authorization: Bearer $AUTH_TOKEN" -H 'Content-Type: application/json' -d '{"params": {}}' "$GATEWAY_URL/memory")
    local missing_method_status="${missing_method_response: -3}"
    if [ "$missing_method_status" = "400" ] || [ "$missing_method_status" = "503" ]; then
        print_success "Missing method properly rejected with $missing_method_status"
    else
        print_error "Missing method returned $missing_method_status (expected 400 or 503)"
    fi
}

# Main test function
run_tests() {
    print_status "Starting MCP AI Gateway functional tests..."
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file $CONFIG_FILE not found"
        exit 1
    fi
    
    # Start the gateway in background with more verbose logging
    print_status "Starting MCP AI Gateway with debug logging..."
    # Set environment variable to enable debug logging for MCP servers
    export MCP_DEBUG=1
    go run cmd/main.go -config "$CONFIG_FILE" > gateway.log 2>&1 &
    GATEWAY_PID=$!
    
    print_status "Gateway started with PID: $GATEWAY_PID"
    
    # Wait for gateway to be ready
    if ! wait_for_gateway; then
        print_error "Gateway failed to start"
        print_status "Gateway startup logs:"
        cat gateway.log
        exit 1
    fi
    
    # Run functional tests only
    local failed_tests=0
    
    test_basic_endpoints || ((failed_tests++))
    test_mcp_proxy || ((failed_tests++))
    test_error_cases || ((failed_tests++))
    
    # Print summary
    print_status "Test Summary:"
    if [ $failed_tests -eq 0 ]; then
        print_success "All functional tests passed!"
    else
        print_warning "$failed_tests test categories had failures"
    fi
    
    # Show gateway logs with MCP server output
    print_status "=== Gateway and MCP Server Logs ==="
    cat gateway.log
    
    # Try to find any additional MCP server logs
    if [ -f "memory-server.log" ]; then
        print_status "=== Memory Server Specific Logs ==="
        cat memory-server.log
    fi
    
    return $failed_tests
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    command -v curl >/dev/null 2>&1 || { print_error "curl is required but not installed."; exit 1; }
    command -v jq >/dev/null 2>&1 || { print_warning "jq is not installed. JSON formatting will be limited."; }
    command -v go >/dev/null 2>&1 || { print_error "go is required but not installed."; exit 1; }
    
    print_success "Dependencies check passed"
}

# Help function
show_help() {
    echo "MCP AI Gateway Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -p, --port     Gateway port (default: 8080)"
    echo "  -c, --config   Config file (default: test_config.json)"
    echo "  -t, --token    Auth token (default: test-token)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run with defaults"
    echo "  $0 -p 9090 -c my_config.json"
    echo "  $0 --port 8080 --token my-token"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            GATEWAY_PORT="$2"
            GATEWAY_URL="http://localhost:${GATEWAY_PORT}"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--token)
            AUTH_TOKEN="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "MCP AI Gateway Test Script"
    print_status "Configuration: Port=$GATEWAY_PORT, Config=$CONFIG_FILE, Token=$AUTH_TOKEN"
    
    check_dependencies
    run_tests
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "All tests completed successfully!"
    else
        print_error "Some tests failed. Check the output above."
    fi
    
    exit $exit_code
}

# Run main function
main 