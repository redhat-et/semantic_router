# MCP AI Gateway

A gateway that bridges LLMs with Model Context Protocol (MCP) servers, providing a unified interface for accessing tools from multiple MCP servers.

## Installation

### Prerequisites
- Go 1.21 or later
- Node.js and npm (for MCP servers using npx)

### Build
```bash
git clone <repository-url>
cd mcp-ai-gateway
go mod tidy
go build -o mcp-ai-gateway ./cmd/main.go
```

## Configuration

Create a `config.json` file:

```json
{
  "mcpProxy": {
    "baseURL": "http://localhost:8080",
    "addr": ":8080",
    "name": "MCP AI Gateway",
    "version": "1.0.0",
    "type": "streamable-http",
    "options": {
      "logEnabled": true,
      "authTokens": ["your-auth-token"]
    }
  },
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "options": {
        "toolFilter": {
          "mode": "allow",
          "list": ["read_file", "write_file"]
        }
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    },
    "remote-server": {
      "url": "https://example.com/mcp/http",
      "transportType": "streamable-http",
      "headers": {
        "Authorization": "Bearer api-key"
      }
    }
  }
}
```

## Usage

### Start the Gateway
```bash
./mcp-ai-gateway --config config.json
```

### Use with OpenAI-Compatible Clients
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/filesystem",
    api_key="your-auth-token"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List files in /tmp"}],
    tools=client.get("/tools")
)
```

### Direct API Calls
```bash
# List available tools
curl http://localhost:8080/filesystem/tools

# Health check
curl http://localhost:8080/health
```

## Command Line Options

```bash
./mcp-ai-gateway --config config.json
./mcp-ai-gateway --help
./mcp-ai-gateway --version
``` 