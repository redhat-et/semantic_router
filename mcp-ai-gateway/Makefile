BINARY_NAME=mcp-ai-gateway
BUILD_DIR=build

.PHONY: build test clean

build:
	@mkdir -p $(BUILD_DIR)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && go build -o $(BUILD_DIR)/$(BINARY_NAME) cmd/main.go

test:
	go test -v ./...

clean:
	rm -rf $(BUILD_DIR) 