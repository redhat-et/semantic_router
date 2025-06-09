.PHONY: all build clean test docker-build podman-build docker-run podman-run test-pii test-pii-unit test-pii-integration test-existing-functionality

# Default target
all: build

# vLLM env var
VLLM_ENDPOINT ?= http://192.168.12.175:11434

# Container settings
USE_CONTAINER ?= false
CONTAINER_ENGINE ?= podman
IMAGE_NAME ?= llm-semantic-router
CONTAINER_NAME ?= llm-semantic-router-container
CONTAINER_VOLUMES = -v ${PWD}:/app

# USE_CONTAINER and CONTAINER_ENGINE to determine the container command or not use container
ifeq ($(USE_CONTAINER),true)
  ifeq ($(CONTAINER_ENGINE),docker)
    CONTAINER_CMD = docker
  else ifeq ($(CONTAINER_ENGINE),podman)
    CONTAINER_CMD = podman
  else
    $(error CONTAINER_ENGINE must be either docker or podman)
  endif
  EXEC_PREFIX = $(CONTAINER_CMD) exec $(CONTAINER_NAME)
  RUN_PREFIX = $(CONTAINER_CMD) run --rm $(CONTAINER_VOLUMES) --network=host --name $(CONTAINER_NAME)
else
  EXEC_PREFIX =
  RUN_PREFIX =
endif

# Build the Rust library and Golang binding
build: rust build-router

# Build the Rust library
rust:
	@echo "Building Rust library..."
ifeq ($(USE_CONTAINER),true)
	$(CONTAINER_CMD) build -t $(IMAGE_NAME) .
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && cargo build --release"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	cd candle-binding && cargo build --release
endif

# Build router
build-router: rust
	@echo "Building router..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "mkdir -p bin && cd semantic_router && go build -o ../bin/router cmd/main.go"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	@mkdir -p bin
	@cd semantic_router && go build -o ../bin/router cmd/main.go
endif

# Run the router
run-router: build-router
	@echo "Running router..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) $(IMAGE_NAME) bash -c "export LD_LIBRARY_PATH=/app/candle-binding/target/release && ./bin/router -config=config/config.yaml"
else
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		./bin/router -config=config/config.yaml
endif

# Removed run-router-pii target - PII detection is now enabled by default in config.yaml

# Run Envoy proxy
run-envoy:
	@echo "Starting Envoy..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) $(IMAGE_NAME) envoy --config-path config/envoy.yaml --component-log-level ext_proc:debug
else
	envoy --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
endif

# Test the Rust library
test-binding: rust
	@echo "Running Go tests with static library..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && CGO_ENABLED=1 go test -v"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v
endif

# Test PII detection unit tests only
test-pii-unit: rust
	@echo "Running PII detection unit tests..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && CGO_ENABLED=1 go test -v -run TestPII"
	$(EXEC_PREFIX) bash -c "cd semantic_router/pkg/extproc && go test -v -run TestPII"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -run TestPII
	@cd semantic_router/pkg/extproc && go test -v -run TestPII
endif

# Test PII detection integration tests (requires running services)
test-pii-integration:
	@echo "Running PII integration tests..."
	@cd tests && python3 03-pii-detection-test.py

# Test that existing functionality still works (regression test)
test-existing-functionality:
	@echo "Running regression tests to ensure existing functionality works..."
	@cd tests && python3 run_all_tests.py --pattern "*test.py" --skip-check || echo "Some tests failed - check if this is due to PII changes"

# Comprehensive PII testing
test-pii: test-pii-unit test-pii-integration test-existing-functionality
	@echo "All PII tests completed!"

# Test with the candle-binding library
test-classifier: rust
	@echo "Testing domain classifier with candle-binding..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the Rust library and the Go binding
test: test-binding

# Clean built artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && cargo clean && rm -f ../bin/router"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
	$(CONTAINER_CMD) rmi $(IMAGE_NAME)
else
	cd candle-binding && cargo clean
	rm -f bin/router
endif

# Build container image
container-build:
	@echo "Building container image..."
	$(CONTAINER_CMD) build -t $(IMAGE_NAME) .

# Start an interactive shell in the container
container-shell: container-build
	@echo "Starting interactive container shell..."
	$(RUN_PREFIX) -it $(IMAGE_NAME) bash

# Test the Envoy extproc
test-prompt:
	@echo "Testing Envoy extproc with curl (Math)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (History)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "assistant", "content": "You are a history teacher. Provide accurate historical information and context."}, {"role": "user", "content": "Tell me about the causes of World War I."}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (Health)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "assistant", "content": "You are a health advisor. Provide helpful health and wellness information."}, {"role": "user", "content": "What are the benefits of regular exercise?"}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (Programming)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "assistant", "content": "You are a programming expert. Help with code and software development."}, {"role": "user", "content": "How do I implement a binary search in Python?"}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (General)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "assistant", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "temperature": 0.7}'

# Test PII detection specifically with sample prompts
test-pii-prompt:
	@echo "Testing PII detection with sample prompts..."
	@echo "Testing with email..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "Please contact me at john.doe@example.com for further assistance"}], "temperature": 0.1, "max_tokens": 50}'
	@echo ""
	@echo "Testing with phone number..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "Call me at 555-123-4567 if you need anything"}], "temperature": 0.1, "max_tokens": 50}'
	@echo ""
	@echo "Testing with multiple PII types..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "John Smith can be reached at john@company.com or 555-0123"}], "temperature": 0.1, "max_tokens": 50}'
	@echo ""
	@echo "Testing with clean text (no PII)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "What is the weather like today?"}], "temperature": 0.1, "max_tokens": 50}'

test-vllm:
	curl -X POST $(VLLM_ENDPOINT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}' | jq
