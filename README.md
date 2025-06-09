# LLM Semantic Router

An Envoy External Processor (ExtProc) that acts as an external **Mixture-of-Models (MoM)** router. It intelligently directs OpenAI API requests to the most suitable backend model from a defined pool based on semantic understanding of the request's intent. This is achieved using BERT classification. Conceptually similar to Mixture-of-Experts (MoE) which lives *within* a model, this system selects the best *entire model* for the nature of the task.

As such, the overall inference accuracy is improved by using a pool of models that are better suited for different types of tasks:

![Model Accuracy](./docs/category_accuracies.png)

The detailed design doc can be found [here](https://docs.google.com/document/d/1BwwRxdf74GuCdG1veSApzMRMJhXeUWcw0wH9YRAmgGw/edit?usp=sharing).

The screenshot below shows the LLM Router dashboard in Grafana.

![LLM Router Dashboard](./docs/grafana_screenshot.png)

The router is implemented in two ways: Golang (with Rust FFI based on Candle) and Python. Benchmarking will be conducted to determine the best implementation.

## Usage

### Build the Project

```bash
make build
```

### Create Dataset (if needed)

```bash
python dual_classifier/create_enhanced_dataset.py
```

### Train Model (if needed)

```bash
python dual_classifier/train_enhanced_model.py --create-dataset --training-strength quick --max-length 256
```

### Start Services (2 terminals)

#### Terminal 1: Run the Envoy Proxy
This listens for incoming requests and uses the ExtProc filter.
```bash
make run-envoy
```

#### Terminal 2: Run the Semantic Router
This builds the Rust binding and the Go router, then starts the ExtProc gRPC server that Envoy communicates with. Includes PII detection via the trained dual classifier.
```bash
make run-router
```

### Test the System

Once both Envoy and the router are running, you can test the routing logic using predefined prompts:

```bash
make test-prompt
```

This will send curl requests simulating different types of user prompts (Math, Creative Writing, General) to the Envoy endpoint (`http://localhost:8801`). The router should direct these to the appropriate backend model configured in `config/config.yaml`.

### Test PII Detection

To test Personally Identifiable Information (PII) detection capabilities:

**Note:** PII detection is enabled by default in `config/config.yaml`.

#### Unit Tests (No Envoy Required)

Test PII detection logic directly without external services:
```bash
make test-pii-unit
```

#### Integration Tests (Requires Envoy + Router)

Make sure both services are running:
```bash
make run-envoy  # In one terminal
make run-router # In another terminal
```

Then test PII detection with sample prompts containing various types of PII:
```bash
make test-pii-prompt
```

This will test detection of:
- Email addresses (`john.doe@example.com`)
- Phone numbers (`555-123-4567`) 
- Multiple PII types together
- Clean text (no PII) as a control

#### Comprehensive PII Testing

Run all PII tests (unit tests, integration tests, and regression tests):
```bash
make test-pii
```

**Note:** The integration and comprehensive tests require both Envoy and the router to be running.

The PII detection system uses BERT-based classification to identify and optionally sanitize sensitive information before routing requests to backend models.

## Testing

A comprehensive test suite is available to validate the functionality of the Semantic Router. The tests follow the data flow through the system, from client request to routing decision.

### Prerequisites

Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

### Running Tests

Make sure both the Envoy proxy and Router are running:
```bash
make run-envoy  # In one terminal
make run-router  # In another terminal
```

Run all tests in sequence:
```bash
python tests/run_all_tests.py
```

Run a specific test:
```bash
python tests/00-client-request-test.py
```

Run only tests matching a pattern:
```bash
python tests/run_all_tests.py --pattern "0*-*.py"
```

Check if services are running without running tests:
```bash
python tests/run_all_tests.py --check-only
```

The test suite includes:
- Basic client request tests
- Envoy ExtProc interaction tests
- Router classification tests
- Semantic cache tests
- Category-specific tests
- Metrics validation tests

