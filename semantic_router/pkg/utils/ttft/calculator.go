package ttft

import (
	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
)

// Calculator handles TTFT (Time To First Token) calculations
type Calculator struct {
	GPUConfig config.GPUConfig
}

// NewCalculator creates a new TTFT calculator
func NewCalculator(gpuConfig config.GPUConfig) *Calculator {
	return &Calculator{
		GPUConfig: gpuConfig,
	}
}

// ComputeBaseTTFT computes base TTFT for a model using the formula based on
// https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/
// TTFT = (2*N*b*s)/(FLOPs) + (2*N)/(HBM)
// Parameters are loaded from config: model-specific (N, b, s) and GPU-specific (FLOPs, HBM)
func (c *Calculator) ComputeBaseTTFT(modelName string, cfg *config.RouterConfig) float64 {
	// Get model-specific parameters from config
	defaultParamCount := 7e9    // Default to 7B if unknown
	defaultBatchSize := 512.0   // Default batch size
	defaultContextSize := 256.0 // Default context size

	// Get model parameters
	N := cfg.GetModelParamCount(modelName, defaultParamCount)
	b := cfg.GetModelBatchSize(modelName, defaultBatchSize)
	s := cfg.GetModelContextSize(modelName, defaultContextSize)

	// Get GPU parameters from config
	FLOPs := c.GPUConfig.FLOPS
	HBM := c.GPUConfig.HBM

	prefillCompute := 2 * N * b * s
	prefillMemory := 2 * N

	TTFT := (prefillCompute/FLOPs + prefillMemory/HBM) * 1000 // ms
	return TTFT
}

// InitializeModelTTFT initializes TTFT map for all models in config
func (c *Calculator) InitializeModelTTFT(cfg *config.RouterConfig) map[string]float64 {
	modelTTFT := make(map[string]float64)

	for _, cat := range cfg.Categories {
		for _, modelScore := range cat.ModelScores {
			if _, ok := modelTTFT[modelScore.Model]; !ok {
				modelTTFT[modelScore.Model] = c.ComputeBaseTTFT(modelScore.Model, cfg)
			}
		}
	}

	if cfg.DefaultModel != "" {
		if _, ok := modelTTFT[cfg.DefaultModel]; !ok {
			modelTTFT[cfg.DefaultModel] = c.ComputeBaseTTFT(cfg.DefaultModel, cfg)
		}
	}

	return modelTTFT
}
