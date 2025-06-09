package pii

import (
	"log"

	"github.com/redhat-et/semantic_route/semantic_router/pkg/config"
)

// PolicyChecker handles PII policy validation
type PolicyChecker struct {
	ModelConfigs map[string]config.ModelParams
}

// NewPolicyChecker creates a new PII policy checker
func NewPolicyChecker(modelConfigs map[string]config.ModelParams) *PolicyChecker {
	return &PolicyChecker{
		ModelConfigs: modelConfigs,
	}
}

// CheckPolicy checks if the detected PII types are allowed for the given model
func (pc *PolicyChecker) CheckPolicy(model string, detectedPII []string) (bool, []string, error) {
	modelConfig, exists := pc.ModelConfigs[model]
	if !exists {
		// If no specific config, allow by default
		log.Printf("No PII policy found for model %s, allowing request", model)
		return true, nil, nil
	}

	policy := modelConfig.PIIPolicy
	var deniedPII []string

	for _, piiType := range detectedPII {
		if piiType == "NO_PII" {
			continue // Skip non-PII content
		}

		// If allow_by_default is true, all PII types are allowed
		if policy.AllowByDefault {
			continue
		}

		// If allow_by_default is false, check if this PII type is explicitly allowed
		isAllowed := false
		for _, allowedPII := range policy.PIITypes {
			if allowedPII == piiType {
				isAllowed = true
				break
			}
		}

		if !isAllowed {
			deniedPII = append(deniedPII, piiType)
		}
	}

	if len(deniedPII) > 0 {
		log.Printf("PII policy violation for model %s: denied PII types %v", model, deniedPII)
		return false, deniedPII, nil
	}

	log.Printf("PII policy check passed for model %s", model)
	return true, nil, nil
}

// FilterModelsForPII filters the list of candidate models based on PII policy compliance
func (pc *PolicyChecker) FilterModelsForPII(candidateModels []string, detectedPII []string) []string {
	var allowedModels []string

	for _, model := range candidateModels {
		allowed, _, err := pc.CheckPolicy(model, detectedPII)
		if err != nil {
			log.Printf("Error checking PII policy for model %s: %v", model, err)
			continue
		}
		if allowed {
			allowedModels = append(allowedModels, model)
		}
	}

	return allowedModels
}

// ExtractAllContent extracts all content from user and non-user messages for PII analysis
func ExtractAllContent(userContent string, nonUserMessages []string) []string {
	var allContent []string
	if userContent != "" {
		allContent = append(allContent, userContent)
	}
	allContent = append(allContent, nonUserMessages...)
	return allContent
}
