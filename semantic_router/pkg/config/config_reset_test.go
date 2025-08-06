package config

import "sync"

// ResetConfig resets the singleton config for testing purposes
// This is needed to ensure test isolation
func ResetConfig() {
	configOnce = sync.Once{}
	config = nil
	configErr = nil
}