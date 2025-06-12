package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redhat-et/semantic_route/semantic_router/pkg/extproc"
)

func main() {
	// Parse command-line flags
	var (
		configPath  = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port        = flag.Int("port", 50051, "Port to listen on")
		metricsPort = flag.Int("metrics-port", 9190, "Port for Prometheus metrics")
	)
	flag.Parse()

	// Check if config file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		log.Fatalf("Config file not found: %s", *configPath)
	}

	// Start metrics server with classification endpoint
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		http.HandleFunc("/classification", extproc.GetClassificationHandler())
		metricsAddr := fmt.Sprintf(":%d", *metricsPort)
		log.Printf("Starting metrics server on %s", metricsAddr)
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	// Create and start the server
	server, err := extproc.NewServer(*configPath, *port)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	log.Printf("Starting LLM Semantic Router ExtProc with config: %s", *configPath)
	if err := server.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
