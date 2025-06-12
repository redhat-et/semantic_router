#!/usr/bin/env python3
"""
Persistent Classification Service
Runs as an HTTP service to avoid model loading overhead on each request.
"""

import json
import sys
import time
import logging
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from dual_classifier import DualClassifier
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ClassificationService:
    """Persistent classification service with model caching."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", port: int = 8888):
        self.device = device
        self.model_path = model_path
        self.port = port
        self.model = None
        self.category_mapping = None
        self.load_time = None
        
        # Load the model once at startup
        self._load_model()
    
    def _load_model(self):
        """Load the dual classifier model once at startup."""
        start_time = time.time()
        try:
            # Auto-detect model path if not provided
            if not self.model_path:
                logger.info("Auto-detecting dual classifier model...")
                model_path = self._auto_detect_model()
                if not model_path:
                    raise RuntimeError("No trained dual classifier model found")
                self.model_path = model_path
            
            # Load training config
            config_path = Path(self.model_path) / "training_config.json"
            if not config_path.exists():
                raise RuntimeError(f"No training config found at {config_path}")
            
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            
            # Get category information
            if 'categories' not in training_config:
                raise RuntimeError("No category information in training config")
            
            categories = training_config['categories']
            num_categories = categories['num_categories']
            self.category_mapping = categories
            
            logger.info(f"Loading model from: {self.model_path}")
            logger.info(f"Categories: {num_categories}")
            logger.info(f"Category mapping: {list(categories['category_to_id'].keys())}")
            
            # Load the model
            self.model = DualClassifier.from_pretrained(self.model_path, num_categories)
            self.model.eval()
            
            self.load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _auto_detect_model(self) -> Optional[str]:
        """Auto-detect the best available model."""
        # Check project root
        finetune_path = Path("finetune-model")
        if finetune_path.exists():
            config_path = finetune_path / "training_config.json"
            if config_path.exists():
                logger.info(f"Found model in finetune-model directory")
                return str(finetune_path.absolute())
        
        # Check parent directory (if running from dual_classifier/)
        finetune_path_parent = Path("../finetune-model")
        if finetune_path_parent.exists():
            config_path = finetune_path_parent / "training_config.json"
            if config_path.exists():
                logger.info(f"Found model in parent finetune-model directory")
                return str(finetune_path_parent.absolute())
        
        logger.error("No finetune-model directory found")
        return None
    
    def classify_text(self, text: str, mode: str = "dual") -> Dict[str, Any]:
        """Classify text using the loaded model."""
        start_time = time.time()
        
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            # Get predictions from the model
            category_probs, pii_probs = self.model.predict(text)
            
            # Process category prediction
            category_id = category_probs.argmax().item()
            category_confidence = category_probs.max().item()
            category_name = self.category_mapping['id_to_category'][str(category_id)]
            
            # Create category scores dictionary
            category_scores = {}
            for cat_id, cat_name in self.category_mapping['id_to_category'].items():
                category_scores[cat_name] = category_probs[0][int(cat_id)].item()
            
            # Process PII prediction
            pii_predictions = pii_probs.argmax(dim=-1)
            has_pii = (pii_predictions == 1).any().item()
            pii_confidence = pii_probs[:, :, 1].max().item()
            
            # Analyze PII tokens
            tokens = text.split()
            pii_tokens = []
            
            if has_pii and len(tokens) > 0:
                actual_length = min(len(tokens), pii_predictions.shape[1])
                for i in range(actual_length):
                    if pii_predictions[0][i].item() == 1:
                        if i < len(tokens):
                            pii_tokens.append({
                                "token": tokens[i],
                                "position": i,
                                "is_pii": True,
                                "confidence": pii_probs[0][i][1].item()
                            })
            
            inference_time = time.time() - start_time
            
            # Return results based on mode
            result = {
                "success": True,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_load_time_s": self.load_time,
                "results": [{
                    "text": text,
                    "category": {
                        "predicted_category": category_name,
                        "confidence": category_confidence,
                        "probabilities": category_scores
                    }
                }]
            }
            
            if mode in ["dual", "pii"]:
                result["results"][0]["pii"] = {
                    "has_pii": has_pii,
                    "pii_token_count": len(pii_tokens),
                    "total_tokens": len(tokens),
                    "tokens": pii_tokens,
                    "confidence": pii_confidence
                }
            
            return result
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}"
            }

class ClassificationRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for classification requests."""
    
    def __init__(self, *args, classification_service=None, **kwargs):
        self.classification_service = classification_service
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests for text classification."""
        try:
            # Parse the request
            if self.path != '/classify':
                self.send_error(404, "Not Found")
                return
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            
            # Parse JSON
            request_data = json.loads(request_body)
            text = request_data.get('text', '')
            mode = request_data.get('mode', 'dual')
            
            if not text:
                self.send_error(400, "Missing 'text' parameter")
                return
            
            # Classify the text
            result = self.classification_service.classify_text(text, mode)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(result)
            self.wfile.write(response_json.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def do_GET(self):
        """Handle GET requests for health checks."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            health_data = {
                "status": "healthy",
                "model_loaded": self.classification_service.model is not None,
                "model_path": self.classification_service.model_path,
                "categories": len(self.classification_service.category_mapping['id_to_category']) if self.classification_service.category_mapping else 0
            }
            
            self.wfile.write(json.dumps(health_data).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")

def create_handler(classification_service):
    """Create a request handler with the classification service."""
    def handler(*args, **kwargs):
        return ClassificationRequestHandler(*args, classification_service=classification_service, **kwargs)
    return handler

def main():
    """Start the classification service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persistent Classification Service")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--model-path", type=str, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    try:
        # Create classification service
        logger.info(f"Starting classification service on port {args.port}")
        service = ClassificationService(
            model_path=args.model_path,
            device=args.device,
            port=args.port
        )
        
        # Create HTTP server
        handler = create_handler(service)
        server = HTTPServer(('localhost', args.port), handler)
        
        logger.info(f"🚀 Classification service ready on http://localhost:{args.port}")
        logger.info("Health check: /health")
        logger.info("Classification: POST /classify")
        
        # Start server
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("Shutting down classification service...")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 