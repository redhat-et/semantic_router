#!/usr/bin/env python3
"""
Enhanced Bridge for Go Integration
Uses the trained dual-purpose classification model for both category classification and PII detection.
"""

import json
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional

from dual_classifier import DualClassifier
# Removed model_detector dependency - using fixed finetune-model path

class EnhancedBridge:
    """
    Enhanced bridge that uses the trained dual classifier model.
    Provides both category classification and PII detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.category_mapping = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the dual classifier model."""
        try:
            # If no model path specified, auto-detect
            if not self.model_path:
                print("Auto-detecting dual classifier model...", file=sys.stderr)
                model_path = self._auto_detect_model()
                if not model_path:
                    raise RuntimeError("No trained dual classifier model found")
                self.model_path = model_path
            
            # Load training config to get categories
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
            
            print(f"Loading model from: {self.model_path}", file=sys.stderr)
            print(f"Categories: {num_categories}", file=sys.stderr)
            print(f"Category mapping: {list(categories['category_to_id'].keys())}", file=sys.stderr)
            
            # Load the model
            self.model = DualClassifier.from_pretrained(self.model_path, num_categories)
            self.model.eval()
            
            print(f"✅ Enhanced dual classifier loaded successfully", file=sys.stderr)
            
        except Exception as e:
            print(f"❌ Failed to load enhanced dual classifier: {e}", file=sys.stderr)
            raise
    
    def _auto_detect_model(self) -> Optional[str]:
        """Auto-detect the best available model."""
        # Check the expected location first (project root)
        finetune_path = Path("../finetune-model")
        if finetune_path.exists():
            config_path = finetune_path / "training_config.json"
            if config_path.exists():
                print(f"Found model in finetune-model directory", file=sys.stderr)
                return str(finetune_path.absolute())
        
        # Also check if we're already in project root
        finetune_path_root = Path("finetune-model")
        if finetune_path_root.exists():
            config_path = finetune_path_root / "training_config.json"
            if config_path.exists():
                print(f"Found model in project root finetune-model directory", file=sys.stderr)
                return str(finetune_path_root.absolute())
        
        # No finetune-model found
        print("No finetune-model directory found", file=sys.stderr)
        return None
    
    def classify_text(self, text: str, mode: str = "dual") -> Dict[str, Any]:
        """
        Classify text using the trained dual classifier.
        
        Args:
            text: Input text to classify
            mode: Classification mode ("category", "pii", or "dual")
            
        Returns:
            Dictionary with classification results
        """
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
            
            # Process PII prediction (simplified approach)
            # Since we're using token-level classification, check if any tokens are classified as PII
            pii_predictions = pii_probs.argmax(dim=-1)  # Shape: (batch_size, seq_len)
            has_pii = (pii_predictions == 1).any().item()  # Check if any token is classified as PII
            pii_confidence = pii_probs[:, :, 1].max().item()  # Max confidence for PII class
            
            # For token-level PII, we need to analyze which tokens were classified as PII
            tokens = text.split()
            pii_tokens = []
            
            if has_pii and len(tokens) > 0:
                # Get the actual sequence length used by the model
                actual_length = min(len(tokens), pii_predictions.shape[1])
                
                for i in range(actual_length):
                    if pii_predictions[0][i].item() == 1:  # Token classified as PII
                        if i < len(tokens):
                            pii_tokens.append({
                                "token": tokens[i],
                                "position": i,
                                "is_pii": True,
                                "confidence": pii_probs[0][i][1].item()
                            })
            
            # Return results based on mode
            if mode == "dual":
                return {
                    "success": True,
                    "results": [{
                        "text": text,
                        "category": {
                            "predicted_category": category_name,
                            "confidence": category_confidence,
                            "probabilities": category_scores
                        },
                        "pii": {
                            "has_pii": has_pii,
                            "pii_token_count": len(pii_tokens),
                            "total_tokens": len(tokens),
                            "tokens": pii_tokens
                        }
                    }]
                }
            
            elif mode == "category":
                return {
                    "success": True,
                    "results": [{
                        "text": text,
                        "category": {
                            "predicted_category": category_name,
                            "confidence": category_confidence,
                            "probabilities": category_scores
                        }
                    }]
                }
            
            elif mode == "pii":
                return {
                    "success": True,
                    "results": [{
                        "text": text,
                        "pii": {
                            "has_pii": has_pii,
                            "pii_token_count": len(pii_tokens),
                            "total_tokens": len(tokens),
                            "tokens": pii_tokens,
                            "confidence": pii_confidence
                        }
                    }]
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown mode: {mode}. Use 'category', 'pii', or 'dual'"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}"
            }


def main():
    """Command line interface compatible with the existing Go bridge expectations"""
    parser = argparse.ArgumentParser(description="Enhanced Bridge for Trained Dual Classifier")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--mode", type=str, default="dual", 
                       choices=["category", "pii", "dual"],
                       help="Classification mode")
    parser.add_argument("--json", action="store_true", default=True,
                       help="Output JSON format (default)")
    parser.add_argument("--model-path", type=str, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    try:
        bridge = EnhancedBridge(model_path=args.model_path, device=args.device)
        result = bridge.classify_text(args.text, args.mode)
        
        # Output JSON to stdout for Go to parse
        print(json.dumps(result))
        
    except Exception as e:
        # Output error in expected format
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main() 