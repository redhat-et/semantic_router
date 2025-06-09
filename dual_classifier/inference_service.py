#!/usr/bin/env python3
"""
Dual Classifier Inference Service
Provides a simple HTTP server or command-line interface for using trained dual classifier models.
Can be called from Go code to perform both category classification and PII detection.
"""

import torch
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from dual_classifier import DualClassifier
# Removed model_detector dependency - using fixed finetune-model path

class DualClassifierInference:
    """
    Inference service for the dual classifier model.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to the model directory. If None, auto-detect best model.
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        self.device = self._get_device(device)
        self.model = None
        self.category_mapping = None
        
        # Auto-detect model if not provided
        if model_path is None:
            model_path = self._auto_detect_model()
            if model_path is None:
                raise ValueError("No trained dual classifier models found")
        
        self.model_path = model_path
        self._load_model()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the best device to use."""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _auto_detect_model(self) -> Optional[str]:
        """Auto-detect the best available model."""
        # Check for finetune-model directory (primary model location)
        finetune_paths = [
            Path("../finetune-model"),  # From dual_classifier directory
            Path("finetune-model"),     # From project root
            Path("./finetune-model")    # Current directory
        ]
        
        for finetune_path in finetune_paths:
            if finetune_path.exists():
                config_path = finetune_path / "training_config.json"
                if config_path.exists():
                    print(f"Found model in {finetune_path}", file=sys.stderr)
                    return str(finetune_path.absolute())
        
        print("No finetune-model directory found", file=sys.stderr)
        return None
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Read training config to get number of categories
            training_config_path = os.path.join(self.model_path, "training_config.json")
            with open(training_config_path, 'r') as f:
                training_config = json.load(f)
            
            num_categories = training_config['categories']['num_categories']
            self.category_mapping = training_config['categories']
            
            # Load the model
            self.model = DualClassifier.from_pretrained(self.model_path, num_categories)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded dual classifier from {self.model_path}", file=sys.stderr)
            print(f"Categories: {list(self.category_mapping['category_to_id'].keys())}", file=sys.stderr)
            print(f"Device: {self.device}", file=sys.stderr)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def classify_category(self, text: Union[str, List[str]]) -> Dict:
        """
        Classify text into categories.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Dictionary with category predictions
        """
        if isinstance(text, str):
            text = [text]
        
        with torch.no_grad():
            category_probs, _ = self.model.predict(text, device=self.device)
            
            # Convert to readable results
            results = []
            for i, probs in enumerate(category_probs):
                predicted_idx = torch.argmax(probs).item()
                predicted_category = self.category_mapping['id_to_category'][str(predicted_idx)]
                confidence = probs[predicted_idx].item()
                
                result = {
                    'text': text[i],
                    'predicted_category': predicted_category,
                    'confidence': confidence,
                    'category_probabilities': {
                        category: probs[idx].item() 
                        for category, idx in self.category_mapping['category_to_id'].items()
                    }
                }
                results.append(result)
        
        return {'results': results}
    
    def detect_pii(self, text: Union[str, List[str]], threshold: float = 0.5) -> Dict:
        """
        Detect PII in text.
        
        Args:
            text: Input text or list of texts
            threshold: Threshold for PII detection
            
        Returns:
            Dictionary with PII detection results
        """
        if isinstance(text, str):
            text = [text]
        
        with torch.no_grad():
            _, pii_probs = self.model.predict(text, device=self.device)
            
            results = []
            for i, probs in enumerate(pii_probs):
                # Get PII probabilities (index 1 is PII, index 0 is non-PII)
                pii_scores = probs[:, 1]  # Shape: (seq_len,)
                
                # Find tokens above threshold
                pii_tokens = (pii_scores > threshold).cpu().numpy()
                
                # Get tokenized text for alignment
                encoded = self.model.encode_text(text[i], device=self.device)
                tokens = self.model.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
                
                # Build result
                token_results = []
                detected_pii_count = 0
                for j, (token, is_pii, score) in enumerate(zip(tokens, pii_tokens, pii_scores)):
                    token_result = {
                        'token': token,
                        'position': j,
                        'is_pii': bool(is_pii),
                        'confidence': float(score)
                    }
                    token_results.append(token_result)
                    if is_pii:
                        detected_pii_count += 1
                
                result = {
                    'text': text[i],
                    'has_pii': detected_pii_count > 0,
                    'pii_token_count': detected_pii_count,
                    'total_tokens': len(tokens),
                    'tokens': token_results
                }
                results.append(result)
        
        return {'results': results}
    
    def classify_dual(self, text: Union[str, List[str]], pii_threshold: float = 0.5) -> Dict:
        """
        Perform both category classification and PII detection.
        
        Args:
            text: Input text or list of texts
            pii_threshold: Threshold for PII detection
            
        Returns:
            Dictionary with both classification and PII detection results
        """
        category_results = self.classify_category(text)
        pii_results = self.detect_pii(text, pii_threshold)
        
        # Combine results
        combined_results = []
        for cat_result, pii_result in zip(category_results['results'], pii_results['results']):
            combined_result = {
                'text': cat_result['text'],
                'category': {
                    'predicted_category': cat_result['predicted_category'],
                    'confidence': cat_result['confidence'],
                    'probabilities': cat_result['category_probabilities']
                },
                'pii': {
                    'has_pii': pii_result['has_pii'],
                    'pii_token_count': pii_result['pii_token_count'],
                    'total_tokens': pii_result['total_tokens'],
                    'tokens': pii_result['tokens']
                }
            }
            combined_results.append(combined_result)
        
        return {'results': combined_results}

def main():
    """
    Command-line interface for the dual classifier inference service.
    """
    parser = argparse.ArgumentParser(description='Dual Classifier Inference Service')
    parser.add_argument('--model-path', help='Path to model directory')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--mode', choices=['category', 'pii', 'dual'], default='dual',
                       help='Classification mode')
    parser.add_argument('--pii-threshold', type=float, default=0.5,
                       help='Threshold for PII detection')
    parser.add_argument('--text', help='Text to classify')
    parser.add_argument('--file', help='File containing text to classify')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference service
        service = DualClassifierInference(args.model_path, args.device)
        
        # Get input text
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r') as f:
                text = f.read().strip()
        else:
            # Read from stdin
            text = sys.stdin.read().strip()
        
        if not text:
            print("Error: No input text provided", file=sys.stderr)
            sys.exit(1)
        
        # Perform classification
        if args.mode == 'category':
            result = service.classify_category(text)
        elif args.mode == 'pii':
            result = service.detect_pii(text, args.pii_threshold)
        else:  # dual
            result = service.classify_dual(text, args.pii_threshold)
        
        # Output result
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            for i, res in enumerate(result['results']):
                if i > 0:
                    print()
                print(f"Text: {res['text']}")
                
                if 'category' in res:
                    cat = res['category']
                    print(f"Category: {cat['predicted_category']} (confidence: {cat['confidence']:.3f})")
                elif 'predicted_category' in res:
                    print(f"Category: {res['predicted_category']} (confidence: {res['confidence']:.3f})")
                
                if 'pii' in res:
                    pii = res['pii']
                    print(f"PII detected: {pii['has_pii']} ({pii['pii_token_count']}/{pii['total_tokens']} tokens)")
                elif 'has_pii' in res:
                    print(f"PII detected: {res['has_pii']} ({res['pii_token_count']}/{res['total_tokens']} tokens)")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 