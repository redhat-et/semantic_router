#!/usr/bin/env python3
"""
Live Demonstration System for Dual-Purpose Semantic Router

This script provides a comprehensive demo showing:
1. Real-time category classification 
2. Real-time PII detection
3. Routing decisions based on both
4. Live user input testing

Perfect for demonstrating the complete semantic router system!
"""

import os
import sys
import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Import our modules
from dual_classifier import DualClassifier
from hardware_detector import detect_and_configure
from enhanced_trainer import EnhancedDualTaskTrainer
from missing_files_detector import check_missing_files

class LiveDemoRouter:
    """
    Live demonstration system for dual-purpose semantic routing.
    Combines category classification and PII detection for real-time routing.
    """
    
    def __init__(self):
        print("🚀 Initializing Live Demo Router...")
        
        # Initialize with default categories (will be updated when model is loaded)
        self.categories = {}
        self.num_categories = 0
        
        # Enhanced PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b',
            'name_formal': r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'account_id': r'\b(?:Account|ID|REF)[-:]?\s*[A-Z0-9]{6,12}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b'
        }
        
        # Generic routing rules (will be updated based on detected categories)
        self.routing_rules = {}
        
        # Load model if available
        self.model = None
        self.model_info = None
        self.load_model()
        
        # Setup routing rules after model is loaded
        self._setup_routing_rules()
        
        print("✅ Live Demo Router initialized!")
        if self.categories:
            print(f"📂 Detected {self.num_categories} categories: {list(self.categories.values())}")
        else:
            print("⚠️ No model loaded - using rule-based classification")
    
    def load_model(self):
        """Load the dual classifier model if available."""
        try:
            # Check for missing model files first
            print("🔍 Checking for available model files...")
            check_missing_files("demo")
            print()
            
            # Scan for available models
            available_models = self._scan_available_models()
            
            if not available_models:
                print("⚠️ No trained models found. Using rule-based classification.")
                return
            
            # Present model selection menu
            selected_model = self._select_model_interactive(available_models)
            
            if not selected_model:
                print("⚠️ No model selected. Using rule-based classification.")
                return
            
            # Load the selected model
            self._load_selected_model(selected_model)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Falling back to rule-based classification.")
    
    def _scan_available_models(self) -> List[Dict]:
        """Scan for available trained models in various directories."""
        available_models = []
        current_dir = Path(".")
        
        # Check for enhanced training directories
        for training_dir in current_dir.glob("enhanced_training_*"):
            if training_dir.is_dir():
                models = self._scan_training_directory(training_dir)
                available_models.extend(models)
        
        # Check for legacy locations
        legacy_models = [
            {
                'path': "dual_classifier_checkpoint.pth",
                'name': "Legacy Checkpoint",
                'type': "checkpoint",
                'directory': ".",
                'description': "Original training checkpoint"
            },
            {
                'path': "trained_model/model.pt", 
                'name': "Legacy Trained Model",
                'type': "model",
                'directory': "trained_model",
                'description': "Legacy trained model"
            }
        ]
        
        for model in legacy_models:
            if os.path.exists(model['path']):
                # Get file size and modification time
                stat = os.stat(model['path'])
                model['size'] = stat.st_size
                model['modified'] = stat.st_mtime
                available_models.append(model)
        
        return available_models
    
    def _scan_training_directory(self, training_dir: Path) -> List[Dict]:
        """Scan a training directory for available models."""
        models = []
        
        # Check for final model
        final_model_dir = training_dir / "final_model"
        if final_model_dir.exists():
            model_files = list(final_model_dir.glob("*.safetensors")) + list(final_model_dir.glob("model.pt")) + list(final_model_dir.glob("pytorch_model.bin"))
            if model_files:
                model_file = model_files[0]  # Take the first one found
                stat = model_file.stat()
                models.append({
                    'path': str(model_file),
                    'name': f"Final Model ({training_dir.name})",
                    'type': "final_model",
                    'directory': str(training_dir),
                    'description': f"Final trained model from {training_dir.name}",
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
        
        # Check for checkpoints
        checkpoints_dir = training_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Look for best model
            best_model = checkpoints_dir / "best_model.pt"
            if best_model.exists():
                stat = best_model.stat()
                models.append({
                    'path': str(best_model),
                    'name': f"Best Model ({training_dir.name})",
                    'type': "best_checkpoint",
                    'directory': str(training_dir),
                    'description': f"Best performing model from {training_dir.name}",
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
            
            # Look for epoch checkpoints
            epoch_checkpoints = sorted(checkpoints_dir.glob("epoch-*.pt"))
            if epoch_checkpoints:
                # Take the latest epoch
                latest_epoch = epoch_checkpoints[-1]
                stat = latest_epoch.stat()
                models.append({
                    'path': str(latest_epoch),
                    'name': f"Latest Epoch ({training_dir.name})",
                    'type': "epoch_checkpoint", 
                    'directory': str(training_dir),
                    'description': f"Latest epoch checkpoint from {training_dir.name}",
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
            
            # Look for step checkpoints
            step_checkpoints = sorted(checkpoints_dir.glob("checkpoint-step-*.pt"))
            if step_checkpoints:
                # Take the latest step
                latest_step = step_checkpoints[-1]
                stat = latest_step.stat()
                models.append({
                    'path': str(latest_step),
                    'name': f"Latest Step ({training_dir.name})",
                    'type': "step_checkpoint",
                    'directory': str(training_dir),
                    'description': f"Latest step checkpoint from {training_dir.name}",
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
        
        return models
    
    def _select_model_interactive(self, available_models: List[Dict]) -> Optional[Dict]:
        """Present an interactive menu for model selection."""
        if not available_models:
            return None
        
        print("\n🤖 Available Trained Models:")
        print("=" * 60)
        
        # Sort models by modification time (newest first)
        available_models.sort(key=lambda x: x.get('modified', 0), reverse=True)
        
        for i, model in enumerate(available_models, 1):
            # Format file size
            size_mb = model.get('size', 0) / (1024 * 1024)
            
            # Format modification time
            mod_time = ""
            if 'modified' in model:
                mod_time = datetime.fromtimestamp(model['modified']).strftime("%Y-%m-%d %H:%M")
            
            # Try to detect categories for this model
            temp_categories = self._detect_categories_from_model_info(model)
            category_info = ""
            if temp_categories:
                category_list = list(temp_categories.values())
                if len(category_list) <= 6:
                    category_info = f" - Categories: {', '.join(category_list)}"
                else:
                    category_info = f" - {len(category_list)} categories: {', '.join(category_list[:3])}..."
            
            print(f"{i:2d}. {model['name']}{category_info}")
            print(f"    📁 {model['description']}")
            print(f"    📄 {model['path']}")
            if size_mb > 0:
                print(f"    💾 Size: {size_mb:.1f} MB")
            if mod_time:
                print(f"    🕒 Modified: {mod_time}")
            print()
        
        print(f"{len(available_models) + 1:2d}. Use rule-based classification (no model)")
        print()
        
        while True:
            try:
                choice = input(f"🎯 Select a model (1-{len(available_models) + 1}) [1]: ").strip()
                
                if not choice:
                    choice = "1"
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(available_models):
                    selected = available_models[choice_num - 1]
                    print(f"✅ Selected: {selected['name']}")
                    return selected
                elif choice_num == len(available_models) + 1:
                    print("✅ Using rule-based classification")
                    return None
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_models) + 1}")
                    
            except ValueError:
                print(f"❌ Invalid input. Please enter a number 1-{len(available_models) + 1}")
            except KeyboardInterrupt:
                print("\n⚠️ Model selection cancelled. Using rule-based classification.")
                return None
    
    def _load_selected_model(self, model_info: Dict):
        """Load the selected model."""
        model_path = model_info['path']
        model_type = model_info['type']
        
        print(f"📦 Loading {model_info['name']}...")
        
        # Store model info for category detection
        self.model_info = model_info
        
        # Try to detect categories before loading the model
        self._setup_categories(model_info)
        
        # Detect hardware capabilities
        capabilities, _ = detect_and_configure()
        device = capabilities.device
        
        # Initialize model with correct number of categories
        if self.num_categories > 0:
            self.model = DualClassifier(num_categories=self.num_categories)
        else:
            # Fallback - will be corrected after loading
            self.model = DualClassifier(num_categories=4)
        
        try:
            # Load model based on type
            if model_type == "final_model":
                # Try loading as saved model directory first
                try:
                    self.model = DualClassifier.from_pretrained(
                        model_info['directory'] + "/final_model",
                        num_categories=self.num_categories if self.num_categories > 0 else None
                    )
                    print(f"✅ Loaded as pretrained model")
                except:
                    # Fall back to checkpoint loading
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"✅ Loaded as checkpoint")
            else:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        
                        # Show training info if available
                        if 'epoch' in checkpoint:
                            print(f"   📊 Trained for {checkpoint['epoch']} epochs")
                        if 'best_val_score' in checkpoint:
                            print(f"   🎯 Best validation score: {checkpoint['best_val_score']:.4f}")
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
            
            # If categories weren't detected before, try again with loaded model
            if not self.categories:
                self._setup_categories(model_info)
            
            self.model.to(device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully on {device}")
            print(f"   🎯 Model: {model_info['name']}")
            print(f"   📁 Path: {model_path}")
            print(f"   📂 Categories ({self.num_categories}): {list(self.categories.values())}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️ Falling back to rule-based classification")
            self.model = None
            # Reset categories for rule-based fallback
            self.categories = {
                0: "business", 1: "science", 2: "technology", 3: "other"
            }
            self.num_categories = len(self.categories)
    
    def detect_pii(self, text: str) -> Dict:
        """
        Detect PII in text using regex patterns.
        
        Returns:
            Dict with PII information including types found and masked text
        """
        words = text.split()
        word_labels = [0] * len(words)
        pii_found = {}
        
        # Check each PII pattern
        for pii_type, pattern in self.pii_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                pii_found[pii_type] = []
                
                for match in matches:
                    pii_found[pii_type].append(match.group())
                    
                    # Mark words that contain PII
                    start_char = match.start()
                    end_char = match.end()
                    
                    char_pos = 0
                    for i, word in enumerate(words):
                        word_start = char_pos
                        word_end = char_pos + len(word)
                        
                        if (word_start < end_char and word_end > start_char):
                            word_labels[i] = 1
                        
                        char_pos = word_end + 1
        
        # Create masked version
        masked_text = text
        for pii_type, values in pii_found.items():
            for value in values:
                mask = f"[{pii_type.upper()}]"
                masked_text = masked_text.replace(value, mask)
        
        has_pii = len(pii_found) > 0
        
        return {
            'has_pii': has_pii,
            'pii_types': list(pii_found.keys()),
            'pii_details': pii_found,
            'word_labels': word_labels,
            'masked_text': masked_text,
            'pii_count': sum(len(v) for v in pii_found.values())
        }
    
    def classify_category(self, text: str) -> Dict:
        """
        Classify text into categories using model or rules.
        
        Returns:
            Dict with category information
        """
        if self.model and self.categories:
            # Use trained model
            try:
                device = next(self.model.parameters()).device
                
                # Use the model's proper tokenizer and encoding method
                encoded = self.model.encode_text(text, device=device)
                
                with torch.no_grad():
                    category_logits, _ = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"]
                    )
                    probabilities = torch.softmax(category_logits, dim=-1)
                    predicted_category = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_category].item()
                
                return {
                    'category_id': predicted_category,
                    'category_name': self.categories.get(predicted_category, 'unknown'),
                    'confidence': confidence,
                    'method': 'neural_model'
                }
                
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Fall back to rule-based
                pass
        
        # Rule-based classification with dynamic keyword mapping
        text_lower = text.lower()
        
        # Create keyword sets for known categories
        keyword_mappings = self._create_keyword_mappings()
        
        # Count keyword matches for each category
        scores = {}
        for category_id, category_name in self.categories.items():
            category_key = category_name.lower().replace(' ', '_')
            keywords = keyword_mappings.get(category_key, [])
            scores[category_id] = sum(1 for kw in keywords if kw in text_lower)
        
        # Add small base score for 'other' or generic categories
        for category_id, category_name in self.categories.items():
            if 'other' in category_name.lower() or 'general' in category_name.lower():
                scores[category_id] = max(scores.get(category_id, 0), 1)
        
        # Find best category
        if scores:
            best_category = max(scores.keys(), key=lambda k: scores[k])
            max_score = scores[best_category]
            
            # If no keywords matched, assign to 'other' or first category
            if max_score == 0:
                other_categories = [cid for cid, cname in self.categories.items() 
                                 if 'other' in cname.lower() or 'general' in cname.lower()]
                best_category = other_categories[0] if other_categories else 0
                max_score = 1
                
            confidence = min(max_score / max(1, len(text_lower.split())), 1.0)
        else:
            # Fallback if no categories
            best_category = 0
            confidence = 0.1
            scores = {0: 1}
        
        return {
            'category_id': best_category,
            'category_name': self.categories.get(best_category, 'unknown'),
            'confidence': confidence,
            'method': 'rule_based',
            'keyword_scores': scores
        }
    
    def _create_keyword_mappings(self) -> Dict[str, List[str]]:
        """Create keyword mappings for rule-based classification."""
        # Comprehensive keyword mappings for various domains
        mappings = {
            # Technology & Computer Science
            'technology': ['technology', 'tech', 'software', 'hardware', 'computer', 'digital', 'app', 'mobile'],
            'computer_science': ['programming', 'algorithm', 'machine learning', 'ai', 'artificial intelligence', 
                               'neural network', 'deep learning', 'python', 'java', 'code', 'software'],
            'engineering': ['engineering', 'design', 'construction', 'mechanical', 'electrical', 'civil',
                          'structural', 'system', 'manufacturing', 'automation', 'robotics'],
            
            # Business & Economics
            'business': ['business', 'corporate', 'company', 'profit', 'revenue', 'sales', 'customer',
                        'management', 'strategy', 'marketing', 'operations', 'enterprise', 'startup'],
            'economics': ['economy', 'economics', 'inflation', 'gdp', 'market', 'stock', 'investment', 
                         'finance', 'monetary', 'fiscal', 'trade', 'demand', 'supply', 'recession'],
            'finance': ['finance', 'financial', 'banking', 'investment', 'money', 'capital', 'asset'],
            
            # Sciences
            'science': ['science', 'scientific', 'research', 'study', 'analysis', 'data', 'experiment'],
            'physics': ['physics', 'quantum', 'relativity', 'mechanics', 'thermodynamics', 'energy',
                       'force', 'gravity', 'electromagnetic', 'particle', 'wave', 'momentum', 'mass'],
            'biology': ['biology', 'cell', 'organism', 'evolution', 'genetics', 'dna', 'species',
                       'ecosystem', 'protein', 'molecular', 'biochemistry', 'anatomy', 'physiology'],
            'chemistry': ['chemistry', 'chemical', 'molecule', 'atom', 'reaction', 'compound', 'element',
                         'periodic', 'organic', 'inorganic', 'catalyst', 'synthesis', 'bond', 'solution'],
            'math': ['math', 'mathematics', 'calculus', 'algebra', 'geometry', 'statistics', 'equation',
                    'theorem', 'proof', 'function', 'derivative', 'integral', 'probability', 'number'],
            'mathematics': ['math', 'mathematics', 'calculus', 'algebra', 'geometry', 'statistics', 'equation',
                           'theorem', 'proof', 'function', 'derivative', 'integral', 'probability', 'number'],
            
            # Social Sciences & Humanities
            'psychology': ['psychology', 'behavior', 'cognitive', 'mental', 'brain', 'emotion', 'learning',
                          'memory', 'personality', 'psychotherapy', 'psychological', 'therapy', 'mind'],
            'philosophy': ['philosophy', 'ethics', 'morality', 'metaphysics', 'epistemology', 'logic',
                          'existence', 'consciousness', 'truth', 'reality', 'kant', 'aristotle', 'plato'],
            'history': ['history', 'historical', 'ancient', 'medieval', 'war', 'civilization', 'culture',
                       'century', 'empire', 'revolution', 'dynasty', 'archaeological'],
            'law': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'contract', 'lawsuit',
                   'legislation', 'constitution', 'rights', 'justice', 'regulation', 'statute'],
            'legal': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'contract', 'lawsuit',
                     'legislation', 'constitution', 'rights', 'justice', 'regulation', 'statute'],
            
            # Health & Medicine
            'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'disease', 'treatment',
                      'patient', 'symptoms', 'diagnosis', 'therapy', 'healthcare', 'clinic', 'surgery'],
            'medical': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'disease', 'treatment',
                       'patient', 'symptoms', 'diagnosis', 'therapy', 'healthcare', 'clinic', 'surgery'],
            'medicine': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'disease', 'treatment',
                        'patient', 'symptoms', 'diagnosis', 'therapy', 'healthcare', 'clinic', 'surgery'],
            
            # News Categories (BBC, AG News, etc.)
            'sport': ['sport', 'sports', 'football', 'soccer', 'basketball', 'tennis', 'game', 'match', 
                     'team', 'player', 'athlete', 'championship', 'league', 'tournament'],
            'sports': ['sport', 'sports', 'football', 'soccer', 'basketball', 'tennis', 'game', 'match', 
                      'team', 'player', 'athlete', 'championship', 'league', 'tournament'],
            'politics': ['politics', 'political', 'government', 'election', 'vote', 'president', 'congress',
                        'senator', 'policy', 'democracy', 'republican', 'democrat', 'campaign'],
            'entertainment': ['entertainment', 'movie', 'film', 'music', 'celebrity', 'actor', 'actress',
                             'concert', 'show', 'television', 'tv', 'hollywood', 'culture', 'art'],
            'world': ['world', 'international', 'global', 'country', 'nation', 'foreign', 'diplomatic'],
            'education': ['education', 'school', 'university', 'student', 'teacher', 'learning', 'academic'],
            
            # Generic/Other
            'other': ['general', 'various', 'miscellaneous', 'different', 'other', 'common'],
            'general': ['general', 'various', 'miscellaneous', 'different', 'other', 'common']
        }
        
        return mappings
    
    def route_query(self, text: str) -> Dict:
        """
        Perform complete routing: category classification + PII detection + routing decision.
        
        Returns:
            Complete routing information
        """
        print(f"\n🔍 Processing query: \"{text}\"")
        
        # Step 1: Classify category
        category_result = self.classify_category(text)
        print(f"📂 Category: {category_result['category_name']} (confidence: {category_result['confidence']:.3f})")
        
        # Step 2: Detect PII
        pii_result = self.detect_pii(text)
        if pii_result['has_pii']:
            print(f"🔒 PII detected: {', '.join(pii_result['pii_types'])} ({pii_result['pii_count']} items)")
        else:
            print("✅ No PII detected")
        
        # Step 3: Determine routing
        category_name = category_result['category_name']
        has_pii = pii_result['has_pii']
        
        # Get routing rule for this category
        routing_rule = self.routing_rules.get(category_name, self.routing_rules['default'])
        target_model = routing_rule['pii' if has_pii else 'clean']
        
        print(f"🎯 Routing to: {target_model}")
        
        # Step 4: Security recommendations
        security_level = "HIGH" if has_pii else "STANDARD"
        recommendations = []
        
        if has_pii:
            recommendations.extend([
                "Enable encryption in transit",
                "Log access for audit",
                "Apply data retention policies"
            ])
            
            if 'ssn' in pii_result['pii_types'] or 'credit_card' in pii_result['pii_types']:
                recommendations.append("Require additional authentication")
                security_level = "CRITICAL"
                
            # Special handling for sensitive categories
            if category_name in ['health', 'law', 'psychology']:
                recommendations.append("Apply strict confidentiality protocols")
                if security_level != "CRITICAL":
                    security_level = "HIGH"
        
        return {
            'query': text,
            'category': category_result,
            'pii': pii_result,
            'routing': {
                'target_model': target_model,
                'security_level': security_level,
                'recommendations': recommendations
            },
            'processing_summary': {
                'safe_to_log': not has_pii,
                'requires_encryption': has_pii,
                'audit_required': has_pii and category_name in ['health', 'law', 'business', 'economics', 'psychology']
            }
        }
    
    def run_demo_queries(self):
        """Run a set of predefined demo queries to show the system in action."""
        print("\n" + "="*80)
        print(f"🎪 LIVE DEMO: Semantic Router with {self.num_categories}-Category Classification + PII Detection")
        print("="*80)
        
        # Create demo queries based on detected categories
        demo_queries = self._generate_demo_queries()
        
        results = []
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 DEMO QUERY {i}/{len(demo_queries)}")
            print("-" * 50)
            
            result = self.route_query(query)
            results.append(result)
            
            # Show masked version if PII detected
            if result['pii']['has_pii']:
                print(f"🎭 Masked query: {result['pii']['masked_text']}")
            
            print()
        
        # Summary statistics
        print("\n" + "="*80)
        print("📊 DEMO SUMMARY")
        print("="*80)
        
        total_queries = len(results)
        pii_queries = sum(1 for r in results if r['pii']['has_pii'])
        
        category_counts = {}
        for result in results:
            cat = result['category']['category_name']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"Total queries processed: {total_queries}")
        print(f"Queries with PII: {pii_queries} ({pii_queries/total_queries*100:.1f}%)")
        print(f"Queries without PII: {total_queries-pii_queries} ({(total_queries-pii_queries)/total_queries*100:.1f}%)")
        
        print(f"\nCategory distribution ({self.num_categories} categories):")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count} queries ({count/total_queries*100:.1f}%)")
        
        print(f"\nSecurity levels:")
        security_counts = {}
        for result in results:
            level = result['routing']['security_level']
            security_counts[level] = security_counts.get(level, 0) + 1
        
        for level, count in sorted(security_counts.items()):
            print(f"  {level}: {count} queries")
        
        return results
    
    def _generate_demo_queries(self) -> List[str]:
        """Generate demo queries based on detected categories."""
        queries = []
        
        # Get available categories
        category_names = list(self.categories.values())
        
        # Base query templates for common categories
        query_templates = {
            'business': [
                "What are effective customer retention strategies for growing businesses?",
                "Business consultation needed. Contact client at john.smith@company.com for details."
            ],
            'technology': [
                "How does cloud computing improve scalability for modern applications?", 
                "Tech support request from user ID: TECH-123456. Call (555) 234-5678."
            ],
            'science': [
                "What are the latest breakthroughs in renewable energy research?",
                "Research collaboration proposal. Email scientist@university.edu for partnership details."
            ],
            'health': [
                "What are the symptoms and treatment options for common allergies?",
                "Patient John Doe (SSN: 123-45-6789) requires specialist consultation."
            ],
            'sports': [
                "How do professional athletes optimize their training for peak performance?",
                "Sports team contract review. Payment via card 4532-1234-5678-9012."
            ],
            'politics': [
                "What factors influence voter turnout in democratic elections?",
                "Political campaign coordination. Contact manager at (555) 987-6543."
            ],
            'entertainment': [
                "What makes a movie successful at the box office?",
                "Entertainment industry meeting at 456 Hollywood Blvd. RSVP required."
            ],
            'world': [
                "How do international trade agreements affect global economics?",
                "Diplomatic briefing scheduled. Contact embassy@foreign.gov for access."
            ],
            'economics': [
                "What factors contribute to inflation in modern economies?",
                "Economic analysis report. Billing address: 789 Wall Street, New York."
            ],
            'education': [
                "What teaching methods are most effective for online learning?",
                "Educational consultation. Student ID: EDU-789012. Phone: (555) 456-7890."
            ]
        }
        
        # Generate queries for detected categories
        for category_name in category_names:
            category_key = category_name.lower().replace(' ', '_')
            
            # Use specific templates if available
            if category_key in query_templates:
                queries.extend(query_templates[category_key])
            elif any(key in category_key for key in query_templates.keys()):
                # Partial match
                for key in query_templates.keys():
                    if key in category_key:
                        queries.extend(query_templates[key])
                        break
            else:
                # Generic queries for unknown categories
                queries.extend([
                    f"What are the key concepts and principles in {category_name}?",
                    f"{category_name.title()} consultation needed. Contact expert at specialist@domain.com."
                ])
        
        # Add some general queries
        queries.extend([
            "What's the weather forecast for this weekend?",
            "General inquiry about services. My credit card 1234-5678-9012-3456 for payment.",
            "How can I improve my productivity while working from home?",
            "Personal appointment booking. Call me at (555) 123-4567 urgently."
        ])
        
        return queries
    
    def interactive_mode(self):
        """Run interactive mode for live testing."""
        print("\n" + "="*80)
        print(f"🖥️  INTERACTIVE MODE: Live Query Testing ({self.num_categories} Categories)")
        print("="*80)
        print("Enter queries to see real-time classification and routing!")
        
        if self.categories:
            category_list = list(self.categories.values())
            if len(category_list) <= 8:
                print(f"Categories: {', '.join(category_list)}")
            else:
                print(f"Categories: {', '.join(category_list[:6])}, ... (+{len(category_list)-6} more)")
        
        print("Type 'quit', 'exit', or 'demo' for special commands.")
        print()
        
        while True:
            try:
                query = input("🔍 Enter your query: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if query.lower() == 'demo':
                    self.run_demo_queries()
                    continue
                    
                if query.lower() == 'categories':
                    print(f"\n📂 Available Categories ({self.num_categories}):")
                    for cat_id, cat_name in self.categories.items():
                        print(f"  {cat_id}: {cat_name}")
                    print()
                    continue
                
                # Process the query
                result = self.route_query(query)
                
                # Show masked version if needed
                if result['pii']['has_pii']:
                    print(f"🎭 Masked: {result['pii']['masked_text']}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _detect_categories_from_model_info(self, model_info: Dict) -> Optional[Dict]:
        """Try to detect categories from the model's training directory."""
        try:
            training_dir = Path(model_info['directory'])
            
            # Check for training history which might contain category info
            history_file = training_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    # History might contain category information
                    if 'categories' in history:
                        return history['categories']
            
            # Check for training config
            config_file = training_dir / "final_model" / "training_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'categories' in config:
                        return config['categories']
            
            # Try to find the original dataset files
            possible_datasets = [
                "real_train_dataset.json",
                "extended_train_dataset.json"
            ]
            
            for dataset_file in possible_datasets:
                dataset_path = Path(dataset_file)
                if dataset_path.exists():
                    return self._detect_categories_from_dataset(dataset_path)
            
        except Exception as e:
            print(f"⚠️ Could not detect categories from model info: {e}")
        
        return None
    
    def _detect_categories_from_dataset(self, dataset_path: Path) -> Optional[Dict]:
        """Extract categories from a dataset file."""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            categories = set()
            for item in data[:100]:  # Check first 100 samples
                if 'category' in item:
                    categories.add(item['category'])
            
            # Create category mapping (alphabetical order for consistency)
            sorted_categories = sorted(categories)
            category_map = {i: cat for i, cat in enumerate(sorted_categories)}
            
            print(f"📊 Detected categories from {dataset_path}: {sorted_categories}")
            return category_map
            
        except Exception as e:
            print(f"⚠️ Could not read dataset {dataset_path}: {e}")
        
        return None
    
    def _detect_categories_from_model_output(self) -> Optional[Dict]:
        """Try to infer categories from model's output dimension."""
        if not self.model:
            return None
        
        try:
            # Get the output dimension from the model
            output_dim = self.model.category_classifier[-1].out_features
            
            # Common category mappings for known datasets
            known_mappings = {
                4: {  # AG News or similar
                    0: "world", 1: "sports", 2: "business", 3: "technology"
                },
                5: {  # BBC News
                    0: "business", 1: "entertainment", 2: "politics", 3: "sport", 4: "tech"
                },
                8: {  # Extended dataset
                    0: "business", 1: "education", 2: "entertainment", 3: "health",
                    4: "politics", 5: "science", 6: "sports", 7: "technology"
                },
                10: {  # Academic subjects
                    0: "biology", 1: "business", 2: "computer science", 3: "economics",
                    4: "engineering", 5: "health", 6: "math", 7: "other", 8: "philosophy", 9: "physics"
                },
                20: {  # 20 Newsgroups simplified
                    i: f"category_{i}" for i in range(20)
                }
            }
            
            if output_dim in known_mappings:
                print(f"📂 Using known category mapping for {output_dim} categories")
                return known_mappings[output_dim]
            else:
                # Generic fallback
                print(f"📂 Using generic category mapping for {output_dim} categories")
                return {i: f"category_{i}" for i in range(output_dim)}
                
        except Exception as e:
            print(f"⚠️ Could not detect categories from model: {e}")
        
        return None
    
    def _setup_categories(self, model_info: Optional[Dict] = None):
        """Setup categories based on available information."""
        categories = None
        
        # Method 1: Try to get from model training info
        if model_info:
            categories = self._detect_categories_from_model_info(model_info)
        
        # Method 2: Try to get from model output dimension
        if not categories:
            categories = self._detect_categories_from_model_output()
        
        # Method 3: Try to get from available datasets
        if not categories:
            for dataset_file in ["real_train_dataset.json", "extended_train_dataset.json"]:
                if Path(dataset_file).exists():
                    categories = self._detect_categories_from_dataset(Path(dataset_file))
                    if categories:
                        break
        
        # Method 4: Fallback to asking user or using defaults
        if not categories:
            print("⚠️ Could not auto-detect categories. Using default academic categories.")
            categories = {
                0: "business", 1: "science", 2: "technology", 3: "other"
            }
        
        self.categories = categories
        self.num_categories = len(categories)
        
    def _setup_routing_rules(self):
        """Setup routing rules based on detected categories."""
        # Always create a default rule first
        self.routing_rules['default'] = {
            'clean': 'general_model',
            'pii': 'secure_general_model'
        }
        
        if not self.categories:
            return
        
        # Create generic routing rules for any category
        for category_name in self.categories.values():
            # Sanitize category name for routing
            clean_name = category_name.lower().replace(' ', '_')
            
            self.routing_rules[category_name] = {
                'clean': f'{clean_name}_model_standard',
                'pii': f'{clean_name}_model_secure'
            }
        
        # Add special rules for sensitive categories
        sensitive_categories = ['health', 'medical', 'law', 'legal', 'psychology', 'business']
        for category_name in self.categories.values():
            if any(sensitive in category_name.lower() for sensitive in sensitive_categories):
                clean_name = category_name.lower().replace(' ', '_')
                self.routing_rules[category_name] = {
                    'clean': f'{clean_name}_model_general',
                    'pii': f'{clean_name}_model_confidential'
                }


def main():
    """Main function to run the live demo."""
    print("🚀 Starting Semantic Router Live Demo...")
    
    # Initialize the demo router
    demo = LiveDemoRouter()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo.run_demo_queries()
        elif sys.argv[1] == '--interactive':
            demo.interactive_mode()
        else:
            print("Usage: python live_demo.py [--demo|--interactive]")
    else:
        # Default: run both demo and interactive
        demo.run_demo_queries()
        demo.interactive_mode()


if __name__ == "__main__":
    main() 