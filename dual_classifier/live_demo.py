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

# Import our modules
from dual_classifier import DualClassifier
from hardware_detector import detect_and_configure
from enhanced_trainer import EnhancedDualTaskTrainer

class LiveDemoRouter:
    """
    Live demonstration system for dual-purpose semantic routing.
    Combines category classification and PII detection for real-time routing.
    """
    
    def __init__(self):
        print("🚀 Initializing Live Demo Router...")
        
        # Updated categories to match the semantic router's 14-category system
        self.categories = {
            0: "economics",
            1: "health", 
            2: "computer science",
            3: "philosophy",
            4: "physics",
            5: "business",
            6: "engineering",
            7: "biology",
            8: "other",
            9: "math",
            10: "psychology",
            11: "chemistry",
            12: "law",
            13: "history"
        }
        
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
        
        # Routing rules based on category + PII (expanded for 14 categories)
        self.routing_rules = {
            'computer science': {
                'clean': 'tech-model-standard',
                'pii': 'tech-model-secure'
            },
            'engineering': {
                'clean': 'engineering-model-standard',
                'pii': 'engineering-model-secure'
            },
            'business': {
                'clean': 'business-model-standard', 
                'pii': 'business-model-encrypted'
            },
            'economics': {
                'clean': 'economics-model-standard',
                'pii': 'economics-model-secure'
            },
            'health': {
                'clean': 'medical-model-general',
                'pii': 'medical-model-hipaa'
            },
            'law': {
                'clean': 'legal-model-standard',
                'pii': 'legal-model-confidential'
            },
            'psychology': {
                'clean': 'psychology-model-standard',
                'pii': 'psychology-model-confidential'
            },
            'math': {
                'clean': 'math-model-standard',
                'pii': 'math-model-secure'
            },
            'physics': {
                'clean': 'physics-model-standard',
                'pii': 'physics-model-secure'
            },
            'chemistry': {
                'clean': 'chemistry-model-standard',
                'pii': 'chemistry-model-secure'
            },
            'biology': {
                'clean': 'biology-model-standard',
                'pii': 'biology-model-secure'
            },
            'philosophy': {
                'clean': 'philosophy-model-standard',
                'pii': 'philosophy-model-secure'
            },
            'history': {
                'clean': 'history-model-standard',
                'pii': 'history-model-secure'
            },
            'default': {
                'clean': 'general-model',
                'pii': 'secure-general-model'
            }
        }
        
        # Load model if available
        self.model = None
        self.load_model()
        
        print("✅ Live Demo Router initialized!")
    
    def load_model(self):
        """Load the dual classifier model if available."""
        try:
            # Check if we have a trained model
            model_path = "dual_classifier_checkpoint.pth"
            if os.path.exists(model_path):
                print("📦 Loading trained model...")
                
                # Detect hardware capabilities
                capabilities = detect_and_configure()
                device = capabilities['device']
                
                # Initialize model
                self.model = DualClassifier(
                    num_categories=len(self.categories),
                    num_pii_classes=2  # binary: PII or not
                )
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(device)
                self.model.eval()
                
                print(f"✅ Model loaded on {device}")
            else:
                print("⚠️ No trained model found. Using rule-based classification.")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Falling back to rule-based classification.")
    
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
        if self.model:
            # Use trained model
            try:
                device = next(self.model.parameters()).device
                
                # Simple tokenization for demo (in real system, use proper tokenizer)
                words = text.lower().split()
                vocab_size = 10000  # Should match training vocab
                word_ids = [hash(word) % vocab_size for word in words]
                
                # Pad/truncate to fixed length
                max_length = 128
                if len(word_ids) > max_length:
                    word_ids = word_ids[:max_length]
                else:
                    word_ids.extend([0] * (max_length - len(word_ids)))
                
                # Convert to tensor
                input_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    category_logits, _ = self.model(input_tensor)
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
        
        # Rule-based classification with expanded keywords for 14 categories
        text_lower = text.lower()
        
        # Economics keywords
        economics_keywords = ['economy', 'economics', 'inflation', 'gdp', 'market', 'stock', 'investment', 
                            'finance', 'monetary', 'fiscal', 'trade', 'demand', 'supply', 'recession']
        
        # Health keywords
        health_keywords = ['health', 'medical', 'doctor', 'hospital', 'medicine', 'disease', 'treatment',
                         'patient', 'symptoms', 'diagnosis', 'therapy', 'healthcare', 'clinic', 'surgery']
        
        # Computer Science keywords
        cs_keywords = ['computer', 'software', 'programming', 'algorithm', 'machine learning', 'ai',
                      'artificial intelligence', 'neural network', 'deep learning', 'python', 'java', 'code']
        
        # Philosophy keywords
        philosophy_keywords = ['philosophy', 'ethics', 'morality', 'metaphysics', 'epistemology', 'logic',
                             'existence', 'consciousness', 'truth', 'reality', 'kant', 'aristotle', 'plato']
        
        # Physics keywords
        physics_keywords = ['physics', 'quantum', 'relativity', 'mechanics', 'thermodynamics', 'energy',
                          'force', 'gravity', 'electromagnetic', 'particle', 'wave', 'momentum', 'mass']
        
        # Business keywords
        business_keywords = ['business', 'corporate', 'company', 'profit', 'revenue', 'sales', 'customer',
                           'management', 'strategy', 'marketing', 'operations', 'enterprise', 'startup']
        
        # Engineering keywords
        engineering_keywords = ['engineering', 'design', 'construction', 'mechanical', 'electrical', 'civil',
                              'structural', 'system', 'technology', 'manufacturing', 'automation', 'robotics']
        
        # Biology keywords
        biology_keywords = ['biology', 'cell', 'organism', 'evolution', 'genetics', 'dna', 'species',
                          'ecosystem', 'protein', 'molecular', 'biochemistry', 'anatomy', 'physiology']
        
        # Math keywords
        math_keywords = ['math', 'mathematics', 'calculus', 'algebra', 'geometry', 'statistics', 'equation',
                       'theorem', 'proof', 'function', 'derivative', 'integral', 'probability', 'number']
        
        # Psychology keywords
        psychology_keywords = ['psychology', 'behavior', 'cognitive', 'mental', 'brain', 'emotion', 'learning',
                             'memory', 'personality', 'psychotherapy', 'psychological', 'therapy', 'mind']
        
        # Chemistry keywords
        chemistry_keywords = ['chemistry', 'chemical', 'molecule', 'atom', 'reaction', 'compound', 'element',
                            'periodic', 'organic', 'inorganic', 'catalyst', 'synthesis', 'bond', 'solution']
        
        # Law keywords
        law_keywords = ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'contract', 'lawsuit',
                       'legislation', 'constitution', 'rights', 'justice', 'regulation', 'statute']
        
        # History keywords
        history_keywords = ['history', 'historical', 'ancient', 'medieval', 'war', 'civilization', 'culture',
                          'century', 'empire', 'revolution', 'dynasty', 'archaeological', 'historical']
        
        # Count keyword matches for all 14 categories
        scores = {
            0: sum(1 for kw in economics_keywords if kw in text_lower),      # economics
            1: sum(1 for kw in health_keywords if kw in text_lower),         # health
            2: sum(1 for kw in cs_keywords if kw in text_lower),             # computer science
            3: sum(1 for kw in philosophy_keywords if kw in text_lower),     # philosophy
            4: sum(1 for kw in physics_keywords if kw in text_lower),        # physics
            5: sum(1 for kw in business_keywords if kw in text_lower),       # business
            6: sum(1 for kw in engineering_keywords if kw in text_lower),    # engineering
            7: sum(1 for kw in biology_keywords if kw in text_lower),        # biology
            8: 1,  # other (always has base score)
            9: sum(1 for kw in math_keywords if kw in text_lower),           # math
            10: sum(1 for kw in psychology_keywords if kw in text_lower),    # psychology
            11: sum(1 for kw in chemistry_keywords if kw in text_lower),     # chemistry
            12: sum(1 for kw in law_keywords if kw in text_lower),           # law
            13: sum(1 for kw in history_keywords if kw in text_lower)        # history
        }
        
        # Find best category
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_category] / max(1, len(text_lower.split()))
        
        return {
            'category_id': best_category,
            'category_name': self.categories[best_category],
            'confidence': min(confidence, 1.0),
            'method': 'rule_based',
            'keyword_scores': scores
        }
    
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
        print("🎪 LIVE DEMO: Semantic Router with 14-Category Classification + PII Detection")
        print("="*80)
        
        demo_queries = [
            # Economics queries
            "What factors contribute to inflation and how does monetary policy affect economic growth?",
            "Stock market analysis for Q3 earnings. Contact analyst at john.doe@finance.com for details.",
            
            # Health queries  
            "What are the symptoms and treatment options for Type 2 diabetes?",
            "Patient John Smith (SSN: 123-45-6789) needs cardiology consultation scheduling.",
            
            # Computer Science queries
            "How do I implement a neural network using PyTorch for image classification?",
            "Debug Python machine learning pipeline. Email me at dev@tech.com with solutions.",
            
            # Philosophy queries
            "Explain Kant's categorical imperative and its implications for modern ethics.",
            "Philosophy research grant application. Contact Dr. Sarah Johnson at 555-123-4567.",
            
            # Physics queries
            "Describe quantum entanglement and its applications in quantum computing.",
            "Research collaboration on particle physics. My card 4532-1234-5678-9012 for expenses.",
            
            # Business queries
            "What are the key strategies for successful customer acquisition and retention?",
            "Business contract review for client at 123 Oak Street, phone (555) 987-6543.",
            
            # Engineering queries
            "Design principles for sustainable civil engineering infrastructure projects.",
            "Engineering consultation needed. Account ID: ENG-789456. Call me urgently.",
            
            # Biology queries
            "Explain the process of cellular respiration and ATP production in mitochondria.",
            "Research data on genetics study. Contact lab at biology@university.edu.",
            
            # Math queries
            "Solve the differential equation dy/dx + 2y = x^2 with initial condition y(0) = 1.",
            "Mathematics tutoring available. Call Prof. Miller at (555) 234-5678.",
            
            # Psychology queries  
            "What are the cognitive behavioral therapy techniques for treating anxiety disorders?",
            "Patient confidential session notes for ID: PSY-123456. Secure access required.",
            
            # Chemistry queries
            "Describe the mechanism of catalytic hydrogenation in organic synthesis.",
            "Chemical research funding application. Contact Dr. Chen at chem@research.org.",
            
            # Law queries
            "What are the constitutional principles governing freedom of speech in democratic societies?",
            "Legal case consultation for client John Anderson, SSN: 987-65-4321. Confidential matter.",
            
            # History queries
            "Analyze the causes and consequences of the Industrial Revolution in 19th century Europe.",
            "Historical archives access request. Email historian@museum.org for permissions.",
            
            # Other/General queries
            "What's the weather forecast for this weekend?",
            "General inquiry about services. My credit card 1234-5678-9012-3456 for payment."
        ]
        
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
        
        print(f"\nCategory distribution (14 categories):")
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
    
    def interactive_mode(self):
        """Run interactive mode for live testing."""
        print("\n" + "="*80)
        print("🖥️  INTERACTIVE MODE: Live Query Testing (14 Categories)")
        print("="*80)
        print("Enter queries to see real-time classification and routing!")
        print("Categories: economics, health, computer science, philosophy, physics,")
        print("           business, engineering, biology, other, math, psychology,")
        print("           chemistry, law, history")
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


def main():
    """Main function to run the live demo."""
    print("🚀 Starting Semantic Router Live Demo (14 Categories)...")
    
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