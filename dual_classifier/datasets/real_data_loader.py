#!/usr/bin/env python3
"""
Real Data Loader for Dual-Purpose Training

This module provides loaders for real datasets suitable for both
category classification and PII detection tasks.
"""

import os
import re
import json
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import random

# Import datasets library for HuggingFace datasets
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
    print("✅ HuggingFace datasets available")
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("❌ HuggingFace datasets not available")


class RealDataCollector:
    """Collect and process real datasets for dual-purpose training."""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b',
            'name_title': r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        }
    
    def detect_pii_with_regex(self, text: str) -> List[int]:
        """Detect PII using regex patterns."""
        words = text.split()
        labels = [0] * len(words)
        
        for pattern_name, pattern in self.pii_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Find which words this match covers
                start_char = match.start()
                end_char = match.end()
                
                char_pos = 0
                for i, word in enumerate(words):
                    word_start = char_pos
                    word_end = char_pos + len(word)
                    
                    if (word_start < end_char and word_end > start_char):
                        labels[i] = 1
                    
                    char_pos = word_end + 1
        
        return labels
    
    def load_ag_news_with_pii(self, num_samples: int = 1000) -> List[Dict]:
        """Load AG News dataset and inject realistic PII for demonstration."""
        if not HF_DATASETS_AVAILABLE:
            print("❌ Cannot load AG News: datasets library not available")
            return []
        
        print("📰 Loading AG News dataset...")
        
        # Load AG News dataset
        try:
            dataset = load_dataset("ag_news")
        except Exception as e:
            print(f"❌ Failed to load AG News: {e}")
            return []
        
        # Category mapping
        categories = {0: "world", 1: "sports", 2: "business", 3: "technology"}
        
        results = []
        
        # PII injection templates with realistic context
        pii_injection_templates = [
            "Contact reporter {name} at {email} for more information about this story.",
            "For interview requests, call {name} at {phone}.",
            "Documents can be sent to {name} at {address}.",
            "Press release issued by {name} (Employee ID: {id}) from the communications department.",
            "For updates on this story, email {email} or call {phone}.",
            "CEO {name} can be reached at {email} for executive comments.",
            "Legal representative {name} (Bar ID: {id}) issued the following statement.",
            "Research conducted by Dr. {name} at {email} shows significant findings.",
            "Spokesperson {name} announced the decision during a call to {phone}.",
            "Financial details available from CFO {name} at {address}."
        ]
        
        # Realistic names, emails, etc.
        names = [
            "John Smith", "Sarah Johnson", "Michael Davis", "Emily Brown", "David Wilson",
            "Jessica Miller", "Christopher Taylor", "Amanda Anderson", "Matthew Jackson", "Lisa Thompson",
            "Dr. Robert Chen", "Prof. Maria Garcia", "Ms. Jennifer Lee", "Mr. Kevin Rodriguez"
        ]
        
        emails = [
            "contact@newsnetwork.com", "info@globalnews.org", "press@mediacorp.com",
            "news@broadcaster.net", "editor@headlines.com", "reporter@newswire.org"
        ]
        
        phones = [
            "555-123-4567", "(212) 555-0123", "1-800-555-0199", 
            "(415) 555-7890", "555.987.6543", "+1-555-123-4567"
        ]
        
        addresses = [
            "123 Main Street, New York", "456 Oak Avenue, Los Angeles", "789 Pine Road, Chicago",
            "321 Elm Boulevard, Houston", "654 Maple Drive, Phoenix", "987 Cedar Lane, Philadelphia"
        ]
        
        employee_ids = [
            "EMP001", "ID-12345", "STAFF789", "REP456", "EXEC123", "NEWS789"
        ]
        
        # Process training data
        processed = 0
        for item in dataset['train']:
            if processed >= num_samples:
                break
                
            text = item['text']
            category = categories[item['label']]
            
            # 40% chance to inject PII (higher than synthetic for realism)
            if random.random() < 0.4:
                # Inject PII
                template = random.choice(pii_injection_templates)
                pii_text = template.format(
                    name=random.choice(names),
                    email=random.choice(emails),
                    phone=random.choice(phones),
                    address=random.choice(addresses),
                    id=random.choice(employee_ids)
                )
                text = f"{text} {pii_text}"
            
            # Detect PII
            pii_labels = self.detect_pii_with_regex(text)
            
            results.append({
                'text': text,
                'category': category,
                'pii_labels': pii_labels,
                'has_pii': any(pii_labels),
                'source': 'ag_news_enhanced'
            })
            
            processed += 1
        
        print(f"✅ Processed {len(results)} AG News samples with PII injection")
        return results
    
    def create_sample_real_data(self, num_samples: int = 200) -> List[Dict]:
        """Create sample real-world-like data for demonstration."""
        print("🔄 Creating sample real-world data...")
        
        # Sample real-world texts with natural PII occurrence
        sample_texts = [
            {
                'text': 'Breaking: Tech giant announces quarterly earnings. CEO John Smith will discuss results in a conference call. Investors can join at (555) 123-4567 or email investor-relations@techcorp.com for details.',
                'category': 'business'
            },
            {
                'text': 'Local team wins championship after defeating rivals 3-2. Coach Mike Johnson credited the victory to months of preparation. For season tickets, contact the box office at tickets@sportsclub.com.',
                'category': 'sports'
            },
            {
                'text': 'New research published in Nature shows promising results for climate change mitigation. Lead researcher Dr. Sarah Chen from Stanford University can be reached at research@stanford.edu for interviews.',
                'category': 'world'
            },
            {
                'text': 'Latest smartphone features advanced AI capabilities and improved battery life. Product manager Jennifer Lee announced the launch during yesterday\'s keynote. Press inquiries: press@innovation.com or (415) 555-0123.',
                'category': 'technology'
            },
            {
                'text': 'Stock market reaches new highs as investors show confidence in economic recovery. Financial analyst Robert Davis suggests caution. His report is available at rqdavis@marketwatch.com.',
                'category': 'business'
            },
            {
                'text': 'Championship game scheduled for Sunday at MetLife Stadium. Tickets available through official website or by calling (212) 555-GAME. Team captain speaks at press conference.',
                'category': 'sports'
            },
            {
                'text': 'International climate summit begins next week in Geneva. Delegates from 50 countries expected to attend. Media credentials available from coordinator@climate-summit.org.',
                'category': 'world'
            },
            {
                'text': 'Revolutionary AI model achieves human-level performance on benchmark tests. Research team led by Prof. Maria Garcia published findings. Technical details at ai-lab@university.edu.',
                'category': 'technology'
            }
        ]
        
        results = []
        
        # Replicate samples to reach desired number
        for i in range(num_samples):
            base_sample = sample_texts[i % len(sample_texts)]
            text = base_sample['text']
            category = base_sample['category']
            
            # Add some variation
            if random.random() < 0.3:
                # Add additional PII
                extra_pii = [
                    " For more information, call (555) 987-6543.",
                    " Additional contact: info@newsdesk.com.",
                    " Follow up with spokesperson at media@company.org.",
                    " Direct line: 1-800-555-0199."
                ]
                text += random.choice(extra_pii)
            
            # Detect PII
            pii_labels = self.detect_pii_with_regex(text)
            
            results.append({
                'text': text,
                'category': category,
                'pii_labels': pii_labels,
                'has_pii': any(pii_labels),
                'source': 'sample_real'
            })
        
        print(f"✅ Created {len(results)} sample real-world texts")
        return results
    
    def create_combined_dataset(
        self, 
        use_ag_news: bool = True,
        ag_news_samples: int = 800,
        sample_data_count: int = 200,
        output_path: str = "real_dual_dataset.json"
    ) -> List[Dict]:
        """Create a combined real dataset from multiple sources."""
        print("🔄 Creating combined real dataset...")
        
        all_data = []
        
        # Add AG News with PII injection (if available)
        if use_ag_news and HF_DATASETS_AVAILABLE:
            ag_data = self.load_ag_news_with_pii(ag_news_samples)
            all_data.extend(ag_data)
        
        # Add sample real-world data
        sample_data = self.create_sample_real_data(sample_data_count)
        all_data.extend(sample_data)
        
        # Shuffle dataset
        random.shuffle(all_data)
        
        # Calculate statistics
        total_samples = len(all_data)
        pii_samples = sum(1 for item in all_data if item['has_pii'])
        
        print(f"\n📊 Combined Dataset Statistics:")
        print(f"   Total samples: {total_samples}")
        print(f"   Samples with PII: {pii_samples} ({pii_samples/total_samples*100:.1f}%)")
        
        # Category distribution
        categories = {}
        for item in all_data:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"   Categories:")
        for cat, count in sorted(categories.items()):
            print(f"     {cat}: {count} ({count/total_samples*100:.1f}%)")
        
        # Source distribution
        sources = {}
        for item in all_data:
            src = item['source']
            sources[src] = sources.get(src, 0) + 1
        
        print(f"   Sources:")
        for src, count in sorted(sources.items()):
            print(f"     {src}: {count} ({count/total_samples*100:.1f}%)")
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n💾 Saved combined dataset to: {output_path}")
        return all_data


def download_and_setup_real_data():
    """Download and setup real datasets for dual-purpose training."""
    print("🚀 Setting up REAL datasets for dual-purpose training")
    print("="*60)
    
    collector = RealDataCollector()
    
    # Create combined dataset
    dataset = collector.create_combined_dataset(
        use_ag_news=True,
        ag_news_samples=800,
        sample_data_count=200
    )
    
    # Split into train/val
    split_point = int(0.8 * len(dataset))
    train_data = dataset[:split_point]
    val_data = dataset[split_point:]
    
    # Save splits
    with open("real_train_dataset.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open("real_val_dataset.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\n✅ Created REAL datasets:")
    print(f"   📄 Training: real_train_dataset.json ({len(train_data)} samples)")
    print(f"   📄 Validation: real_val_dataset.json ({len(val_data)} samples)")
    
    # Show some examples
    print(f"\n📝 Sample entries:")
    for i, sample in enumerate(train_data[:3]):
        pii_words = [word for word, label in zip(sample['text'].split(), sample['pii_labels']) if label == 1]
        print(f"   {i+1}. Category: {sample['category']}")
        print(f"      Text: {sample['text'][:100]}...")
        print(f"      PII detected: {pii_words if pii_words else 'None'}")
        print()
    
    return train_data, val_data


if __name__ == "__main__":
    # Demo the real data collection
    download_and_setup_real_data() 