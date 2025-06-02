#!/usr/bin/env python3
"""
Download and prepare AG News dataset for dual classifier training.
High quality 4-category news dataset.
"""

import json
import os
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def download_agnews():
    """
    Download AG News dataset using Hugging Face datasets.
    """
    if not DATASETS_AVAILABLE:
        print("❌ 'datasets' library not installed.")
        print("💡 Install with: pip install datasets")
        return False
    
    print("📰 Downloading AG News Dataset...")
    
    try:
        # Download the dataset
        print("⏳ Fetching dataset from Hugging Face...")
        dataset = load_dataset("ag_news")
        
        # Category mapping
        category_mapping = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
        
        # Prepare training samples
        train_samples = []
        for item in dataset['train']:
            train_samples.append({
                'text': item['text'][:600] + ("..." if len(item['text']) > 600 else ""),
                'category': category_mapping[item['label']]
            })
        
        # Prepare test samples (use as validation)
        val_samples = []
        for item in dataset['test']:
            val_samples.append({
                'text': item['text'][:600] + ("..." if len(item['text']) > 600 else ""),
                'category': category_mapping[item['label']]
            })
        
        print(f"📊 Training samples: {len(train_samples)}")
        print(f"📊 Validation samples: {len(val_samples)}")
        
        # Show category distribution
        category_counts = {}
        for sample in train_samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📊 Training category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"   {cat}: {count} samples")
        
        # Take subset if dataset is too large
        if len(train_samples) > 10000:
            print(f"📊 Large dataset detected. Taking subset for faster training...")
            train_samples = train_samples[:8000]
            val_samples = val_samples[:2000]
            print(f"📊 Reduced to {len(train_samples)} training, {len(val_samples)} validation")
        
        # Save datasets
        with open("real_train_dataset.json", "w") as f:
            json.dump(train_samples, f, indent=2)
        
        with open("real_val_dataset.json", "w") as f:
            json.dump(val_samples, f, indent=2)
        
        print(f"\n✅ Created training dataset: {len(train_samples)} samples")
        print(f"✅ Created validation dataset: {len(val_samples)} samples")
        print(f"📂 Files saved as: real_train_dataset.json, real_val_dataset.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have internet connection and 'datasets' library installed")
        return False

if __name__ == "__main__":
    print("🚀 AG News Dataset Downloader")
    print("=" * 35)
    print("📊 High-quality 4-category news dataset")
    print("📂 Categories: World, Sports, Business, Technology")
    print()
    
    if download_agnews():
        print(f"\n🎉 Dataset preparation complete!")
        print(f"📂 4 high-quality categories")
        print(f"🚀 You can now run: python enhanced_trainer.py normal")
    else:
        print("❌ Failed to download dataset") 