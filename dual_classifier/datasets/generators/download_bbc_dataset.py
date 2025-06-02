#!/usr/bin/env python3
"""
Download and prepare BBC News dataset for dual classifier training.
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
import zipfile
from sklearn.model_selection import train_test_split

def download_bbc_dataset():
    """
    Download BBC News dataset from official source.
    """
    print("📰 Downloading BBC News Dataset...")
    
    # BBC News dataset URL (from official Kaggle or other sources)
    # This is a publicly available dataset
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    
    # Create directory
    os.makedirs("datasets", exist_ok=True)
    
    # Download file
    response = requests.get(url)
    if response.status_code == 200:
        with open("datasets/bbc-fulltext.zip", "wb") as f:
            f.write(response.content)
        print("✅ Dataset downloaded successfully")
    else:
        print(f"❌ Failed to download dataset. Status code: {response.status_code}")
        return False
    
    # Extract zip file
    with zipfile.ZipFile("datasets/bbc-fulltext.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/")
    
    print("✅ Dataset extracted")
    return True

def prepare_bbc_dataset():
    """
    Prepare BBC dataset in the format expected by the dual classifier.
    """
    print("🔄 Preparing BBC News dataset...")
    
    dataset_path = Path("datasets/bbc")
    if not dataset_path.exists():
        print("❌ BBC dataset not found. Please download first.")
        return False
    
    samples = []
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            for file_path in category_path.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # Take first 500 characters to avoid very long texts
                        if len(content) > 500:
                            content = content[:500] + "..."
                        
                        samples.append({
                            'text': content,
                            'category': category
                        })
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
    
    print(f"📊 Loaded {len(samples)} samples from {len(categories)} categories")
    
    # Show category distribution
    category_counts = {}
    for sample in samples:
        cat = sample['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("📊 Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count} samples")
    
    # Split into train/validation
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=42, stratify=[s['category'] for s in samples]
    )
    
    # Save datasets
    with open("real_train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    
    with open("real_val_dataset.json", "w") as f:
        json.dump(val_samples, f, indent=2)
    
    print(f"✅ Created training dataset: {len(train_samples)} samples")
    print(f"✅ Created validation dataset: {len(val_samples)} samples")
    print(f"📂 Files saved as: real_train_dataset.json, real_val_dataset.json")
    
    return True

if __name__ == "__main__":
    print("🚀 BBC News Dataset Downloader")
    print("=" * 40)
    
    # Download dataset
    if download_bbc_dataset():
        # Prepare dataset
        if prepare_bbc_dataset():
            print(f"\n🎉 Dataset preparation complete!")
            print(f"📂 Categories: business, entertainment, politics, sport, tech")
            print(f"🚀 You can now run: python enhanced_trainer.py normal")
        else:
            print("❌ Failed to prepare dataset")
    else:
        print("❌ Failed to download dataset") 