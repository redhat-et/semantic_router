#!/usr/bin/env python3
"""
Download and prepare 20 Newsgroups dataset for dual classifier training.
This dataset has 20 categories, making it great for testing classifier robustness.
"""

import json
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import re
import traceback

def clean_text(text):
    """Clean and truncate text for better training."""
    # Remove headers and footers
    text = re.sub(r'^.*?Subject:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?From:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Organization:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Lines:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Distribution:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Keywords:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?NNTP-Posting-Host:.*?\n', '', text, flags=re.MULTILINE)
    
    # Remove quoted text (lines starting with >)
    lines = text.split('\n')
    clean_lines = [line for line in lines if not line.strip().startswith('>')]
    text = '\n'.join(clean_lines)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long
    if len(text) > 800:
        text = text[:800] + "..."
    
    return text

def download_20newsgroups():
    """
    Download 20 Newsgroups dataset using sklearn.
    """
    print("📰 Downloading 20 Newsgroups Dataset...")
    
    # Categories in 20 newsgroups (simplified names)
    category_mapping = {
        'alt.atheism': 'atheism',
        'comp.graphics': 'graphics',
        'comp.os.ms-windows.misc': 'windows',
        'comp.sys.ibm.pc.hardware': 'hardware',
        'comp.sys.mac.hardware': 'mac',
        'comp.windows.x': 'x_windows',
        'misc.forsale': 'forsale',
        'rec.autos': 'autos',
        'rec.motorcycles': 'motorcycles',
        'rec.sport.baseball': 'baseball',
        'rec.sport.hockey': 'hockey',
        'sci.crypt': 'cryptography',
        'sci.electronics': 'electronics',
        'sci.med': 'medicine',
        'sci.space': 'space',
        'soc.religion.christian': 'christian',
        'talk.politics.guns': 'guns',
        'talk.politics.mideast': 'mideast',
        'talk.politics.misc': 'politics',
        'talk.religion.misc': 'religion'
    }
    
    try:
        # Download the dataset
        print("⏳ Fetching dataset from sklearn...")
        print("   This may take a few minutes...")
        
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42,
            download_if_missing=True
        )
        
        print(f"✅ Downloaded {len(newsgroups.data)} samples")
        print(f"📂 Categories: {len(newsgroups.target_names)}")
        
        # Prepare samples with batch processing to avoid memory issues
        samples = []
        batch_size = 1000
        total_samples = len(newsgroups.data)
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            print(f"📊 Processing batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}...")
            
            for i in range(batch_start, batch_end):
                try:
                    text = newsgroups.data[i]
                    target = newsgroups.target[i]
                    
                    if text is None or target is None:
                        continue
                    
                    original_category = newsgroups.target_names[target]
                    clean_category = category_mapping.get(original_category, original_category)
                    
                    # Clean the text
                    cleaned_text = clean_text(str(text))
                    
                    # Skip very short texts
                    if len(cleaned_text) < 50:
                        continue
                    
                    samples.append({
                        'text': cleaned_text,
                        'category': clean_category
                    })
                except Exception as e:
                    # Skip problematic samples instead of crashing
                    continue
        
        print(f"📊 Processed {len(samples)} valid samples")
        
        # Show category distribution
        category_counts = {}
        for sample in samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📊 Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"   {cat}: {count} samples")
        
        # Split into train/validation with error handling
        try:
            train_samples, val_samples = train_test_split(
                samples, test_size=0.2, random_state=42, 
                stratify=[s['category'] for s in samples]
            )
        except Exception as e:
            print(f"⚠️ Stratified split failed, using random split: {str(e)}")
            train_samples, val_samples = train_test_split(
                samples, test_size=0.2, random_state=42
            )
        
        # Save datasets
        with open("real_train_dataset.json", "w", encoding='utf-8') as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        
        with open("real_val_dataset.json", "w", encoding='utf-8') as f:
            json.dump(val_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Created training dataset: {len(train_samples)} samples")
        print(f"✅ Created validation dataset: {len(val_samples)} samples")
        print(f"📂 Files saved as: real_train_dataset.json, real_val_dataset.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 20 Newsgroups Dataset Downloader")
    print("=" * 45)
    print("⚠️  This dataset has 20 categories - great for testing!")
    print("📊 Expected categories: atheism, graphics, hardware, etc.")
    print()
    
    try:
        if download_20newsgroups():
            print(f"\n🎉 Dataset preparation complete!")
            print(f"📂 20 categories available")
            print(f"🚀 You can now run: python enhanced_trainer.py normal")
            print(f"💡 Consider using 'intensive' or 'maximum' strength for this complex dataset")
        else:
            print("❌ Failed to download dataset")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc() 