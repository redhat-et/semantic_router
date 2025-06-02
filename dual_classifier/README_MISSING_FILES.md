# Missing Files Guide for Dual Classifier

## Overview

This project excludes large files from Git for performance and storage reasons. When you clone the repository, you'll be missing some files needed for training and demos. This guide explains what's missing and how to get them.

## What Files Are Missing?

### 1. Dataset Files (Required for Training)
- `real_train_dataset.json` (~7.6MB) - Training dataset
- `real_val_dataset.json` (~1.9MB) - Validation dataset

### 2. Model Files (Required for Live Demo)
- `training_outputs/**/*.pt` (100MB-800MB each) - Trained model checkpoints
- `training_outputs/**/model.pt` (258MB each) - Final trained models
- `training_outputs/**/vocab.txt` (226KB each) - Vocabulary files

## Quick Start

### For Training
```bash
# Check what's missing
python missing_files_detector.py training

# Generate dataset (choose one):
python datasets/generators/download_agnews.py      # 4 categories, smaller
python datasets/generators/download_20newsgroups.py  # 20 categories, comprehensive
python datasets/generators/download_bbc_dataset.py   # 5 categories, medium

# Start training
python enhanced_trainer.py normal
```

### For Live Demo
```bash
# Check what's missing
python missing_files_detector.py demo

# If no models found, train one first:
python datasets/generators/download_agnews.py  # Generate dataset
python enhanced_trainer.py quick               # Quick training

# Run demo (works with or without models)
python live_demo.py
```

## Automatic Detection

The system automatically detects missing files and provides helpful instructions:

- **enhanced_trainer.py** - Checks for dataset files before training
- **live_demo.py** - Checks for model files at startup  
- **missing_files_detector.py** - Standalone checker with detailed help

## Dataset Options

### 1. AG News (Recommended for beginners)
```bash
python datasets/generators/download_agnews.py
```
- 4 categories: World, Sports, Business, Technology
- Size: ~3MB train + 0.8MB validation
- Training time: Fast

### 2. 20 Newsgroups (Comprehensive)
```bash
python datasets/generators/download_20newsgroups.py
```
- 20 categories: atheism, graphics, hardware, etc.
- Size: ~7.6MB train + 1.9MB validation  
- Training time: Longer, better for testing

### 3. BBC News (Balanced)
```bash
python datasets/generators/download_bbc_dataset.py
```
- 5 categories: business, entertainment, politics, sport, tech
- Size: ~1.5MB train + 0.4MB validation
- Training time: Medium

### 4. Custom Dataset
```bash
python datasets/real_data_loader.py
```
- Your own data
- Follow the prompts to configure
- Size: Variable

## Training Options

After generating datasets, train with different intensities:

```bash
python enhanced_trainer.py quick      # 2 epochs, fast
python enhanced_trainer.py normal     # 5 epochs, balanced  
python enhanced_trainer.py intensive  # 10 epochs, high quality
python enhanced_trainer.py maximum    # 20 epochs, best quality
```

## File Locations

The system looks for files in these locations:

### Datasets
- `./real_train_dataset.json`
- `./real_val_dataset.json`
- `./datasets/real_train_dataset.json`
- `./datasets/real_val_dataset.json`

### Models
- `./training_outputs/trained_model/model.pt`
- `./training_outputs/task3_demo_output/final_model/model.pt`
- `./training_outputs/**/checkpoints/best_model.pt`
- `./dual_classifier_checkpoint.pth` (legacy)
- `./trained_model/model.pt` (legacy)

## Why Are These Files Excluded?

1. **Size**: Model files can be 100MB-800MB each
2. **Performance**: Large files slow down git operations
3. **Storage**: GitHub has file size limits
4. **Regeneration**: These files can be automatically generated
5. **Personalization**: Different users may want different datasets

## Troubleshooting

### "No dataset files found"
```bash
python missing_files_detector.py training
# Follow the dataset generation instructions
```

### "No model files found"  
```bash
python missing_files_detector.py demo
# Either train a new model or use rule-based mode
```

### "Import error for missing_files_detector"
Make sure you're running from the dual_classifier directory:
```bash
cd dual_classifier
python enhanced_trainer.py
```

## Rule-Based Fallback

The live demo works without trained models by using rule-based classification:
- Category classification based on keywords
- PII detection using regex patterns
- Less accurate but demonstrates the system

Train a model for much better accuracy!

## Need Help?

Run the missing files detector for detailed, context-specific help:
```bash
python missing_files_detector.py           # General check
python missing_files_detector.py training  # Training-specific
python missing_files_detector.py demo      # Demo-specific
``` 