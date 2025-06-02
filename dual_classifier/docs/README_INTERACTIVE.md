# 🚀 Enhanced Interactive Dual Classifier Training

The enhanced trainer now features **interactive dataset and strength selection** with arrow key navigation!

## ✨ New Features

### 📂 Interactive Dataset Selection
- **Auto-detects** available datasets in your directory
- **Arrow key navigation** for easy selection (if `inquirer` is installed)
- **Fallback** to numbered selection if `inquirer` is not available
- **Smart recommendations** if no datasets are found

### ⚡ Interactive Strength Selection
- **Arrow key navigation** through training strengths
- **Clear descriptions** with epoch counts
- **Default selection** (Normal) for quick starts

### 🔍 Dataset Analysis
- **Automatic category detection** from your dataset
- **Sample count display** for both training and validation
- **Category listing** to verify your data

## 🚀 Quick Start

### 1. Install Dependencies (Recommended)
```bash
python install_requirements.py
```

### 2. Download a Dataset (if you don't have one)
```bash
# Choose one:
python download_bbc_dataset.py      # 5 categories (recommended)
python download_20newsgroups.py     # 20 categories (advanced)
python download_agnews.py           # 4 categories (high quality)
```

### 3. Run Interactive Training
```bash
python enhanced_trainer.py
```

## 📊 What You'll See

### Dataset Selection Screen
```
🔍 Detecting available datasets...

📊 Found 2 available dataset(s):
? 📂 Select dataset
> Custom Dataset - 4 categories, 1000 samples
  Extended Dataset (8 categories) - 8 categories, 800 samples
```

### Strength Selection Screen
```
? ⚡ Select training strength
  QUICK - Fast training for testing and prototyping (2 epochs)
> NORMAL - Balanced training for good results in reasonable time (5 epochs)
  INTENSIVE - Thorough training for high-quality results (10 epochs)
  MAXIMUM - Maximum quality training - may take hours (20 epochs)
```

### Training Configuration Summary
```
🚀 Training Configuration:
   📂 Dataset: real_train_dataset.json
   🎯 Training strength: NORMAL
   📊 Categories: 5
   🔄 Expected epochs: 5
   📁 Output directory: ./enhanced_training_normal
   ⏱️ Early stopping patience: 5
```

## 🛠️ Available Datasets

The system automatically detects these dataset files:
- `real_train_dataset.json` / `real_val_dataset.json` - Your custom dataset
- `extended_train_dataset.json` / `extended_val_dataset.json` - Generated 8-category dataset

## 🎯 Training Strengths

| Strength | Epochs | Description | Best For |
|----------|--------|-------------|----------|
| **QUICK** | 2 | Fast testing | Prototyping, testing setup |
| **NORMAL** | 5 | Balanced training | Most use cases |
| **INTENSIVE** | 10 | High quality | Production models |
| **MAXIMUM** | 20 | Maximum quality | Research, competition |

## 🚨 Troubleshooting

### No Datasets Found?
If you see "❌ No datasets found!", run one of the download scripts:
```bash
python download_bbc_dataset.py      # Recommended start
```

### No Arrow Keys?
Install `inquirer` for better navigation:
```bash
pip install inquirer
```
The system will fallback to numbered selection without it.

### Training Fails?
- Check you have enough memory for the selected batch size
- Try "quick" strength for initial testing
- Ensure your dataset format is correct (JSON with 'text' and 'category' fields)

## 📁 Output Structure

After training, you'll find:
```
enhanced_training_normal/
├── checkpoints/          # Training checkpoints
├── final_model/         # Your trained model
└── training_history.json # Training metrics
```

## 🎉 Next Steps

1. Use your trained model in `live_demo.py`
2. Experiment with different datasets and strengths
3. Compare results across different configurations
4. Deploy your best model!

---

**Happy Training! 🚀** 