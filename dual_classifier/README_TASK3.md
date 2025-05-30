# Task 3: Enhanced Training Pipeline with Hardware Detection

This directory contains the complete implementation of **Task 3: Implement Training Pipeline for Dual-Purpose Model** with advanced hardware capability detection and real dataset support.

## 🎯 Task 3 Overview

Task 3 transitions the project from synthetic to real datasets and adds production-ready training capabilities:

- ✅ **Hardware Capability Detection**: Automatically detects and optimizes for your machine's capabilities
- ✅ **Real Dataset Support**: Load datasets in JSON, CSV, CoNLL formats with automatic detection
- ✅ **Enhanced Training Pipeline**: Mixed precision, gradient accumulation, checkpointing
- ✅ **Machine Compatibility**: Ensures code runs successfully on different hardware configurations
- ✅ **Comprehensive Metrics**: Detailed evaluation and monitoring

## 🏗️ Architecture

```
dual_classifier/
├── dual_classifier.py          # Core DualClassifier model (from Task 2)
├── trainer.py                  # Basic trainer components (from Task 2) 
├── data_generator.py           # Synthetic data generator (from Task 2)
├── hardware_detector.py        # NEW: Hardware capability detection
├── dataset_loaders.py          # NEW: Real dataset loading utilities
├── enhanced_trainer.py         # NEW: Enhanced training pipeline
├── task3_demo.py              # NEW: Complete Task 3 demonstration
└── README_TASK3.md            # This file
```

## 🔧 New Components

### 1. Hardware Detector (`hardware_detector.py`)

Automatically detects hardware capabilities and provides optimal training configurations:

```python
from hardware_detector import detect_and_configure

# Automatic hardware detection
capabilities, config = detect_and_configure()

# Shows device, memory, optimal batch size, mixed precision support, etc.
```

**Features:**
- GPU/CPU detection with memory analysis
- Apple Silicon (MPS) support
- Mixed precision capability detection
- Optimal batch size calculation
- Training time estimation
- Hardware-specific warnings and recommendations

### 2. Dataset Loaders (`dataset_loaders.py`)

Support for real datasets with automatic format detection:

```python
from dataset_loaders import load_custom_dataset

# Load any supported format
texts, categories, pii_labels, info = load_custom_dataset("dataset.json")
```

**Supported Formats:**
- **JSON**: Single or multi-document format with configurable field names
- **CSV**: With configurable column mapping and delimiters
- **CoNLL**: Named Entity Recognition format support
- **Auto-detection**: Automatically identifies format from file extension/content

### 3. Enhanced Trainer (`enhanced_trainer.py`)

Production-ready training pipeline with advanced features:

```python
from enhanced_trainer import EnhancedDualTaskTrainer

trainer = EnhancedDualTaskTrainer(
    model=model,
    train_dataset_path="train.json",
    val_dataset_path="val.json",
    auto_detect_hardware=True
)

history = trainer.train(num_epochs=3, save_best_model=True)
```

**Features:**
- Automatic hardware optimization
- Mixed precision training (when supported)
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling with warmup
- Automatic checkpointing and model selection
- Comprehensive metrics (accuracy, F1, combined scores)
- Training history tracking

## 🚀 Quick Start

### 1. Run the Complete Demo

The easiest way to see Task 3 in action:

```bash
cd dual_classifier
python task3_demo.py
```

This will:
- Detect your hardware capabilities
- Create sample datasets in multiple formats
- Run the enhanced training pipeline
- Show model inference results
- Display comprehensive metrics

### 2. Use with Your Own Data

```python
# For JSON format
from enhanced_trainer import EnhancedDualTaskTrainer
from dual_classifier import DualClassifier

model = DualClassifier(num_categories=5)
trainer = EnhancedDualTaskTrainer(
    model=model,
    train_dataset_path="your_train.json",
    val_dataset_path="your_val.json"
)
history = trainer.train(num_epochs=5)
```

### 3. Manual Hardware Configuration

```python
# Override automatic hardware detection
trainer = EnhancedDualTaskTrainer(
    model=model,
    train_dataset_path="train.json",
    auto_detect_hardware=False,
    batch_size=16,
    use_mixed_precision=True,
    learning_rate=3e-5
)
```

## 📊 Dataset Format Examples

### JSON Format
```json
[
  {
    "text": "Contact support at help@company.com for assistance",
    "category": "technology"
  },
  {
    "text": "Patient John Doe requires immediate attention",
    "category": "healthcare"
  }
]
```

### CSV Format
```csv
text,category
"AI and machine learning are advancing rapidly",technology
"Patient records must be kept confidential",healthcare
```

### Custom Field Mapping
```python
# For custom JSON structure
texts, categories, pii_labels, info = load_custom_dataset(
    "dataset.json",
    text_field="content",
    category_field="label"
)

# For custom CSV structure  
texts, categories, pii_labels, info = load_custom_dataset(
    "dataset.csv",
    text_column=0,
    category_column=2
)
```

## 🔍 Hardware Capability Detection

The hardware detector automatically optimizes training for your system:

### Supported Configurations

| Hardware | Device | Mixed Precision | Batch Size | Notes |
|----------|--------|----------------|------------|-------|
| NVIDIA GPU (≥8GB) | `cuda` | ✅ FP16 | 8-32 | Optimal performance |
| NVIDIA GPU (<8GB) | `cuda` | ✅ FP16 | 2-4 | Memory-constrained |
| Apple Silicon M1/M2 | `mps` | ❌ | 4-8 | Good performance |
| Intel/AMD CPU | `cpu` | ❌ | 1-4 | Slower but compatible |

### Sample Output
```
🔍 Hardware Detection Results:
┌─ Device: CUDA
├─ Device Name: NVIDIA RTX 3080
├─ Available Memory: 10.0GB
├─ CPU Cores: 16
├─ Mixed Precision: ✅ Supported
├─ Recommended Batch Size: 16
├─ Gradient Accumulation Steps: 1
└─ DataLoader Workers: 4

⏱️ Estimated training time for 1000 samples: ~2 minutes
```

## 🏃‍♂️ Performance Optimizations

### Automatic Optimizations
- **Memory Management**: Optimal batch sizes based on available memory
- **Mixed Precision**: FP16 training when hardware supports it
- **Gradient Accumulation**: Achieve large effective batch sizes on limited memory
- **DataLoader Tuning**: Optimal number of workers based on CPU cores
- **Device Placement**: Automatic tensors placement with non-blocking transfers

### Manual Optimizations
```python
# Override for specific needs
trainer = EnhancedDualTaskTrainer(
    model=model,
    train_dataset_path="train.json",
    batch_size=32,           # Larger batch size
    gradient_accumulation_steps=2,  # Effective batch = 64
    use_mixed_precision=True,       # Force FP16
    num_workers=8,          # More DataLoader workers
    pin_memory=True         # Faster GPU transfers
)
```

## 📈 Training Monitoring

### Real-time Progress
```
Epoch 1/3: 100%|██████████| 25/25 [00:45<00:00, 1.81s/it, loss=0.1234, cat_loss=0.0567, pii_loss=0.0667, lr=2.00e-05]
```

### Comprehensive Metrics
```
📈 Epoch 2/3
   📊 Train Loss: 0.1234
   📊 Val Loss: 0.1456
   📊 Category Acc: 0.8750
   📊 PII F1: 0.7890
   📊 Combined Score: 0.8320
✅ New best model saved! Combined score: 0.8320
```

### Saved Outputs
```
task3_output/
├── checkpoints/
│   ├── best_model.pt
│   ├── epoch-1.pt
│   └── epoch-2.pt
├── final_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_config.json
└── training_history.json
```

## 🧪 Testing

### Hardware Detection Test
```python
from hardware_detector import detect_and_configure

capabilities, config = detect_and_configure()
print(f"Detected device: {capabilities.device}")
print(f"Optimal batch size: {config['batch_size']}")
```

### Dataset Loading Test
```python
from dataset_loaders import load_custom_dataset

texts, categories, pii_labels, info = load_custom_dataset("test.json")
print(f"Loaded {len(texts)} samples")
print(f"Categories: {info.num_categories}")
print(f"Has PII labels: {info.has_pii_labels}")
```

### Training Pipeline Test
```python
# Quick training test
trainer = EnhancedDualTaskTrainer(
    model=DualClassifier(num_categories=3),
    train_dataset_path="small_train.json",
    output_dir="./test_output"
)
history = trainer.train(num_epochs=1)
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
```python
# Solution: Reduce batch size or enable gradient accumulation
trainer = EnhancedDualTaskTrainer(
    model=model,
    batch_size=4,  # Smaller batch size
    gradient_accumulation_steps=4  # Maintain effective batch size
)
```

**Issue**: `MPS backend not available`
```python
# Solution: Fallback to CPU automatically handled
# Or force CPU:
trainer = EnhancedDualTaskTrainer(
    model=model,
    auto_detect_hardware=False,
    device='cpu'
)
```

**Issue**: `Dataset format not recognized`
```python
# Solution: Specify format explicitly
texts, categories, pii_labels, info = load_custom_dataset(
    "dataset.txt",
    format="json"  # Force format
)
```

### Performance Tips

1. **Use GPU when available**: Automatic detection handles this
2. **Enable mixed precision**: Automatically enabled when supported
3. **Optimize batch size**: Hardware detector finds optimal size
4. **Monitor memory usage**: Watch for warnings in output
5. **Use gradient accumulation**: For large effective batch sizes

## 🎯 Task 3 Completion Checklist

- ✅ **Hardware Detection**: Automatic capability detection and optimization
- ✅ **Real Dataset Loading**: Support for JSON, CSV, CoNLL formats  
- ✅ **Enhanced Training**: Mixed precision, gradient accumulation, checkpointing
- ✅ **Machine Compatibility**: Runs on CPU, CUDA, MPS devices
- ✅ **Format Standardization**: Consistent data format handling
- ✅ **Comprehensive Demo**: Complete working example
- ✅ **Documentation**: Detailed usage instructions
- ✅ **Error Handling**: Graceful fallbacks and warnings

## 🔗 Integration with Previous Tasks

Task 3 builds upon and enhances the foundation from Tasks 1-2:

- **Task 1**: Repository setup and structure ✅
- **Task 2**: DualClassifier model and basic training ✅  
- **Task 3**: Enhanced training with hardware detection ✅

The enhanced trainer maintains backward compatibility with Task 2 components while adding significant new capabilities for production use.

---

**Ready to train your dual-purpose classifier with automatic hardware optimization!** 🚀 