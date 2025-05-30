#!/usr/bin/env python3
"""
Task 3 Demo: Enhanced Training Pipeline with Hardware Detection

This demo showcases the complete Task 3 implementation:
1. Hardware capability detection
2. Real dataset loading with multiple formats
3. Enhanced training pipeline with mixed precision
4. Automatic optimization based on hardware
5. Comprehensive metrics and checkpointing

Run this script to see Task 3 in action!
"""

import os
import sys
import json
import torch
import warnings
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dual_classifier import DualClassifier
from hardware_detector import detect_and_configure, estimate_training_time
from dataset_loaders import RealDatasetLoader, load_custom_dataset
from enhanced_trainer import EnhancedDualTaskTrainer, create_sample_real_dataset


def create_demo_datasets():
    """Create comprehensive demo datasets in multiple formats."""
    print("📊 Creating demo datasets in multiple formats...")
    
    # JSON format dataset
    json_data = []
    categories = ['technology', 'healthcare', 'finance', 'education', 'retail']
    
    # Technology samples
    tech_samples = [
        {"text": "AI and machine learning are revolutionizing software development", "category": "technology"},
        {"text": "Contact our AI team at ai-support@techcorp.com for implementation details", "category": "technology"},
        {"text": "Cloud computing infrastructure scales automatically based on demand", "category": "technology"},
        {"text": "Call our support team at 555-TECH-123 for technical assistance", "category": "technology"},
        {"text": "Cybersecurity protocols protect sensitive data from breaches", "category": "technology"},
    ]
    
    # Healthcare samples  
    health_samples = [
        {"text": "Telemedicine allows remote patient consultations via video calls", "category": "healthcare"},
        {"text": "Patient John Doe with ID 12345 requires immediate attention", "category": "healthcare"},
        {"text": "Electronic health records improve patient care coordination", "category": "healthcare"},
        {"text": "Contact Dr. Smith at dsmith@hospital.org for consultation", "category": "healthcare"},
        {"text": "Medical imaging technology enhances diagnostic accuracy", "category": "healthcare"},
    ]
    
    # Finance samples
    finance_samples = [
        {"text": "Cryptocurrency markets show high volatility patterns", "category": "finance"},
        {"text": "Account holder Sarah Johnson (SSN: 123-45-6789) requested statement", "category": "finance"},
        {"text": "Digital banking services offer 24/7 account access", "category": "finance"},
        {"text": "For investment advice, email advisor@financefirm.com", "category": "finance"},
        {"text": "Blockchain technology ensures transaction transparency", "category": "finance"},
    ]
    
    # Education samples
    edu_samples = [
        {"text": "Online learning platforms increase educational accessibility", "category": "education"},
        {"text": "Student Emily Wilson (ID: STU456789) enrolled in Computer Science", "category": "education"},
        {"text": "Virtual classrooms enable remote learning experiences", "category": "education"},
        {"text": "Contact admissions at admissions@university.edu for applications", "category": "education"},
        {"text": "Adaptive learning systems personalize educational content", "category": "education"},
    ]
    
    # Retail samples
    retail_samples = [
        {"text": "E-commerce platforms drive online retail growth", "category": "retail"},
        {"text": "Customer order #ORD-789456 shipped to 123 Main St, Boston MA", "category": "retail"},
        {"text": "Supply chain optimization reduces delivery times", "category": "retail"},
        {"text": "For returns, contact support at returns@retailstore.com", "category": "retail"},
        {"text": "Inventory management systems track product availability", "category": "retail"},
    ]
    
    # Combine all samples
    all_samples = tech_samples + health_samples + finance_samples + edu_samples + retail_samples
    
    # Create training set (80% of data)
    train_size = int(0.8 * len(all_samples))
    train_data = all_samples[:train_size]
    val_data = all_samples[train_size:]
    
    # Save JSON datasets
    with open("demo_train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open("demo_val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    # Create CSV format dataset
    import csv
    with open("demo_train.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'category'])
        writer.writeheader()
        writer.writerows(train_data)
    
    with open("demo_val.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'category'])
        writer.writeheader()
        writer.writerows(val_data)
    
    print(f"✅ Created datasets:")
    print(f"   📄 JSON format: demo_train.json ({len(train_data)} samples), demo_val.json ({len(val_data)} samples)")
    print(f"   📄 CSV format: demo_train.csv ({len(train_data)} samples), demo_val.csv ({len(val_data)} samples)")
    
    return len(train_data), len(val_data)


def demo_hardware_detection():
    """Demonstrate hardware detection capabilities."""
    print("\n" + "="*60)
    print("🔍 HARDWARE DETECTION DEMO")
    print("="*60)
    
    capabilities, config = detect_and_configure()
    
    print(f"\n💡 Training recommendations:")
    print(f"   ⚙️ Optimal batch size: {config['batch_size']}")
    print(f"   ⚙️ Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"   ⚙️ Mixed precision: {config['use_mixed_precision']}")
    print(f"   ⚙️ Learning rate: {config['learning_rate']}")
    print(f"   ⚙️ DataLoader workers: {config['num_workers']}")
    
    # Show estimated training time
    estimated_time = estimate_training_time(100, capabilities, num_epochs=3)
    print(f"   ⏱️ Est. training time (100 samples, 3 epochs): {estimated_time}")
    
    return capabilities, config


def demo_dataset_loading():
    """Demonstrate real dataset loading capabilities."""
    print("\n" + "="*60)
    print("📊 DATASET LOADING DEMO")
    print("="*60)
    
    loader = RealDatasetLoader()
    
    # Test JSON format
    print("\n📄 Loading JSON dataset...")
    texts, categories, pii_labels, info = load_custom_dataset("demo_train.json")
    loader.print_dataset_info(info)
    
    print(f"Sample data:")
    for i in range(min(3, len(texts))):
        print(f"   {i+1}. Text: {texts[i][:50]}...")
        print(f"      Category: {categories[i]}")
        print(f"      PII detected: {any(pii_labels[i])}")
    
    # Test CSV format  
    print("\n📄 Loading CSV dataset...")
    csv_texts, csv_categories, csv_pii_labels, csv_info = load_custom_dataset("demo_train.csv")
    print(f"✅ CSV loaded: {len(csv_texts)} samples")
    
    # Test format detection
    print("\n🔍 Testing automatic format detection...")
    detected_format = loader.detect_format("demo_train.json")
    print(f"   demo_train.json detected as: {detected_format}")
    detected_format = loader.detect_format("demo_train.csv")
    print(f"   demo_train.csv detected as: {detected_format}")
    
    return texts, categories, pii_labels


def demo_enhanced_training(capabilities, config):
    """Demonstrate enhanced training pipeline."""
    print("\n" + "="*60)
    print("🚀 ENHANCED TRAINING DEMO")
    print("="*60)
    
    # Initialize model
    print("🤖 Initializing DualClassifier model...")
    model = DualClassifier(num_categories=5)  # 5 categories in our demo
    
    print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create enhanced trainer
    print("\n⚙️ Setting up EnhancedDualTaskTrainer...")
    trainer = EnhancedDualTaskTrainer(
        model=model,
        train_dataset_path="demo_train.json",
        val_dataset_path="demo_val.json",
        auto_detect_hardware=True,
        output_dir="./task3_demo_output",
        category_weight=1.0,
        pii_weight=1.5,  # Slightly higher weight for PII detection
    )
    
    # Show training configuration
    print(f"\n📋 Training Configuration:")
    for key, value in trainer.config.items():
        print(f"   {key}: {value}")
    
    # Run training
    print(f"\n🎯 Starting training...")
    try:
        # Train for 2 epochs for demo purposes
        history = trainer.train(num_epochs=2, save_best_model=True)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📈 Final metrics:")
        if history['val_loss']:
            print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
            print(f"   Final category accuracy: {history['val_category_acc'][-1]:.4f}")
            print(f"   Final PII F1 score: {history['val_pii_f1'][-1]:.4f}")
        
        return True, trainer
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False, None


def demo_model_inference(trainer):
    """Demonstrate trained model inference."""
    if not trainer:
        print("⚠️ Skipping inference demo - no trained model available")
        return
    
    print("\n" + "="*60)
    print("🎯 MODEL INFERENCE DEMO")
    print("="*60)
    
    # Test samples with different categories and PII content
    test_samples = [
        "Machine learning algorithms improve software performance",
        "Contact support at help@company.com for technical issues",
        "Patient records contain sensitive medical information",
        "Call Dr. Johnson at 555-MED-HELP for appointments",
        "Investment portfolios require careful risk assessment",
        "Account number 123456789 shows unusual transaction activity"
    ]
    
    model = trainer.model
    model.eval()
    
    print("🔍 Running inference on test samples...")
    
    with torch.no_grad():
        for i, text in enumerate(test_samples, 1):
            # Tokenize
            inputs = model.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=model.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(trainer.device)
            attention_mask = inputs['attention_mask'].to(trainer.device)
            
            # Forward pass
            category_logits, pii_logits = model(input_ids, attention_mask)
            
            # Get predictions
            category_pred = torch.argmax(category_logits, dim=1).item()
            pii_preds = torch.argmax(pii_logits, dim=2)[0]
            
            # Count PII tokens
            valid_length = attention_mask[0].sum().item()
            pii_tokens = pii_preds[:valid_length].sum().item()
            
            print(f"\n   {i}. Text: {text}")
            print(f"      Category: {category_pred}")
            print(f"      PII tokens detected: {pii_tokens}/{valid_length}")
            print(f"      PII ratio: {pii_tokens/valid_length:.2%}")


def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_train.json", "demo_val.json",
        "demo_train.csv", "demo_val.csv"
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
    
    print(f"\n🧹 Cleaned up demo files")


def main():
    """Run the complete Task 3 demo."""
    print("🚀 TASK 3: ENHANCED TRAINING PIPELINE DEMO")
    print("="*60)
    print("This demo showcases:")
    print("✅ Hardware capability detection")
    print("✅ Real dataset loading (JSON, CSV)")
    print("✅ Enhanced training with mixed precision")
    print("✅ Automatic hardware optimization")
    print("✅ Comprehensive metrics and checkpointing")
    print("✅ Model inference")
    
    try:
        # Create datasets
        train_size, val_size = create_demo_datasets()
        
        # Demo hardware detection
        capabilities, config = demo_hardware_detection()
        
        # Demo dataset loading
        texts, categories, pii_labels = demo_dataset_loading()
        
        # Demo enhanced training
        success, trainer = demo_enhanced_training(capabilities, config)
        
        # Demo inference
        if success:
            demo_model_inference(trainer)
        
        print("\n" + "="*60)
        print("🎉 TASK 3 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"   ✅ Hardware detection: {capabilities.device.upper()}")
        print(f"   ✅ Dataset loading: {train_size} train, {val_size} val samples")
        print(f"   ✅ Training pipeline: {'Success' if success else 'Failed'}")
        print(f"   ✅ Mixed precision: {config['use_mixed_precision']}")
        print(f"   ✅ Optimal batch size: {config['batch_size']}")
        
        # Show output directory contents
        output_dir = Path("./task3_demo_output")
        if output_dir.exists():
            print(f"\n📁 Generated outputs in {output_dir}:")
            for item in output_dir.rglob("*"):
                if item.is_file():
                    print(f"   📄 {item.relative_to(output_dir)}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_demo_files()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main() 