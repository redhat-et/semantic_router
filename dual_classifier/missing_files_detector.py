#!/usr/bin/env python3
"""
Missing Files Detector for Dual Classifier

This module detects missing large files that were excluded from git
and provides helpful instructions on how to obtain them.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MissingFilesDetector:
    """
    Detects missing files and provides download/generation instructions.
    """
    
    def __init__(self):
        self.current_dir = Path(".")
        self.dataset_generators = {
            "20newsgroups": {
                "file": "datasets/generators/download_20newsgroups.py",
                "description": "20 Newsgroups dataset (20 categories)",
                "command": "python datasets/generators/download_20newsgroups.py",
                "expected_files": ["real_train_dataset.json", "real_val_dataset.json"],
                "size": "~7.6MB train + 1.9MB val"
            },
            "agnews": {
                "file": "datasets/generators/download_agnews.py", 
                "description": "AG News dataset (4 categories)",
                "command": "python datasets/generators/download_agnews.py",
                "expected_files": ["real_train_dataset.json", "real_val_dataset.json"],
                "size": "~3MB train + 0.8MB val"
            },
            "bbc": {
                "file": "datasets/generators/download_bbc_dataset.py",
                "description": "BBC News dataset (5 categories)", 
                "command": "python datasets/generators/download_bbc_dataset.py",
                "expected_files": ["real_train_dataset.json", "real_val_dataset.json"],
                "size": "~1.5MB train + 0.4MB val"
            },
            "custom": {
                "file": "datasets/real_data_loader.py",
                "description": "Custom dataset from your data",
                "command": "python datasets/real_data_loader.py",
                "expected_files": ["real_train_dataset.json", "real_val_dataset.json"],
                "size": "Variable"
            }
        }
    
    def check_dataset_files(self) -> Tuple[bool, List[str]]:
        """
        Check if required dataset files exist.
        
        Returns:
            (files_exist, missing_files)
        """
        required_files = ["real_train_dataset.json", "real_val_dataset.json"]
        missing_files = []
        
        for file in required_files:
            # Check in current directory
            if not os.path.exists(file):
                # Check in datasets directory  
                datasets_path = self.current_dir / "datasets" / file
                if not datasets_path.exists():
                    missing_files.append(file)
        
        return len(missing_files) == 0, missing_files
    
    def check_model_files(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check if any trained model files exist.
        
        Returns:
            (models_exist, missing_locations, available_models)
        """
        model_locations = [
            "training_outputs/trained_model/model.pt",
            "training_outputs/task3_demo_output/final_model/model.pt", 
            "dual_classifier_checkpoint.pth",
            "trained_model/model.pt"
        ]
        
        checkpoint_locations = [
            "training_outputs/task3_demo_output/checkpoints/best_model.pt",
            "training_outputs/task3_demo_output/checkpoints/epoch-1.pt",
            "training_outputs/task3_demo_output/checkpoints/epoch-2.pt"
        ]
        
        available_models = []
        missing_locations = []
        
        all_locations = model_locations + checkpoint_locations
        
        for location in all_locations:
            if os.path.exists(location):
                available_models.append(location)
            else:
                missing_locations.append(location)
        
        return len(available_models) > 0, missing_locations, available_models
    
    def print_dataset_help(self, missing_files: List[str]):
        """Print helpful instructions for missing dataset files."""
        print("❌ Missing Dataset Files:")
        for file in missing_files:
            print(f"   • {file}")
        
        print("\n📋 To generate dataset files, choose one of these options:")
        print("=" * 60)
        
        for name, info in self.dataset_generators.items():
            if os.path.exists(info["file"]):
                print(f"\n🔄 Option {len([x for x in self.dataset_generators.keys() if x <= name])}: {info['description']}")
                print(f"   📂 Expected size: {info['size']}")
                print(f"   ⚡ Run: {info['command']}")
                print(f"   📄 Generates: {', '.join(info['expected_files'])}")
        
        print(f"\n💡 Recommended for beginners: python datasets/generators/download_agnews.py")
        print(f"💡 For more categories: python datasets/generators/download_20newsgroups.py")
        print(f"💡 For custom data: python datasets/real_data_loader.py")
        
        print(f"\n⚠️  Note: These files are large and excluded from git for performance.")
        print(f"🚀 After generating, you can run training with: python enhanced_trainer.py")
    
    def print_model_help(self, missing_locations: List[str], available_models: List[str]):
        """Print helpful instructions for missing model files."""
        if available_models:
            print("✅ Found existing model files:")
            for model in available_models:
                size = self._get_file_size(model)
                print(f"   • {model} ({size})")
            print("\n🚀 You can run the live demo: python live_demo.py")
        else:
            print("❌ No trained model files found.")
            print("\n📋 To get model files, you have these options:")
            print("=" * 50)
            
            print("\n🔄 Option 1: Train a new model")
            print("   1️⃣ First generate dataset files (see dataset help above)")
            print("   2️⃣ Run training: python enhanced_trainer.py normal")
            print("   3️⃣ Models will be saved to training_outputs/")
            print("   💡 Training strengths: quick, normal, intensive, maximum")
            
            print("\n🔄 Option 2: Use existing checkpoints (if available)")
            print("   📂 Check for *.pt files in:")
            for location in missing_locations[:5]:  # Show first 5
                print(f"      • {location}")
            
            print(f"\n⚠️  Note: Model files are large (100MB-800MB) and excluded from git.")
            print(f"🎯 The live_demo.py will work without models using rule-based classification.")
    
    def _get_file_size(self, filepath: str) -> str:
        """Get human-readable file size."""
        try:
            size = os.path.getsize(filepath)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f}{unit}"
                size /= 1024
            return f"{size:.1f}TB"
        except:
            return "unknown size"
    
    def check_vocab_files(self) -> Tuple[bool, List[str]]:
        """Check for vocab files (usually auto-generated with models)."""
        vocab_locations = [
            "training_outputs/trained_model/vocab.txt",
            "training_outputs/task3_demo_output/final_model/vocab.txt"
        ]
        
        missing_vocab = []
        for location in vocab_locations:
            if not os.path.exists(location):
                missing_vocab.append(location)
        
        return len(missing_vocab) == 0, missing_vocab
    
    def print_vocab_help(self, missing_vocab: List[str]):
        """Print help for missing vocab files."""
        print("ℹ️  Missing vocab files (auto-generated during training):")
        for file in missing_vocab:
            print(f"   • {file}")
        print("💡 These are automatically created when you train a model.")
    
    def run_full_check(self, context: str = "general") -> bool:
        """
        Run a full check and print helpful messages.
        
        Args:
            context: "training", "demo", or "general"
            
        Returns:
            True if all required files exist for the context
        """
        print(f"🔍 Checking for missing files (context: {context})...")
        print("=" * 50)
        
        all_good = True
        
        # Check datasets (critical for training)
        datasets_exist, missing_datasets = self.check_dataset_files()
        if not datasets_exist and context in ["training", "general"]:
            self.print_dataset_help(missing_datasets)
            all_good = False
            print()
        
        # Check models (critical for live demo)
        models_exist, missing_models, available_models = self.check_model_files()
        if not models_exist and context in ["demo", "general"]:
            self.print_model_help(missing_models, available_models)
            all_good = False
            print()
        elif models_exist and context in ["demo", "general"]:
            print("✅ Model files found - live demo should work!")
            print()
        
        # Check vocab (informational)
        vocab_exist, missing_vocab = self.check_vocab_files()
        if not vocab_exist and len(missing_vocab) > 0:
            self.print_vocab_help(missing_vocab)
            print()
        
        if all_good:
            print("✅ All required files are present!")
            if context == "training":
                print("🚀 Ready for training: python enhanced_trainer.py normal")
            elif context == "demo":
                print("🚀 Ready for demo: python live_demo.py")
            else:
                print("🚀 Ready for both training and demo!")
        
        return all_good

def check_missing_files(context: str = "general"):
    """
    Convenience function to check for missing files.
    
    Args:
        context: "training", "demo", or "general"
    """
    detector = MissingFilesDetector()
    return detector.run_full_check(context)

if __name__ == "__main__":
    import sys
    
    context = "general"
    if len(sys.argv) > 1:
        context = sys.argv[1]
    
    print("🔍 Dual Classifier Missing Files Detector")
    print("=" * 45)
    print()
    
    check_missing_files(context) 