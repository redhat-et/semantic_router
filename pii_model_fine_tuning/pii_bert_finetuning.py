"""
PII Classification Fine-tuning with Multiple BERT Models
Uses the simplified Hugging Face Transformers approach with AutoModelForSequenceClassification.

Usage:
    # Train with default model (MiniLM)
    python pii_bert_finetuning.py --mode train

    # Train with BERT base
    python pii_bert_finetuning.py --mode train --model bert-base

    # Train with DeBERTa v3
    python pii_bert_finetuning.py --mode train --model deberta-v3-base

    # Train with ModernBERT (recommended for best performance)
    python pii_bert_finetuning.py --mode train --model modernbert-base

    # Train with custom epochs and batch size
    python pii_bert_finetuning.py --mode train --model modernbert-base --epochs 10 --batch-size 32

    # Quick training for testing
    python pii_bert_finetuning.py --mode train --model distilbert --epochs 2 --batch-size 8

    # Test inference with trained model
    python pii_bert_finetuning.py --mode test --model bert-base

Supported models:
    - bert-base, bert-large: Standard BERT models
    - roberta-base, roberta-large: RoBERTa models
    - deberta-v3-base, deberta-v3-large: DeBERTa v3 models
    - modernbert-base, modernbert-large: ModernBERT models (recommended)
    - minilm: Lightweight MiniLM model (default for compatibility)
    - distilbert: Distilled BERT
    - electra-base, electra-large: ELECTRA models

Features:
    - Automatic classification head via AutoModelForSequenceClassification
    - Simplified training with Hugging Face Trainer
    - Built-in evaluation metrics (F1 score, accuracy)
    - Support for multiple BERT-based architectures
    - Automatic device detection (GPU/CPU)
"""

import os
import json
import torch
import numpy as np
import requests
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    __version__ as transformers_version
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check transformers version and compatibility
def check_transformers_compatibility():
    """Check transformers version and provide helpful messages."""
    logger.info(f"Transformers version: {transformers_version}")
    
    # Parse version to determine parameter names
    version_parts = transformers_version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    # For versions < 4.19, use evaluation_strategy; for >= 4.19, use eval_strategy
    if major < 4 or (major == 4 and minor < 19):
        return "evaluation_strategy"
    else:
        return "eval_strategy"

# Device configuration - prioritize GPU if available
def get_device():
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        logger.warning("No GPU detected. Using CPU. For better performance, ensure CUDA is installed.")
    
    logger.info(f"Using device: {device}")
    return device

# Model configurations for different BERT variants
MODEL_CONFIGS = {
    'bert-base': 'bert-base-uncased',
    'bert-large': 'bert-large-uncased',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
    'deberta-v3-base': 'microsoft/deberta-v3-base',
    'deberta-v3-large': 'microsoft/deberta-v3-large',
    'modernbert-base': 'answerdotai/ModernBERT-base',
    'modernbert-large': 'answerdotai/ModernBERT-large',
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2',  # Default fallback
    'distilbert': 'distilbert-base-uncased',
    'electra-base': 'google/electra-base-discriminator',
    'electra-large': 'google/electra-large-discriminator'
}

# Metrics computation function for Trainer
def compute_metrics(eval_pred):
    """Compute F1 score and accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": accuracy}

class PII_Dataset:
    """Dataset class for PII sequence classification fine-tuning."""
    
    def __init__(self, data_dir="presidio_pii_data"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing the generated PII data
        """
        self.data_dir = Path(data_dir)
        self.label2id = {}
        self.id2label = {}
        
    def download_presidio_dataset(self):
        """Download the Microsoft Presidio research dataset."""
        url = "https://raw.githubusercontent.com/microsoft/presidio-research/refs/heads/master/data/synth_dataset_v2.json"
        dataset_path = "presidio_synth_dataset_v2.json"
        
        if not Path(dataset_path).exists():
            logger.info(f"Downloading Presidio dataset from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"Dataset downloaded to {dataset_path}")
        else:
            logger.info(f"Dataset already exists at {dataset_path}")
        
        return dataset_path
        
    def load_presidio_json(self, file_path):
        """Load and parse Presidio JSON format and convert to sequence classification."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for sample in data:
            text = sample['full_text']
            spans = sample.get('spans', [])
            
            # Extract entity types from spans
            entity_types = [span['entity_type'] for span in spans]
            
            if not entity_types:
                dominant_label = 'NO_PII'
            else:
                # If multiple PII types, choose the most frequent one
                # For classification, we assign one label per text
                dominant_label = max(set(entity_types), key=entity_types.count)
            
            texts.append(text)
            labels.append(dominant_label)
        
        return texts, labels
    
    def split_dataset(self, texts, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Split the dataset into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Check class distribution and filter rare classes if needed
        class_counts = Counter(labels)
        
        # Remove classes with less than 3 samples for stratified splitting
        min_samples_per_class = 3
        filtered_data = []
        for text, label in zip(texts, labels):
            if class_counts[label] >= min_samples_per_class:
                filtered_data.append((text, label))
        
        if len(filtered_data) < len(texts):
            removed_count = len(texts) - len(filtered_data)
            rare_classes = [cls for cls, count in class_counts.items() if count < min_samples_per_class]
            logger.warning(f"Removed {removed_count} samples from rare classes: {rare_classes}")
        
        # Unpack filtered data
        filtered_texts, filtered_labels = zip(*filtered_data) if filtered_data else ([], [])
        filtered_texts, filtered_labels = list(filtered_texts), list(filtered_labels)
        
        try:
            # First split: train and temp (val + test)
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                filtered_texts, filtered_labels, test_size=(val_ratio + test_ratio), 
                random_state=random_state, stratify=filtered_labels
            )
            
            # Second split: val and test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=(1 - val_size), 
                random_state=random_state, stratify=temp_labels
            )
            
        except ValueError as e:
            # Fall back to non-stratified splitting if stratified fails
            logger.warning(f"Stratified split failed: {e}. Using random split instead.")
            
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                filtered_texts, filtered_labels, test_size=(val_ratio + test_ratio), 
                random_state=random_state
            )
            
            val_size = val_ratio / (val_ratio + test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=(1 - val_size), 
                random_state=random_state
            )
        
        return {
            'train': (train_texts, train_labels),
            'validation': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
    
    def create_label_mappings(self, all_labels):
        """Create label to ID mappings."""
        unique_labels = sorted(list(set(all_labels)))
        
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        logger.info(f"Created mappings for {len(unique_labels)} labels: {unique_labels}")
        
    def prepare_datasets(self):
        """Prepare train/validation/test datasets from Presidio dataset."""
        
        # Download dataset if needed
        dataset_path = self.download_presidio_dataset()
        
        # Load the full dataset
        logger.info("Loading Presidio dataset...")
        texts, labels = self.load_presidio_json(dataset_path)
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Label distribution: {dict(sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True))}")
        
        # Split the dataset
        logger.info("Splitting dataset into train/validation/test...")
        datasets = self.split_dataset(texts, labels)
        
        train_texts, train_labels = datasets['train']
        val_texts, val_labels = datasets['validation']
        test_texts, test_labels = datasets['test']
        
        # Create label mappings
        all_labels = train_labels + val_labels + test_labels
        self.create_label_mappings(all_labels)
        
        # Convert labels to IDs
        train_label_ids = [self.label2id[label] for label in train_labels]
        val_label_ids = [self.label2id[label] for label in val_labels]
        test_label_ids = [self.label2id[label] for label in test_labels]
        
        return {
            'train': (train_texts, train_label_ids),
            'validation': (val_texts, val_label_ids),
            'test': (test_texts, test_label_ids)
        }

# Function to predict PII type using the classification model
def predict_pii_type(model, tokenizer, text, idx_to_label_map, device):
    """Predict PII type for a given text."""
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted_idx = torch.max(probabilities, dim=-1)
    
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    
    predicted_pii_type = idx_to_label_map.get(predicted_idx, "Unknown PII Type")
    
    return predicted_pii_type, confidence

# Evaluate on validation set using the classification model
def evaluate_pii_classifier(model, tokenizer, texts_list, true_label_indices_list, idx_to_label_map, device):
    """Evaluate the PII classifier on a dataset."""
    correct = 0
    total = len(texts_list)
    
    if total == 0:
        return 0.0

    for text, true_label_idx in zip(texts_list, true_label_indices_list):
        predicted_pii_type, _ = predict_pii_type(model, tokenizer, text, idx_to_label_map, device)
        true_pii_type = idx_to_label_map.get(true_label_idx)
        
        if true_pii_type == predicted_pii_type:
            correct += 1
    
    return correct / total

def main(model_name="minilm", num_epochs=5, batch_size=16):
    """Main function to demonstrate PII classification fine-tuning."""
    
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    # Set up device (GPU if available)
    device = get_device()
    
    model_path = MODEL_CONFIGS[model_name]
    logger.info(f"Using model: {model_name} ({model_path})")
    logger.info(f"Training configuration: {num_epochs} epochs, batch size {batch_size}")
    
    logger.info("Loading Presidio PII dataset...")
    dataset_loader = PII_Dataset()
    datasets = dataset_loader.prepare_datasets()
    
    train_texts, train_categories = datasets['train']
    val_texts, val_categories = datasets['validation']
    test_texts, test_categories = datasets['test']
    
    unique_categories = list(dataset_loader.label2id.keys())
    category_to_idx = dataset_loader.label2id
    idx_to_category = dataset_loader.id2label

    logger.info(f"Found {len(unique_categories)} unique PII types: {unique_categories}")
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_texts)}")
    logger.info(f"  Validation: {len(val_texts)}")
    logger.info(f"  Test: {len(test_texts)}")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    num_labels = len(unique_categories)
    
    # Suppress the expected warning about newly initialized classifier weights
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*classifier.*newly initialized.*")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            label2id=category_to_idx,
            id2label=idx_to_category
        )
    
    # Move model to device
    model.to(device)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_categories})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_categories})
    test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_categories})
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Check transformers version compatibility
    eval_strategy_param = check_transformers_compatibility()
    
    # Training arguments
    output_model_path = f"pii_classifier_{model_name}_model"
    training_args_dict = {
        "output_dir": output_model_path,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": f"{output_model_path}/logs",
        "logging_steps": 100,
        eval_strategy_param: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "save_total_limit": 2,
        "report_to": [],
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting PII classification fine-tuning with {model_name}...")

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Save the PII type mapping
    pii_mapping_path = os.path.join(output_model_path, "pii_type_mapping.json")
    with open(pii_mapping_path, "w") as f:
        json.dump({
            "label_to_idx": category_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_category.items()} # JSON keys must be strings
        }, f)

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_accuracy = evaluate_pii_classifier(model, tokenizer, val_texts, val_categories, idx_to_category, device)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_accuracy = evaluate_pii_classifier(model, tokenizer, test_texts, test_categories, idx_to_category, device)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Print final results
    print("\n" + "="*50)
    print("PII Classification Fine-tuning Completed!")
    print("="*50)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, tokenizer, idx_to_category

def demo_inference(model_name="minilm"):
    """Demonstrate inference with the trained model."""
    
    # Set up device (GPU if available)
    device = get_device()
    
    model_path = f"./pii_classifier_{model_name}_model"
    if not Path(model_path).exists():
        logger.error(f"Trained model not found at {model_path}. Please run training first with --model {model_name}")
        return
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    
    mapping_path = os.path.join(model_path, "pii_type_mapping.json")
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
        idx_to_label = {int(k): v for k, v in mappings["idx_to_label"].items()}
    
    print("\n" + "="*50)
    print("PII Detection Test")
    print("="*50)
    
    test_texts = [
        "My name is John Smith and my email is john.smith@example.com.",
        "Please call me at (555) 123-4567 for more information.",
        "My social security number is 123-45-6789.",
        "Visit our website at https://example.com for more details.",
        "This is a normal sentence without any personal information.",
        "My credit card number is 4532-1234-5678-9012.",
        "I live at 123 Main Street, New York, NY 10001.",
        "My date of birth is January 15, 1990."
    ]
    
    for text in test_texts:
        predicted_pii_type, confidence = predict_pii_type(model, tokenizer, text, idx_to_label, device)
        print(f"Text: {text}")
        print(f"Predicted PII Type: {predicted_pii_type}, Confidence: {confidence:.4f}")
        print("---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Classification Fine-tuning")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: 'train' to fine-tune model, 'test' to run inference")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="minilm", 
                       help="Model to use for fine-tuning (e.g., bert-base, roberta-base, etc.)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training and evaluation batch size (default: 8)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main(args.model, args.epochs, args.batch_size)
    elif args.mode == "test":
        demo_inference(args.model) 