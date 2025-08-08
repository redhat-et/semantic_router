"""
PII Classification Fine-tuning with Multiple BERT Models
Supports both sequence classification and token classification (NER-style) for PII detection.

Usage:
    # Train sequence classification (default) with MiniLM and Presidio dataset
    python pii_bert_finetuning.py --mode train

    # Train token classification for entity location detection
    python pii_bert_finetuning.py --mode train --task-type token

    # Train with BERT base and AI4Privacy dataset for token classification
    python pii_bert_finetuning.py --mode train --model bert-base --dataset ai4privacy --task-type token

    # Train with DeBERTa v3 and AI4Privacy dataset for sequence classification
    python pii_bert_finetuning.py --mode train --model deberta-v3-base --dataset ai4privacy --task-type sequence

    # Train with ModernBERT for token classification (recommended for best performance)
    python pii_bert_finetuning.py --mode train --model modernbert-base --dataset ai4privacy --task-type token

    # Train with custom target accuracy and batch size for token classification
    python pii_bert_finetuning.py --mode train --model modernbert-base --dataset ai4privacy --task-type token --target-accuracy 0.98 --batch-size 32

    # Quick training for testing with lower accuracy target
    python pii_bert_finetuning.py --mode train --model distilbert --dataset presidio --target-accuracy 0.85 --batch-size 8 --task-type sequence

    # Test inference with trained sequence classification model
    python pii_bert_finetuning.py --mode test --model bert-base --dataset ai4privacy --task-type sequence

    # Test inference with trained token classification model
    python pii_bert_finetuning.py --mode test --model bert-base --dataset ai4privacy --task-type token

    # Train with custom maximum epochs and patience
    python pii_bert_finetuning.py --mode train --model bert-base --dataset ai4privacy --max-epochs 100 --patience 5 --task-type token

    # Force restart training from scratch (ignore checkpoints)
    python pii_bert_finetuning.py --mode train --model bert-base --dataset ai4privacy --force-restart --task-type token

Supported models:
    - bert-base, bert-large: Standard BERT models
    - roberta-base, roberta-large: RoBERTa models
    - deberta-v3-base, deberta-v3-large: DeBERTa v3 models
    - modernbert-base, modernbert-large: ModernBERT models (recommended)
    - minilm: Lightweight MiniLM model (default for compatibility)
    - distilbert: Distilled BERT
    - electra-base, electra-large: ELECTRA models

Supported datasets:
    - presidio: Microsoft Presidio research dataset (default)
    - ai4privacy: AI4Privacy PII masking dataset (300k samples, multilingual)

Features:
    - Dual task support: Sequence classification and Token classification (NER-style)
    - Automatic model selection: AutoModelForSequenceClassification or AutoModelForTokenClassification
    - Token classification with BIO tagging for precise PII entity location detection
    - Simplified training with Hugging Face Trainer
    - Built-in evaluation metrics (F1 score, accuracy, precision, recall)
    - Accuracy-based early stopping with configurable target and patience
    - Support for multiple BERT-based architectures
    - Support for multiple PII datasets with intelligent caching
    - Automatic checkpoint resuming on training failures
    - Force restart option to ignore existing checkpoints
    - Automatic device detection (GPU/CPU)
    - Entity extraction with confidence scores for token classification
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
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    __version__ as transformers_version
)
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
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

# Metrics computation function for Trainer (sequence classification)
def compute_metrics(eval_pred):
    """Compute F1 score and accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": accuracy}

# Metrics computation function for token classification
def compute_metrics_token_classification(eval_pred):
    """Compute metrics for token classification (NER-style evaluation)."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (usually -100) and flatten
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:  # -100 is typically used for ignored tokens
                true_predictions.append(pred_id)
                true_labels.append(label_id)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Custom early stopping callback based on accuracy
class AccuracyEarlyStoppingCallback(TrainerCallback):
    """Custom callback to stop training when target accuracy is reached."""
    
    def __init__(self, target_accuracy=0.95, patience=3):
        """
        Initialize the callback.
        
        Args:
            target_accuracy: Target accuracy to reach (default: 0.95)
            patience: Number of evaluations to wait after reaching target before stopping
        """
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.wait_count = 0
        self.best_accuracy = 0.0
        self.target_reached = False
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after each evaluation."""
        if logs is None:
            return
            
        current_accuracy = logs.get("eval_accuracy", 0.0)
        
        # Update best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            
        # Check if target accuracy is reached
        if current_accuracy >= self.target_accuracy:
            if not self.target_reached:
                logger.info(f"Target accuracy {self.target_accuracy:.4f} reached! Current accuracy: {current_accuracy:.4f}")
                self.target_reached = True
                self.wait_count = 0
            else:
                self.wait_count += 1
                
            # Stop training after patience evaluations at target accuracy
            if self.wait_count >= self.patience:
                logger.info(f"Stopping training - target accuracy maintained for {self.patience} evaluations")
                control.should_training_stop = True
        else:
            # Reset if we drop below target
            if self.target_reached:
                logger.info(f"Accuracy dropped below target ({current_accuracy:.4f} < {self.target_accuracy:.4f}). Continuing training...")
                self.target_reached = False
                self.wait_count = 0

class PII_Dataset:
    """Dataset class for PII classification fine-tuning (both sequence and token classification)."""
    
    def __init__(self, data_dir="presidio_pii_data", dataset_type="presidio", task_type="sequence"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing the generated PII data
            dataset_type: Type of dataset to use ("presidio" or "ai4privacy")
            task_type: Type of task ("sequence" for classification, "token" for NER-style token classification)
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.task_type = task_type
        self.label2id = {}
        self.id2label = {}
        
    def clear_cache(self):
        """Clear the cached AI4Privacy dataset."""
        cache_dir = Path("./ai4privacy_cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info("Cleared AI4Privacy dataset cache")
        
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
        
    def load_presidio_json(self, file_path, tokenizer=None):
        """Load and parse Presidio JSON format for both sequence and token classification."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for sample in data:
            text = sample['full_text']
            spans = sample.get('spans', [])
            
            texts.append(text)
            
            if self.task_type == "token":
                # For token classification, store the spans for later processing
                labels.append(spans)
            else:
                # For sequence classification, extract dominant entity type
                entity_types = [span['entity_type'] for span in spans]
                
                if not entity_types:
                    dominant_label = 'NO_PII'
                else:
                    # If multiple PII types, choose the most frequent one
                    dominant_label = max(set(entity_types), key=entity_types.count)
                
                labels.append(dominant_label)
        
        return texts, labels
    
    def load_ai4privacy_dataset(self):
        """Load and parse AI4Privacy PII masking dataset from Hugging Face."""
        logger.info("Loading AI4Privacy PII masking dataset from Hugging Face...")
        
        # Check if cached dataset exists
        cache_dir = Path("./ai4privacy_cache")
        cache_file = cache_dir / f"processed_dataset_{self.task_type}.json"
        
        if cache_file.exists():
            logger.info("Found cached AI4Privacy dataset, loading from cache...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    texts = cached_data['texts']
                    labels = cached_data['labels']
                    logger.info(f"Loaded {len(texts)} samples from cache")
                    return texts, labels
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Re-downloading...")
        
        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset("ai4privacy/pii-masking-300k")
            
            texts = []
            labels = []
            
            # Process training split
            if 'train' in dataset:
                train_data = dataset['train']
                logger.info(f"Processing {len(train_data)} training samples...")
                
                for sample in train_data:
                    source_text = sample['source_text']
                    privacy_mask = sample.get('privacy_mask', [])
                    
                    # Extract PII types from privacy mask
                    if isinstance(privacy_mask, str):
                        try:
                            privacy_mask = json.loads(privacy_mask)
                        except json.JSONDecodeError:
                            privacy_mask = []
                    
                    texts.append(source_text)
                    
                    if self.task_type == "token":
                        # For token classification, convert privacy mask to span format
                        spans = []
                        for mask_item in privacy_mask:
                            # Handle different possible key structures in AI4Privacy dataset
                            entity_type = None
                            start_pos = None
                            end_pos = None
                            
                            # Try different key names for entity type
                            if 'label' in mask_item:
                                entity_type = mask_item['label']
                            elif 'entity_type' in mask_item:
                                entity_type = mask_item['entity_type']
                            
                            # Try different key names for start position
                            if 'start' in mask_item:
                                start_pos = mask_item['start']
                            elif 'start_position' in mask_item:
                                start_pos = mask_item['start_position']
                            
                            # Try different key names for end position
                            if 'end' in mask_item:
                                end_pos = mask_item['end']
                            elif 'end_position' in mask_item:
                                end_pos = mask_item['end_position']
                            
                            # Only add span if we have all required information
                            if entity_type and start_pos is not None and end_pos is not None:
                                spans.append({
                                    'entity_type': entity_type,
                                    'start': start_pos,
                                    'end': end_pos,
                                    'value': mask_item.get('value', '')  # Store the actual PII value if available
                                })
                        labels.append(spans)
                    else:
                        # For sequence classification
                        if not privacy_mask or len(privacy_mask) == 0:
                            dominant_label = 'NO_PII'
                        else:
                            # Extract entity types from privacy mask
                            entity_types = [mask_item['label'] for mask_item in privacy_mask if 'label' in mask_item]
                            
                            if not entity_types:
                                dominant_label = 'NO_PII'
                            else:
                                # For classification, choose the most frequent PII type
                                dominant_label = max(set(entity_types), key=entity_types.count)
                        
                        labels.append(dominant_label)
            
            # If validation split exists, add it to the dataset
            if 'validation' in dataset:
                val_data = dataset['validation']
                logger.info(f"Processing {len(val_data)} validation samples...")
                
                for sample in val_data:
                    source_text = sample['source_text']
                    privacy_mask = sample.get('privacy_mask', [])
                    
                    # Extract PII types from privacy mask
                    if isinstance(privacy_mask, str):
                        try:
                            privacy_mask = json.loads(privacy_mask)
                        except json.JSONDecodeError:
                            privacy_mask = []
                    
                    texts.append(source_text)
                    
                    if self.task_type == "token":
                        # For token classification, convert privacy mask to span format
                        spans = []
                        for mask_item in privacy_mask:
                            # Handle different possible key structures in AI4Privacy dataset
                            entity_type = None
                            start_pos = None
                            end_pos = None
                            
                            # Try different key names for entity type
                            if 'label' in mask_item:
                                entity_type = mask_item['label']
                            elif 'entity_type' in mask_item:
                                entity_type = mask_item['entity_type']
                            
                            # Try different key names for start position
                            if 'start' in mask_item:
                                start_pos = mask_item['start']
                            elif 'start_position' in mask_item:
                                start_pos = mask_item['start_position']
                            
                            # Try different key names for end position
                            if 'end' in mask_item:
                                end_pos = mask_item['end']
                            elif 'end_position' in mask_item:
                                end_pos = mask_item['end_position']
                            
                            # Only add span if we have all required information
                            if entity_type and start_pos is not None and end_pos is not None:
                                spans.append({
                                    'entity_type': entity_type,
                                    'start': start_pos,
                                    'end': end_pos,
                                    'value': mask_item.get('value', '')  # Store the actual PII value if available
                                })
                        labels.append(spans)
                    else:
                        # For sequence classification
                        if not privacy_mask or len(privacy_mask) == 0:
                            dominant_label = 'NO_PII'
                        else:
                            # Extract entity types from privacy mask
                            entity_types = [mask_item['label'] for mask_item in privacy_mask if 'label' in mask_item]
                            
                            if not entity_types:
                                dominant_label = 'NO_PII'
                            else:
                                # For classification, choose the most frequent PII type
                                dominant_label = max(set(entity_types), key=entity_types.count)
                        
                        labels.append(dominant_label)
            
            logger.info(f"Loaded {len(texts)} total samples from AI4Privacy dataset")
            
            # Cache the processed dataset for future use
            try:
                cache_dir.mkdir(exist_ok=True)
                cache_data = {
                    'texts': texts,
                    'labels': labels,
                    'timestamp': str(Path(__file__).stat().st_mtime),  # File modification time for cache invalidation
                    'dataset_size': len(texts)
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Cached processed dataset to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load AI4Privacy dataset: {e}")
            logger.info("Falling back to Presidio dataset...")
            return self.load_presidio_json(self.download_presidio_dataset())
    
    def split_dataset(self, texts, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Split the dataset into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        if self.task_type == "token":
            # For token classification, we can't easily do stratified splitting based on entity types
            # So we'll do simple random splitting
            logger.info("Using random split for token classification task")
            
            # First split: train and temp (val + test)
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts, labels, test_size=(val_ratio + test_ratio), random_state=random_state
            )
            
            # Second split: val and test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=(1 - val_size), random_state=random_state
            )
            
        else:
            # For sequence classification, use stratified splitting
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
        if self.task_type == "token":
            # Check if we have pre-tokenized BIO labels
            if (all_labels and isinstance(all_labels[0], list) and 
                len(all_labels[0]) > 0 and isinstance(all_labels[0][0], str)):
                # We have pre-tokenized BIO labels, extract unique labels
                unique_labels = set()
                for bio_labels in all_labels:
                    unique_labels.update(bio_labels)
                unique_labels = sorted(list(unique_labels))
                logger.info("Using pre-tokenized BIO labels for mappings")
            else:
                # For token classification with spans, we need to create BIO tags
                entity_types = set()
                for spans_list in all_labels:
                    if isinstance(spans_list, list):
                        for span in spans_list:
                            if isinstance(span, dict) and 'entity_type' in span:
                                entity_types.add(span['entity_type'])
                
                # Create BIO tags
                unique_labels = ['O']  # Outside tag
                for entity_type in sorted(entity_types):
                    unique_labels.extend([f'B-{entity_type}', f'I-{entity_type}'])
                logger.info("Created BIO tags from entity spans")
        else:
            # For sequence classification
            unique_labels = sorted(list(set(all_labels)))
        
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        logger.info(f"Created mappings for {len(unique_labels)} labels: {unique_labels}")
    
    def create_token_labels(self, text, spans, tokenizer, max_length=512):
        """Create token-level BIO labels for a given text and spans."""
        # Tokenize the text
        tokenized = tokenizer(text, truncation=True, padding=False, max_length=max_length, return_offsets_mapping=True)
        tokens = tokenized['input_ids']
        offset_mapping = tokenized['offset_mapping']
        
        # Initialize all labels as 'O' (Outside)
        labels = ['O'] * len(tokens)
        
        # Process each span
        for span in spans:
            # Handle different span formats - check what keys are available
            start_char = None
            end_char = None
            entity_type = None
            
            if isinstance(span, dict):
                # Try different possible key names for start position
                if 'start' in span:
                    start_char = span['start']
                elif 'start_position' in span:
                    start_char = span['start_position']
                elif 'start_pos' in span:
                    start_char = span['start_pos']
                elif 'begin' in span:
                    start_char = span['begin']
                elif 'start_char' in span:
                    start_char = span['start_char']
                elif 'offset' in span:
                    start_char = span['offset']
                
                # Try different possible key names for end position
                if 'end' in span:
                    end_char = span['end']
                elif 'end_position' in span:
                    end_char = span['end_position']
                elif 'end_pos' in span:
                    end_char = span['end_pos']
                elif 'finish' in span:
                    end_char = span['finish']
                elif 'end_char' in span:
                    end_char = span['end_char']
                elif 'length' in span and start_char is not None:
                    end_char = start_char + span['length']
                
                # Try different possible key names for entity type
                if 'entity_type' in span:
                    entity_type = span['entity_type']
                elif 'label' in span:
                    entity_type = span['label']
                elif 'type' in span:
                    entity_type = span['type']
                elif 'entity_label' in span:
                    entity_type = span['entity_label']
                elif 'tag' in span:
                    entity_type = span['tag']
            
            # Skip if we couldn't find the required information
            if start_char is None or end_char is None or entity_type is None:
                logger.warning(f"Skipping span with missing information. Span keys: {list(span.keys()) if isinstance(span, dict) else 'Not a dict'}, Span: {span}")
                continue
            
            # Find tokens that overlap with this span
            token_start_idx = None
            token_end_idx = None
            
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens (they have offset (0,0))
                if token_start == 0 and token_end == 0 and i > 0:
                    continue
                
                # Check if token overlaps with span
                if token_start < end_char and token_end > start_char:
                    if token_start_idx is None:
                        token_start_idx = i
                    token_end_idx = i
            
            # Assign BIO labels
            if token_start_idx is not None and token_end_idx is not None:
                for i in range(token_start_idx, token_end_idx + 1):
                    if i == token_start_idx:
                        labels[i] = f'B-{entity_type}'
                    else:
                        labels[i] = f'I-{entity_type}'
        
        # Convert labels to IDs
        label_ids = [self.label2id.get(label, self.label2id['O']) for label in labels]
        
        return label_ids
    
    def load_ai4privacy_pretokenized(self):
        """Load AI4Privacy dataset using pre-tokenized BIO labels if available."""
        logger.info("Attempting to load AI4Privacy dataset with pre-tokenized BIO labels...")
        
        try:
            dataset = load_dataset("ai4privacy/pii-masking-300k")
            
            texts = []
            labels = []
            
            # Check if the dataset has pre-tokenized BIO labels
            if 'train' in dataset:
                train_data = dataset['train']
                sample = train_data[0]
                
                # Check if pre-tokenized columns exist
                has_pretokenized = 'mbert_text_tokens' in sample and 'mbert_bio_labels' in sample
                
                if has_pretokenized and self.task_type == "token":
                    logger.info("Found pre-tokenized BIO labels, using them directly...")
                    
                    for sample in train_data:
                        # Get the original text by joining tokens (remove special BERT tokens)
                        tokens = sample['mbert_text_tokens']
                        bio_labels = sample['mbert_bio_labels']
                        
                        # Filter out BERT special tokens and reconstruct text
                        filtered_tokens = []
                        filtered_labels = []
                        
                        for token, label in zip(tokens, bio_labels):
                            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                                # Convert subword tokens back to text
                                if token.startswith('##'):
                                    if filtered_tokens:
                                        filtered_tokens[-1] += token[2:]  # Remove ## prefix
                                else:
                                    filtered_tokens.append(token)
                                    filtered_labels.append(label)
                        
                        # Reconstruct text from tokens
                        text = ' '.join(filtered_tokens)
                        texts.append(text)
                        
                        # Convert BIO labels to our format
                        # Note: This assumes the BIO labels are already in the correct format
                        labels.append(filtered_labels)
                    
                    # Add validation data if available
                    if 'validation' in dataset:
                        val_data = dataset['validation']
                        for sample in val_data:
                            tokens = sample['mbert_text_tokens']
                            bio_labels = sample['mbert_bio_labels']
                            
                            filtered_tokens = []
                            filtered_labels = []
                            
                            for token, label in zip(tokens, bio_labels):
                                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                                    if token.startswith('##'):
                                        if filtered_tokens:
                                            filtered_tokens[-1] += token[2:]
                                    else:
                                        filtered_tokens.append(token)
                                        filtered_labels.append(label)
                            
                            text = ' '.join(filtered_tokens)
                            texts.append(text)
                            labels.append(filtered_labels)
                    
                    logger.info(f"Loaded {len(texts)} samples using pre-tokenized BIO labels")
                    return texts, labels
                
        except Exception as e:
            logger.warning(f"Could not load pre-tokenized labels: {e}")
        
        # Fall back to regular loading method
        logger.info("Falling back to regular AI4Privacy loading method...")
        return None, None
        
    def prepare_datasets(self, tokenizer=None):
        """Prepare train/validation/test datasets from selected dataset."""
        
        # Load the appropriate dataset based on dataset_type
        if self.dataset_type == "ai4privacy":
            logger.info("Loading AI4Privacy dataset...")
            # Try pre-tokenized approach first for token classification
            if self.task_type == "token":
                texts, labels = self.load_ai4privacy_pretokenized()
                if texts is None or labels is None:
                    # Fall back to regular method
                    texts, labels = self.load_ai4privacy_dataset()
            else:
                texts, labels = self.load_ai4privacy_dataset()
        else:
            # Default to Presidio dataset
            dataset_path = self.download_presidio_dataset()
            logger.info("Loading Presidio dataset...")
            texts, labels = self.load_presidio_json(dataset_path, tokenizer)
        
        logger.info(f"Loaded {len(texts)} samples")
        
        if self.task_type == "sequence":
            logger.info(f"Label distribution: {dict(sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True))}")
        else:
            # For token classification, show entity type distribution
            entity_types = []
            
            # Check if we have pre-tokenized BIO labels or span dictionaries
            if labels and isinstance(labels[0], list):
                if len(labels[0]) > 0 and isinstance(labels[0][0], str):
                    # We have pre-tokenized BIO labels
                    for bio_labels in labels:
                        for label in bio_labels:
                            if label != 'O' and label.startswith(('B-', 'I-')):
                                entity_type = label[2:]  # Remove B- or I- prefix
                                entity_types.append(entity_type)
                else:
                    # We have span dictionaries
                    for spans in labels:
                        for span in spans:
                            if isinstance(span, dict) and 'entity_type' in span:
                                entity_types.append(span['entity_type'])
            
            if entity_types:
                logger.info(f"Entity type distribution: {dict(sorted([(entity, entity_types.count(entity)) for entity in set(entity_types)], key=lambda x: x[1], reverse=True))}")
        
        # Split the dataset
        logger.info("Splitting dataset into train/validation/test...")
        datasets = self.split_dataset(texts, labels)
        
        train_texts, train_labels = datasets['train']
        val_texts, val_labels = datasets['validation']
        test_texts, test_labels = datasets['test']
        
        # Create label mappings
        all_labels = train_labels + val_labels + test_labels
        self.create_label_mappings(all_labels)
        
        if self.task_type == "token":
            # For token classification, handle labels
            if tokenizer is None:
                raise ValueError("Tokenizer is required for token classification")
            
            train_label_ids = []
            val_label_ids = []
            test_label_ids = []
            
            # Check if we already have BIO labels (from pre-tokenized data)
            if (train_labels and isinstance(train_labels[0], list) and 
                len(train_labels[0]) > 0 and isinstance(train_labels[0][0], str)):
                # We have pre-tokenized BIO labels, convert them to IDs
                logger.info("Using pre-tokenized BIO labels")
                
                for bio_labels in train_labels:
                    label_ids = [self.label2id.get(label, self.label2id.get('O', 0)) for label in bio_labels]
                    train_label_ids.append(label_ids)
                
                for bio_labels in val_labels:
                    label_ids = [self.label2id.get(label, self.label2id.get('O', 0)) for label in bio_labels]
                    val_label_ids.append(label_ids)
                    
                for bio_labels in test_labels:
                    label_ids = [self.label2id.get(label, self.label2id.get('O', 0)) for label in bio_labels]
                    test_label_ids.append(label_ids)
            else:
                # We have span-based labels, convert them to token labels
                logger.info("Converting spans to token labels")
                
                for text, spans in zip(train_texts, train_labels):
                    token_labels = self.create_token_labels(text, spans, tokenizer)
                    train_label_ids.append(token_labels)
                
                for text, spans in zip(val_texts, val_labels):
                    token_labels = self.create_token_labels(text, spans, tokenizer)
                    val_label_ids.append(token_labels)
                    
                for text, spans in zip(test_texts, test_labels):
                    token_labels = self.create_token_labels(text, spans, tokenizer)
                    test_label_ids.append(token_labels)
        else:
            # For sequence classification, convert labels to IDs
            train_label_ids = [self.label2id[label] for label in train_labels]
            val_label_ids = [self.label2id[label] for label in val_labels]
            test_label_ids = [self.label2id[label] for label in test_labels]
        
        return {
            'train': (train_texts, train_label_ids),
            'validation': (val_texts, val_label_ids),
            'test': (test_texts, test_label_ids)
        }

# Function to predict PII type using the sequence classification model
def predict_pii_type(model, tokenizer, text, idx_to_label_map, device):
    """Predict PII type for a given text using sequence classification."""
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

# Function to predict token-level PII labels and extract entities
def predict_pii_tokens(model, tokenizer, text, idx_to_label_map, device):
    """Predict token-level PII labels and extract entity spans."""
    model.eval()
    
    # Tokenize input with offset mapping to reconstruct spans
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, 
                      max_length=512, return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')[0]  # Remove from inputs, keep for span reconstruction
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)[0]  # Remove batch dimension
    probabilities = torch.softmax(logits, dim=-1)[0]
    
    # Convert predictions to labels
    predicted_labels = [idx_to_label_map.get(pred.item(), 'O') for pred in predictions]
    
    # Extract entities from BIO tags
    entities = []
    current_entity = None
    
    for i, (label, offset, prob) in enumerate(zip(predicted_labels, offset_mapping, probabilities)):
        # Skip special tokens (they have offset (0,0))
        if offset[0] == 0 and offset[1] == 0 and i > 0:
            continue
            
        if label.startswith('B-'):
            # Beginning of new entity
            if current_entity is not None:
                entities.append(current_entity)
            
            entity_type = label[2:]  # Remove 'B-' prefix
            current_entity = {
                'entity_type': entity_type,
                'start': offset[0].item(),
                'end': offset[1].item(),
                'text': text[offset[0]:offset[1]],
                'confidence': prob[predictions[i]].item()
            }
        elif label.startswith('I-') and current_entity is not None:
            # Inside current entity
            entity_type = label[2:]  # Remove 'I-' prefix
            if entity_type == current_entity['entity_type']:
                # Extend current entity
                current_entity['end'] = offset[1].item()
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
                # Update confidence with average
                current_entity['confidence'] = (current_entity['confidence'] + prob[predictions[i]].item()) / 2
        else:
            # Outside entity or different entity type
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget the last entity
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities

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

def main(model_name="minilm", max_epochs=50, batch_size=16, dataset_type="presidio", force_restart=False, target_accuracy=0.95, patience=3, task_type="sequence"):
    """Main function to demonstrate PII classification fine-tuning with accuracy-based early stopping."""
    
    # Validate model name and task type
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    if task_type not in ["sequence", "token"]:
        logger.error(f"Unknown task type: {task_type}. Available types: ['sequence', 'token']")
        return
    
    # Set up device (GPU if available)
    device = get_device()
    
    model_path = MODEL_CONFIGS[model_name]
    logger.info(f"Using model: {model_name} ({model_path})")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Training configuration: max {max_epochs} epochs, batch size {batch_size}")
    logger.info(f"Early stopping: target accuracy {target_accuracy:.4f}, patience {patience}")
    logger.info(f"Dataset: {dataset_type}")
    if force_restart:
        logger.info("Force restart enabled - will start training from scratch")
    
    # Load tokenizer first (needed for token classification dataset preparation)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    logger.info(f"Loading {dataset_type} PII dataset...")
    dataset_loader = PII_Dataset(dataset_type=dataset_type, task_type=task_type)
    datasets = dataset_loader.prepare_datasets(tokenizer)
    
    train_texts, train_categories = datasets['train']
    val_texts, val_categories = datasets['validation']
    test_texts, test_categories = datasets['test']
    
    unique_categories = list(dataset_loader.label2id.keys())
    category_to_idx = dataset_loader.label2id
    idx_to_category = dataset_loader.id2label

    logger.info(f"Found {len(unique_categories)} unique labels: {unique_categories}")
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_texts)}")
    logger.info(f"  Validation: {len(val_texts)}")
    logger.info(f"  Test: {len(test_texts)}")

    num_labels = len(unique_categories)
    
    # Load model based on task type
    logger.info(f"Loading {task_type} classification model...")
    
    # Suppress the expected warning about newly initialized classifier weights
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*classifier.*newly initialized.*")
        if task_type == "sequence":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                label2id=category_to_idx,
                id2label=idx_to_category
            )
        else:  # token classification
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                label2id=category_to_idx,
                id2label=idx_to_category
            )
    
    # Move model to device
    model.to(device)
    
    # Create datasets and tokenization function based on task type
    if task_type == "sequence":
        # For sequence classification
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
        train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_categories})
        val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_categories})
        test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_categories})
    else:
        # For token classification
        def tokenize_function(examples):
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            # Pad/truncate labels to match tokenized input length
            labels = []
            for label_list in examples["labels"]:
                # Ensure labels match the tokenized length
                if len(label_list) < 512:
                    # Pad with -100 (ignored index)
                    padded_labels = label_list + [-100] * (512 - len(label_list))
                else:
                    # Truncate
                    padded_labels = label_list[:512]
                labels.append(padded_labels)
            tokenized["labels"] = labels
            return tokenized
        
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
    output_model_path = f"pii_classifier_{model_name}_{dataset_type}_{task_type}_model"
    
    # Check for existing checkpoints
    checkpoint_dir = None
    if not force_restart and Path(output_model_path).exists():
        # Look for checkpoint directories
        checkpoints = [d for d in Path(output_model_path).iterdir() 
                      if d.is_dir() and d.name.startswith('checkpoint-')]
        if checkpoints:
            # Get the latest checkpoint
            try:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                checkpoint_dir = str(latest_checkpoint)
                logger.info(f"Found existing checkpoint: {checkpoint_dir}")
                logger.info("Training will resume from this checkpoint")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing checkpoint names: {e}. Starting from scratch.")
        else:
            logger.info(f"Output directory {output_model_path} exists but no checkpoints found")
    elif force_restart and Path(output_model_path).exists():
        logger.info(f"Force restart enabled - ignoring existing checkpoints in {output_model_path}")
    
    training_args_dict = {
        "output_dir": output_model_path,
        "num_train_epochs": max_epochs,  # Maximum epochs (will stop early if target accuracy reached)
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": f"{output_model_path}/logs",
        "logging_steps": 100,
        eval_strategy_param: "epoch",  # Evaluate every epoch to check accuracy
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",  # Use accuracy as the primary metric
        "save_total_limit": 3,  # Keep more checkpoints for resuming
        "report_to": [],
    }
    
    # Only add resume_from_checkpoint if we found a checkpoint
    if checkpoint_dir:
        training_args_dict["resume_from_checkpoint"] = checkpoint_dir
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Create early stopping callback
    early_stopping_callback = AccuracyEarlyStoppingCallback(
        target_accuracy=target_accuracy,
        patience=patience
    )
    
    # Create trainer with early stopping callback
    metrics_function = compute_metrics_token_classification if task_type == "token" else compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_function,
        callbacks=[early_stopping_callback],
    )

    logger.info(f"Starting PII classification fine-tuning with {model_name}...")

    # Train the model with error handling
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.info("You can resume training by running the same command again (checkpoints will be automatically detected)")
        raise

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
    if task_type == "sequence":
        val_accuracy = evaluate_pii_classifier(model, tokenizer, val_texts, val_categories, idx_to_category, device)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_accuracy = evaluate_pii_classifier(model, tokenizer, test_texts, test_categories, idx_to_category, device)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Print final results
        print("\n" + "="*50)
        print("PII Sequence Classification Fine-tuning Completed!")
        print("="*50)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        # For token classification, we'll do a simpler evaluation
        logger.info("Token classification training completed!")
        print("\n" + "="*50)
        print("PII Token Classification Fine-tuning Completed!")
        print("="*50)
        print("Model can now predict token-level PII entities and their locations.")
    
    return model, tokenizer, idx_to_category

def demo_inference(model_name="minilm", dataset_type="presidio", task_type="sequence"):
    """Demonstrate inference with the trained model."""
    
    # Set up device (GPU if available)
    device = get_device()
    
    model_path = f"./pii_classifier_{model_name}_{dataset_type}_{task_type}_model"
    if not Path(model_path).exists():
        logger.error(f"Trained model not found at {model_path}. Please run training first with --model {model_name} --dataset {dataset_type} --task-type {task_type}")
        return
    
    # Load model and tokenizer based on task type
    if task_type == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    
    mapping_path = os.path.join(model_path, "pii_type_mapping.json")
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
        idx_to_label = {int(k): v for k, v in mappings["idx_to_label"].items()}
    
    print("\n" + "="*50)
    if task_type == "sequence":
        print("PII Sequence Classification Test")
    else:
        print("PII Token Classification Test")
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
        print(f"Text: {text}")
        
        if task_type == "sequence":
            predicted_pii_type, confidence = predict_pii_type(model, tokenizer, text, idx_to_label, device)
            print(f"Predicted PII Type: {predicted_pii_type}, Confidence: {confidence:.4f}")
        else:
            entities = predict_pii_tokens(model, tokenizer, text, idx_to_label, device)
            if entities:
                print("Detected PII entities:")
                for entity in entities:
                    print(f"  - {entity['entity_type']}: '{entity['text']}' (confidence: {entity['confidence']:.4f})")
            else:
                print("No PII entities detected.")
        
        print("---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Classification Fine-tuning")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: 'train' to fine-tune model, 'test' to run inference")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="minilm", 
                       help="Model to use for fine-tuning (e.g., bert-base, roberta-base, etc.)")
    parser.add_argument("--dataset", choices=["presidio", "ai4privacy"], default="presidio",
                       help="Dataset to use: 'presidio' for Microsoft Presidio dataset, 'ai4privacy' for AI4Privacy PII masking dataset")
    parser.add_argument("--task-type", choices=["sequence", "token"], default="sequence",
                       help="Task type: 'sequence' for classification, 'token' for token-level entity recognition")
    parser.add_argument("--max-epochs", type=int, default=10,
                       help="Maximum number of training epochs (default: 10, training will stop early if target accuracy reached)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training and evaluation batch size (default: 8)")
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                       help="Target accuracy to reach before stopping training (default: 0.95)")
    parser.add_argument("--patience", type=int, default=3,
                       help="Number of evaluations to wait after reaching target accuracy before stopping (default: 3)")
    parser.add_argument("--force-restart", action="store_true",
                       help="Force restart training from scratch, ignoring any existing checkpoints")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear the AI4Privacy dataset cache before training")
    
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        dataset_loader = PII_Dataset()
        dataset_loader.clear_cache()
        if args.mode != "train":  # If only clearing cache, exit
            logger.info("Cache cleared successfully")
            exit(0)
    
    if args.mode == "train":
        main(args.model, args.max_epochs, args.batch_size, args.dataset, args.force_restart, args.target_accuracy, args.patience, args.task_type)
    elif args.mode == "test":
        demo_inference(args.model, args.dataset, args.task_type) 