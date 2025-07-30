"""
Multitask BERT Fine-tuning with Multiple Base Models
Motivated by research papers that explain the benefits of multitask learning in resource efficiency

Usage:
    # Train with default model (MiniLM)
    python multitask_bert_training.py --model minilm

    # Train with BERT base
    python multitask_bert_training.py --model bert-base

    # Train with DeBERTa v3
    python multitask_bert_training.py --model deberta-v3-base

    # Train with ModernBERT
    python multitask_bert_training.py --model modernbert-base

Supported models:
    - bert-base, bert-large: Standard BERT models
    - roberta-base, roberta-large: RoBERTa models
    - deberta-v3-base, deberta-v3-large: DeBERTa v3 models
    - modernbert-base, modernbert-large: ModernBERT models
    - minilm: Lightweight sentence transformer (default)
    - distilbert: Distilled BERT
    - electra-base, electra-large: ELECTRA models
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, InputExample
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import logging
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class MultitaskBertModel(nn.Module):
    """
    Multitask BERT model with shared base model and task-specific classification heads.
    """
    
    def __init__(self, base_model_name, task_configs):
        """
        Initialize multitask BERT model.
        
        Args:
            base_model_name: Name/path of the base BERT model
            task_configs: Dict mapping task names to their configurations
                         {"task_name": {"num_classes": int, "weight": float}}
        """
        super(MultitaskBertModel, self).__init__()
        
        # Shared BERT base model
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Task-specific classification heads
        self.task_heads = nn.ModuleDict()
        self.task_configs = task_configs
        
        hidden_size = self.bert.config.hidden_size
        
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = nn.Linear(hidden_size, config["num_classes"])
    
    def forward(self, input_ids, attention_mask, task_name=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task_name: Specific task to run (if None, runs all tasks)
            
        Returns:
            Dict mapping task names to their logits
        """
        # Shared BERT base model
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence length
        token_embeddings = bert_output.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific classification heads
        outputs = {}
        if task_name:
            # Run only specific task
            outputs[task_name] = self.task_heads[task_name](pooled_output)
        else:
            # Run all tasks
            for task in self.task_heads:
                outputs[task] = self.task_heads[task](pooled_output)
        
        return outputs

class MultitaskDataset(Dataset):
    """Dataset class for multitask learning."""
    
    def __init__(self, samples, tokenizer, max_length=512):
        """
        Initialize dataset.
        
        Args:
            samples: List of (text, task_name, label) tuples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, task_name, label = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task_name': task_name,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultitaskTrainer:
    """Trainer for multitask BERT model."""
    
    def __init__(self, model, tokenizer, task_configs, device='cuda'):
        self.model = model.to(device) if model is not None else None
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        self.device = device
        
        # Initialize label mappings
        self.jailbreak_label_mapping = None
        
        # Task-specific loss functions
        self.loss_fns = {task: nn.CrossEntropyLoss() for task in task_configs}
    
    def prepare_datasets(self):
        """Prepare datasets for all tasks."""
        all_samples = []
        
        # Load and prepare each task's data
        datasets = {}
        
        # Category Classification (MMLU-Pro)
        logger.info("Loading MMLU-Pro dataset for category classification...")
        try:
            mmlu_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
            questions = mmlu_dataset["test"]["question"]
            categories = mmlu_dataset["test"]["category"]
            
            # Create category mapping
            unique_categories = sorted(list(set(categories)))
            category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
            
            # Add samples with progress bar
            logger.info("Processing MMLU-Pro samples...")
            for question, category in zip(questions, categories):
                all_samples.append((question, "category", category_to_idx[category]))
            
            datasets["category"] = {
                "label_mapping": {"label_to_idx": category_to_idx, "idx_to_label": {v: k for k, v in category_to_idx.items()}}
            }
            
        except Exception as e:
            logger.warning(f"Failed to load MMLU-Pro: {e}")
        
        # PII Detection
        logger.info("Loading PII dataset...")
        try:
            pii_samples = self._load_pii_dataset()
            if pii_samples:
                # Create PII label mapping first
                pii_labels = sorted(list(set([label for _, label in pii_samples])))
                pii_to_idx = {label: idx for idx, label in enumerate(pii_labels)}
                
                # Add mapped PII samples directly with progress bar
                logger.info("Processing PII samples...")
                for text, label in tqdm(pii_samples, desc="PII Dataset"):
                    all_samples.append((text, "pii", pii_to_idx[label]))
                
                datasets["pii"] = {
                    "label_mapping": {"label_to_idx": pii_to_idx, "idx_to_label": {v: k for k, v in pii_to_idx.items()}}
                }
                logger.info(f"Added {len(pii_samples)} PII samples to training")
        except Exception as e:
            logger.warning(f"Failed to load PII dataset: {e}")
        
        # Jailbreak Detection (real dataset from HuggingFace)
        logger.info("Loading real jailbreak dataset...")
        jailbreak_samples = self._load_jailbreak_dataset()
        logger.info("Processing jailbreak samples...")
        for text, label in tqdm(jailbreak_samples, desc="Jailbreak Dataset"):
            all_samples.append((text, "jailbreak", label))
        
        datasets["jailbreak"] = {
            "label_mapping": self.jailbreak_label_mapping
        }
        
        # Split data into train/val
        logger.info("Splitting dataset into train/validation...")
        train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
        
        return train_samples, val_samples, datasets
    
    def _load_pii_dataset(self):
        """Load PII dataset (improved version with better data handling)."""
        # Download Presidio dataset
        url = "https://raw.githubusercontent.com/microsoft/presidio-research/refs/heads/master/data/synth_dataset_v2.json"
        dataset_path = "presidio_synth_dataset_v2.json"
        
        if not Path(dataset_path).exists():
            logger.info(f"Downloading Presidio dataset...")
            response = requests.get(url)
            response.raise_for_status()
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        labels_count = defaultdict(int)
        
        # Collect all samples and count labels
        all_samples = []
        for sample in tqdm(data, desc="Processing PII data"):
            text = sample['full_text']
            spans = sample.get('spans', [])
            
            if not spans:
                label = 'NO_PII'
            else:
                entity_types = [span['entity_type'] for span in spans]
                label = max(set(entity_types), key=entity_types.count)
            
            all_samples.append((text, label))
            labels_count[label] += 1

        logger.info(f"Using {len(all_samples)} PII samples for training")
        logger.info(f"PII label distribution: {dict(sorted([(label, sum(1 for _, l in all_samples if l == label)) for label in set(l for _, l in all_samples)], key=lambda x: x[1], reverse=True))}")
        
        return all_samples
    
    def _load_jailbreak_dataset(self):
        """Load real jailbreak classification dataset from HuggingFace."""
        dataset_name = "jackhhao/jailbreak-classification"
        
        try:
            logger.info(f"Loading jailbreak dataset from HuggingFace: {dataset_name}")
            jailbreak_dataset = load_dataset(dataset_name)
            
            texts = []
            labels = []
            
            # Process train split
            if 'train' in jailbreak_dataset:
                for sample in tqdm(jailbreak_dataset['train'], desc="Processing jailbreak train"):
                    texts.append(sample['prompt'])
                    labels.append(sample['type'])
            
            # Process test split if available
            if 'test' in jailbreak_dataset:
                for sample in tqdm(jailbreak_dataset['test'], desc="Processing jailbreak test"):
                    texts.append(sample['prompt'])
                    labels.append(sample['type'])
            
            logger.info(f"Loaded {len(texts)} jailbreak samples")
            
            # Create label mapping
            unique_labels = sorted(list(set(labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            logger.info(f"Jailbreak label distribution: {dict(sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True))}")
            logger.info(f"Jailbreak labels: {unique_labels}")
            
            # Convert labels to indices
            label_indices = [label_to_idx[label] for label in labels]
            
            # Combine texts with label indices
            samples = list(zip(texts, label_indices))
            
            # Store label mapping for later use
            self.jailbreak_label_mapping = {
                "label_to_idx": label_to_idx,
                "idx_to_label": {idx: label for label, idx in label_to_idx.items()}
            }
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load jailbreak dataset: {e}")
            logger.warning("Falling back to synthetic jailbreak data...")
            return []
    
    
    def train(self, train_samples, val_samples, num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the multitask model."""
        
        # Create datasets
        train_dataset = MultitaskDataset(train_samples, self.tokenizer)
        val_dataset = MultitaskDataset(val_samples, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop with progress bars
        self.model.train()
        
        # Overall epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)
        
        for epoch in epoch_pbar:
            total_loss = 0
            task_losses = defaultdict(float)
            task_counts = defaultdict(int)
            
            # Batch progress bar for current epoch
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=1, leave=False)
            
            for batch in batch_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_names = batch['task_name']
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses for each task in the batch
                batch_loss = 0
                for i, task_name in enumerate(task_names):
                    task_logits = outputs[task_name][i:i+1]  # Get logits for this sample
                    task_label = labels[i:i+1]
                    
                    # Apply task weight
                    task_weight = self.task_configs[task_name].get("weight", 1.0)
                    task_loss = self.loss_fns[task_name](task_logits, task_label) * task_weight
                    
                    batch_loss += task_loss
                    task_losses[task_name] += task_loss.item()
                    task_counts[task_name] += 1
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += batch_loss.item()
                
                # Update batch progress bar with current loss
                current_avg_loss = total_loss / (batch_pbar.n + 1)
                batch_pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
            
            # Close batch progress bar
            batch_pbar.close()
            
            # Log epoch results
            avg_loss = total_loss / len(train_loader)
            
            # Prepare task loss summary
            task_loss_summary = {}
            for task_name in task_losses:
                avg_task_loss = task_losses[task_name] / task_counts[task_name]
                task_loss_summary[task_name] = avg_task_loss
            
            # Validation
            logger.info("Running validation...")
            val_accuracy = self.evaluate(val_loader)
            
            # Update epoch progress bar with summary
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'val_acc': f'{val_accuracy if isinstance(val_accuracy, (int, float)) else "N/A"}'
            })
            
            # Log detailed results
            logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            for task_name, avg_task_loss in task_loss_summary.items():
                logger.info(f"  {task_name} loss: {avg_task_loss:.4f}")
            logger.info(f"  Validation accuracy: {val_accuracy}")
        
        # Close epoch progress bar
        epoch_pbar.close()
        logger.info("Training completed!")
    
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        
        task_correct = defaultdict(int)
        task_total = defaultdict(int)
        
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc="Validating", position=1, leave=False)
            
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_names = batch['task_name']
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                for i, task_name in enumerate(task_names):
                    task_logits = outputs[task_name][i:i+1]
                    task_label = labels[i:i+1]
                    
                    predicted = torch.argmax(task_logits, dim=1)
                    
                    task_correct[task_name] += (predicted == task_label).sum().item()
                    task_total[task_name] += 1
                
                # Update validation progress with current accuracies
                if len(task_total) > 0:
                    current_acc = sum(task_correct.values()) / sum(task_total.values())
                    val_pbar.set_postfix({'acc': f'{current_acc:.3f}'})
            
            val_pbar.close()
        
        # Calculate accuracies
        accuracies = {}
        for task_name in task_correct:
            accuracies[task_name] = task_correct[task_name] / task_total[task_name]
        
        self.model.train()
        return accuracies
    
    def save_model(self, output_path):
        """Save the trained model and configurations."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        # Save task configurations
        with open(os.path.join(output_path, "task_configs.json"), "w") as f:
            json.dump(self.task_configs, f, indent=2)
        
        # Save model config
        model_config = {
            "base_model_name": self.model.bert.config.name_or_path,
            "hidden_size": self.model.bert.config.hidden_size,
            "model_type": "multitask_bert"
        }
        
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Model saved to {output_path}")

def main(model_name="minilm", num_epochs=5, batch_size=16):
    """Main training function."""
    
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    # Auto-optimize batch size based on model if using default
    if batch_size == 0:  # Default batch size
        optimal_batch_sizes = {
            'minilm': 20,           # Lightweight model
            'bert-base': 12,        # Medium model  
            'bert-large': 6,        # Large model
            'roberta-base': 12,     # Similar to BERT-base
            'roberta-large': 6,     # Large model
            'deberta-v3-base': 10,  # Slightly larger than BERT-base
            'deberta-v3-large': 6,  # Large model
            'modernbert-base': 12,  # Optimized architecture
            'modernbert-large': 6,  # Large model
            'distilbert': 16,       # Smaller than BERT-base
            'electra-base': 12,     # Similar to BERT-base
            'electra-large': 6      # Large model
        }
        
        optimized_batch_size = optimal_batch_sizes.get(model_name, 12)
        if optimized_batch_size != batch_size:
            logger.info(f"🚀 Auto-optimizing batch size for {model_name}: {batch_size} → {optimized_batch_size}")
            batch_size = optimized_batch_size
    
    # Configuration
    base_model_name = MODEL_CONFIGS[model_name]
    output_path = f"./multitask_bert_model_{model_name}"
    
    logger.info(f"Using model: {model_name} ({base_model_name})")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize trainer
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Create a temporary trainer to load datasets and determine configurations
    temp_trainer = MultitaskTrainer(None, tokenizer, {}, device)
    
    # Prepare data to determine actual task configurations
    logger.info("Preparing datasets...")
    print("📊 Loading and preparing datasets...")
    train_samples, val_samples, label_mappings = temp_trainer.prepare_datasets()
    
    # Determine task configurations based on actual data
    task_configs = {}
    
    # Category classification
    if "category" in label_mappings:
        num_category_classes = len(label_mappings["category"]["label_mapping"]["label_to_idx"])
        task_configs["category"] = {"num_classes": num_category_classes, "weight": 1.0}
        logger.info(f"Category task: {num_category_classes} classes")
    
    # PII detection  
    if "pii" in label_mappings:
        num_pii_classes = len(label_mappings["pii"]["label_mapping"]["label_to_idx"])
        task_configs["pii"] = {"num_classes": num_pii_classes, "weight": 3.0}  # Increased weight
        logger.info(f"PII task: {num_pii_classes} classes")
    
    # Jailbreak detection
    if "jailbreak" in label_mappings:
        num_jailbreak_classes = len(label_mappings["jailbreak"]["label_mapping"]["label_to_idx"])
        task_configs["jailbreak"] = {"num_classes": num_jailbreak_classes, "weight": 2.0}
        logger.info(f"Jailbreak task: {num_jailbreak_classes} classes")
    
    logger.info(f"Final task configurations: {task_configs}")
    
    # Now initialize the actual model with correct configurations
    model = MultitaskBertModel(base_model_name, task_configs)
    
    # Create the real trainer
    trainer = MultitaskTrainer(model, tokenizer, task_configs, device)
    
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    
    # Train model
    logger.info("Starting multitask training...")
    print("🚀 Starting multitask training...")
    trainer.train(train_samples, val_samples, num_epochs=num_epochs, batch_size=batch_size)  # Increased epochs
    
    # Save model
    trainer.save_model(output_path)
    
    # Save label mappings
    with open(os.path.join(output_path, "label_mappings.json"), "w") as f:
        json.dump(label_mappings, f, indent=2)
    
    logger.info("Multitask training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multitask BERT Training")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="minilm", 
                       help="Model to use for multitask training (e.g., bert-base, roberta-base, etc.)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for training (if 0, auto-optimize based on model)")
    args = parser.parse_args()
    
    main(args.model, args.epochs, args.batch_size) 