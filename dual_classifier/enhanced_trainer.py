import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json
import os
import time
import warnings
from pathlib import Path

# Import our custom modules
from dual_classifier import DualClassifier
from hardware_detector import HardwareDetector, HardwareCapabilities, detect_and_configure, estimate_training_time
from dataset_loaders import RealDatasetLoader, DatasetInfo, load_custom_dataset
from missing_files_detector import check_missing_files

# Interactive selection support
try:
    import inquirer
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False
    print("💡 For enhanced interactive selection, install: pip install inquirer")


class DualTaskDataset(Dataset):
    """
    Dataset for dual-task learning with category classification and PII detection.
    """
    
    def __init__(
        self,
        texts: List[str],
        category_labels: List[int],
        pii_labels: List[List[int]],  # Token-level PII labels
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.category_labels = category_labels
        self.pii_labels = pii_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        category_label = self.category_labels[idx]
        pii_label = self.pii_labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare PII labels to match tokenized length
        # Note: This is simplified - in practice you'd need proper token alignment
        pii_labels_padded = pii_label[:self.max_length]
        if len(pii_labels_padded) < self.max_length:
            pii_labels_padded.extend([0] * (self.max_length - len(pii_labels_padded)))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'category_label': torch.tensor(category_label, dtype=torch.long),
            'pii_labels': torch.tensor(pii_labels_padded, dtype=torch.long)
        }


class DualTaskLoss(nn.Module):
    """
    Combined loss function for dual-task learning.
    """
    
    def __init__(self, category_weight: float = 1.0, pii_weight: float = 1.0):
        super().__init__()
        self.category_weight = category_weight
        self.pii_weight = pii_weight
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.pii_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
        
    def forward(
        self,
        category_logits: torch.Tensor,
        pii_logits: torch.Tensor,
        category_labels: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for both tasks.
        
        Returns:
            total_loss, category_loss, pii_loss
        """
        # Category classification loss
        category_loss = self.category_loss_fn(category_logits, category_labels)
        
        # PII detection loss - only compute loss for attended tokens
        # Reshape for loss computation
        pii_logits_flat = pii_logits.view(-1, pii_logits.size(-1))
        pii_labels_flat = pii_labels.view(-1)
        
        # Mask out padded tokens
        attention_mask_flat = attention_mask.view(-1)
        pii_labels_masked = pii_labels_flat.clone()
        pii_labels_masked[attention_mask_flat == 0] = -100
        
        pii_loss = self.pii_loss_fn(pii_logits_flat, pii_labels_masked)
        
        # Combined loss
        total_loss = (self.category_weight * category_loss + 
                     self.pii_weight * pii_loss)
        
        return total_loss, category_loss, pii_loss


class TrainingStrengthConfig:
    """
    Training strength configurations for different training intensities.
    """
    
    CONFIGS = {
        "quick": {
            "description": "Fast training for testing and prototyping",
            "num_epochs": 2,
            "learning_rate_multiplier": 2.0,  # Higher LR for faster convergence
            "batch_size_multiplier": 2.0,     # Larger batches for speed
            "gradient_accumulation_divider": 2, # Less accumulation for speed
            "checkpoint_steps_multiplier": 2.0, # Less frequent checkpoints
            "eval_steps_multiplier": 2.0,     # Less frequent evaluation
            "early_stopping_patience": 3,
            "warmup_ratio": 0.05  # Less warmup
        },
        "normal": {
            "description": "Balanced training for good results in reasonable time",
            "num_epochs": 5,
            "learning_rate_multiplier": 1.0,  # Standard LR
            "batch_size_multiplier": 1.0,     # Standard batch size
            "gradient_accumulation_divider": 1, # Standard accumulation
            "checkpoint_steps_multiplier": 1.0, # Standard checkpointing
            "eval_steps_multiplier": 1.0,     # Standard evaluation
            "early_stopping_patience": 5,
            "warmup_ratio": 0.1
        },
        "intensive": {
            "description": "Thorough training for high-quality results",
            "num_epochs": 10,
            "learning_rate_multiplier": 0.7,  # Lower LR for stability
            "batch_size_multiplier": 0.8,     # Smaller batches for precision
            "gradient_accumulation_divider": 1, # Standard accumulation
            "checkpoint_steps_multiplier": 0.5, # More frequent checkpoints
            "eval_steps_multiplier": 0.5,     # More frequent evaluation
            "early_stopping_patience": 8,
            "warmup_ratio": 0.15  # More warmup
        },
        "maximum": {
            "description": "Maximum quality training - may take hours",
            "num_epochs": 20,
            "learning_rate_multiplier": 0.5,  # Very conservative LR
            "batch_size_multiplier": 0.6,     # Smaller batches
            "gradient_accumulation_divider": 1, # Standard accumulation
            "checkpoint_steps_multiplier": 0.3, # Very frequent checkpoints
            "eval_steps_multiplier": 0.3,     # Very frequent evaluation
            "early_stopping_patience": 12,
            "warmup_ratio": 0.2   # Extensive warmup
        }
    }
    
    @classmethod
    def get_config(cls, strength: str) -> Dict:
        """Get configuration for specified training strength."""
        if strength not in cls.CONFIGS:
            available = list(cls.CONFIGS.keys())
            raise ValueError(f"Training strength '{strength}' not available. Choose from: {available}")
        return cls.CONFIGS[strength].copy()
    
    @classmethod
    def list_strengths(cls) -> Dict[str, str]:
        """List available training strengths with descriptions."""
        return {name: config["description"] for name, config in cls.CONFIGS.items()}


class EnhancedDualTaskTrainer:
    """
    Enhanced trainer for dual-purpose classifier with hardware detection and real dataset support.
    
    Key features:
    - Automatic hardware capability detection
    - Real dataset loading with format detection
    - Mixed precision training support
    - Gradient accumulation for large effective batch sizes
    - Automatic checkpointing and recovery
    - Comprehensive metrics and monitoring
    - Configurable training strength levels
    """
    
    def __init__(
        self,
        model: DualClassifier,
        train_dataset_path: Optional[str] = None,
        val_dataset_path: Optional[str] = None,
        train_dataset: Optional[DualTaskDataset] = None,
        val_dataset: Optional[DualTaskDataset] = None,
        auto_detect_hardware: bool = True,
        training_strength: str = "normal",
        category_weight: float = 1.0,
        pii_weight: float = 1.0,
        output_dir: str = "./enhanced_training_output",
        **override_config
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: DualClassifier model
            train_dataset_path: Path to training dataset file (alternative to train_dataset)
            val_dataset_path: Path to validation dataset file (alternative to val_dataset)
            train_dataset: Pre-loaded training dataset
            val_dataset: Pre-loaded validation dataset
            auto_detect_hardware: Whether to automatically detect and configure hardware
            training_strength: Training intensity level ("quick", "normal", "intensive", "maximum")
            category_weight: Weight for category classification loss
            pii_weight: Weight for PII detection loss
            output_dir: Directory for outputs and checkpoints
            **override_config: Override hardware-detected configuration
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get training strength configuration
        self.training_strength = training_strength
        self.strength_config = TrainingStrengthConfig.get_config(training_strength)
        
        print(f"🎯 Training Strength: {training_strength}")
        print(f"   {self.strength_config['description']}")
        print(f"   Expected epochs: {self.strength_config['num_epochs']}")
        
        # Hardware detection and configuration
        if auto_detect_hardware:
            print("🔍 Detecting hardware capabilities...")
            self.capabilities, self.config = detect_and_configure()
            
            # Apply training strength modifications
            self._apply_strength_config()
            
            # Apply any user overrides
            self.config.update(override_config)
        else:
            # Use default configuration
            self.capabilities = None
            self.config = {
                'device': 'cpu',
                'batch_size': 8,
                'gradient_accumulation_steps': 1,
                'use_mixed_precision': False,
                'num_workers': 0,
                'pin_memory': False,
                'learning_rate': 2e-5,
                'warmup_steps': 100,
                'max_grad_norm': 1.0,
                'checkpoint_steps': 500,
                'eval_steps': 250,
                'fp16': False,
                'bf16': False,
            }
            self._apply_strength_config()
            self.config.update(override_config)
        
        # Set device
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Load datasets
        self.train_dataset, self.val_dataset = self._prepare_datasets(
            train_dataset_path, val_dataset_path, train_dataset, val_dataset
        )
        
        # Setup training components
        self._setup_training_components(category_weight, pii_weight)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_category_acc': [],
            'val_pii_f1': [],
            'learning_rates': [],
            'epochs': []
        }
        
        print(f"✅ Enhanced trainer initialized")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   🎯 Training samples: {len(self.train_dataset) if self.train_dataset else 0}")
        print(f"   🎯 Validation samples: {len(self.val_dataset) if self.val_dataset else 0}")
        print(f"   ⚙️ Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print(f"   ⚙️ Learning rate: {self.config['learning_rate']:.2e}")
    
    def _apply_strength_config(self):
        """Apply training strength configuration to base config."""
        # Adjust learning rate
        base_lr = self.config.get('learning_rate', 2e-5)
        self.config['learning_rate'] = base_lr * self.strength_config['learning_rate_multiplier']
        
        # Adjust batch size
        base_batch = self.config.get('batch_size', 8)
        new_batch = int(base_batch * self.strength_config['batch_size_multiplier'])
        self.config['batch_size'] = max(1, new_batch)
        
        # Adjust gradient accumulation
        base_accum = self.config.get('gradient_accumulation_steps', 1)
        new_accum = max(1, base_accum // self.strength_config['gradient_accumulation_divider'])
        self.config['gradient_accumulation_steps'] = new_accum
        
        # Adjust checkpoint frequency
        base_checkpoint = self.config.get('checkpoint_steps', 500)
        new_checkpoint = int(base_checkpoint * self.strength_config['checkpoint_steps_multiplier'])
        self.config['checkpoint_steps'] = max(50, new_checkpoint)
        
        # Adjust evaluation frequency
        base_eval = self.config.get('eval_steps', 250)
        new_eval = int(base_eval * self.strength_config['eval_steps_multiplier'])
        self.config['eval_steps'] = max(25, new_eval)
        
        # Adjust warmup
        base_warmup = self.config.get('warmup_steps', 100)
        warmup_ratio = self.strength_config['warmup_ratio']
        # We'll calculate actual warmup steps later when we know dataset size
        self.config['warmup_ratio'] = warmup_ratio
        
        # Store early stopping patience
        self.config['early_stopping_patience'] = self.strength_config['early_stopping_patience']
        
        print(f"   ⚙️ Strength adjustments applied:")
        print(f"      Learning rate: {self.config['learning_rate']:.2e}")
        print(f"      Batch size: {self.config['batch_size']}")
        print(f"      Gradient accumulation: {self.config['gradient_accumulation_steps']}")
        print(f"      Checkpoint every: {self.config['checkpoint_steps']} steps")
        print(f"      Evaluate every: {self.config['eval_steps']} steps")
    
    def _prepare_datasets(
        self,
        train_path: Optional[str],
        val_path: Optional[str],
        train_dataset: Optional[DualTaskDataset],
        val_dataset: Optional[DualTaskDataset]
    ) -> Tuple[Optional[DualTaskDataset], Optional[DualTaskDataset]]:
        """Prepare training and validation datasets."""
        
        # Use provided datasets if available
        if train_dataset is not None:
            return train_dataset, val_dataset
        
        # Check for missing files if no paths provided
        if not train_path and not val_path:
            print("⚠️  No dataset paths provided. Checking for available dataset files...")
            if not check_missing_files("training"):
                print("\n💡 Run one of the dataset generators first, then retry training.")
                raise FileNotFoundError("Required dataset files not found. See instructions above.")
            
            # Auto-detect dataset files if they exist
            if os.path.exists("real_train_dataset.json"):
                train_path = "real_train_dataset.json"
                print(f"✅ Auto-detected training dataset: {train_path}")
            elif os.path.exists("datasets/real_train_dataset.json"):
                train_path = "datasets/real_train_dataset.json"
                print(f"✅ Auto-detected training dataset: {train_path}")
            
            if os.path.exists("real_val_dataset.json"):
                val_path = "real_val_dataset.json"
                print(f"✅ Auto-detected validation dataset: {val_path}")
            elif os.path.exists("datasets/real_val_dataset.json"):
                val_path = "datasets/real_val_dataset.json"
                print(f"✅ Auto-detected validation dataset: {val_path}")
        
        # Load from paths if provided
        if train_path:
            print(f"📊 Loading training dataset from: {train_path}")
            train_texts, train_categories, train_pii_labels, train_info = load_custom_dataset(
                train_path, tokenizer=self.model.tokenizer
            )
            
            # Print dataset info
            loader = RealDatasetLoader()
            loader.print_dataset_info(train_info)
            
            # Estimate training time
            if self.capabilities:
                estimated_time = estimate_training_time(
                    len(train_texts), self.capabilities, num_epochs=3
                )
                print(f"⏱️  Estimated training time: {estimated_time}")
            
            train_dataset = DualTaskDataset(
                train_texts, train_categories, train_pii_labels,
                self.model.tokenizer, max_length=self.model.max_length
            )
        else:
            train_dataset = None
        
        if val_path:
            print(f"📊 Loading validation dataset from: {val_path}")
            val_texts, val_categories, val_pii_labels, val_info = load_custom_dataset(
                val_path, tokenizer=self.model.tokenizer
            )
            
            val_dataset = DualTaskDataset(
                val_texts, val_categories, val_pii_labels,
                self.model.tokenizer, max_length=self.model.max_length
            )
        else:
            val_dataset = None
        
        return train_dataset, val_dataset
    
    def _setup_training_components(self, category_weight: float, pii_weight: float):
        """Setup loss function, optimizer, scheduler, and data loaders."""
        
        # Loss function
        self.loss_fn = DualTaskLoss(category_weight, pii_weight)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # Data loaders
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                drop_last=self.config.get('dataloader_drop_last', False)
            )
        else:
            self.train_loader = None
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory']
            )
        else:
            self.val_loader = None
        
        # Scheduler (setup after knowing number of steps)
        if self.train_loader:
            total_steps = len(self.train_loader) * 3  # Assume 3 epochs for now
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config['warmup_steps']
            )
        else:
            self.scheduler = None
        
        # Mixed precision scaler
        if self.config['use_mixed_precision'] and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with advanced features."""
        if not self.train_loader:
            raise ValueError("No training dataset provided")
        
        self.model.train()
        total_loss = 0
        total_category_loss = 0
        total_pii_loss = 0
        num_batches = 0
        
        # Setup progress bar
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            category_labels = batch['category_label'].to(self.device, non_blocking=True)
            pii_labels = batch['pii_labels'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    category_logits, pii_logits = self.model(input_ids, attention_mask)
                    loss, cat_loss, pii_loss = self.loss_fn(
                        category_logits, pii_logits, category_labels, pii_labels, attention_mask
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                category_logits, pii_logits = self.model(input_ids, attention_mask)
                loss, cat_loss, pii_loss = self.loss_fn(
                    category_logits, pii_logits, category_labels, pii_labels, attention_mask
                )
                # Scale loss for gradient accumulation
                loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass
                loss.backward()
            
            # Update metrics
            total_loss += loss.item()
            total_category_loss += cat_loss.item()
            total_pii_loss += pii_loss.item()
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Checkpointing
                if self.global_step % self.config['checkpoint_steps'] == 0:
                    self._save_checkpoint(f"checkpoint-step-{self.global_step}")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cat_loss': f'{cat_loss.item():.4f}',
                'pii_loss': f'{pii_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}' if self.scheduler else f'{self.config["learning_rate"]:.2e}'
            })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_cat_loss = total_category_loss / num_batches
        avg_pii_loss = total_pii_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_category_loss': avg_cat_loss,
            'train_pii_loss': avg_pii_loss
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set with detailed metrics."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0
        total_category_loss = 0
        total_pii_loss = 0
        num_batches = 0
        
        all_category_preds = []
        all_category_labels = []
        all_pii_preds = []
        all_pii_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                category_labels = batch['category_label'].to(self.device, non_blocking=True)
                pii_labels = batch['pii_labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        category_logits, pii_logits = self.model(input_ids, attention_mask)
                        loss, cat_loss, pii_loss = self.loss_fn(
                            category_logits, pii_logits, category_labels, pii_labels, attention_mask
                        )
                else:
                    category_logits, pii_logits = self.model(input_ids, attention_mask)
                    loss, cat_loss, pii_loss = self.loss_fn(
                        category_logits, pii_logits, category_labels, pii_labels, attention_mask
                    )
                
                # Update loss metrics
                total_loss += loss.item()
                total_category_loss += cat_loss.item()
                total_pii_loss += pii_loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                category_preds = torch.argmax(category_logits, dim=1)
                all_category_preds.extend(category_preds.cpu().numpy())
                all_category_labels.extend(category_labels.cpu().numpy())
                
                # PII predictions (only for non-padded tokens)
                pii_preds = torch.argmax(pii_logits, dim=2)
                for i in range(len(input_ids)):
                    mask = attention_mask[i].cpu().numpy()
                    valid_length = mask.sum()
                    all_pii_preds.extend(pii_preds[i][:valid_length].cpu().numpy())
                    all_pii_labels.extend(pii_labels[i][:valid_length].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        avg_cat_loss = total_category_loss / num_batches
        avg_pii_loss = total_pii_loss / num_batches
        
        category_acc = accuracy_score(all_category_labels, all_category_preds)
        pii_f1 = f1_score(all_pii_labels, all_pii_preds, average='weighted', zero_division=0)
        
        # Combined score for model selection
        combined_score = (category_acc + pii_f1) / 2
        
        return {
            'val_loss': avg_loss,
            'val_category_loss': avg_cat_loss,
            'val_pii_loss': avg_pii_loss,
            'val_category_acc': category_acc,
            'val_pii_f1': pii_f1,
            'val_combined_score': combined_score
        }
    
    def train(self, num_epochs: Optional[int] = None, save_best_model: bool = True):
        """
        Train the model for specified epochs.
        
        Args:
            num_epochs: Number of epochs (uses training strength default if None)
            save_best_model: Whether to save the best model during training
        """
        # Use training strength default if not specified
        if num_epochs is None:
            num_epochs = self.strength_config['num_epochs']
        
        print(f"\n🚀 Starting enhanced training for {num_epochs} epochs")
        print(f"   📊 Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"   📊 Validation samples: {len(self.val_dataset)}")
        print(f"   ⚙️ Device: {self.device}")
        print(f"   ⚙️ Mixed precision: {self.config['use_mixed_precision']}")
        print(f"   ⚙️ Batch size: {self.config['batch_size']}")
        print(f"   ⚙️ Gradient accumulation: {self.config['gradient_accumulation_steps']}")
        print(f"   ⚙️ Training strength: {self.training_strength}")
        
        # Estimate training time
        if self.capabilities and self.train_dataset:
            estimated_time = estimate_training_time(
                len(self.train_dataset), self.capabilities, num_epochs=num_epochs
            )
            print(f"   ⏱️  Estimated training time: {estimated_time}")
        
        start_time = time.time()
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Evaluation
            if self.val_dataset:
                val_metrics = self.evaluate()
                
                # Early stopping check
                if val_metrics['val_combined_score'] > self.best_val_score:
                    self.best_val_score = val_metrics['val_combined_score']
                    early_stopping_counter = 0
                    
                    if save_best_model:
                        self._save_checkpoint("best_model")
                        print(f"✅ New best model saved! Combined score: {self.best_val_score:.4f}")
                else:
                    early_stopping_counter += 1
                
                # Check early stopping
                patience = self.config['early_stopping_patience']
                if early_stopping_counter >= patience:
                    print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                    break
                
                # Log metrics
                print(f"   📊 Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"   📊 Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"   📊 Category Acc: {val_metrics['val_category_acc']:.4f}")
                print(f"   📊 PII F1: {val_metrics['val_pii_f1']:.4f}")
                print(f"   📊 Combined Score: {val_metrics['val_combined_score']:.4f}")
                print(f"   📊 Early stopping: {early_stopping_counter}/{patience}")
                
                # Update history
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_category_acc'].append(val_metrics['val_category_acc'])
                self.training_history['val_pii_f1'].append(val_metrics['val_pii_f1'])
            else:
                print(f"   📊 Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['epochs'].append(epoch + 1)
            if self.scheduler:
                self.training_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch-{epoch + 1}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.1f} seconds")
        
        # Save final model and history
        self._save_final_model()
        self._save_training_history()
        
        return self.training_history
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_score': self.best_val_score,
            'config': self.config,
            'training_history': self.training_history
        }, checkpoint_path)
    
    def _save_final_model(self):
        """Save the final trained model."""
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        
        # Save model using the DualClassifier's method
        self.model.save_pretrained(str(final_model_dir))
        
        # Save configuration
        config_path = final_model_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def _save_training_history(self):
        """Save training history."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = checkpoint['training_history']
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")


def detect_available_datasets():
    """
    Detect available datasets in the current directory and datasets/ subdirectory.
    Returns a list of dataset info dictionaries.
    """
    datasets = []
    
    # Standard dataset files to look for
    dataset_files = [
        ("real_train_dataset.json", "real_val_dataset.json", "Custom Dataset", "Your custom dataset files"),
        ("extended_train_dataset.json", "extended_val_dataset.json", "Extended Dataset (8 categories)", "Generated multi-category dataset"),
    ]
    
    # Look in both current directory and datasets/ subdirectory
    search_paths = ["./", "./datasets/"]
    
    for search_path in search_paths:
        for train_file, val_file, name, description in dataset_files:
            train_path = os.path.join(search_path, train_file)
            val_path = os.path.join(search_path, val_file)
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                # Try to get category info
                try:
                    with open(train_path, 'r') as f:
                        sample_data = json.load(f)
                    categories = set()
                    for item in sample_data[:100]:
                        if 'category' in item:
                            categories.add(item['category'])
                    
                    datasets.append({
                        'name': name,
                        'description': description,
                        'train_path': train_path,
                        'val_path': val_path,
                        'categories': len(categories),
                        'category_list': sorted(categories),
                        'samples': len(sample_data)
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {train_path}: {e}")
    
    return datasets

def interactive_dataset_selection():
    """
    Interactive dataset selection with arrow key navigation.
    """
    print("🔍 Detecting available datasets...")
    datasets = detect_available_datasets()
    
    if not datasets:
        print("\n❌ No datasets found!")
        print("💡 Available dataset downloaders:")
        print("   📰 python datasets/generators/download_bbc_dataset.py      (5 categories: business, entertainment, politics, sport, tech)")
        print("   📰 python datasets/generators/download_20newsgroups.py     (20 categories: various topics)")
        print("   📰 python datasets/generators/download_agnews.py           (4 categories: world, sports, business, technology)")
        print("   🛠️  python datasets/generators/create_multi_category_dataset.py  (8 categories: custom generated)")
        print("\n🚀 Run one of these scripts first, then come back!")
        return None, None
    
    print(f"\n📊 Found {len(datasets)} available dataset(s):")
    
    if INQUIRER_AVAILABLE:
        # Create choices for inquirer
        choices = []
        for i, dataset in enumerate(datasets):
            choice_text = f"{dataset['name']} - {dataset['categories']} categories, {dataset['samples']} samples"
            choices.append((choice_text, i))
        
        questions = [
            inquirer.List('dataset',
                message="📂 Select dataset",
                choices=choices,
                default=choices[0][1] if choices else None,
            ),
        ]
        
        try:
            answers = inquirer.prompt(questions)
            if answers and 'dataset' in answers:
                selected_idx = answers['dataset']
                selected_dataset = datasets[selected_idx]
                print(f"\n✅ Selected: {selected_dataset['name']}")
                print(f"   📊 Categories ({selected_dataset['categories']}): {', '.join(selected_dataset['category_list'])}")
                return selected_dataset['train_path'], selected_dataset['val_path']
        except KeyboardInterrupt:
            print("\n👋 Selection cancelled.")
            return None, None
    else:
        # Fallback to numbered selection
        for i, dataset in enumerate(datasets):
            print(f"   {i+1}. {dataset['name']}")
            print(f"      📊 Categories: {dataset['categories']} ({', '.join(dataset['category_list'])})")
            print(f"      📄 Samples: {dataset['samples']}")
            print(f"      📁 Files: {dataset['train_path']}, {dataset['val_path']}")
            print()
        
        while True:
            try:
                choice = input(f"Select dataset (1-{len(datasets)}) [1]: ").strip()
                if not choice:
                    choice = "1"
                
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    selected_dataset = datasets[idx]
                    print(f"\n✅ Selected: {selected_dataset['name']}")
                    return selected_dataset['train_path'], selected_dataset['val_path']
                else:
                    print(f"❌ Please enter a number between 1 and {len(datasets)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Selection cancelled.")
                return None, None
    
    return None, None

def interactive_strength_selection():
    """
    Interactive training strength selection with arrow key navigation.
    """
    strengths = TrainingStrengthConfig.list_strengths()
    
    print("\n🎯 Available Training Strengths:")
    for strength, description in strengths.items():
        epochs = TrainingStrengthConfig.get_config(strength)['num_epochs']
        print(f"   {strength.upper()}: {description} ({epochs} epochs)")
    
    if INQUIRER_AVAILABLE:
        # Create choices for inquirer
        choices = []
        for strength, description in strengths.items():
            epochs = TrainingStrengthConfig.get_config(strength)['num_epochs']
            choice_text = f"{strength.upper()} - {description} ({epochs} epochs)"
            choices.append((choice_text, strength))
        
        questions = [
            inquirer.List('strength',
                message="⚡ Select training strength",
                choices=choices,
                default='normal',
            ),
        ]
        
        try:
            answers = inquirer.prompt(questions)
            if answers and 'strength' in answers:
                selected_strength = answers['strength']
                print(f"\n✅ Selected training strength: {selected_strength.upper()}")
                return selected_strength
        except KeyboardInterrupt:
            print("\n👋 Selection cancelled.")
            return None
    else:
        # Fallback to numbered selection
        strength_list = list(strengths.keys())
        for i, strength in enumerate(strength_list):
            epochs = TrainingStrengthConfig.get_config(strength)['num_epochs']
            marker = " (default)" if strength == 'normal' else ""
            print(f"   {i+1}. {strength.upper()}: {strengths[strength]} ({epochs} epochs){marker}")
        
        while True:
            try:
                choice = input(f"Select strength (1-{len(strength_list)}) [2 for normal]: ").strip()
                if not choice:
                    choice = "2"  # Default to normal
                
                idx = int(choice) - 1
                if 0 <= idx < len(strength_list):
                    selected_strength = strength_list[idx]
                    print(f"\n✅ Selected training strength: {selected_strength.upper()}")
                    return selected_strength
                else:
                    print(f"❌ Please enter a number between 1 and {len(strength_list)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Selection cancelled.")
                return None
    
    return 'normal'  # Default fallback


def create_sample_real_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample real dataset in JSON format for testing.
    
    This simulates what a real dataset might look like.
    """
    import random
    
    categories = ['technology', 'science', 'politics', 'sports', 'business']
    
    samples = []
    for i in range(num_samples):
        category = random.choice(categories)
        
        # Create sample texts with potential PII
        base_texts = {
            'technology': [
                "How does artificial intelligence work?",
                "The latest smartphone features are impressive",
                "Cloud computing is transforming businesses",
                "Cybersecurity threats are increasing",
                "Machine learning algorithms are complex"
            ],
            'science': [
                "Climate change affects global temperatures",
                "DNA research reveals new insights",
                "Space exploration continues to advance",
                "Medical breakthroughs save lives",
                "Quantum physics explains reality"
            ],
            'politics': [
                "Election results vary by region",
                "Policy changes affect citizens",
                "Government spending increases annually",
                "International relations remain complex",
                "Political debates shape public opinion"
            ],
            'sports': [
                "The championship game was exciting",
                "Athletes train for months",
                "Team performance exceeded expectations",
                "Sports statistics reveal trends",
                "Coaching strategies influence outcomes"
            ],
            'business': [
                "Market trends affect stock prices",
                "Company profits increased quarterly",
                "Economic indicators show growth",
                "Investment strategies vary widely",
                "Business partnerships drive success"
            ]
        }
        
        text = random.choice(base_texts[category])
        
        # Occasionally add PII for testing
        if random.random() < 0.3:
            pii_additions = [
                " Contact John Smith at john.smith@company.com",
                " Call 555-123-4567 for more information",
                " Visit our office at 123 Main Street, New York",
                " Email support@business.com for help",
                " Reach out to Sarah Johnson for details"
            ]
            text += random.choice(pii_additions)
        
        sample = {
            'text': text,
            'category': category
        }
        
        samples.append(sample)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Created sample dataset with {num_samples} samples at {output_path}")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Enhanced trainer with real data and interactive selection
    print("🚀 Enhanced Dual-Purpose Classifier Training")
    print("=" * 60)
    
    # Interactive dataset selection
    train_path, val_path = interactive_dataset_selection()
    if not train_path or not val_path:
        print("👋 Exiting...")
        sys.exit(0)
    
    # Interactive strength selection
    training_strength = interactive_strength_selection()
    if not training_strength:
        print("👋 Exiting...")
        sys.exit(0)
    
    # Check how many categories are in the selected data
    try:
        with open(train_path, 'r') as f:
            sample_data = json.load(f)
        categories = set()
        for item in sample_data[:100]:  # Check first 100 samples
            if 'category' in item:
                categories.add(item['category'])
        num_categories = len(categories)
        print(f"\n📂 Dataset Analysis:")
        print(f"   Categories ({num_categories}): {sorted(categories)}")
        print(f"   Training samples: {len(sample_data)}")
        
        # Show validation info too
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        print(f"   Validation samples: {len(val_data)}")
        
    except Exception as e:
        print(f"⚠️ Could not analyze dataset, using default (4): {e}")
        num_categories = 4
    
    # Initialize model with correct number of categories
    print(f"\n🧠 Initializing model with {num_categories} categories...")
    model = DualClassifier(num_categories=num_categories)
    
    # Create output directory based on strength
    output_dir = f"./enhanced_training_{training_strength}"
    
    # Create enhanced trainer with selected parameters
    print(f"🏋️  Setting up enhanced trainer...")
    try:
        trainer = EnhancedDualTaskTrainer(
            model=model,
            train_dataset_path=train_path,
            val_dataset_path=val_path,
            auto_detect_hardware=True,
            training_strength=training_strength,
            output_dir=output_dir
        )
        
        # Show training summary
        strength_config = TrainingStrengthConfig.get_config(training_strength)
        expected_epochs = strength_config['num_epochs']
        
        print(f"\n🚀 Training Configuration:")
        print(f"   📂 Dataset: {train_path}")
        print(f"   🎯 Training strength: {training_strength.upper()}")
        print(f"   📊 Categories: {num_categories}")
        print(f"   🔄 Expected epochs: {expected_epochs}")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   ⏱️  Early stopping patience: {strength_config['early_stopping_patience']}")
        
        # Final confirmation
        confirm = input(f"\n🔥 Start training with these settings? (Y/n): ").strip().lower()
        if confirm and confirm not in ['y', 'yes', '']:
            print("👋 Training cancelled.")
            sys.exit(0)
        
        # Start training
        print(f"\n🔥 Starting {training_strength} training...")
        history = trainer.train(save_best_model=True)
        
        # Show final results
        if history['val_category_acc']:
            final_acc = history['val_category_acc'][-1]
            final_f1 = history['val_pii_f1'][-1]
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Final Results:")
            print(f"   Category Accuracy: {final_acc:.3f}")
            print(f"   PII F1 Score: {final_f1:.3f}")
            print(f"   Model saved to: {output_dir}/final_model/")
            
            # Show improvement
            if len(history['val_category_acc']) > 1:
                initial_acc = history['val_category_acc'][0]
                improvement = final_acc - initial_acc
                print(f"   Improvement: +{improvement:.3f} accuracy")
        else:
            print("✅ Training completed (no validation metrics)")
            
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Show helpful tips
        print(f"\n💡 Training Tips:")
        print("   - Try 'quick' strength for faster testing")
        print("   - Ensure you have enough memory for the batch size")
        print("   - Check that your datasets are in the correct format")
        print("   - GPU training is much faster if available")
        print("   - Install inquirer for better selection: pip install inquirer")
    
    print(f"\n📁 Output saved to: {output_dir}/")
    print("🚀 Use your trained model in live_demo.py for testing!") 