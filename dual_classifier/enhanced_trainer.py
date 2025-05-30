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
from trainer import DualTaskDataset, DualTaskLoss  # Use existing classes
from hardware_detector import HardwareDetector, HardwareCapabilities, detect_and_configure, estimate_training_time
from dataset_loaders import RealDatasetLoader, DatasetInfo, load_custom_dataset


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
    """
    
    def __init__(
        self,
        model: DualClassifier,
        train_dataset_path: Optional[str] = None,
        val_dataset_path: Optional[str] = None,
        train_dataset: Optional[DualTaskDataset] = None,
        val_dataset: Optional[DualTaskDataset] = None,
        auto_detect_hardware: bool = True,
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
            category_weight: Weight for category classification loss
            pii_weight: Weight for PII detection loss
            output_dir: Directory for outputs and checkpoints
            **override_config: Override hardware-detected configuration
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardware detection and configuration
        if auto_detect_hardware:
            print("🔍 Detecting hardware capabilities...")
            self.capabilities, self.config = detect_and_configure()
            
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
    
    def train(self, num_epochs: int = 3, save_best_model: bool = True):
        """Train the model for specified epochs."""
        print(f"\n🚀 Starting enhanced training for {num_epochs} epochs")
        print(f"   📊 Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"   📊 Validation samples: {len(self.val_dataset)}")
        print(f"   ⚙️ Device: {self.device}")
        print(f"   ⚙️ Mixed precision: {self.config['use_mixed_precision']}")
        print(f"   ⚙️ Batch size: {self.config['batch_size']}")
        print(f"   ⚙️ Gradient accumulation: {self.config['gradient_accumulation_steps']}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Evaluation
            if self.val_dataset:
                val_metrics = self.evaluate()
                
                # Model selection based on combined score
                if save_best_model and val_metrics['val_combined_score'] > self.best_val_score:
                    self.best_val_score = val_metrics['val_combined_score']
                    self._save_checkpoint("best_model")
                    print(f"✅ New best model saved! Combined score: {self.best_val_score:.4f}")
                
                # Log metrics
                print(f"   📊 Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"   📊 Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"   📊 Category Acc: {val_metrics['val_category_acc']:.4f}")
                print(f"   📊 PII F1: {val_metrics['val_pii_f1']:.4f}")
                print(f"   📊 Combined Score: {val_metrics['val_combined_score']:.4f}")
                
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
    # Demo the enhanced trainer
    print("🚀 Enhanced Trainer Demo")
    
    # Create sample datasets
    create_sample_real_dataset("sample_train.json", 80)
    create_sample_real_dataset("sample_val.json", 20)
    
    # Initialize model
    model = DualClassifier(num_categories=5)  # 5 categories in our sample data
    
    # Create enhanced trainer
    trainer = EnhancedDualTaskTrainer(
        model=model,
        train_dataset_path="sample_train.json",
        val_dataset_path="sample_val.json",
        auto_detect_hardware=True,
        output_dir="./enhanced_training_demo"
    )
    
    # Train the model
    try:
        history = trainer.train(num_epochs=2, save_best_model=True)
        print("✅ Enhanced training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        # Clean up demo files
        import os
        for file in ["sample_train.json", "sample_val.json"]:
            if os.path.exists(file):
                os.remove(file) 