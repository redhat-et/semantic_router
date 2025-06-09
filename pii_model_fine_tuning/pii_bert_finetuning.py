import os
import json
import torch
import numpy as np
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom cross entropy loss compatible with sentence-transformers
class PIIClassificationLoss(torch.nn.Module):
    def __init__(self, model):
        super(PIIClassificationLoss, self).__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, sentence_features, labels):
        embeddings = self.model(sentence_features[0])['sentence_embedding']
        
        for i in range(1, len(sentence_features)):
            emb = self.model(sentence_features[i])['sentence_embedding']
            embeddings = torch.cat((embeddings, emb.unsqueeze(0)))
        
        label_tensor = torch.tensor(labels, dtype=torch.long, device=embeddings.device)
        
        return self.loss_fn(embeddings, label_tensor)
        
    def __call__(self, sentence_features, labels):
        return self.forward(sentence_features, labels)

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

# Function to predict PII type using the linear classification model
def predict_pii_type(model, text, idx_to_label_map):
    """Predict PII type for a given text."""
    logits_np = model.encode(text, show_progress_bar=False)
    logits_tensor = torch.tensor(logits_np)

    probabilities = torch.softmax(logits_tensor, dim=0) 

    confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()

    predicted_pii_type = idx_to_label_map.get(predicted_idx, "Unknown PII Type")
    
    return predicted_pii_type, confidence

# Evaluate on validation set using the linear classification model
def evaluate_pii_classifier(model, texts_list, true_label_indices_list, idx_to_label_map):
    """Evaluate the PII classifier on a dataset."""
    correct = 0
    total = len(texts_list)
    
    if total == 0:
        return 0.0

    for text, true_label_idx in zip(texts_list, true_label_indices_list):
        predicted_pii_type, _ = predict_pii_type(model, text, idx_to_label_map)
        true_pii_type = idx_to_label_map.get(true_label_idx)
        
        if true_pii_type == predicted_pii_type:
            correct += 1
    
    return correct / total

def main():
    """Main function to demonstrate PII classification fine-tuning."""
    
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

    # TODO: use a better base model that supports token classification
    word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L12-v2')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=len(unique_categories),
        activation_function=torch.nn.Identity()
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_samples = [(text, category) for text, category in zip(train_texts, train_categories)]

    train_loss = PIIClassificationLoss(model)

    num_epochs = 8
    batch_size = 16
    train_examples = []
    for text, category_idx in train_samples:
        train_examples.append(InputExample(texts=[text], label=category_idx))

    warmup_steps = int(len(train_examples) * num_epochs * 0.1 / batch_size)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    output_model_path = "pii_classifier_linear_model"
    os.makedirs(output_model_path, exist_ok=True)

    logger.info("Starting PII classification fine-tuning...")

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=None,
        output_path=output_model_path,
        show_progress_bar=True
    )

    # Save the PII type mapping
    pii_mapping_path = os.path.join(output_model_path, "pii_type_mapping.json")
    with open(pii_mapping_path, "w") as f:
        json.dump({
            "label_to_idx": category_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_category.items()} # JSON keys must be strings
        }, f)

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_accuracy = evaluate_pii_classifier(model, val_texts, val_categories, idx_to_category)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_accuracy = evaluate_pii_classifier(model, test_texts, test_categories, idx_to_category)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save(output_model_path)

    # Print final results
    print("\n" + "="*50)
    print("PII Classification Fine-tuning Completed!")
    print("="*50)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, idx_to_category

def demo_inference():
    """Demonstrate inference with the trained model."""
    
    model_path = "./pii_classifier_linear_model"
    if not Path(model_path).exists():
        logger.error("Trained model not found. Please run training first.")
        return
    
    model = SentenceTransformer(model_path)
    
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
        predicted_pii_type, confidence = predict_pii_type(model, text, idx_to_label)
        print(f"Text: {text}")
        print(f"Predicted PII Type: {predicted_pii_type}, Confidence: {confidence:.4f}")
        print("---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Classification Fine-tuning")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: 'train' to fine-tune model, 'test' to run inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main()
    elif args.mode == "test":
        demo_inference() 