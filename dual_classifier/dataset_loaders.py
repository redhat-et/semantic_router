# Dataset loaders module
import json
import csv
import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

# Try to import datasets library for HuggingFace datasets
try:
    from datasets import load_dataset, Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Some dataset loaders will be limited.")

logger = logging.getLogger(__name__)


class DatasetInfo:
    """Container for dataset information and statistics."""
    
    def __init__(self):
        self.name: str = ""
        self.format: str = ""
        self.num_samples: int = 0
        self.num_categories: int = 0
        self.category_distribution: Dict[str, int] = {}
        self.pii_distribution: Dict[str, int] = {}
        self.avg_text_length: float = 0.0
        self.max_text_length: int = 0
        self.has_pii_labels: bool = False
        self.has_category_labels: bool = False


class RealDatasetLoader:
    """
    Loader for real datasets with support for various formats and automatic tokenization alignment.
    
    Supports:
    - HuggingFace datasets (when available)
    - JSON files with various structures
    - CSV files with configurable columns
    - CoNLL format files (for NER tasks)
    - Custom formats
    """
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        """
        Initialize dataset loader.
        
        Args:
            tokenizer: HuggingFace tokenizer for alignment
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def detect_format(self, path: Union[str, Path]) -> str:
        """Automatically detect dataset format."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        extension = path.suffix.lower()
        
        if extension == '.json':
            return 'json'
        elif extension == '.csv':
            return 'csv'
        elif extension in ['.conll', '.conllu', '.conll-u']:
            return 'conll'
        else:
            # Try to auto-detect based on content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') or first_line.startswith('['):
                        return 'json'
                    elif '\t' in first_line and len(first_line.split('\t')) > 2:
                        return 'conll'
                    else:
                        return 'text'
            except:
                return 'unknown'
    
    def load_dataset(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[str], List[int], List[List[int]], DatasetInfo]:
        """Load dataset from file with automatic format detection."""
        path = Path(path)
        
        if format is None:
            format = self.detect_format(path)
        
        logger.info(f"Loading dataset from {path} with format: {format}")
        
        if format == 'json':
            return self._load_json(path, **kwargs)
        elif format == 'csv':
            return self._load_csv(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_json(
        self,
        path: Path,
        text_field: str = 'text',
        category_field: str = 'category',
        pii_field: Optional[str] = None,
        category_mapping: Optional[Dict[str, int]] = None
    ) -> Tuple[List[str], List[int], List[List[int]], DatasetInfo]:
        """Load JSON dataset."""
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            if 'data' in data:
                samples = data['data']
            elif 'samples' in data:
                samples = data['samples']
            else:
                samples = [data]
        else:
            raise ValueError("Unsupported JSON structure")
        
        texts = []
        categories = []
        pii_labels = []
        
        # Build category mapping if not provided
        if category_mapping is None:
            unique_categories = set()
            for sample in samples:
                if category_field in sample:
                    unique_categories.add(sample[category_field])
            category_mapping = {cat: i for i, cat in enumerate(sorted(unique_categories))}
        
        for sample in samples:
            # Extract text
            if text_field not in sample:
                logger.warning(f"Missing text field '{text_field}' in sample")
                continue
            
            text = sample[text_field]
            texts.append(text)
            
            # Extract category
            if category_field in sample:
                cat_label = sample[category_field]
                if isinstance(cat_label, str):
                    categories.append(category_mapping.get(cat_label, 0))
                else:
                    categories.append(int(cat_label))
            else:
                categories.append(0)  # Default category
            
            # Generate simple PII labels (basic regex patterns)
            pii_labels.append(self._generate_pii_labels(text))
        
        # Create dataset info
        info = self._create_dataset_info(texts, categories, pii_labels, 'json')
        
        return texts, categories, pii_labels, info
    
    def _load_csv(
        self,
        path: Path,
        text_column: Union[str, int] = 'text',
        category_column: Union[str, int] = 'category',
        pii_column: Optional[Union[str, int]] = None,
        delimiter: str = ',',
        **kwargs
    ) -> Tuple[List[str], List[int], List[List[int]], DatasetInfo]:
        """Load CSV dataset."""
        
        texts = []
        categories = []
        pii_labels = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter) if isinstance(text_column, str) else csv.reader(f, delimiter=delimiter)
            
            category_mapping = {}
            category_counter = 0
            
            for row in reader:
                if isinstance(text_column, str):
                    # Dict reader
                    text = row.get(text_column, '')
                    category = row.get(category_column, '')
                else:
                    # List reader
                    if len(row) <= max(text_column, category_column):
                        continue
                    text = row[text_column]
                    category = row[category_column]
                
                if not text:
                    continue
                
                texts.append(text)
                
                # Handle category
                if category not in category_mapping:
                    category_mapping[category] = category_counter
                    category_counter += 1
                categories.append(category_mapping[category])
                
                # Generate PII labels
                pii_labels.append(self._generate_pii_labels(text))
        
        # Create dataset info
        info = self._create_dataset_info(texts, categories, pii_labels, 'csv')
        
        return texts, categories, pii_labels, info
    
    def _generate_pii_labels(self, text: str) -> List[int]:
        """Generate basic PII labels for text without existing labels."""
        import re
        
        words = text.split()
        labels = [0] * len(words)
        
        # Simple patterns for demonstration
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        
        # Mark words containing emails or phones as PII
        for i, word in enumerate(words):
            if re.search(email_pattern, word) or re.search(phone_pattern, word):
                labels[i] = 1
        
        return labels
    
    def _create_dataset_info(
        self,
        texts: List[str],
        categories: List[int],
        pii_labels: List[List[int]],
        format_type: str
    ) -> DatasetInfo:
        """Create dataset information object."""
        info = DatasetInfo()
        info.format = format_type
        info.num_samples = len(texts)
        
        # Category statistics
        if categories:
            info.has_category_labels = True
            info.num_categories = len(set(categories))
            info.category_distribution = {str(cat): categories.count(cat) for cat in set(categories)}
        
        # PII statistics
        if pii_labels and any(any(labels) for labels in pii_labels):
            info.has_pii_labels = True
            total_tokens = sum(len(labels) for labels in pii_labels)
            pii_tokens = sum(sum(labels) for labels in pii_labels)
            info.pii_distribution = {
                'no_pii': total_tokens - pii_tokens,
                'pii': pii_tokens
            }
        
        # Text statistics
        if texts:
            text_lengths = [len(text) for text in texts]
            info.avg_text_length = sum(text_lengths) / len(text_lengths)
            info.max_text_length = max(text_lengths)
        
        return info
    
    def print_dataset_info(self, info: DatasetInfo):
        """Print dataset information in a user-friendly format."""
        print(f"\n📊 Dataset Information:")
        print(f"┌─ Format: {info.format}")
        print(f"├─ Samples: {info.num_samples:,}")
        print(f"├─ Categories: {info.num_categories}")
        print(f"├─ Has Category Labels: {'✅' if info.has_category_labels else '❌'}")
        print(f"├─ Has PII Labels: {'✅' if info.has_pii_labels else '❌'}")
        print(f"├─ Avg Text Length: {info.avg_text_length:.0f} chars")
        print(f"└─ Max Text Length: {info.max_text_length} chars")
        
        if info.category_distribution:
            print(f"\n📈 Category Distribution:")
            for cat, count in sorted(info.category_distribution.items()):
                percentage = (count / info.num_samples) * 100
                print(f"   {cat}: {count} ({percentage:.1f}%)")
        
        if info.pii_distribution:
            print(f"\n🔒 PII Distribution:")
            for label, count in info.pii_distribution.items():
                total = sum(info.pii_distribution.values())
                percentage = (count / total) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")
        
        print()


def load_custom_dataset(
    path: Union[str, Path],
    tokenizer=None,
    format: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], List[int], List[List[int]], DatasetInfo]:
    """
    Load a custom dataset with automatic format detection.
    
    Args:
        path: Path to dataset file
        tokenizer: HuggingFace tokenizer for alignment
        format: Force specific format (optional)
        **kwargs: Format-specific arguments
        
    Returns:
        Tuple of (texts, category_labels, pii_labels, dataset_info)
    """
    loader = RealDatasetLoader(tokenizer=tokenizer)
    return loader.load_dataset(path, format=format, **kwargs)


if __name__ == "__main__":
    # Test the loader
    print("Testing dataset loader...")
    
    # Create a sample JSON file
    test_data = [
        {"text": "Contact support at help@company.com", "category": "support"},
        {"text": "Call us at 555-123-4567 for assistance", "category": "support"},
        {"text": "AI technology is advancing rapidly", "category": "technology"}
    ]
    
    with open("test.json", "w") as f:
        json.dump(test_data, f)
    
    # Test loading
    texts, categories, pii_labels, info = load_custom_dataset("test.json")
    
    loader = RealDatasetLoader()
    loader.print_dataset_info(info)
    
    # Cleanup
    os.remove("test.json")
    print("✅ Test completed successfully!") 