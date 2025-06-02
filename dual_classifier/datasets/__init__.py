# Dataset utilities for dual classifier
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import real_data_loader from this directory
try:
    from .real_data_loader import RealDatasetLoader
except ImportError:
    RealDatasetLoader = None

# Import dataset_loaders from parent directory
try:
    from dataset_loaders import load_custom_dataset
except ImportError:
    load_custom_dataset = None

__all__ = ['RealDatasetLoader', 'load_custom_dataset']
