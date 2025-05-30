import torch
import psutil
import platform
import warnings
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareCapabilities:
    """Container for hardware capability information."""
    
    def __init__(self):
        self.device: str = "cpu"
        self.device_name: str = ""
        self.total_memory_gb: float = 0.0
        self.available_memory_gb: float = 0.0
        self.cpu_count: int = 0
        self.supports_mixed_precision: bool = False
        self.recommended_batch_size: int = 1
        self.gradient_accumulation_steps: int = 1
        self.max_workers: int = 0
        self.memory_fraction: float = 0.8
        self.warnings: list = []


class HardwareDetector:
    """
    Detects hardware capabilities and provides optimal training configurations.
    
    This module helps ensure code can run successfully on different hardware configurations
    by automatically detecting system limitations and providing appropriate fallbacks.
    """
    
    def __init__(self, model_size_mb: float = 250.0):
        """
        Initialize hardware detector.
        
        Args:
            model_size_mb: Estimated model size in MB (DistilBERT ~250MB)
        """
        self.model_size_mb = model_size_mb
    
    def detect_capabilities(self) -> HardwareCapabilities:
        """
        Detect hardware capabilities and return optimal configuration.
        
        Returns:
            HardwareCapabilities object with optimal settings
        """
        capabilities = HardwareCapabilities()
        
        # Detect CPU information
        capabilities.cpu_count = psutil.cpu_count(logical=True)
        
        # Detect system memory
        memory_info = psutil.virtual_memory()
        capabilities.total_memory_gb = memory_info.total / (1024**3)
        capabilities.available_memory_gb = memory_info.available / (1024**3)
        
        # Detect GPU capabilities
        gpu_info = self._detect_gpu()
        if gpu_info['available']:
            capabilities.device = "cuda"
            capabilities.device_name = gpu_info['name']
            capabilities.supports_mixed_precision = gpu_info['supports_fp16']
            
            # Calculate optimal batch size for GPU
            gpu_memory_gb = gpu_info['memory_gb']
            capabilities.recommended_batch_size = self._calculate_gpu_batch_size(gpu_memory_gb)
            capabilities.gradient_accumulation_steps = max(1, 8 // capabilities.recommended_batch_size)
            capabilities.memory_fraction = 0.85  # Use more GPU memory
            capabilities.max_workers = min(4, capabilities.cpu_count // 2)
            
            if gpu_memory_gb < 4.0:
                capabilities.warnings.append(
                    f"GPU has only {gpu_memory_gb:.1f}GB memory. Consider using CPU for large models."
                )
        else:
            # CPU configuration
            capabilities.device = "cpu"
            capabilities.device_name = platform.processor() or "Unknown CPU"
            capabilities.supports_mixed_precision = False
            
            # Calculate optimal batch size for CPU
            capabilities.recommended_batch_size = self._calculate_cpu_batch_size(
                capabilities.available_memory_gb
            )
            capabilities.gradient_accumulation_steps = max(1, 16 // capabilities.recommended_batch_size)
            capabilities.memory_fraction = 0.7  # Conservative CPU memory usage
            capabilities.max_workers = 0  # Avoid multiprocessing issues on CPU
            
            if capabilities.available_memory_gb < 4.0:
                capabilities.warnings.append(
                    f"System has only {capabilities.available_memory_gb:.1f}GB available RAM. "
                    "Training may be very slow or fail."
                )
        
        # Add system-specific warnings
        self._add_system_warnings(capabilities)
        
        return capabilities
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU availability and capabilities."""
        gpu_info = {
            'available': False,
            'name': '',
            'memory_gb': 0.0,
            'supports_fp16': False
        }
        
        if not torch.cuda.is_available():
            return gpu_info
        
        try:
            gpu_info['available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get GPU memory
            props = torch.cuda.get_device_properties(0)
            gpu_info['memory_gb'] = props.total_memory / (1024**3)
            
            # Check mixed precision support (requires Tensor Cores)
            gpu_info['supports_fp16'] = (
                props.major >= 7 or  # Volta and newer
                (props.major == 6 and props.minor >= 1)  # Pascal with Tensor Cores
            )
            
        except Exception as e:
            logger.warning(f"Error detecting GPU capabilities: {e}")
            gpu_info['available'] = False
        
        return gpu_info
    
    def _calculate_gpu_batch_size(self, gpu_memory_gb: float) -> int:
        """Calculate optimal batch size for GPU based on available memory."""
        # Conservative estimates for DistilBERT training
        # These are rough estimates - actual memory usage depends on sequence length
        if gpu_memory_gb >= 24:  # A100, RTX 4090, etc.
            return 32
        elif gpu_memory_gb >= 16:  # V100, RTX 3080, etc.
            return 16
        elif gpu_memory_gb >= 11:  # RTX 2080 Ti, RTX 3060, etc.
            return 8
        elif gpu_memory_gb >= 8:   # RTX 2070, GTX 1080, etc.
            return 4
        elif gpu_memory_gb >= 6:   # RTX 2060, GTX 1060, etc.
            return 2
        else:  # < 6GB
            return 1
    
    def _calculate_cpu_batch_size(self, available_memory_gb: float) -> int:
        """Calculate optimal batch size for CPU based on available memory."""
        # Conservative estimates for CPU training
        if available_memory_gb >= 16:
            return 4
        elif available_memory_gb >= 8:
            return 2
        else:
            return 1
    
    def _add_system_warnings(self, capabilities: HardwareCapabilities):
        """Add system-specific warnings and recommendations."""
        # Check for M1/M2 Macs
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            capabilities.warnings.append(
                "Running on Apple Silicon. Use MPS backend if available (torch.backends.mps)."
            )
            # Check for MPS availability
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                capabilities.device = "mps"
        
        # Check for older CUDA versions
        if capabilities.device == "cuda":
            cuda_version = torch.version.cuda
            if cuda_version and float(cuda_version[:3]) < 11.0:
                capabilities.warnings.append(
                    f"CUDA version {cuda_version} is older. Consider upgrading for better performance."
                )
        
        # Memory warnings
        memory_per_batch = self.model_size_mb * capabilities.recommended_batch_size / 1024
        if memory_per_batch > capabilities.available_memory_gb * capabilities.memory_fraction:
            capabilities.warnings.append(
                f"Estimated memory usage ({memory_per_batch:.1f}GB) may exceed available memory. "
                "Consider reducing batch size."
            )
    
    def print_capabilities(self, capabilities: HardwareCapabilities):
        """Print detected capabilities in a user-friendly format."""
        print("\n🔍 Hardware Detection Results:")
        print(f"┌─ Device: {capabilities.device.upper()}")
        print(f"├─ Device Name: {capabilities.device_name}")
        print(f"├─ Available Memory: {capabilities.available_memory_gb:.1f}GB")
        print(f"├─ CPU Cores: {capabilities.cpu_count}")
        print(f"├─ Mixed Precision: {'✅ Supported' if capabilities.supports_mixed_precision else '❌ Not supported'}")
        print(f"├─ Recommended Batch Size: {capabilities.recommended_batch_size}")
        print(f"├─ Gradient Accumulation Steps: {capabilities.gradient_accumulation_steps}")
        print(f"└─ DataLoader Workers: {capabilities.max_workers}")
        
        if capabilities.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in capabilities.warnings:
                print(f"   • {warning}")
        
        print()
    
    def get_training_config(self, capabilities: HardwareCapabilities) -> Dict[str, Any]:
        """
        Get training configuration based on hardware capabilities.
        
        Args:
            capabilities: Detected hardware capabilities
            
        Returns:
            Dictionary with training configuration
        """
        config = {
            'device': capabilities.device,
            'batch_size': capabilities.recommended_batch_size,
            'gradient_accumulation_steps': capabilities.gradient_accumulation_steps,
            'use_mixed_precision': capabilities.supports_mixed_precision,
            'num_workers': capabilities.max_workers,
            'pin_memory': capabilities.device == "cuda",
            'memory_fraction': capabilities.memory_fraction,
            
            # Training hyperparameters adjusted for hardware
            'learning_rate': 2e-5 if capabilities.device == "cuda" else 1e-5,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'checkpoint_steps': 500,
            'eval_steps': 250,
            'save_total_limit': 3,
        }
        
        # Adjust based on device capability
        if capabilities.device == "cpu":
            config.update({
                'dataloader_drop_last': True,  # Avoid small batches that might cause issues
                'fp16': False,
                'bf16': False,
            })
        elif capabilities.device == "cuda":
            config.update({
                'fp16': capabilities.supports_mixed_precision,
                'bf16': False,  # Can be enabled for A100
                'dataloader_drop_last': False,
            })
        elif capabilities.device == "mps":
            config.update({
                'fp16': False,  # MPS doesn't support fp16 yet
                'bf16': False,
                'dataloader_drop_last': True,
            })
        
        return config


def detect_and_configure() -> Tuple[HardwareCapabilities, Dict[str, Any]]:
    """
    Convenience function to detect hardware and get training configuration.
    
    Returns:
        Tuple of (capabilities, training_config)
    """
    detector = HardwareDetector()
    capabilities = detector.detect_capabilities()
    config = detector.get_training_config(capabilities)
    
    # Print results
    detector.print_capabilities(capabilities)
    
    # Show warnings if any
    if capabilities.warnings:
        for warning in capabilities.warnings:
            warnings.warn(warning, UserWarning)
    
    return capabilities, config


def estimate_training_time(
    num_samples: int,
    capabilities: HardwareCapabilities,
    num_epochs: int = 3
) -> str:
    """
    Estimate training time based on hardware and dataset size.
    
    Args:
        num_samples: Number of training samples
        capabilities: Hardware capabilities
        num_epochs: Number of training epochs
        
    Returns:
        Estimated training time as string
    """
    # Rough estimates based on empirical observations
    if capabilities.device == "cuda":
        samples_per_second = 50 * capabilities.recommended_batch_size
    elif capabilities.device == "mps":
        samples_per_second = 25 * capabilities.recommended_batch_size
    else:  # CPU
        samples_per_second = 5 * capabilities.recommended_batch_size
    
    total_samples = num_samples * num_epochs
    estimated_seconds = total_samples / samples_per_second
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"~{estimated_seconds/60:.0f} minutes"
    else:
        return f"~{estimated_seconds/3600:.1f} hours"


if __name__ == "__main__":
    # Demo the hardware detection
    capabilities, config = detect_and_configure()
    
    print("\n📊 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n⏱️  Estimated training time for 1000 samples: {estimate_training_time(1000, capabilities)}") 