#!/usr/bin/env python3
"""
Install required dependencies for enhanced interactive training.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Installing Enhanced Training Dependencies")
    print("=" * 45)
    
    packages = [
        ("inquirer", "Interactive arrow-key selection"),
        ("requests", "Dataset downloading"),
        ("pandas", "Data processing"),
        ("scikit-learn", "Dataset splitting and metrics"),
        ("datasets", "Hugging Face datasets (optional)")
    ]
    
    for package, description in packages:
        print(f"📦 Installing {package} - {description}")
        if install_package(package):
            print(f"   ✅ {package} installed successfully")
        else:
            print(f"   ⚠️ Failed to install {package} (might already be installed)")
        print()
    
    print("🎉 Installation complete!")
    print("🚀 You can now run: python enhanced_trainer.py")

if __name__ == "__main__":
    main() 