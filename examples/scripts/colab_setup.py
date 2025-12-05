#!/usr/bin/env python3
"""
LEAF-YOLO Google Colab Setup Script
Quick setup for running LEAF-YOLO in Google Colab
"""

import os
import sys
import subprocess
import torch

def print_header():
    """Print welcome header"""
    print("🍃 LEAF-YOLO Google Colab Setup")
    print("=" * 50)
    print("Setting up LEAF-YOLO for Google Colab...")
    print()

def check_environment():
    """Check Colab environment"""
    print("🔍 Checking environment...")
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("⚠️  Not running in Google Colab (but that's OK)")
        in_colab = False
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        device = 'cuda'
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
        if in_colab:
            print("💡 To enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU")
        device = 'cpu'
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # Check disk space
    if in_colab:
        disk_usage = os.statvfs('/content')
        free_space = disk_usage.f_bavail * disk_usage.f_frsize / 1024**3
        print(f"💽 Available Space: {free_space:.1f} GB")
    
    print(f"🎯 Using device: {device}")
    print()
    
    return device, in_colab

def install_requirements():
    """Install LEAF-YOLO requirements"""
    print("📦 Installing requirements...")
    
    # Basic requirements
    requirements = [
        "torch>=1.8.0",
        "torchvision>=0.9.0", 
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "matplotlib>=3.2.2",
        "scipy>=1.4.1",
        "tqdm>=4.41.0",
        "pandas>=1.1.4",
        "seaborn>=0.11.0"
    ]
    
    # Colab-specific packages
    colab_packages = [
        "roboflow",
        "supervision", 
        "wandb"
    ]
    
    try:
        for req in requirements:
            print(f"Installing {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", req], 
                         check=True, capture_output=True)
        
        for pkg in colab_packages:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], 
                         check=False, capture_output=True)  # Don't fail if these don't install
        
        print("✅ Requirements installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some packages failed to install: {e}")
        print("This might not affect basic functionality.")
    
    print()

def setup_leafyolo():
    """Setup LEAF-YOLO"""
    print("🚀 Setting up LEAF-YOLO...")
    
    # Check if we're already in LEAF-YOLO directory
    if os.path.exists('leafyolo') and os.path.exists('leafyolo_cli.py'):
        print("✅ LEAF-YOLO already available")
        return
    
    # Clone repository if not available
    try:
        print("📥 Cloning LEAF-YOLO repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/your_username/LEAF-YOLO.git"
        ], check=True, capture_output=True)
        
        # Change to LEAF-YOLO directory
        os.chdir('LEAF-YOLO')
        print("✅ LEAF-YOLO repository cloned")
        
    except subprocess.CalledProcessError:
        print("❌ Failed to clone repository")
        print("💡 Make sure you have internet connection and try again")
        return False
    
    print()
    return True

def test_installation():
    """Test LEAF-YOLO installation"""
    print("🧪 Testing LEAF-YOLO installation...")
    
    try:
        # Test imports
        from leafyolo import LEAFYOLO
        from leafyolo.utils.config import get_config
        print("✅ LEAF-YOLO imported successfully")
        
        # Test configuration
        config = get_config('detect', 'leafyolo_n')
        print("✅ Configuration system working")
        
        # Test model creation
        model = LEAFYOLO('detect', variant='leafyolo_n')
        print("✅ Model creation successful")
        
        print("🎉 LEAF-YOLO is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("💡 Try restarting runtime and running setup again")
        return False

def show_usage_examples():
    """Show basic usage examples"""
    print()
    print("📚 Quick Usage Examples:")
    print("-" * 30)
    
    print("""
# 1. Basic Object Detection
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect', variant='leafyolo_s')
results = model.predict('your_image.jpg')

# 2. Train Custom Model  
model.train(data='dataset.yaml', epochs=100, device='cuda')

# 3. Evaluate Model
metrics = model.val(data='test_dataset.yaml')

# 4. Export Model
model.export(format='onnx')
""")

def create_sample_notebook():
    """Create a sample notebook for quick testing"""
    print("📓 Creating sample notebook...")
    
    notebook_content = """
# LEAF-YOLO Quick Test

## Import and Setup
from leafyolo import LEAFYOLO
import torch
import matplotlib.pyplot as plt

## Create Model
model = LEAFYOLO('detect', variant='leafyolo_s')
print(f"Model loaded with {len(model.names)} classes")

## Test with Random Image
test_image = torch.randn(1, 3, 640, 640)
results = model.predict(test_image)
print("✅ Model working correctly!")

## Upload and Test Your Own Image
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    results = model.predict(filename)
    # Display results here
    break
"""
    
    with open('quick_test.py', 'w') as f:
        f.write(notebook_content)
    
    print("✅ Sample notebook created: quick_test.py")

def main():
    """Main setup function"""
    print_header()
    
    # Check environment
    device, in_colab = check_environment()
    
    # Install requirements
    install_requirements()
    
    # Setup LEAF-YOLO
    if not setup_leafyolo():
        return
    
    # Test installation
    if not test_installation():
        return
    
    # Create sample notebook
    create_sample_notebook()
    
    # Show usage examples
    show_usage_examples()
    
    print()
    print("🎯 Next Steps:")
    print("1. Try the examples above")
    print("2. Upload your own images for detection")
    print("3. Explore the complete training tutorial")
    print("4. Check out the documentation")
    print()
    print("🎉 Happy detecting with LEAF-YOLO!")

if __name__ == "__main__":
    main()
