# 🛠️ LEAF-YOLO Development Tools

<div align="center">

**⚙️ Developer Toolkit**

*Essential tools for development, testing, and deployment*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Testing Guide](https://img.shields.io/badge/🧪%20See-Testing%20Guide-blue?style=for-the-badge)](../tests/README.md)

</div>

---

## 📋 Table of Contents

- [🔧 Development Tools](#-development-tools)
- [🧪 Testing Tools](#-testing-tools)
- [📊 Analysis Tools](#-analysis-tools)
- [🚀 Deployment Tools](#-deployment-tools)
- [⚙️ Configuration](#️-configuration)

---

## 🔧 **Development Tools**

### **Code Quality Tools**

#### **🎨 Makefile** - Build Automation
```bash
# Available commands
make help           # Show all available commands
make install        # Install dependencies
make test          # Run all tests
make test-fast     # Run fast tests only
make lint          # Run code linting
make format        # Format code
make clean         # Clean up generated files
make docs          # Generate documentation
```

#### **📝 Makefile Commands**

| Command | Description | Usage |
|---------|-------------|--------|
| `make install` | Install all dependencies | Development setup |
| `make test` | Run complete test suite | Before commits |
| `make test-fast` | Quick unit tests | During development |
| `make lint` | Check code quality | Code review |
| `make format` | Auto-format code | Code cleanup |
| `make ci` | CI pipeline simulation | Pre-push |

### **Example Makefile Usage**

```bash
# Initial setup
make install

# Development workflow
make test-fast      # Quick check
make format         # Clean up code
make lint          # Check quality
make test          # Full testing
```

---

## 🧪 **Testing Tools**

### **📋 pytest.ini** - Test Configuration

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    benchmark: Performance benchmarks
    gpu: Requires GPU
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
    -v
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

### **Test Execution**

```bash
# Run different test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m "not slow"        # Skip slow tests
pytest -m benchmark         # Performance tests
pytest -m gpu               # GPU tests

# Coverage reporting
pytest --cov=leafyolo --cov-report=html

# Parallel testing
pytest -n auto             # Use all CPU cores
pytest -n 4                # Use 4 processes
```

---

## 📊 **Analysis Tools**

### **🔍 Code Analysis**

#### **Quality Checks**
```bash
# Flake8 - Style guide enforcement
flake8 leafyolo/ scripts/ tests/

# Black - Code formatting
black leafyolo/ scripts/ tests/

# isort - Import sorting
isort leafyolo/ scripts/ tests/

# mypy - Type checking
mypy leafyolo/
```

#### **Performance Analysis**
```python
# tools/profiler.py - Performance profiling
python tools/profiler.py --model leafyolo_s --input test_image.jpg

# tools/memory_profiler.py - Memory usage analysis
python tools/memory_profiler.py --model leafyolo_m --batch-size 16

# tools/benchmark.py - Speed benchmarking  
python tools/benchmark.py --variants leafyolo_n leafyolo_s leafyolo_m
```

### **📈 Metrics Collection**

```python
# tools/collect_metrics.py
"""
Collect comprehensive model metrics:
- Inference speed across different hardware
- Memory usage patterns
- Model accuracy on various datasets
- Export format compatibility
"""

# Usage
python tools/collect_metrics.py \
    --models leafyolo_n leafyolo_s leafyolo_m \
    --datasets coco visdrone \
    --formats pytorch onnx tflite \
    --devices cpu cuda
```

---

## 🚀 **Deployment Tools**

### **📦 Export & Conversion**

```python
# tools/export_all_formats.py
"""
Export model to all supported formats:
- PyTorch (.pt)
- ONNX (.onnx) 
- TensorRT (.engine)
- TensorFlow Lite (.tflite)
- CoreML (.mlmodel)
"""

# Usage
python tools/export_all_formats.py \
    --model best.pt \
    --img-size 640 \
    --batch-size 1 \
    --device cuda
```

### **🔧 Optimization Tools**

```python
# tools/optimize_model.py
"""
Model optimization techniques:
- Pruning for smaller models
- Quantization for faster inference
- Knowledge distillation
- Architecture search
"""

# Usage  
python tools/optimize_model.py \
    --model leafyolo_m.pt \
    --method pruning \
    --target-size 50  # 50% of original size
```

---

## ⚙️ **Configuration**

### **🔧 Development Environment**

#### **.gitignore** - Git Ignore Rules
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# LEAF-YOLO specific
runs/
datasets/
weights/
*.pt
*.onnx
*.engine

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

#### **pyproject.toml** - Modern Python Configuration
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]

[project]
name = "leafyolo"
version = "1.0.0"
description = "Lightweight, Efficient, Accurate, Fast YOLO for Aerial Imagery"
authors = [{name = "LEAF-YOLO Team"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.8"

dependencies = [
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "numpy>=1.18.5",
    "opencv-python>=4.1.2",
    "pillow>=7.1.2",
    "pyyaml>=5.3.1",
    "matplotlib>=3.2.2",
    "scipy>=1.4.1",
    "tqdm>=4.41.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.10",
    "pytest-xdist>=2.0",
    "black>=21.0",
    "flake8>=3.8",
    "isort>=5.0",
    "mypy>=0.910",
]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

---

## 🛠️ **Utility Scripts**

### **Development Helpers**

```python
# tools/setup_dev_env.py
"""
Set up complete development environment:
- Install dependencies
- Setup pre-commit hooks  
- Configure IDE settings
- Download test datasets
"""

# tools/check_installation.py
"""
Verify LEAF-YOLO installation:
- Check dependencies
- Test import statements
- Validate model loading
- Run basic inference test
"""

# tools/clean_project.py
"""
Clean project directory:
- Remove __pycache__ directories
- Clean build artifacts
- Remove temporary files
- Reset git to clean state
"""
```

### **Automation Scripts**

```bash
# tools/run_experiments.py
"""
Automated experiment runner:
- Train multiple model variants
- Test on different datasets  
- Compare performance metrics
- Generate comparison reports
"""

# Usage
python tools/run_experiments.py \
    --variants leafyolo_n leafyolo_s leafyolo_m \
    --datasets coco visdrone custom \
    --epochs 100 \
    --output-dir experiments/
```

---

## 📊 **Monitoring & Logging**

### **Performance Monitoring**

```python
# tools/monitor_training.py
"""
Real-time training monitoring:
- GPU utilization tracking
- Memory usage monitoring
- Training speed analysis
- Early stopping suggestions
"""

# tools/log_analyzer.py
"""
Training log analysis:
- Parse training logs
- Generate progress reports
- Identify bottlenecks
- Suggest optimizations
"""
```

### **System Health Checks**

```python
# tools/system_check.py
"""
System compatibility check:
- Hardware detection
- Driver version verification
- Performance benchmarking
- Compatibility warnings
"""

# Usage
python tools/system_check.py --verbose
```

---

## 🎯 **Quick Commands Reference**

### **Daily Development**

```bash
# Start development session
make install            # Install dependencies
make test-fast         # Quick test
code .                 # Open in VS Code

# Before committing
make format            # Format code
make lint              # Check quality  
make test              # Full test suite
git add . && git commit -m "feat: new feature"

# Before pushing
make ci                # Simulate CI pipeline
git push origin main
```

### **Release Preparation**

```bash
# Performance analysis
python tools/benchmark.py --all-variants

# Export all formats
python tools/export_all_formats.py --model best.pt

# Generate documentation
make docs

# Clean and package
make clean
python setup.py sdist bdist_wheel
```

---

## 🔧 **Troubleshooting Tools**

### **Debug Helpers**

```python
# tools/debug_model.py
"""
Model debugging utilities:
- Layer-by-layer output inspection
- Gradient flow analysis
- Memory leak detection
- Performance bottleneck identification
"""

# tools/fix_common_issues.py
"""
Automated issue resolution:
- Fix import path problems
- Resolve dependency conflicts
- Clean corrupted cache files
- Reset configuration files
"""
```

---

<div align="center">

**🛠️ Development Tools Ready!**

*Everything you need for efficient LEAF-YOLO development*

[![Setup Environment](https://img.shields.io/badge/⚙️%20Setup-Dev%20Environment-green?style=for-the-badge)](#-development-tools)
[![Run Tests](https://img.shields.io/badge/🧪%20Run-All%20Tests-blue?style=for-the-badge)](#-testing-tools)

</div>
