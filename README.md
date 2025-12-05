# 🍃 LEAF-YOLO
### *Lightweight, Efficient, Accurate, Fast YOLO for Aerial Imagery*

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/your_username/LEAF-YOLO?style=social)](https://github.com/Gaurav14cs17/LEAF-YOLO/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/your_username/LEAF-YOLO?style=social)](https://github.com/Gaurav14cs17/LEAF-YOLO/network)

**🚁 Professional Object Detection for Drones, UAVs, and Edge Devices**

[![Quick Start](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb)
[![Complete Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)

<img src="./figure/params.png" width="60%" alt="LEAF-YOLO Performance"/>

</div>

---

## 📋 Table of Contents

- [🎯 What is LEAF-YOLO?](#-what-is-leaf-yolo)
- [🌟 Why Choose LEAF-YOLO?](#-why-choose-leaf-yolo)
- [🚀 Quick Start](#-quick-start-5-minutes-to-success)
- [🏆 Performance](#-performance-that-matters)
- [🛠️ How to Use](#️-how-to-use-leaf-yolo)
- [🎮 Command Line Interface](#-command-line-interface-for-quick-tasks)
- [☁️ Google Colab Integration](#️-google-colab-integration)
- [📚 Model Variants](#-model-variants-choose-your-fighter)
- [📖 Documentation](#-learning--examples)
- [🎨 Advanced Features](#-advanced-features)
- [🤝 Getting Help](#-getting-help--contributing)
- [📄 License & Citation](#-license--citation)

---

---

## 🎯 What is LEAF-YOLO?

LEAF-YOLO is a **game-changing object detection model** designed specifically for **aerial imagery and edge devices**. Think of it as a super-smart AI that can spot tiny objects in drone footage while being lightweight enough to run on your phone or edge computer.

<details>
<summary><strong>🔍 Technical Details</strong></summary>

LEAF-YOLO introduces novel architectural improvements including:
- **Efficient Cross Stage Partial (CSP) connections** for better gradient flow
- **Spatial Pyramid Pooling (SPP)** for multi-scale feature extraction  
- **Coordinate attention mechanisms** for precise spatial localization
- **Lightweight backbone** optimized for aerial imagery characteristics

</details>

### 🌟 Why Choose LEAF-YOLO?

<table>
<tr>
<td align="center"><strong>🪶 Ultra-Lightweight</strong></td>
<td align="center"><strong>⚡ Lightning Fast</strong></td>
<td align="center"><strong>🎯 Highly Accurate</strong></td>
</tr>
<tr>
<td align="center">Only 1.2M parameters for nano version</td>
<td align="center">Real-time on mobile & edge devices</td>
<td align="center">Beats larger models on small objects</td>
</tr>
<tr>
<td align="center"><strong>🚁 Aerial Optimized</strong></td>
<td align="center"><strong>🔧 Easy to Use</strong></td>
<td align="center"><strong>📦 Ready to Deploy</strong></td>
</tr>
<tr>
<td align="center">Purpose-built for drone imagery</td>
<td align="center">Simple Python API - no PhD required</td>
<td align="center">Export to ONNX, TensorRT, CoreML+</td>
</tr>
</table>

---

## 🚀 Quick Start (5 Minutes to Success!)

### 💻 **Local Installation**
```bash
# Create a fresh environment (recommended)
conda create -n leafyolo python=3.9
conda activate leafyolo

# Clone and install
git clone https://github.com/Gaurav14cs17/LEAF-YOLO.git
cd LEAF-YOLO
pip install -r requirements.txt

# Run your first detection
python leafyolo_cli.py predict --source your_image.jpg
```

### ☁️ **Google Colab (Recommended for Beginners)**

**🎯 Quick Start (5 minutes)**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb)

**🏋️ Complete Training Tutorial**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)

### 🐍 **Python API**
```python
from leafyolo import LEAFYOLO

# Load a pre-trained model
model = LEAFYOLO('detect', variant='leafyolo_n')

# Detect objects in your images
results = model.predict('your_image.jpg')

# Train on your custom dataset
model.train(data='dataset.yaml', epochs=100, device='0')
```

**New to AI?** Start with our [**Google Colab tutorials**](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb) - no installation required! 🎓

---

## 🏆 Performance That Matters

### Real-World Results on VisDrone Dataset
*The numbers that actually matter for your projects:*

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **LEAF-YOLO-N** | 1.2M | **16ms** | 39.7% AP50 | 📱 Mobile/Edge |
| **LEAF-YOLO** | 4.3M | **22ms** | 48.3% AP50 | 🖥️ General Use |
| HIC-YOLOv5 | 9.3M | 30ms | 43% AP50 | 💻 Comparison |
| EdgeYOLO-S | 9.3M | 38ms | 40.8% AP50 | 💻 Comparison |

### Edge Device Performance 📱
*Running on Jetson AGX Xavier (real hardware):*

- **LEAF-YOLO-N**: 56 FPS (single image) 🚀
- **LEAF-YOLO**: 32 FPS (single image) ⚡
- Memory usage: < 500MB 💾

**Translation**: You can process drone footage in real-time on affordable hardware!

---

## 🛠️ How to Use LEAF-YOLO

### 🎯 Object Detection (Most Common)
```python
from leafyolo import LEAFYOLO

# Initialize detector
detector = LEAFYOLO('detect', variant='leafyolo_m')

# Detect in single image
results = detector('drone_image.jpg')

# Detect in video
results = detector('drone_video.mp4', save=True)

# Detect from webcam
results = detector(0)  # Camera index
```

### 🏋️ Training Your Own Model
```python
# Prepare your dataset (YOLO format)
# dataset.yaml should point to your images and labels

model = LEAFYOLO('detect', variant='leafyolo_s')
results = model.train(
    data='dataset.yaml',
    epochs=300,
    img_size=640,
    batch_size=16,
    device='0'  # GPU device
)

print(f"Best model saved at: {results['best']}")
```

### 📊 Evaluating Performance
```python
# Test your model's accuracy
metrics = model.val(data='test_dataset.yaml')

print(f"mAP@0.5: {metrics['map50']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

### 📤 Export for Deployment
```python
# Export to different formats
model.export(format='onnx')        # ONNX for cross-platform
model.export(format='tensorrt')    # TensorRT for NVIDIA GPUs
model.export(format='coreml')      # CoreML for iOS
model.export(format='tflite')      # TensorFlow Lite for mobile
```

---

## 🎮 Command Line Interface (For Quick Tasks)

```bash
# Train a model
python leafyolo_cli.py train --task detect --variant leafyolo_m --data coco --epochs 100

# Run detection on images
python leafyolo_cli.py predict --model best.pt --source images/ --save

# Evaluate model performance  
python leafyolo_cli.py val --model best.pt --data coco

# Export to ONNX
python leafyolo_cli.py export --model best.pt --format onnx

# Get system info
python leafyolo_cli.py info
```

## ☁️ Google Colab Integration

**Perfect for learning, experimentation, and quick prototyping!**

### 🚀 **Quick Start (5 minutes)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb)

- ✅ **No installation required** - runs in your browser
- ✅ **Free GPU access** - train models faster  
- ✅ **Pre-configured environment** - everything set up for you
- ✅ **Interactive tutorials** - learn by doing

### 🏋️ **Complete Training Tutorial**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)

**What you'll learn:**
- 📊 **Dataset preparation** and validation
- 🏋️ **Model training** from scratch
- 📈 **Performance monitoring** and optimization  
- 🎯 **Model evaluation** and testing
- 📤 **Export for deployment**

### 🛠️ **Colab Setup Script**
```python
# Quick setup for any Colab notebook
!wget https://raw.githubusercontent.com/Gaurav14cs17/LEAF-YOLO/main/colab_setup.py
!python colab_setup.py

# Now you can use LEAF-YOLO!
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect', variant='leafyolo_s')
```

---

## 📚 Model Variants (Choose Your Fighter)

| Variant | Parameters | Use Case | When to Choose |
|---------|------------|----------|----------------|
| `leafyolo_n` | 1.2M | 📱 Mobile apps, IoT devices | Ultra-lightweight needed |
| `leafyolo_s` | 2.5M | 🖥️ Edge computers, drones | Balanced speed/accuracy |
| `leafyolo_m` | 4.3M | 💻 General purpose | **Recommended start** |
| `leafyolo_l` | 8.1M | 🖥️ Server deployment | Higher accuracy needed |
| `leafyolo_x` | 12.9M | 🏢 Data center inference | Maximum accuracy |

**Not sure which to pick?** Start with `leafyolo_m` - it's the sweet spot for most projects!

---

## 🎓 Learning & Examples

### 📖 Documentation Structure
```
LEAF-YOLO/
├── 📄 README.md              # You are here! 
├── 📄 CONFIG_GUIDE.md        # Configuration deep-dive
├── 📂 leafyolo/              # Main code (see leafyolo/README.md)
├── 📂 tests/                 # Testing framework (see tests/README.md)
├── 📂 data/                  # Sample datasets (see data/README.md)
└── 📄 requirements.txt       # What you need to install
```

### 🎯 Real-World Examples

#### Drone Surveillance System
```python
# Monitor a construction site with drone footage
model = LEAFYOLO('detect', variant='leafyolo_s')

# Process live drone stream
for frame in drone_stream:
    detections = model(frame, conf_thres=0.5)
    
    # Alert if humans detected in restricted area
    for detection in detections:
        if detection.class_name == 'person':
            send_alert(f"Human detected at {detection.location}")
```

#### Wildlife Conservation
```python
# Count animals in aerial wildlife surveys
model = LEAFYOLO('detect', variant='leafyolo_m')

# Process survey images
for image_path in survey_images:
    results = model(image_path)
    
    # Count different species
    species_count = {}
    for detection in results:
        species = detection.class_name
        species_count[species] = species_count.get(species, 0) + 1
    
    print(f"Found: {species_count}")
```

---

## 🏗️ Project Architecture (The Inside Scoop)

LEAF-YOLO is built like a modern software project - clean, modular, and easy to understand:

```
📦 LEAF-YOLO Architecture
├── 🧠 leafyolo/models/        # The AI brains
├── ⚙️ leafyolo/engine/        # Training & inference engines  
├── 🧱 leafyolo/nn/            # Neural network building blocks
├── 🛠️ leafyolo/utils/         # Helper functions & tools
├── 📊 leafyolo/data/          # Data loading & processing
└── 🧪 tests/                  # Quality assurance (130+ tests!)
```

**Each folder has its own README** explaining what's inside and how to use it!

---

## 🎨 Advanced Features

### 🔧 Adaptive Configuration System
```python
# Automatically adapts to your task
model = LEAFYOLO('detect', variant='leafyolo_m', nc=20)  # 20 classes
model = LEAFYOLO('segment', variant='leafyolo_s')        # Segmentation  
model = LEAFYOLO('classify', variant='leafyolo_n')       # Classification

# Everything configures itself! ✨
```

### 📈 Built-in Training Optimizations
- **Mixed Precision Training**: Faster training with less memory
- **Model EMA**: Better final performance
- **Smart Learning Rate**: Cosine annealing with warmup
- **Data Augmentation**: Mosaic, MixUp, and more
- **Progress Tracking**: Beautiful training logs and metrics

### 🎯 Professional Deployment
```python
# Production-ready features
model.export(
    format='onnx',
    dynamic=True,      # Variable batch sizes
    simplify=True,     # Optimize for speed
    opset_version=11   # Compatibility
)
```

---

## 🤝 Getting Help & Contributing

### 💬 Need Help?
- 📖 **Start here**: Read the folder README files  
- 🐛 **Found a bug?**: [Open an issue](https://github.com/Gaurav14cs17/LEAF-YOLO/issues)
- 💡 **Have ideas?**: [Start a discussion](https://github.com/Gaurav14cs17/LEAF-YOLO/discussions)
- 📧 **Direct contact**: gourav14cs17.email@example.com

### 🎯 Quick Troubleshooting

**Model not loading?** 
```python
# Try this:
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect', variant='leafyolo_n', verbose=True)
```

**Training too slow?** 
```python
# Use smaller images and batch size:
model.train(data='dataset.yaml', img_size=416, batch_size=8)
```

**Out of memory?**
```python
# Reduce batch size:
model.train(data='dataset.yaml', batch_size=4)
```

### 🌟 Contributing
We love contributions! Whether you're:
- 🐛 Fixing bugs
- ✨ Adding features  
- 📖 Improving docs
- 🧪 Writing tests

Check out each folder's README for contribution guidelines!

---

## 📄 License & Citation

### 📜 License
This project is licensed under the Apache 2.0 License - use it freely in your projects!

### 🎓 Citation
If LEAF-YOLO helps your research or project, please cite our paper:

```bibtex
@article{NGHIEM2025200484,
title = {LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on Aerial Imagery},
journal = {Intelligent Systems with Applications},
volume = {25},
pages = {200484},
year = {2025},
author = {Van Quang Nghiem and Huy Hoang Nguyen and Minh Son Hoang},
}
```

---

## 🙏 Acknowledgments

LEAF-YOLO stands on the shoulders of giants. Special thanks to:

- **🏗️ YOLOv7 Team** - For the excellent foundation
- **🚀 YOLOv5 Community** - For training pipeline innovations  
- **🧠 PyTorch Team** - For the amazing deep learning framework
- **👥 Open Source Community** - For making AI accessible to everyone

---

<div align="center">

## 🚀 Ready to Get Started?

**[⬇️ Install LEAF-YOLO](#-quick-start-5-minutes-to-success) • [📖 Read the Docs](#-learning--examples) • [🎯 See Examples](#-real-world-examples)**

### ⭐ **Star this repo if LEAF-YOLO helps your project!** ⭐

*Built with ❤️ for the aerial AI community*

</div>
