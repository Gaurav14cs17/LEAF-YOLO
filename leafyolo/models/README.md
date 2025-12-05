# 🧠 LEAF-YOLO Models

<div align="center">

**🎭 The Brain Center of LEAF-YOLO**

*Where intelligence lives - your AI model collection*

[![Back to Package](https://img.shields.io/badge/←%20Back%20to-Package%20README-blue?style=for-the-badge)](../README.md)
[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's in Here?](#-whats-in-here)
- [🌟 The Star: LEAFYOLO Class](#-the-star-leafyolo-class)
- [🎭 Model Variants](#-model-variants-choose-your-fighter)
- [🎯 Different Tasks](#-different-tasks-same-interface)
- [🔧 Advanced Configuration](#-advanced-configuration)
- [📊 Understanding Performance](#-understanding-model-performance)
- [🔄 Model Lifecycle](#-model-lifecycle)
- [🐛 Troubleshooting](#-troubleshooting)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's in Here?

This directory contains the **smart models** that can see and understand images. It's like having a collection of different experts, each specialized for different tasks.

```
models/
├── 📄 leafyolo.py       # The main AI brain - your go-to model
├── 🎯 detect/           # Object detection specialists  
│   └── yolo.py          # The detection expert
└── 📄 __init__.py       # Makes everything work together
```

## 🌟 The Star: LEAFYOLO Class

### What Is It?
The `LEAFYOLO` class is like having a super-smart assistant that can:
- 👀 **See objects** in any image or video
- 🎓 **Learn new things** from your training data
- 🏃 **Work fast** on phones, computers, or servers
- 🔄 **Adapt automatically** to different tasks

It's the **one class to rule them all** - everything else in LEAF-YOLO revolves around this!

### Why Is It Special?
```python
from leafyolo import LEAFYOLO

# This one line creates a complete AI system:
model = LEAFYOLO('detect', variant='leafyolo_m')

# It automatically:
# ✅ Configures itself for object detection
# ✅ Chooses the right neural network architecture  
# ✅ Sets up optimal parameters
# ✅ Prepares for training or inference
# ✅ Handles all the complex stuff behind the scenes
```

## 🚀 Quick Start Examples

### 1. **Instant Object Detection**
```python
from leafyolo import LEAFYOLO

# Load a pre-trained model (downloads automatically if needed)
detector = LEAFYOLO('detect', variant='leafyolo_s')

# Detect objects in your image
results = detector('my_drone_photo.jpg')

# Look at what it found
for detection in results[0].boxes:
    print(f"Found {detection.cls} with confidence {detection.conf:.2f}")
```

### 2. **Train Your Own AI**
```python
from leafyolo import LEAFYOLO

# Create a fresh model for your specific task
model = LEAFYOLO('detect', variant='leafyolo_m', nc=10)  # 10 custom classes

# Train it on your data
results = model.train(
    data='my_custom_dataset.yaml',
    epochs=100,
    device='0'  # Use GPU
)

print(f"Training complete! Your model is saved at: {results['best']}")
```

### 3. **Evaluate Performance**
```python
from leafyolo import LEAFYOLO

# Load your trained model
model = LEAFYOLO('path/to/your/best_model.pt')

# Test how good it is
metrics = model.val(data='test_dataset.yaml')

print(f"Your model's accuracy: {metrics['map50']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

## 🎭 Model Variants (Choose Your Fighter)

LEAF-YOLO comes in different sizes, like T-shirt sizes but for AI models:

### 📱 **leafyolo_n** (Nano - The Speedster)
- **Size**: 1.2M parameters (tiny!)
- **Best for**: Mobile apps, IoT devices, real-time applications
- **Speed**: ⚡⚡⚡⚡⚡ (Super fast)
- **Accuracy**: ⭐⭐⭐ (Good)
- **When to use**: When speed matters more than perfect accuracy

```python
# Perfect for mobile deployment
mobile_model = LEAFYOLO('detect', variant='leafyolo_n')
mobile_model.export(format='tflite')  # For Android/iOS
```

### 🏃 **leafyolo_s** (Small - The Balanced One)  
- **Size**: 2.5M parameters  
- **Best for**: Edge computers, drones, embedded systems
- **Speed**: ⚡⚡⚡⚡ (Fast)
- **Accuracy**: ⭐⭐⭐⭐ (Very good)
- **When to use**: Sweet spot between speed and accuracy

```python
# Great for drones and edge devices
drone_model = LEAFYOLO('detect', variant='leafyolo_s')
```

### 💻 **leafyolo_m** (Medium - The Recommended Choice)
- **Size**: 4.3M parameters
- **Best for**: General purpose, desktop applications, servers
- **Speed**: ⚡⚡⚡ (Good)  
- **Accuracy**: ⭐⭐⭐⭐⭐ (Excellent)
- **When to use**: **Start here!** Best overall performance

```python
# The goldilocks choice - just right for most projects
general_model = LEAFYOLO('detect', variant='leafyolo_m')
```

### 🖥️ **leafyolo_l** (Large - The Accurate One)
- **Size**: 8.1M parameters
- **Best for**: Server deployment, accuracy-critical applications
- **Speed**: ⚡⚡ (Moderate)
- **Accuracy**: ⭐⭐⭐⭐⭐⭐ (Excellent+)
- **When to use**: When you need higher accuracy and have computing power

### 🏢 **leafyolo_x** (Extra Large - The Perfectionist)
- **Size**: 12.9M parameters (big!)
- **Best for**: Data centers, research, maximum accuracy needed
- **Speed**: ⚡ (Slower but thorough)
- **Accuracy**: ⭐⭐⭐⭐⭐⭐⭐ (Best possible)
- **When to use**: When accuracy is everything and speed is not critical

## 🎯 Different Tasks, Same Interface

### Object Detection (Most Common)
```python
# Find and locate objects in images
detector = LEAFYOLO('detect', variant='leafyolo_m')
results = detector('image.jpg')

# Results include: bounding boxes, confidence scores, class labels
```

### Image Segmentation (Pixel-Level Detail)
```python  
# Get exact pixel outlines of objects
segmenter = LEAFYOLO('segment', variant='leafyolo_s')
results = segmenter('image.jpg')

# Results include: masks, contours, precise object boundaries
```

### Classification (What's in the Image?)
```python
# Classify entire images into categories
classifier = LEAFYOLO('classify', variant='leafyolo_n') 
results = classifier('image.jpg')

# Results include: top predicted classes, confidence scores
```

## 🔧 Advanced Configuration

### Custom Classes
```python
# Train on your specific objects
model = LEAFYOLO('detect', variant='leafyolo_m', nc=5)  # 5 custom classes

# Or specify class names
model = LEAFYOLO('detect', variant='leafyolo_s', 
                 names=['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
```

### Pre-trained Models
```python
# Start from existing weights (transfer learning)
model = LEAFYOLO('path/to/pretrained/model.pt')

# Fine-tune for your specific task
model.train(data='your_data.yaml', epochs=50)
```

### Multi-Task Models
```python
# The same model architecture can handle different tasks
detection_model = LEAFYOLO('detect', variant='leafyolo_m')
segmentation_model = LEAFYOLO('segment', variant='leafyolo_m')
classification_model = LEAFYOLO('classify', variant='leafyolo_m')
```

## 📊 Understanding Model Performance

### Speed Benchmarks (on RTX 3090)
| Model | Parameters | Speed | Use Case |
|-------|------------|--------|----------|
| leafyolo_n | 1.2M | 16ms | 📱 Mobile/Real-time |
| leafyolo_s | 2.5M | 19ms | 🚁 Drones/Edge |
| leafyolo_m | 4.3M | 22ms | 💻 General Purpose |
| leafyolo_l | 8.1M | 28ms | 🖥️ Server/Accuracy |
| leafyolo_x | 12.9M | 35ms | 🏢 Maximum Quality |

### Accuracy Benchmarks (VisDrone Dataset)
| Model | mAP@0.5 | Small Objects | Medium Objects |
|-------|---------|---------------|----------------|
| leafyolo_n | 39.7% | 14.0% | 30.6% |
| leafyolo_s | 42.1% | 16.2% | 33.8% |
| leafyolo_m | 48.3% | 20.0% | 38.0% |
| leafyolo_l | 51.2% | 23.1% | 41.5% |
| leafyolo_x | 53.8% | 25.7% | 44.2% |

## 🔄 Model Lifecycle

### 1. **Creation**
```python
# Create a fresh model
model = LEAFYOLO('detect', variant='leafyolo_m')
```

### 2. **Training**  
```python
# Teach it to recognize your objects
model.train(data='dataset.yaml', epochs=100)
```

### 3. **Validation**
```python
# Test how well it learned
metrics = model.val(data='test_data.yaml')
```

### 4. **Deployment**
```python
# Export for production use
model.export(format='onnx')
```

### 5. **Inference**
```python
# Use it in real applications
results = model('new_images/')
```

## 🐛 Troubleshooting

### **Model Won't Load?**
```python
# Try with verbose output to see what's happening
model = LEAFYOLO('detect', verbose=True)

# Check if file exists
import os
print(os.path.exists('your_model.pt'))

# Try absolute path
model = LEAFYOLO('/full/path/to/model.pt')
```

### **Out of Memory During Training?**
```python
# Use smaller variant
model = LEAFYOLO('detect', variant='leafyolo_n')

# Or reduce batch size
model.train(data='dataset.yaml', batch_size=8)
```

### **Predictions Too Slow?**
```python
# Use faster variant
model = LEAFYOLO('detect', variant='leafyolo_n')

# Or smaller image size
results = model('image.jpg', img_size=416)

# Or use GPU
model = LEAFYOLO('detect', device='0')
```

## 💡 Pro Tips

1. **Start with leafyolo_m**: It's the best balance for most projects
2. **Use Transfer Learning**: Start from pre-trained weights, then fine-tune
3. **Monitor GPU Memory**: Larger models need more memory
4. **Save Checkpoints**: Training can take hours - save progress regularly
5. **Validate Frequently**: Check performance every few epochs
6. **Export for Deployment**: Convert to ONNX/TensorRT for production

## 🎓 Next Steps

1. **Experiment**: Try different variants and see which works best for your use case
2. **Read the Code**: Check out `leafyolo.py` to understand how it works
3. **Customize**: Modify architectures in `detect/yolo.py` for advanced use cases
4. **Train**: Create your own models with custom datasets
5. **Deploy**: Export your trained models for real-world applications

---

**Remember**: The model is just the beginning - the real magic happens when you train it on your own data and deploy it to solve real problems! 🚀

**Need help?** Check out the other README files in the `engine/`, `nn/`, and `utils/` directories for more detailed information about specific components.
