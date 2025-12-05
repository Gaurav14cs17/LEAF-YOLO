# 🧱 Neural Network Modules

<div align="center">

**🎲 The LEGO Box of LEAF-YOLO**

*Smart building blocks for amazing AI models*

[![Back to NN](https://img.shields.io/badge/←%20Back%20to-Neural%20Networks-blue?style=for-the-badge)](../README.md)
[![Back to Package](https://img.shields.io/badge/←%20Back%20to-Package-orange?style=for-the-badge)](../../README.md)
[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main-green?style=for-the-badge)](../../../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What Are These Modules?](#-what-are-these-modules)
- [🔧 common.py - The Essential Toolkit](#-commonpy---the-essential-toolkit)
- [🧠 heads.py - The Smart Detectors](#-headspy---the-smart-detectors)
- [⚡ activations.py - The Spark Plugs](#-activationspy---the-spark-plugs)
- [🎯 attention.py - The Focus Enhancer](#-attentionpy---the-focus-enhancer)
- [📍 cooratt.py - Position-Aware Helper](#-coorattpy---the-position-aware-helper)
- [🌟 SE.py - The Feature Enhancer](#-sepy---the-feature-enhancer)
- [🧪 experimental.py - Innovation Lab](#-experimentalpy---the-innovation-lab)
- [🎨 Building Your Own Models](#-building-your-own-models)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What Are These Modules?

Think of these as **specialized tools in a toolbox** - each one does a specific job really well, and when you combine them cleverly, you get powerful AI models that can see and understand images.

```
modules/
├── 🔧 common.py           # Essential building blocks (Conv, C3, SPP, etc.)
├── 🧠 heads.py            # Smart detection heads (finds objects)
├── ⚡ activations.py      # Activation functions (brings neurons to life)
├── 🎯 attention.py        # Attention mechanisms (focus on what matters)
├── 📍 cooratt.py         # Coordinate attention (spatial awareness)
├── 🌟 SE.py              # Squeeze-Excitation (feature enhancement)
├── 🧪 experimental.py     # Cutting-edge experimental components
└── 📄 __init__.py         # Makes everything work together
```

## 🔧 **common.py** - The Essential Toolkit

This file is like your **basic toolbox** - it contains the fundamental pieces you'll use in almost every AI model.

### 🏗️ **Core Building Blocks**

#### **Conv** - The Foundation
**What it does**: The basic building block of all computer vision AI
**Think of it as**: A smart filter that learns to detect features (edges, textures, shapes)
**When to use**: Everywhere! It's the bread and butter of AI vision

```python
from leafyolo.nn.modules.common import Conv

# Create a basic convolution layer
conv = Conv(
    c1=3,    # Input channels (RGB = 3)
    c2=64,   # Output channels (64 different features to detect)
    k=3,     # Kernel size (3x3 filter)
    s=1,     # Stride (how much to move the filter each step)
    p=1      # Padding (border handling)
)

# This learns to detect 64 different patterns in your image!
```

#### **C3** - The Efficiency Expert  
**What it does**: Processes features efficiently using Cross Stage Partial connections
**Think of it as**: A smart note-taker that keeps important info while discarding redundancy
**Why special**: This is LEAF-YOLO's secret sauce for being lightweight yet accurate

```python
from leafyolo.nn.modules.common import C3

# Create an efficient processing block
c3 = C3(
    c1=128,  # Input channels
    c2=256,  # Output channels  
    n=3      # Number of bottleneck layers inside
)

# This efficiently processes 128 features into 256 richer features
```

#### **SPP** - The Multi-Scale Master
**What it does**: Spatial Pyramid Pooling - looks at features at different scales
**Think of it as**: Having multiple magnifying glasses to see both big and small details
**When to use**: Before detection heads to capture objects of different sizes

```python
from leafyolo.nn.modules.common import SPP

# Create multi-scale feature extraction
spp = SPP(
    c1=512,           # Input channels
    c2=512,           # Output channels
    k=(5, 9, 13)      # Different pooling kernel sizes
)

# This sees features at 3 different scales simultaneously
```

### 🎛️ **Advanced Components**

#### **Focus** - The Downsampler
**What it does**: Reduces image size while preserving all information
**Think of it as**: Compressing a high-res photo without losing important details
**When to use**: At the very beginning to make processing faster

```python
from leafyolo.nn.modules.common import Focus

# Efficiently downsample input images
focus = Focus(
    c1=3,    # RGB input
    c2=32,   # Output features
    k=3      # Kernel size
)

# Turns 640x640x3 image into 320x320x32 feature map
```

## 🧠 **heads.py** - The Smart Detectors

This is where your AI model **makes decisions** about what it sees in images.

### 🎯 **Detect** - Object Detection Head
**What it does**: Finds objects and draws bounding boxes around them
**Think of it as**: A detective that spots objects and marks their locations
**Output**: "I see a car at coordinates (100,150) with 95% confidence"

```python
from leafyolo.nn.modules.heads import Detect

# Create an object detection head
detect_head = Detect(
    nc=80,                    # Number of classes (COCO has 80)
    anchors=[[...], [...], [...]],  # Anchor boxes for different scales
    ch=[256, 512, 1024]      # Input channel sizes from backbone
)

# This turns feature maps into object detections
```

### 🖼️ **Segment** - Segmentation Head  
**What it does**: Creates pixel-perfect masks around objects
**Think of it as**: An artist that traces exact outlines of everything in the image
**Output**: "Here's the exact shape of every object, pixel by pixel"

### 🏷️ **Classify** - Classification Head
**What it does**: Decides what's in the entire image  
**Think of it as**: A critic that gives the whole image a single label
**Output**: "This entire image contains a dog"

## ⚡ **activations.py** - The Spark Plugs

Activation functions are like **spark plugs in an engine** - they bring your neural network to life!

### **SiLU** (Swish) - The Smooth Operator
**What it does**: Smooth, efficient activation that helps gradients flow well
**Think of it as**: A smooth gear transition in a car
**Why use it**: Better training stability and performance

### **Hardswish** - The Mobile-Friendly Version
**What it does**: Approximation of Swish that runs faster on mobile devices
**Think of it as**: A lightweight version optimized for phones
**Why use it**: When deploying to mobile or edge devices

## 🎯 **attention.py** - The Focus Enhancer

Attention mechanisms help your AI **focus on what's important** instead of looking at everything equally.

### **CBAM** - Convolutional Block Attention Module
**What it does**: Learns to focus on important spatial locations AND important feature channels
**Think of it as**: Having both a spotlight (spatial) and a volume control (channels)
**Effect**: Makes your model much smarter at focusing on relevant parts

```python
from leafyolo.nn.modules.attention import CBAM

# Add smart attention to any layer
attention = CBAM(
    channels=256,      # Number of input channels
    reduction=16       # How much to compress for efficiency
)

# Now your model pays attention to what matters most!
```

## 📍 **cooratt.py** - The Position-Aware Helper

Coordinate Attention helps your AI understand **where things are located** in the image.

### **CoordAtt** - Coordinate Attention
**What it does**: Encodes positional information along with features
**Think of it as**: Having GPS coordinates embedded in every feature
**Why important**: Helps with precise object localization

```python
from leafyolo.nn.modules.cooratt import CoordAtt

# Add position awareness
coord_att = CoordAtt(
    inp=256,     # Input channels
    oup=256      # Output channels  
)

# Now your model knows WHERE features are, not just WHAT they are
```

## 🌟 **SE.py** - The Feature Enhancer

Squeeze-and-Excitation modules help your AI **boost important features** while suppressing noise.

### **SEAttention** - Feature Channel Attention
**What it does**: Learns which feature channels are most important
**Think of it as**: Having a volume mixer that boosts useful features and quiets noise
**Effect**: Significant accuracy improvement with minimal computational cost

```python
from leafyolo.nn.modules.SE import SEAttention

# Enhance feature channels intelligently
se = SEAttention(
    channels=512,      # Input/output channels
    reduction=16       # Compression ratio for efficiency
)

# Automatically highlights the most useful features
```

## 🧪 **experimental.py** - The Innovation Lab

This file contains **cutting-edge experimental components** - new ideas being tested!

### **CrossConv** - Cross Convolution
**What it does**: Efficient convolution with factorized kernels
**Status**: Experimental but promising for efficiency

### **MixConv2d** - Mixed Convolutions
**What it does**: Uses multiple kernel sizes in a single layer
**Status**: Interesting for multi-scale feature extraction

## 🎨 Building Your Own Models

### **Simple Example**: Custom Block
```python
import torch.nn as nn
from leafyolo.nn.modules.common import Conv, C3
from leafyolo.nn.modules.attention import CBAM

class MyCustomBlock(nn.Module):
    """A custom block combining basic components"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv(channels, channels, k=3)
        self.c3 = C3(channels, channels, n=2)
        self.attention = CBAM(channels)
    
    def forward(self, x):
        x = self.conv(x)      # Basic feature extraction
        x = self.c3(x)        # Efficient processing
        x = self.attention(x)  # Smart focusing
        return x

# Use your custom block
custom_block = MyCustomBlock(256)
```

### **Advanced Example**: Custom Detection Head
```python
from leafyolo.nn.modules.heads import Detect
from leafyolo.nn.modules.common import Conv

class MyCustomDetectionHead(nn.Module):
    """A custom detection head with extra processing"""
    
    def __init__(self, nc, ch):
        super().__init__()
        self.extra_conv = Conv(ch[0], ch[0], k=3)  # Extra processing
        self.detect = Detect(nc=nc, ch=ch)         # Standard detection
    
    def forward(self, x):
        x[0] = self.extra_conv(x[0])  # Process first feature map
        return self.detect(x)         # Standard detection
```

## 📊 Performance Considerations

### **Speed Impact** (⚡ = faster)
- **Conv**: ⚡⚡⚡⚡ (Very fast, use everywhere)
- **C3**: ⚡⚡⚡ (Fast, efficient)
- **Attention (CBAM/SE)**: ⚡⚡ (Moderate, use selectively)
- **SPP**: ⚡⚡⚡ (Fast, use before heads)

### **Accuracy Impact** (⭐ = better accuracy)
- **More C3 blocks**: ⭐⭐⭐⭐ (Higher accuracy)
- **Attention mechanisms**: ⭐⭐⭐⭐⭐ (Significant improvement)
- **Larger channels**: ⭐⭐⭐ (Better but diminishing returns)

### **Memory Usage** (📦 = more memory)
- **Bigger channels**: 📦📦📦 (Linear increase)
- **More layers**: 📦📦 (Moderate increase)
- **Attention**: 📦 (Small increase, big benefit)

## 🔧 Practical Tips

### **For Mobile/Edge Deployment**
```python
# Use lightweight components
conv = Conv(c1, c2, k=3, s=1)  # Basic convolution
c3 = C3(c1, c2, n=1)           # Single bottleneck
# Skip heavy attention layers
```

### **For Maximum Accuracy**
```python
# Use more powerful components  
c3 = C3(c1, c2, n=3)           # Multiple bottlenecks
attention = CBAM(channels)      # Add attention
se = SEAttention(channels)      # Feature enhancement
```

### **For Balanced Performance**
```python
# LEAF-YOLO's approach
c3 = C3(c1, c2, n=3)           # Efficient processing
spp = SPP(c1, c2, k=(5,9,13))  # Multi-scale pooling  
# Selective attention at key points
```

## 🐛 Common Issues & Solutions

### **Shape Mismatches**
```python
# Always check tensor shapes
print(f"Input shape: {x.shape}")
x = your_module(x)
print(f"Output shape: {x.shape}")

# Common issue: channel mismatch
# Solution: Make sure c1 matches input, c2 matches expected output
```

### **Out of Memory**
```python
# Reduce channel sizes
Conv(c1=64, c2=128)  # Instead of c2=256

# Use fewer blocks
C3(c1, c2, n=1)  # Instead of n=3

# Skip some attention layers
```

### **Model Too Slow**
```python
# Profile your model to find bottlenecks
import torch
x = torch.randn(1, 3, 640, 640)

# Time each component
import time
start = time.time()
output = your_module(x)
print(f"Time: {time.time() - start:.4f}s")
```

## 💡 Pro Tips

1. **Start Simple**: Begin with Conv + C3, add complexity gradually
2. **Measure Everything**: Profile speed, memory, and accuracy
3. **Attention is Powerful**: Small computational cost, big accuracy gain
4. **Channel Progression**: Usually 32→64→128→256→512
5. **Batch Normalization**: Always included in Conv (handles normalization)
6. **Activation Functions**: SiLU is the default, HardSwish for mobile

---

**Remember**: These modules are like LEGO blocks - the fun is in combining them creatively to build something amazing! Start with the basics (Conv, C3) and gradually add more sophisticated components (attention, SPP) as you learn what works for your specific use case. 🚀🧱
