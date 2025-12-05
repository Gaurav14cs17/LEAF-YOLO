# 📊 Data Pipeline

<div align="center">

**🍽️ The Dining Room of LEAF-YOLO**

*Where we prepare perfect meals for your hungry AI*

[![Back to Package](https://img.shields.io/badge/←%20Back%20to-Package%20README-blue?style=for-the-badge)](../README.md)
[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's This All About?](#-whats-this-all-about)
- [📊 datasets.py - The Master Chef](#-datasetspy---the-master-chef)
- [🎨 Data Augmentation - Adding Flavor](#-data-augmentation---adding-flavor)
- [🔧 Data Loading Functions](#-data-loading-functions)
- [📋 Data Formats & Structure](#-data-formats--structure)
- [🚀 Practical Examples](#-practical-examples)
- [🔧 Performance Optimization](#-performance-optimization)
- [🐛 Common Issues & Solutions](#-common-issues--solutions)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's This All About?

The data pipeline is like having a **smart chef** that knows exactly how to:
- 🥘 **Prepare ingredients**: Load and preprocess images
- 🍽️ **Serve portions**: Create perfect batches for training
- 🧂 **Add seasoning**: Apply data augmentation for variety
- 📋 **Quality control**: Validate data formats and labels

```
data/
├── 📊 datasets.py       # The master chef - handles all data operations
└── 📄 __init__.py       # Makes everything accessible
```

## 📊 **datasets.py** - The Master Chef

This file is the **heart of the data pipeline** - it contains all the classes and functions that load, process, and serve data to your AI models.

### 🍽️ **Main Data Classes**

#### **LoadImagesAndLabels** - The Training Chef
**What it does**: Loads training data with images and their corresponding labels
**Think of it as**: A chef who prepares complete meals (image + label pairs)
**When to use**: During training when you have both images and ground truth labels

```python
from leafyolo.data.datasets import LoadImagesAndLabels

# Create a dataset for training
dataset = LoadImagesAndLabels(
    path='path/to/train/images',
    img_size=640,           # Resize images to this size
    batch_size=16,          # How many images per batch
    augment=True,           # Apply data augmentation
    hyp=hyperparameters,    # Augmentation settings
    rect=False,             # Use rectangular training
    cache_images=False      # Cache images in memory (faster but uses more RAM)
)

print(f"Dataset contains {len(dataset)} images")
```

#### **LoadImages** - The Inference Server
**What it does**: Loads individual images or folders of images for prediction
**Think of it as**: A waiter serving individual dishes (just images, no labels needed)
**When to use**: During inference when you just want to detect objects

```python
from leafyolo.data.datasets import LoadImages

# Load images for prediction
dataset = LoadImages('path/to/images/', img_size=640)

for path, img, im0s, vid_cap in dataset:
    # path: image file path
    # img: processed image tensor (ready for model)
    # im0s: original image (for visualization)
    # vid_cap: video capture object (if processing video)
    print(f"Processing: {path}")
```

#### **LoadStreams** - The Live Feed Handler
**What it does**: Handles live video streams from webcams or IP cameras
**Think of it as**: A live cooking show chef working in real-time
**When to use**: For real-time detection from cameras or video streams

```python
from leafyolo.data.datasets import LoadStreams

# Handle multiple camera streams
sources = ['0', '1', 'rtsp://camera1.ip', 'http://camera2.ip']
dataset = LoadStreams(sources, img_size=640)

for path, img, im0s, vid_cap in dataset:
    # Real-time processing of multiple streams
    print(f"Live frame from: {path}")
```

### 🎨 **Data Augmentation - Adding Flavor**

Data augmentation is like **adding spices to cooking** - it creates variety to help your AI learn better.

#### **Why Augmentation Matters**
- 🔄 **More data**: Creates variations from limited images
- 🛡️ **Robustness**: Teaches AI to handle different conditions
- 🎯 **Better accuracy**: Prevents overfitting to specific conditions

#### **LEAF-YOLO Augmentation Techniques**

##### **Mosaic Augmentation** 
```python
# Combines 4 images into one training sample
# [Image1] [Image2]
# [Image3] [Image4]
# Creates rich multi-object scenes for training
```

##### **MixUp**
```python
# Blends two images together
# image_mixed = alpha * image1 + (1-alpha) * image2
# Helps with gradient smoothness and generalization
```

##### **Copy-Paste** 
```python
# Copies objects from one image and pastes into another
# Increases object instances and scene variety
```

##### **Geometric Transformations**
```python
# Random transformations applied to images:
# - Rotation: ±10 degrees
# - Scaling: 0.5x to 1.5x
# - Translation: ±10% of image size
# - Flipping: Horizontal flip 50% chance
# - Perspective: Small perspective changes
```

##### **Color Augmentation**
```python
# Color space modifications:
# - HSV adjustments: Hue, Saturation, Value
# - Brightness: ±20%
# - Contrast: ±20%
# - Gamma correction: Simulate different lighting
```

### 🔧 **Data Loading Functions**

#### **create_dataloader** - The Kitchen Manager
```python
from leafyolo.data.datasets import create_dataloader

# Create a complete data pipeline for training
dataloader, dataset = create_dataloader(
    path='data/train',      # Path to training data
    img_size=640,           # Image size
    batch_size=16,          # Batch size
    stride=32,              # Model stride (usually 32 for YOLO)
    single_cls=False,       # Single class detection?
    hyp=hyperparameters,    # Augmentation hyperparameters
    augment=True,           # Enable augmentation
    cache=False,            # Cache images in memory
    pad=0.0,                # Padding
    rect=False,             # Rectangular training
    rank=-1,                # For distributed training
    workers=8,              # Number of data loading workers
    image_weights=False,    # Weight sampling by image content
    quad=False,             # Quad-image mosaic
    prefix=colorstr('train: ')  # Logging prefix
)

# Use in training loop
for batch_i, (imgs, targets, paths, _) in enumerate(dataloader):
    # imgs: batch of preprocessed images
    # targets: batch of labels/annotations  
    # paths: list of image file paths
    # Train your model here!
```

### 📋 **Data Formats & Structure**

#### **Expected Directory Structure**
```
your_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
└── dataset.yaml
```

#### **Label Format (YOLO format)**
Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```
- **class_id**: 0, 1, 2, ... (integer class index)
- **x_center, y_center**: Center coordinates (0-1, relative to image size)
- **width, height**: Box dimensions (0-1, relative to image size)

#### **Dataset Configuration (dataset.yaml)**
```yaml
# Dataset configuration
path: /path/to/dataset          # Dataset root directory
train: images/train             # Training images (relative to path)
val: images/val                 # Validation images (relative to path)

# Classes
nc: 80                          # Number of classes
names: ['person', 'bicycle', 'car', ...]  # Class names
```

### 🚀 **Practical Examples**

#### **Simple Image Loading**
```python
from leafyolo.data.datasets import LoadImages

# Load a single image
dataset = LoadImages('photo.jpg', img_size=640)
path, img, im0, vid_cap = next(iter(dataset))

print(f"Original shape: {im0.shape}")    # (H, W, C) - for display
print(f"Processed shape: {img.shape}")   # (C, H, W) - for model
```

#### **Batch Processing**
```python
from leafyolo.data.datasets import create_dataloader

# Create dataloader for batch processing  
dataloader, _ = create_dataloader(
    'images_folder/',
    img_size=640,
    batch_size=8,
    stride=32,
    augment=False  # No augmentation for inference
)

# Process in batches
for imgs, targets, paths, shapes in dataloader:
    # imgs.shape: (batch_size, 3, 640, 640)
    # Process batch with your model
    predictions = model(imgs)
```

#### **Real-time Camera Processing**
```python
from leafyolo.data.datasets import LoadStreams

# Set up camera stream
dataset = LoadStreams('0', img_size=640)  # Webcam

# Real-time loop
for path, img, im0s, vid_cap in dataset:
    # img: preprocessed for model (1, 3, 640, 640)
    # im0s: original frames for display
    
    # Run detection
    results = model(img)
    
    # Display results
    cv2.imshow('LEAF-YOLO Live', im0s[0])
    if cv2.waitKey(1) == ord('q'):
        break
```

#### **Custom Augmentation**
```python
# Modify augmentation parameters
hyp = {
    'degrees': 10.0,        # Rotation degrees
    'translate': 0.1,       # Translation fraction
    'scale': 0.5,           # Scale factor
    'shear': 0.0,           # Shear degrees
    'perspective': 0.0,     # Perspective change
    'flipud': 0.0,          # Vertical flip probability
    'fliplr': 0.5,          # Horizontal flip probability
    'mosaic': 1.0,          # Mosaic probability
    'mixup': 0.15,          # MixUp probability
    'copy_paste': 0.3       # Copy-paste probability
}

# Use custom augmentation
dataloader, _ = create_dataloader(
    'train_data/',
    hyp=hyp,
    augment=True
)
```

### 🔧 **Performance Optimization**

#### **Memory vs Speed Trade-offs**
```python
# Fast loading (more memory usage)
dataset = LoadImagesAndLabels(
    path='data/',
    cache_images=True,      # Cache in RAM (faster access)
    workers=8               # More parallel workers
)

# Memory efficient (slower loading)  
dataset = LoadImagesAndLabels(
    path='data/',
    cache_images=False,     # Load from disk each time
    workers=2               # Fewer workers to save memory
)
```

#### **Distributed Training Setup**
```python
# For multi-GPU training
dataloader = create_dataloader(
    path='train/',
    batch_size=16,
    rank=local_rank,        # GPU rank in distributed setup
    world_size=world_size   # Total number of GPUs
)
```

### 🐛 **Common Issues & Solutions**

#### **Dataset Not Found**
```python
# Check your paths
import os
print(os.path.exists('path/to/images'))
print(os.listdir('path/to/images'))

# Common issues:
# ❌ Wrong file paths
# ❌ Missing images/ or labels/ folders
# ❌ Incorrect dataset.yaml configuration
```

#### **Label Format Errors**
```python
# Validate your label format
# Each line should be: class_id x_center y_center width height
# All values should be between 0 and 1 (except class_id)

# Example validation:
with open('labels/image1.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        
        # Check ranges
        assert 0 <= x <= 1, f"x_center out of range: {x}"
        assert 0 <= y <= 1, f"y_center out of range: {y}"
        assert 0 <= w <= 1, f"width out of range: {w}"
        assert 0 <= h <= 1, f"height out of range: {h}"
```

#### **Out of Memory During Loading**
```python
# Reduce memory usage
dataloader = create_dataloader(
    path='data/',
    batch_size=8,           # Smaller batches
    cache_images=False,     # Don't cache in RAM
    workers=2               # Fewer workers
)
```

#### **Slow Data Loading**
```python
# Speed up data loading
dataloader = create_dataloader(
    path='data/',
    workers=8,              # More parallel workers
    cache_images=True,      # Cache frequently used images
    pin_memory=True         # Faster GPU transfer
)
```

### 💡 **Pro Tips**

1. **Start Small**: Test with a few images before processing entire datasets
2. **Validate Data**: Always check label format and image-label correspondence
3. **Monitor Loading**: Watch for bottlenecks in data loading vs. training
4. **Use Augmentation**: But not too much - find the right balance
5. **Cache Wisely**: Cache small datasets, stream large ones
6. **Workers Matter**: More workers = faster loading (up to a point)

### 🎓 **Understanding Data Flow**

```mermaid
Raw Images + Labels
        ↓
   LoadImagesAndLabels
        ↓
    Preprocessing
    (resize, normalize)
        ↓
    Augmentation
    (mosaic, mixup, etc.)
        ↓
     Batching
        ↓
    DataLoader
        ↓
   Training Loop
```

### 🌟 **Advanced Features**

#### **Rectangular Training**
```python
# More efficient training for images with different aspect ratios
dataloader = create_dataloader(
    path='data/',
    rect=True,              # Enable rectangular training
    batch_size=16
)
# Groups images by aspect ratio for more efficient batching
```

#### **Image Weighting**
```python
# Sample images based on detection difficulty
dataloader = create_dataloader(
    path='data/',
    image_weights=True      # Weight sampling by image content
)
# Focuses training on harder examples
```

#### **Multi-Scale Training**
```python
# Train with different image sizes for robustness
sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]
for epoch in range(epochs):
    if epoch % 10 == 0:  # Change size every 10 epochs
        img_size = random.choice(sizes)
        # Recreate dataloader with new size
```

---

**Remember**: Good data is the foundation of great AI! Spend time understanding your data, cleaning it, and setting up efficient pipelines. The data pipeline might not be glamorous, but it's absolutely crucial for success. 📊🚀

**The key insight**: Your model is only as good as your data. Master the data pipeline, and you'll be able to train better models faster!
