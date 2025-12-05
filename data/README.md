# 📊 Dataset Configurations

<div align="center">

**🎛️ Data Command Center**

*Configuration files that tell your AI where to find data*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Config Guide](https://img.shields.io/badge/📖%20Read-Config%20Guide-blue?style=for-the-badge)](../CONFIG_GUIDE.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's in Here?](#-whats-in-here)
- [📄 Configuration File Format](#-configuration-file-format)
- [🌟 Pre-configured Datasets](#-pre-configured-datasets)
- [🎯 Creating Custom Configurations](#-how-to-create-your-own-dataset-configuration)
- [🚀 Using Your Datasets](#-using-your-dataset-configuration)
- [🎨 Popular Dataset Examples](#-popular-dataset-examples)
- [🔧 Advanced Configuration Options](#-advanced-configuration-options)
- [🐛 Common Issues & Solutions](#-common-issues--solutions)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's in Here?

This directory contains **dataset configuration files** - think of them as instruction manuals that tell LEAF-YOLO:
- 📁 **Where your images are located** 
- 🏷️ **What objects to look for**
- 📊 **How your data is organized**
- 🎯 **Which classes to detect**

```
data/
├── 📄 coco.yaml        # COCO dataset configuration (80 common objects)
├── 📄 visdrone.yaml    # VisDrone dataset configuration (aerial imagery)
└── 📄 your_data.yaml   # Your custom dataset configuration
```

## 📄 **Configuration File Format**

Each `.yaml` file is a simple text file that describes your dataset in a format LEAF-YOLO understands.

### 🏗️ **Basic Structure**
```yaml
# Dataset Information
path: /path/to/your/dataset    # Root directory of your dataset
train: images/train            # Training images folder (relative to path)
val: images/val               # Validation images folder (relative to path)
test: images/test             # Test images folder (optional)

# Class Information  
nc: 80                        # Number of classes
names: ['person', 'bicycle', 'car', ...]  # List of class names
```

### 🌟 **Real Example: COCO Dataset**
```yaml
# COCO 2017 Dataset Configuration
# Download from: https://cocodataset.org/

path: ../datasets/coco        # Dataset root
train: train2017.txt          # Training image list  
val: val2017.txt             # Validation image list
test: test-dev2017.txt       # Test image list

# Classes (80 common objects)
nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        # ... and 54 more classes
       ]
```

### 🚁 **Real Example: VisDrone Dataset**
```yaml
# VisDrone Dataset - Aerial Imagery Object Detection  
# Perfect for drone and UAV applications
# Download from: http://aiskyeye.com/

path: ../datasets/VisDrone    # Dataset root
train: VisDrone2019-DET-train # Training images
val: VisDrone2019-DET-val     # Validation images  
test: VisDrone2019-DET-test   # Test images

# Classes (10 aerial objects)
nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 
        'tricycle', 'awning-tricycle', 'bus', 'motor']
```

## 🎯 **How to Create Your Own Dataset Configuration**

### Step 1: **Organize Your Data**
```
your_dataset/
├── images/
│   ├── train/           # Training images
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/             # Validation images  
│   │   ├── img501.jpg
│   │   ├── img502.jpg
│   │   └── ...
│   └── test/            # Test images (optional)
│       ├── img801.jpg
│       └── ...
└── labels/
    ├── train/           # Training labels (YOLO format)
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    ├── val/             # Validation labels
    │   ├── img501.txt
    │   ├── img502.txt
    │   └── ...
    └── test/            # Test labels (optional)
        ├── img801.txt
        └── ...
```

### Step 2: **Create Configuration File**
Create a new file called `my_dataset.yaml`:

```yaml
# My Custom Dataset Configuration

# Path to dataset root directory
path: /path/to/your_dataset

# Relative paths to image folders
train: images/train
val: images/val
test: images/test

# Number of classes you want to detect
nc: 5

# Names of your classes (in order, starting from 0)
names: ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Optional: Additional information
download: |
  # Instructions for downloading your dataset
  echo "Download from: https://your-dataset-url.com"
  echo "Extract to: /path/to/your_dataset"
```

### Step 3: **Verify Label Format**
Each label file (`.txt`) should contain one line per object:
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
```

**Important**: All coordinates should be **normalized** (between 0 and 1):
- `x_center`, `y_center`: Object center relative to image size
- `width`, `height`: Object size relative to image size

## 🚀 **Using Your Dataset Configuration**

### **For Training**
```python
from leafyolo import LEAFYOLO

# Create model for your dataset
model = LEAFYOLO('detect', variant='leafyolo_m', nc=5)  # 5 classes

# Train using your configuration
results = model.train(
    data='data/my_dataset.yaml',  # Path to your config file
    epochs=100,
    batch_size=16,
    device='0'
)

print(f"Training complete! Best model: {results['best']}")
```

### **For Validation**
```python
# Test your trained model
model = LEAFYOLO('path/to/your/trained/model.pt')

# Validate using your configuration
metrics = model.val(data='data/my_dataset.yaml')

print(f"Accuracy: {metrics['map50']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

### **For Prediction**
```python  
# Use your trained model for detection
model = LEAFYOLO('path/to/your/trained/model.pt')

# Predict on new images
results = model.predict('new_images/')

# Results will use your class names automatically!
for r in results:
    for detection in r.boxes:
        class_name = model.names[int(detection.cls)]
        confidence = detection.conf
        print(f"Found {class_name} with {confidence:.2f} confidence")
```

## 🎨 **Popular Dataset Examples**

### 🏙️ **COCO (Common Objects in Context)**
- **What**: 80 everyday objects (people, animals, vehicles, furniture)
- **Size**: 118K training images, 5K validation images
- **Best for**: General object detection, indoor/outdoor scenes
- **Download**: https://cocodataset.org/

```yaml
# Use pre-configured COCO
model.train(data='coco')  # Automatically uses coco.yaml
```

### 🚁 **VisDrone (Drone/Aerial Imagery)**  
- **What**: 10 objects commonly seen from drones (cars, people, trucks)
- **Size**: 6.5K training images, 548 validation images
- **Best for**: Aerial photography, surveillance, traffic monitoring
- **Download**: http://aiskyeye.com/

```yaml
# Use pre-configured VisDrone
model.train(data='visdrone')  # Automatically uses visdrone.yaml
```

### 🚗 **Custom Vehicle Dataset Example**
```yaml
# vehicles.yaml - Custom vehicle detection
path: ../datasets/vehicles
train: images/train  
val: images/val

nc: 6
names: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van']

# Optional metadata
description: "Custom vehicle detection dataset"
url: "https://my-dataset.com"
version: "1.0"
```

### 🏠 **Custom Indoor Objects Example**
```yaml
# indoor_objects.yaml - Home object detection  
path: ../datasets/indoor_objects
train: train.txt  # Can also use .txt files with image lists
val: val.txt

nc: 8
names: ['chair', 'table', 'sofa', 'bed', 'tv', 'laptop', 'phone', 'book']

# Training configuration hints
img_size: 640
batch_size: 16
epochs: 200
```

## 🔧 **Advanced Configuration Options**

### 📋 **Using Image Lists**
Instead of folder paths, you can use `.txt` files that list image paths:

```yaml
# Using image list files
path: /dataset/root
train: train_images.txt  # File containing training image paths
val: val_images.txt      # File containing validation image paths
```

**train_images.txt** content:
```
images/train/img001.jpg
images/train/img002.jpg  
images/train/img003.jpg
# ... more image paths
```

### 🎯 **Class Mapping**
```yaml
# Advanced class configuration
nc: 3
names: ['vehicle', 'person', 'animal']

# Optional: Map multiple classes to single detection class
class_mapping:
  vehicle: ['car', 'truck', 'bus', 'motorcycle']  # All become 'vehicle'
  person: ['person', 'pedestrian', 'cyclist']      # All become 'person' 
  animal: ['dog', 'cat', 'bird', 'horse']         # All become 'animal'
```

### 🌐 **Download Scripts**
```yaml
# Automatic dataset download
path: ../datasets/my_dataset
train: images/train
val: images/val

nc: 5  
names: ['class1', 'class2', 'class3', 'class4', 'class5']

# Auto-download script
download: |
  # Download and extract dataset
  wget https://example.com/dataset.zip
  unzip dataset.zip -d ../datasets/
  mv dataset my_dataset
  echo "Dataset ready!"
```

## 🐛 **Common Issues & Solutions**

### **❌ "Dataset not found" Error**
```python
# Check your paths
import os
dataset_path = '/path/to/dataset'
print(f"Dataset exists: {os.path.exists(dataset_path)}")
print(f"Contents: {os.listdir(dataset_path)}")

# Common fixes:
# 1. Check absolute vs relative paths
# 2. Ensure images/ and labels/ folders exist
# 3. Verify yaml file is in correct location
```

### **❌ "No images found" Error**  
```yaml
# Make sure your yaml paths are correct
path: /absolute/path/to/dataset     # Use absolute path if unsure
train: images/train                 # Should contain .jpg, .png files
val: images/val                     # Should contain .jpg, .png files

# Check image extensions
# LEAF-YOLO supports: .jpg, .jpeg, .png, .bmp, .tif, .tiff
```

### **❌ "Label format error"**
```python
# Check label format in your .txt files
# Each line should be: class_id x_center y_center width height
# Example: 0 0.5 0.5 0.3 0.4

# Validate labels
with open('labels/train/sample.txt', 'r') as f:
    for line_num, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Error on line {line_num}: {line}")
        
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        
        # Check ranges
        assert 0 <= x <= 1, f"x_center out of range: {x}"
        assert 0 <= y <= 1, f"y_center out of range: {y}"  
        assert 0 <= w <= 1, f"width out of range: {w}"
        assert 0 <= h <= 1, f"height out of range: {h}"
```

### **❌ "Class count mismatch"**
```yaml
# Make sure nc matches your actual number of classes
nc: 5                                           # Number of classes
names: ['class0', 'class1', 'class2', 'class3', 'class4']  # Must have exactly 5 names

# Check your label files - class IDs should be 0, 1, 2, 3, 4 (not 1, 2, 3, 4, 5)
```

## 💡 **Pro Tips**

### **Dataset Preparation**
1. **Start Small**: Test with 10-20 images first, then scale up
2. **Check Quality**: Manually verify a few label files are correct
3. **Balance Classes**: Try to have roughly equal numbers of each class
4. **Augmentation**: LEAF-YOLO automatically augments data during training

### **Configuration Best Practices**
1. **Use Absolute Paths**: Avoid confusion with relative paths
2. **Descriptive Names**: Use clear, descriptive class names
3. **Version Control**: Keep track of dataset versions and changes
4. **Documentation**: Add comments explaining your dataset

### **Performance Optimization**
1. **Image Size**: 640x640 is the standard, but you can use 320, 416, or 1280
2. **Cache**: For small datasets, enable caching for faster loading
3. **Workers**: Use more data loading workers for faster training

## 🎓 **Learning Path**

### **Beginner**: Using Existing Datasets
1. **Start with COCO**: `model.train(data='coco')`
2. **Try VisDrone**: `model.train(data='visdrone')`  
3. **Understand**: Look at the .yaml files to see how they work

### **Intermediate**: Custom Datasets
1. **Create**: Your own dataset with proper folder structure
2. **Configure**: Write your own .yaml configuration file
3. **Train**: Use your custom dataset for training

### **Advanced**: Complex Configurations  
1. **Optimize**: Use image lists, class mapping, download scripts
2. **Integrate**: Combine multiple datasets
3. **Automate**: Create scripts to generate configurations

---

**Remember**: Good data configuration is the foundation of successful AI training! Take time to organize your data properly and write clear configuration files. Your future self will thank you when everything works smoothly! 📊🚀

**The key insight**: The configuration file is like a recipe - it tells LEAF-YOLO exactly how to find and understand your data. Get this right, and everything else becomes much easier!
