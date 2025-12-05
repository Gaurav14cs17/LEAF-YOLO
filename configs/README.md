# ⚙️ LEAF-YOLO Configuration Files

<div align="center">

**🎛️ Configuration Hub**

*All configuration files and dataset setups*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Documentation](https://img.shields.io/badge/📚%20Config-Guide-blue?style=for-the-badge)](../docs/configuration.md)

</div>

---

## 📋 Table of Contents

- [📄 Configuration Files](#-configuration-files)
- [📊 Dataset Configurations](#-dataset-configurations) 
- [🎯 Model Configurations](#-model-configurations)
- [🚀 Quick Start](#-quick-start)
- [🔧 Custom Configuration](#-custom-configuration)
- [💡 Best Practices](#-best-practices)

---

## 📄 **Configuration Files**

### 🎯 **Main Configuration**

```yaml
# configs/default.yaml - Default LEAF-YOLO configuration

# Task configuration
task: detect                    # detect, segment, classify

# Model variant
variant: leafyolo_m            # leafyolo_n, leafyolo_s, leafyolo_m, leafyolo_l, leafyolo_x

# Training parameters
epochs: 100
batch_size: 16
img_size: 640
device: auto                   # auto, cpu, cuda, 0, 1, 2, 3

# Dataset
data: configs/datasets/coco.yaml
nc: 80                         # Number of classes

# Optimization
optimizer: AdamW
lr0: 0.001
momentum: 0.9
weight_decay: 0.0005

# Augmentation
hsv_h: 0.015
hsv_s: 0.7  
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3
```

---

## 📊 **Dataset Configurations**

### 🌟 **Popular Datasets**

<table>
<tr>
<td width="50%">

**📁 [COCO Dataset](datasets/coco.yaml)**
```yaml
# COCO 2017 Dataset
path: ../datasets/coco
train: train2017.txt
val: val2017.txt
test: test-dev2017.txt

nc: 80
names: ['person', 'bicycle', 'car', ...]
```

</td>
<td width="50%">

**📁 [VisDrone Dataset](datasets/visdrone.yaml)**
```yaml
# VisDrone Dataset - Aerial Imagery  
path: ../datasets/visdrone
train: VisDrone2019-DET-train
val: VisDrone2019-DET-val

nc: 10  
names: ['pedestrian', 'people', 'bicycle', ...]
```

</td>
</tr>
</table>

### 📂 **Directory Structure**

```
configs/datasets/
├── coco.yaml              # COCO dataset config
├── visdrone.yaml          # VisDrone dataset config
├── custom_template.yaml   # Template for custom datasets
├── cityscapes.yaml        # Cityscapes dataset
├── pascal_voc.yaml        # Pascal VOC dataset
└── imagenet.yaml          # ImageNet classification
```

---

## 🎯 **Model Configurations**

### 🏗️ **Model Variants**

```yaml
# configs/models/variants.yaml

variants:
  leafyolo_n:
    depth_multiple: 0.33
    width_multiple: 0.25
    max_channels: 1024
    
  leafyolo_s:
    depth_multiple: 0.33
    width_multiple: 0.50
    max_channels: 1024
    
  leafyolo_m:
    depth_multiple: 0.67
    width_multiple: 0.75
    max_channels: 1024
    
  leafyolo_l:
    depth_multiple: 1.0
    width_multiple: 1.0  
    max_channels: 1024
    
  leafyolo_x:
    depth_multiple: 1.33
    width_multiple: 1.25
    max_channels: 1024
```

---

## 🚀 **Quick Start**

### **Using Default Configuration**
```python
from leafyolo import LEAFYOLO

# Use default configuration
model = LEAFYOLO('detect', variant='leafyolo_m')
```

### **Using Custom Configuration**
```python
from leafyolo import LEAFYOLO

# Use specific config file
model = LEAFYOLO(config='configs/custom_config.yaml')

# Or override specific parameters
model = LEAFYOLO('detect', variant='leafyolo_s', nc=20)
```

### **Training with Configuration**
```python
# Train with config file
results = model.train(
    data='configs/datasets/visdrone.yaml',
    epochs=100,
    batch_size=16
)
```

---

## 🔧 **Custom Configuration**

### **Creating Custom Dataset Config**

1. **Copy template:**
```bash
cp configs/datasets/custom_template.yaml configs/datasets/my_dataset.yaml
```

2. **Edit configuration:**
```yaml
# configs/datasets/my_dataset.yaml

# Dataset path (absolute or relative to config file)
path: ../datasets/my_dataset

# Image directories (relative to path)
train: images/train
val: images/val
test: images/test

# Class information
nc: 5
names: ['class1', 'class2', 'class3', 'class4', 'class5']

# Optional: Download script
download: |
  # Commands to download and setup dataset
  echo "Download your dataset here"
```

3. **Use in training:**
```python
model.train(data='configs/datasets/my_dataset.yaml')
```

### **Creating Custom Training Config**

```yaml
# configs/my_training_config.yaml

# Model configuration
task: detect
variant: leafyolo_m
nc: 5

# Dataset
data: configs/datasets/my_dataset.yaml

# Training parameters
epochs: 200
batch_size: 32
img_size: 640
device: '0'

# Optimization
optimizer: AdamW
lr0: 0.01
weight_decay: 0.0005

# Learning rate schedule
lrf: 0.1              # Final learning rate (lr0 * lrf)
cos_lr: true          # Cosine learning rate schedule

# Augmentation (adjust for your data)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0         # Rotation degrees
translate: 0.1        # Translation fraction  
scale: 0.9            # Scale range (1-scale to 1+scale)
fliplr: 0.5           # Horizontal flip probability

# Advanced settings
patience: 50          # Early stopping patience
save_period: 10       # Save checkpoint every N epochs
workers: 8            # Data loading workers
```

---

## 💡 **Best Practices**

### 🎯 **Dataset Configuration Tips**

```yaml
# ✅ Good practices
path: ../datasets/my_data     # Use relative paths
train: images/train           # Clear directory names
val: images/val

# ❌ Avoid
path: /absolute/path/dataset  # Hard-coded absolute paths
train: train_imgs_final_v2    # Unclear naming
```

### ⚡ **Training Configuration Tips**

```yaml
# For fast prototyping
epochs: 50
batch_size: 32
img_size: 416
patience: 10

# For production quality
epochs: 300
batch_size: 16  
img_size: 640
patience: 50
```

### 🚀 **Hardware-Specific Settings**

```yaml
# For high-end GPU (RTX 3090, A100)
batch_size: 32
img_size: 640
workers: 16

# For mid-range GPU (RTX 3060, RTX 2070)  
batch_size: 16
img_size: 640
workers: 8

# For mobile/edge deployment training
batch_size: 8
img_size: 416
workers: 4
```

---

## 📚 **Configuration Reference**

### 🔗 **Related Documentation**
- [📖 Configuration Guide](../docs/configuration.md) - Complete configuration reference
- [🎓 Training Tutorial](../examples/notebooks/) - Step-by-step training
- [📊 Dataset Preparation](../docs/dataset-preparation.md) - Prepare your data

### ⚙️ **Configuration Validation**

```python
# Validate your configuration
from leafyolo.utils.config import validate_config

config_path = 'configs/my_config.yaml'
is_valid, errors = validate_config(config_path)

if is_valid:
    print("✅ Configuration is valid!")
else:
    print("❌ Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### 🐛 **Common Issues**

<details>
<summary><strong>Dataset not found</strong></summary>

```yaml
# Problem: Dataset path incorrect
path: ../datasets/my_data

# Solution: Check path exists
# Use absolute path if needed
path: /full/path/to/dataset
```

</details>

<details>
<summary><strong>Class count mismatch</strong></summary>

```yaml
# Problem: nc doesn't match actual classes
nc: 5
names: ['class1', 'class2', 'class3']  # Only 3 names!

# Solution: Make sure nc matches names length
nc: 3
names: ['class1', 'class2', 'class3']
```

</details>

<details>
<summary><strong>Out of memory</strong></summary>

```yaml
# Problem: Batch size too large
batch_size: 64  # Too big for your GPU

# Solution: Reduce batch size
batch_size: 16
# Or reduce image size
img_size: 416
```

</details>

---

<div align="center">

**⚙️ Ready to Configure?**

*Start with our templates and customize for your needs*

[![Config Guide](https://img.shields.io/badge/📖%20Read-Config%20Guide-blue?style=for-the-badge)](../docs/configuration.md)

</div>
