# 🛠️ Utility Functions

<div align="center">

**🔧 The Swiss Army Knife of LEAF-YOLO**

*Your trusty toolbox of specialized instruments*

[![Back to Package](https://img.shields.io/badge/←%20Back%20to-Package%20README-blue?style=for-the-badge)](../README.md)
[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's in Your Toolbox?](#-whats-in-your-toolbox)
- [⚙️ config.py - The Smart Brain](#️-configpy---the-smart-brain)
- [🔧 general.py - The Multi-Tool](#-generalpy---the-multi-tool)
- [🧮 loss.py - The AI Teacher](#-losspy---the-ai-teacher)
- [🎨 plots.py - The Visual Artist](#-plotspy---the-visual-artist)
- [🔥 torch_utils.py - PyTorch Optimizer](#-torch_utilspy---the-pytorch-optimizer)
- [⚓ autoanchor.py - Anchor Optimizer](#-autoanchorpy---the-anchor-optimizer)
- [📊 metrics/ - Performance Analyst](#-metrics---the-performance-analyst)
- [📞 callbacks/ - Event Handlers](#-callbacks---the-event-handlers)
- [🚀 Practical Usage Examples](#-practical-usage-examples)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's in Your Toolbox?

This directory contains all the **helper functions and tools** that support the main AI functionality. It's like having a workshop full of useful gadgets that make your life easier.

```
utils/
├── ⚙️ config.py            # Smart configuration manager (the brain)
├── 🔧 general.py           # General utilities (the multitool)
├── 🧮 loss.py              # Loss functions (the teacher)
├── 🎨 plots.py             # Visualization tools (the artist)
├── 🔥 torch_utils.py       # PyTorch helpers (the optimizer)
├── ⚓ autoanchor.py         # Automatic anchor generation
├── 🌐 google_utils.py      # Cloud storage utilities
├── ➕ add_nms.py           # Non-Maximum Suppression tools
├── 📊 metrics/             # Performance measurement tools
├── 📞 callbacks/           # Training event handlers
└── 📄 __init__.py          # Makes everything accessible
```

## ⚙️ **config.py** - The Smart Brain

This is the **intelligent configuration manager** that automatically sets up everything for you. It's like having a smart assistant that knows exactly how to configure your AI for any task.

### 🧠 **What It Does**
- **Adapts automatically**: Detects what task you want and configures everything
- **Manages variants**: Switches between nano, small, medium, large, and xlarge models
- **Handles datasets**: Knows about COCO, VisDrone, and custom datasets
- **Validates settings**: Catches configuration mistakes before they cause problems

### 🚀 **How to Use It**
```python
from leafyolo.utils.config import get_config

# Automatically configure for object detection
config = get_config('detect', 'leafyolo_m')
print("✅ Perfect detection setup ready!")

# Configure for segmentation  
config = get_config('segment', 'leafyolo_s', nc=20)  # 20 custom classes
print("✅ Segmentation model configured!")

# Configure for classification
config = get_config('classify', 'leafyolo_n', nc=1000)  # ImageNet classes
print("✅ Classification system ready!")
```

### 💡 **Pro Tip**: You rarely need to call this directly - the main LEAFYOLO class uses it automatically!

## 🔧 **general.py** - The Multi-Tool

This file is like having a **super-powered multi-tool** - it contains dozens of useful functions that are used throughout LEAF-YOLO.

### 🎯 **Key Functions**

#### **Bounding Box Operations**
```python
from leafyolo.utils.general import xyxy2xywh, xywh2xyxy, clip_coords

# Convert between different bounding box formats
bbox_xyxy = [100, 150, 200, 250]  # [x1, y1, x2, y2]
bbox_xywh = xyxy2xywh(bbox_xyxy)  # [x_center, y_center, width, height]

# Clip coordinates to image boundaries  
clipped = clip_coords(bbox_xyxy, img_shape=(640, 640))
```

#### **Non-Maximum Suppression** 
```python
from leafyolo.utils.general import non_max_suppression

# Remove overlapping duplicate detections
predictions = model(image)  # Raw model output
clean_detections = non_max_suppression(
    predictions,
    conf_thres=0.25,    # Minimum confidence
    iou_thres=0.45      # Maximum overlap allowed
)
```

#### **IoU Calculations**
```python
from leafyolo.utils.general import bbox_iou, box_iou

# Calculate how much two bounding boxes overlap
box1 = [100, 100, 200, 200]
box2 = [150, 150, 250, 250] 
overlap = bbox_iou(box1, box2)
print(f"Boxes overlap by {overlap:.2%}")
```

## 🧮 **loss.py** - The AI Teacher

Loss functions are like **teachers that grade your AI's homework** - they tell the model how wrong it is and help it learn to do better.

### 📚 **What Loss Functions Do**
- **Detection Loss**: Teaches the model to find objects and draw accurate boxes
- **Classification Loss**: Teaches the model to identify objects correctly
- **Confidence Loss**: Teaches the model to be appropriately confident

### 🎓 **How It Works**
```python
from leafyolo.utils.loss import ComputeLoss

# Create a loss calculator for your model
loss_fn = ComputeLoss(model)

# During training, the loss function grades the model's predictions
predictions = model(images)
targets = load_ground_truth(images)

# Calculate how wrong the predictions are
loss, loss_items = loss_fn(predictions, targets)

# The optimizer uses this to improve the model
optimizer.zero_grad()
loss.backward()      # Calculate gradients
optimizer.step()     # Update model weights
```

### 💡 **Human Analogy**: 
Imagine you're learning to draw. The loss function is like an art teacher who:
- Says "Your circle is too oval" (shape loss)
- Says "Your cat is in the wrong place" (position loss)  
- Says "That's definitely not a dog" (classification loss)

## 🎨 **plots.py** - The Visual Artist

This file creates **beautiful visualizations** to help you understand what your AI is seeing and doing.

### 🖼️ **What It Can Visualize**

#### **Detection Results**
```python
from leafyolo.utils.plots import plot_one_box

# Draw bounding boxes on images
for detection in results:
    plot_one_box(
        box=detection.xyxy,
        img=image,
        label=f"{detection.class_name} {detection.conf:.2f}",
        color=(255, 0, 0)  # Red box
    )
```

#### **Training Progress**
```python
from leafyolo.utils.plots import plot_results

# Visualize training metrics over time
plot_results('runs/train/exp/results.csv')
# Creates beautiful charts showing loss curves, accuracy, etc.
```

#### **Data Analysis**
```python
from leafyolo.utils.plots import plot_labels

# Analyze your dataset
plot_labels(labels='path/to/labels/', save_dir='analysis/')
# Shows class distributions, box sizes, etc.
```

## 🔥 **torch_utils.py** - The PyTorch Optimizer

This file contains **PyTorch-specific utilities** that make training faster and more efficient.

### ⚡ **Key Features**

#### **Model EMA (Exponential Moving Average)**
```python
from leafyolo.utils.torch_utils import ModelEMA

# Create a smoothed version of your model for better results
ema = ModelEMA(model)

# During training, update the EMA model
for batch in dataloader:
    # ... normal training ...
    ema.update(model)  # Keep a smoothed version

# Use EMA model for final predictions (usually better!)
final_model = ema.ema
```

#### **Smart Device Selection**
```python
from leafyolo.utils.torch_utils import select_device

# Automatically choose the best available device
device = select_device('0')  # Try GPU 0, fallback to CPU
device = select_device('')   # Auto-select best available
device = select_device('cpu')  # Force CPU

print(f"Using device: {device}")
```

## ⚓ **autoanchor.py** - The Anchor Optimizer

Anchors are like **template boxes** that help your AI find objects more efficiently. This file automatically optimizes them for your specific dataset.

### 🎯 **What It Does**
```python
from leafyolo.utils.autoanchor import check_anchors

# Automatically optimize anchors for your dataset
model = LEAFYOLO('detect')
dataset = load_your_dataset()

# This improves detection accuracy by matching anchors to your data
check_anchors(dataset, model=model, thr=4.0, img_size=640)
```

### 💡 **Why This Matters**
Different datasets have different object sizes. A dataset of:
- **Aerial images**: Needs tiny anchors for small objects
- **Close-up photos**: Needs larger anchors for big objects
- **Mixed scenes**: Needs a variety of anchor sizes

Auto-anchor optimization matches the anchors to your specific data!

## 📊 **metrics/** - The Performance Analyst

This subdirectory contains tools for **measuring how good your AI actually is**.

### 📈 **Key Metrics**

#### **mAP (mean Average Precision)**
```python
from leafyolo.utils.metrics.metrics import ap_per_class

# Calculate detailed accuracy metrics
tp, fp, precision, recall, f1, ap, ap_class = ap_per_class(...)

print(f"Overall accuracy (mAP@0.5): {ap.mean():.3f}")
print(f"Precision: {precision.mean():.3f}")
print(f"Recall: {recall.mean():.3f}")
```

#### **Confusion Matrix**
```python
from leafyolo.utils.metrics.metrics import ConfusionMatrix

# See exactly where your model makes mistakes
confusion = ConfusionMatrix(nc=80)  # 80 classes
confusion.process_batch(detections, labels)
confusion.plot(save_dir='analysis/')  # Beautiful confusion matrix plot
```

## 📞 **callbacks/** - The Event Handlers

Callbacks are like **event listeners** that react to things happening during training.

### 🎯 **WandB Integration**
```python
from leafyolo.utils.callbacks.wandb_logging.wandb_utils import WandbLogger

# Automatically log training to Weights & Biases
logger = WandbLogger(project='my-leafyolo-project')

# During training, it automatically:
# ✅ Logs metrics and loss curves
# ✅ Saves model checkpoints  
# ✅ Tracks system performance
# ✅ Creates beautiful dashboards
```

## 🚀 Practical Usage Examples

### **Complete Detection Pipeline**
```python
from leafyolo import LEAFYOLO
from leafyolo.utils.general import non_max_suppression
from leafyolo.utils.plots import plot_one_box
import cv2

# Load model
model = LEAFYOLO('detect')

# Load and process image
img = cv2.imread('your_image.jpg')
results = model(img)

# Clean up overlapping detections
clean_results = non_max_suppression(results, conf_thres=0.25)

# Visualize results
for detection in clean_results[0]:  # First image
    plot_one_box(
        detection[:4],  # Box coordinates
        img, 
        label=f"Class {int(detection[5])} {detection[4]:.2f}",
        color=(0, 255, 0)
    )

cv2.imshow('Results', img)
cv2.waitKey(0)
```

### **Training with Monitoring**
```python
from leafyolo import LEAFYOLO
from leafyolo.utils.callbacks.wandb_logging.wandb_utils import WandbLogger

# Set up training with monitoring
model = LEAFYOLO('detect', variant='leafyolo_m')

# Train with automatic logging and visualization
results = model.train(
    data='dataset.yaml',
    epochs=100,
    wandb=True,           # Enable WandB logging
    project='my-project', # Project name
    name='experiment-1'   # Experiment name
)

# Beautiful training plots and metrics automatically created!
```

## 🐛 Troubleshooting Common Issues

### **Configuration Problems**
```python
# Check your configuration
from leafyolo.utils.config import get_config
config = get_config('detect', 'leafyolo_m')
print(config)  # See exactly what's being configured

# Validate configuration
from leafyolo.utils.config import ConfigManager
manager = ConfigManager()
manager.validate_config(config)  # Catches common errors
```

### **Bounding Box Issues**
```python
# Visualize your bounding boxes to debug
from leafyolo.utils.plots import plot_one_box
import cv2

# Check if boxes are in the right format and location
img = cv2.imread('test.jpg')
box = [100, 100, 200, 200]  # [x1, y1, x2, y2]
plot_one_box(box, img, label='Test Box')
cv2.imshow('Debug', img)
```

### **Loss Not Decreasing**
```python
# Check your loss components
loss, loss_items = loss_fn(predictions, targets)
print(f"Box loss: {loss_items[0]}")      # Should decrease
print(f"Object loss: {loss_items[1]}")   # Should decrease  
print(f"Class loss: {loss_items[2]}")    # Should decrease

# If losses aren't decreasing, check:
# - Learning rate (maybe too high/low)
# - Data quality (correct labels?)
# - Model complexity (too simple/complex?)
```

## 💡 Pro Tips

1. **Use the Config Manager**: Let it handle complex configurations automatically
2. **Visualize Everything**: Use plots.py to understand what's happening
3. **Monitor Training**: WandB integration makes tracking easy
4. **Check Your Data**: Use label analysis tools to understand your dataset
5. **Optimize Anchors**: Run autoanchor for better performance on custom data
6. **Debug with Plots**: When something's wrong, visualize it first

## 🎓 Learning Path

### **Beginner**: Understanding the Basics
1. **Start with**: `config.py` - Learn how configuration works
2. **Explore**: `general.py` - Understand common operations
3. **Visualize**: `plots.py` - See what your model is doing

### **Intermediate**: Training and Optimization  
1. **Study**: `loss.py` - Understand how models learn
2. **Use**: `torch_utils.py` - Optimize training performance
3. **Monitor**: `callbacks/` - Track training progress

### **Advanced**: Custom Development
1. **Extend**: Add your own utility functions
2. **Optimize**: Create custom loss functions
3. **Integrate**: Add new monitoring and visualization tools

---

**Remember**: Utilities are the unsung heroes of AI - they make everything else possible! Master these tools, and you'll be able to debug problems, optimize performance, and understand what's happening under the hood. 🔧🚀

**The key insight**: Don't try to memorize every function - instead, know what capabilities exist so you can find the right tool when you need it!
