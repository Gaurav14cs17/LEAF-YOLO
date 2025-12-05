# LEAF-YOLO Unified Configuration System

This guide explains the new unified configuration system that adapts to different tasks automatically.

## 🎯 Overview

The LEAF-YOLO configuration system uses a **single configuration file** (`leafyolo.yaml`) that automatically adapts based on the task type. No more multiple config files for different tasks!

## 📁 Clean Project Structure

```
LEAF-YOLO/
├── leafyolo/                 # Main package (Ultralytics-style)
│   ├── engine/              # Training & inference engines
│   ├── models/              # Model architectures
│   ├── nn/                  # Neural network components
│   │   ├── modules/         # Building blocks
│   │   └── tasks/           # Task-specific models
│   ├── utils/               # Utilities including config.py
│   └── data/                # Data loading
├── leafyolo.yaml            # ✨ Single adaptive config
├── leafyolo_cli.py          # Simple CLI interface
└── *_ultralytics.py         # Entry point scripts
```

## ⚙️ Configuration System

### Single Config File (`leafyolo.yaml`)
```yaml
# Automatically adapts to tasks: detect, segment, classify
task: 'detect'  # Auto-detected or explicitly set

# Model variants
variants:
  leafyolo_n:   # Nano
  leafyolo_s:   # Small  
  leafyolo_m:   # Medium (default)
  leafyolo_l:   # Large
  leafyolo_x:   # Extra Large

# Task-specific heads
head_detect:    # Detection head
head_segment:   # Segmentation head  
head_classify:  # Classification head

# Built-in datasets
datasets:
  coco:         # COCO dataset
  visdrone:     # VisDrone dataset
```

### Adaptive Configuration Manager
```python
from leafyolo.utils.config import get_config

# Get task-specific config
config = get_config(task='detect', variant='leafyolo_m', nc=80)
config = get_config(task='segment', variant='leafyolo_s', nc=91)  
config = get_config(task='classify', variant='leafyolo_n', nc=1000)
```

## 🚀 Usage Examples

### 1. Simple API (Recommended)
```python
from leafyolo import LEAFYOLO

# Auto-configure for detection
model = LEAFYOLO('detect', variant='leafyolo_m')

# Auto-configure for segmentation
model = LEAFYOLO('segment', variant='leafyolo_s', nc=91)

# Auto-configure for classification
model = LEAFYOLO('classify', variant='leafyolo_n', nc=1000)

# Train, predict, validate
model.train(data='coco', epochs=100)
results = model.predict('images/')
metrics = model.val(data='coco')
```

### 2. CLI Interface
```bash
# Train detection model
python leafyolo_cli.py train --task detect --variant leafyolo_m --data coco --epochs 100

# Run prediction
python leafyolo_cli.py predict --model leafyolo.pt --source images/ --save

# Validate model
python leafyolo_cli.py val --model best.pt --data coco

# Export model
python leafyolo_cli.py export --model best.pt --format onnx

# Show config info
python leafyolo_cli.py info --config-info
```

### 3. Traditional Scripts
```bash
# Training
python train_ultralytics.py --task detect --variant leafyolo_m --data coco --epochs 100

# Prediction  
python predict_ultralytics.py --model best.pt --source images/ --save

# Validation
python val_ultralytics.py --model best.pt --data coco
```

## 🎛️ Available Options

### Model Variants
- `leafyolo_n` - Nano (fastest, smallest)
- `leafyolo_s` - Small (balanced)  
- `leafyolo_m` - Medium (default, good balance)
- `leafyolo_l` - Large (higher accuracy)
- `leafyolo_x` - Extra Large (best accuracy)

### Tasks
- `detect` - Object detection (default)
- `segment` - Instance segmentation
- `classify` - Image classification

### Built-in Datasets
- `coco` - COCO dataset (80 classes)
- `visdrone` - VisDrone dataset (10 classes)

## 🔧 Advanced Configuration

### Custom Configuration
```python
from leafyolo import LEAFYOLO

# Override specific parameters
model = LEAFYOLO('detect', 
                variant='leafyolo_m',
                nc=80,
                depth_multiple=1.0,
                width_multiple=1.0)

# Custom hyperparameters
model.train(data='coco', 
           epochs=300,
           lr0=0.01,
           batch_size=32,
           img_size=640)
```

### Programmatic Config Access
```python
from leafyolo.utils.config import config_manager

# List available options
print("Variants:", config_manager.list_variants())
print("Datasets:", config_manager.list_datasets())  
print("Tasks:", config_manager.list_tasks())

# Get specific configurations
detect_config = config_manager.get_config('detect', 'leafyolo_m')
dataset_config = config_manager.get_dataset_config('coco')
hyperparams = config_manager.get_hyperparameters('detect')
```

## ✨ Key Benefits

1. **Single Source of Truth**: One config file for all tasks
2. **Automatic Adaptation**: Config adapts based on task type
3. **No Duplication**: Eliminates redundant configuration files
4. **Easy to Use**: Simple API with sensible defaults  
5. **Fully Compatible**: Works with existing Ultralytics patterns
6. **Clean Structure**: Organized, professional codebase

## 🔄 Migration from Old System

**Before:**
```python
# Old way - multiple config files
model = Model('yolov7-tiny.yaml')           # Detection
model = Model('yolov7-tiny-seg.yaml')       # Segmentation  
model = Model('yolov7-tiny-cls.yaml')       # Classification
```

**After:**
```python
# New way - single adaptive system
model = LEAFYOLO('detect')     # Auto-configures for detection
model = LEAFYOLO('segment')    # Auto-configures for segmentation
model = LEAFYOLO('classify')   # Auto-configures for classification
```

## 📊 File Count Reduction

- **Before**: 47+ configuration files, complex structure
- **After**: 1 config file, clean organized structure
- **Reduction**: ~95% fewer configuration files! 🎉
