# 🛠️ LEAF-YOLO Scripts

<div align="center">

**⚡ Ready-to-Use Scripts**

*Command-line tools for training, prediction, and deployment*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Examples](https://img.shields.io/badge/📔%20See-Examples-blue?style=for-the-badge)](../examples/README.md)

</div>

---

## 📋 Table of Contents

- [🚀 Quick Start Scripts](#-quick-start-scripts)
- [🎯 Main Scripts](#-main-scripts)
- [🔧 Utility Scripts](#-utility-scripts)
- [💻 Command Examples](#-command-examples)
- [⚙️ Advanced Usage](#️-advanced-usage)

---

## 🚀 **Quick Start Scripts**

### **One-Line Commands**

```bash
# Quick training
python scripts/train.py --data coco --variant leafyolo_m --epochs 100

# Quick prediction
python scripts/predict.py --source images/ --model best.pt

# Quick validation
python scripts/validate.py --data coco --model best.pt

# CLI interface
python scripts/cli.py --help
```

---

## 🎯 **Main Scripts**

### 📄 **Script Overview**

<table>
<tr>
<td width="50%">

**🏋️ [train.py](train.py)**
- Complete training pipeline
- Multi-GPU support
- Progress monitoring
- Automatic checkpointing

**🎯 [predict.py](predict.py)**  
- Single image or batch prediction
- Video processing support
- Real-time webcam inference
- Multiple output formats

</td>
<td width="50%">

**📊 [validate.py](validate.py)**
- Model evaluation and testing
- Performance metrics calculation
- Comparison between models
- Error analysis tools

**🎮 [cli.py](cli.py)**
- Unified command-line interface
- All-in-one tool
- Interactive help
- Simplified commands

</td>
</tr>
</table>

---

## 🏋️ **Training Script**

### **Basic Training**
```bash
# Train with default settings
python scripts/train.py --data configs/datasets/coco.yaml --variant leafyolo_m

# Train with custom settings
python scripts/train.py \
    --data configs/datasets/visdrone.yaml \
    --variant leafyolo_s \
    --epochs 200 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

### **Advanced Training Options**
```bash
# Multi-GPU training
python scripts/train.py \
    --data coco \
    --variant leafyolo_l \
    --device 0,1,2,3 \
    --batch-size 64

# Resume training
python scripts/train.py \
    --data coco \
    --resume runs/train/exp/weights/last.pt

# Transfer learning
python scripts/train.py \
    --data custom_dataset.yaml \
    --weights leafyolo_m.pt \
    --freeze 10
```

### **Training Parameters**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--data` | Dataset config path | None | `coco`, `visdrone.yaml` |
| `--variant` | Model variant | `leafyolo_m` | `leafyolo_n`, `leafyolo_s` |
| `--epochs` | Training epochs | 100 | 300 |
| `--batch-size` | Batch size | 16 | 32 |
| `--img-size` | Image size | 640 | 416, 1280 |
| `--device` | Device to use | `auto` | `cpu`, `0`, `0,1,2,3` |
| `--weights` | Initial weights | None | `leafyolo_m.pt` |
| `--resume` | Resume training | None | `last.pt` |

---

## 🎯 **Prediction Script**

### **Basic Prediction**
```bash
# Predict single image
python scripts/predict.py --source image.jpg --model best.pt

# Predict multiple images
python scripts/predict.py --source images/ --model best.pt

# Predict video
python scripts/predict.py --source video.mp4 --model best.pt

# Webcam prediction
python scripts/predict.py --source 0 --model best.pt
```

### **Advanced Prediction Options**
```bash
# Custom confidence threshold
python scripts/predict.py \
    --source images/ \
    --model best.pt \
    --conf-thres 0.5 \
    --iou-thres 0.45

# Save results  
python scripts/predict.py \
    --source images/ \
    --model best.pt \
    --save \
    --save-txt \
    --save-conf

# Specific classes only
python scripts/predict.py \
    --source images/ \
    --model best.pt \
    --classes 0 1 2  # person, bicycle, car
```

### **Prediction Parameters**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--source` | Input source | None | `image.jpg`, `images/`, `0` |
| `--model` | Model path | None | `best.pt`, `leafyolo_s.onnx` |
| `--conf-thres` | Confidence threshold | 0.25 | 0.1, 0.5 |
| `--iou-thres` | IoU threshold | 0.45 | 0.3, 0.6 |
| `--classes` | Filter classes | None | `0 1 2`, `person car` |
| `--save` | Save results | False | `--save` |
| `--device` | Device | `auto` | `cpu`, `0` |

---

## 📊 **Validation Script**

### **Basic Validation**
```bash
# Validate model
python scripts/validate.py --data configs/datasets/coco.yaml --model best.pt

# Validate with custom settings
python scripts/validate.py \
    --data visdrone.yaml \
    --model best.pt \
    --img-size 640 \
    --batch-size 32
```

### **Validation Parameters**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--data` | Dataset config | None | `coco.yaml` |
| `--model` | Model path | None | `best.pt` |
| `--img-size` | Image size | 640 | 416, 1280 |
| `--batch-size` | Batch size | 32 | 16, 64 |
| `--conf-thres` | Confidence threshold | 0.001 | 0.1 |
| `--iou-thres` | IoU threshold | 0.6 | 0.45 |
| `--device` | Device | `auto` | `cpu`, `0` |

---

## 🎮 **CLI Interface**

### **Unified Command Interface**
```bash
# Show all available commands
python scripts/cli.py --help

# Training
python scripts/cli.py train --data coco --variant leafyolo_m --epochs 100

# Prediction  
python scripts/cli.py predict --source images/ --model best.pt

# Validation
python scripts/cli.py val --data coco --model best.pt

# Export
python scripts/cli.py export --model best.pt --format onnx

# System info
python scripts/cli.py info
```

---

## 🔧 **Utility Scripts**

### 📁 **Setup Scripts**

```bash
scripts/setup/
├── install_dependencies.py    # Install all dependencies
├── download_models.py         # Download pre-trained models
├── setup_datasets.py          # Setup common datasets
└── verify_installation.py     # Verify installation
```

### 🛠️ **Tool Scripts**

```bash
scripts/tools/
├── convert_format.py          # Convert between data formats
├── analyze_dataset.py         # Dataset analysis and statistics
├── benchmark.py               # Performance benchmarking
└── export_model.py            # Export to different formats
```

---

## 💻 **Command Examples**

### **Complete Workflow Example**

```bash
# 1. Setup environment
python scripts/setup/install_dependencies.py

# 2. Download pre-trained model
python scripts/setup/download_models.py --variant leafyolo_m

# 3. Train on custom dataset
python scripts/train.py \
    --data configs/datasets/custom.yaml \
    --variant leafyolo_m \
    --epochs 100 \
    --batch-size 16

# 4. Validate trained model
python scripts/validate.py \
    --data configs/datasets/custom.yaml \
    --model runs/train/exp/weights/best.pt

# 5. Run predictions
python scripts/predict.py \
    --source test_images/ \
    --model runs/train/exp/weights/best.pt \
    --save

# 6. Export for deployment
python scripts/tools/export_model.py \
    --model runs/train/exp/weights/best.pt \
    --format onnx tflite
```

### **Batch Processing Example**

```bash
# Process multiple datasets
for dataset in coco visdrone custom; do
    python scripts/train.py \
        --data configs/datasets/${dataset}.yaml \
        --variant leafyolo_s \
        --epochs 50 \
        --name ${dataset}_experiment
done

# Compare results
python scripts/tools/compare_results.py \
    --experiments coco_experiment visdrone_experiment custom_experiment
```

---

## ⚙️ **Advanced Usage**

### **Environment Variables**
```bash
# Set default device
export LEAFYOLO_DEVICE=0

# Set default data directory
export LEAFYOLO_DATA_DIR=/path/to/datasets

# Use in scripts
python scripts/train.py --data $LEAFYOLO_DATA_DIR/coco.yaml
```

### **Configuration Files**
```bash
# Use configuration file instead of command line arguments
python scripts/train.py --config configs/training/high_accuracy.yaml

# Override specific parameters
python scripts/train.py \
    --config configs/training/fast_training.yaml \
    --epochs 200 \
    --batch-size 32
```

### **Distributed Training**
```bash
# Single node, multiple GPUs
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    scripts/train.py \
    --data coco \
    --variant leafyolo_l \
    --batch-size 64

# Multiple nodes (advanced)
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    --nproc_per_node 4 \
    scripts/train.py
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

<details>
<summary><strong>Script not found</strong></summary>

```bash
# Make sure you're in the project root
cd /path/to/LEAF-YOLO
python scripts/train.py --help

# Or use absolute path
python /path/to/LEAF-YOLO/scripts/train.py --help
```

</details>

<details>
<summary><strong>Import errors</strong></summary>

```bash
# Make sure LEAF-YOLO is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/LEAF-YOLO"
python scripts/train.py

# Or install in development mode
pip install -e .
```

</details>

<details>
<summary><strong>Permission errors</strong></summary>

```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with python explicitly
python scripts/train.py
```

</details>

---

## 📚 **Additional Resources**

### 🔗 **Related Documentation**
- [📖 Configuration Guide](../docs/configuration.md) - Complete config reference
- [🎓 Examples & Tutorials](../examples/README.md) - Step-by-step guides
- [🧪 Testing Framework](../tests/README.md) - Test your setup

### 💡 **Pro Tips**

1. **Use configuration files** for complex setups
2. **Start with smaller models** for experimentation  
3. **Monitor GPU memory usage** during training
4. **Save checkpoints frequently** for long training runs
5. **Use validation scripts** to compare models

---

<div align="center">

**⚡ Ready to Get Started?**

*Choose your script and start building!*

[![Train Now](https://img.shields.io/badge/🏋️%20Start-Training-green?style=for-the-badge)](train.py)
[![Predict Now](https://img.shields.io/badge/🎯%20Start-Predicting-blue?style=for-the-badge)](predict.py)

</div>
