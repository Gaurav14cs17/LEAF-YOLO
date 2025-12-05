# 🍃 LEAF-YOLO Main Package

<div align="center">

**Welcome to the heart of LEAF-YOLO!** 🚀

*This is where all the magic happens - the engine room of your AI model*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-blue?style=for-the-badge)](../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's in Here?](#-whats-in-here)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🎓 Learning Path](#-learning-path)
- [🔧 Common Use Cases](#-common-use-cases)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 Next Steps](#-next-steps)

---

## 🎯 What's in Here?

This directory contains the **core LEAF-YOLO package** - everything you need to detect objects, train models, and deploy AI solutions. It's organized like a well-structured house where everything has its place.

<table>
<tr>
<td width="50%">

**📂 Directory Structure**
```
leafyolo/
├── 🧠 models/          # AI brains
├── ⚙️ engine/          # Workhorses  
├── 🧱 nn/              # LEGO blocks
├── 🛠️ utils/           # Toolbox
├── 📊 data/            # Data magic
└── 📄 __init__.py      # Front door
```

</td>
<td width="50%">

**🔗 Quick Links**
- [📖 Models Guide](./models/README.md)
- [📖 Engine Guide](./engine/README.md) 
- [📖 Neural Networks](./nn/README.md)
- [📖 Utils Guide](./utils/README.md)
- [📖 Data Pipeline](./data/README.md)

</td>
</tr>
</table>

## 🚀 Quick Start

### The Simple Way (Recommended)
```python
from leafyolo import LEAFYOLO

# Create your AI detector in one line!
model = LEAFYOLO('detect', variant='leafyolo_m')

# Detect objects in any image
results = model('your_image.jpg')

# That's it! 🎉
```

### The Advanced Way (For Power Users)
```python
# Import specific components
from leafyolo.models.leafyolo import LEAFYOLO
from leafyolo.engine.trainer import LeafTrainer
from leafyolo.utils.config import get_config

# Build your custom pipeline
config = get_config('detect', 'leafyolo_s', nc=20)
model = LEAFYOLO(config)
trainer = LeafTrainer(model=model)
```

## 🏗️ Architecture Overview

### 🧠 **models/** - The Smart Stuff
This is where your AI models live. The main `LEAFYOLO` class is your best friend - it handles everything from loading models to running predictions.

**What it does**: Creates, loads, and manages your AI models
**When to use**: Every time you want to use LEAF-YOLO (which is always!)

### ⚙️ **engine/** - The Powerhouse  
The engine is like your car's motor - it does the heavy lifting. Training? Check. Inference? Check. Validation? Double check.

**What it does**: Trains models, runs predictions, evaluates performance
**When to use**: When you need to train custom models or run batch processing

### 🧱 **nn/** - Building Blocks
Neural Networks made simple! This contains all the LEGO blocks we use to build our AI models. Convolutions, attention mechanisms, detection heads - it's all here.

**What it does**: Provides the fundamental neural network components
**When to use**: When building custom architectures or understanding how LEAF-YOLO works

### 🛠️ **utils/** - Your Toolbox
Every good project needs utilities. Configuration management, image processing, mathematical operations - all the helpful tools live here.

**What it does**: Provides helper functions and utilities
**When to use**: Throughout the entire pipeline (these are used everywhere!)

### 📊 **data/** - Data Pipeline
Getting data into your model is crucial. This handles loading images, processing datasets, and feeding data to your AI models.

**What it does**: Loads and preprocesses your training and inference data
**When to use**: When training models or processing large datasets

## 🎓 Learning Path

### 👶 **Beginner**: Just Want to Detect Objects?
1. **Start with**: `models/leafyolo.py` - The main LEAFYOLO class
2. **Read**: `utils/config.py` - Understanding configuration
3. **Try**: Simple detection examples (see main README)

### 🧑‍💻 **Intermediate**: Want to Train Your Own Model?
1. **Explore**: `engine/trainer/` - Training pipeline
2. **Understand**: `data/datasets.py` - Data loading
3. **Configure**: `utils/config.py` - Custom configurations

### 🚀 **Advanced**: Want to Build Custom Models?
1. **Study**: `nn/modules/` - Neural network components  
2. **Experiment**: `nn/tasks/detect.py` - Model architectures
3. **Extend**: `models/` - Create your own model variants

## 🔧 Common Use Cases

### 1. Quick Object Detection
```python
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect')
results = model('image.jpg')
```

### 2. Custom Training
```python
from leafyolo.engine.trainer import LeafTrainer
trainer = LeafTrainer(model=your_model, data='dataset.yaml')
trainer.train(epochs=100)
```

### 3. Batch Processing
```python
from leafyolo.engine.predictor import LeafPredictor
predictor = LeafPredictor(model=your_model)
results = predictor.predict('folder_of_images/')
```

### 4. Model Evaluation
```python
from leafyolo.engine.validator import LeafValidator  
validator = LeafValidator(model=your_model)
metrics = validator.validate('test_dataset.yaml')
```

## 🐛 Troubleshooting

### **Import Errors?**
```python
# Make sure you're in the right directory
import sys
sys.path.append('/path/to/LEAF-YOLO')
from leafyolo import LEAFYOLO
```

### **Model Not Loading?**
```python
# Try with verbose mode to see what's happening
model = LEAFYOLO('detect', verbose=True)
```

### **Configuration Issues?**
```python
# Check your configuration
from leafyolo.utils.config import get_config
config = get_config('detect', 'leafyolo_m')
print(config)  # See what's being loaded
```

## 📚 Next Steps

1. **Explore Each Subdirectory**: Each folder has its own README with detailed explanations
2. **Try the Examples**: Start with simple detection, then move to training
3. **Read the Code**: The code is well-documented and human-readable
4. **Experiment**: Modify configurations and see what happens!

## 💡 Pro Tips

- **Start Small**: Begin with `leafyolo_n` for quick experiments
- **Use GPU**: Set `device='0'` to use your GPU for faster processing
- **Save Results**: Add `save=True` to automatically save detection results
- **Monitor Progress**: Training shows beautiful progress bars and metrics

---

**Remember**: Every expert was once a beginner. Don't be afraid to experiment and learn by doing! 🚀

**Next**: Check out the individual README files in each subdirectory for detailed information about specific components.
