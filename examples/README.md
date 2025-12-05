# 📔 LEAF-YOLO Examples & Tutorials

<div align="center">

**🎯 Learn by Doing**

*Interactive examples and step-by-step tutorials*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Documentation](https://img.shields.io/badge/📚%20Full-Documentation-blue?style=for-the-badge)](../docs/README.md)

</div>

---

## 📋 Table of Contents

- [📓 Jupyter Notebooks](#-jupyter-notebooks)
- [🐍 Python Scripts](#-python-scripts)  
- [🚀 Quick Start](#-quick-start)
- [🎓 Learning Path](#-learning-path)
- [💡 Tips & Tricks](#-tips--tricks)

---

## 📓 **Jupyter Notebooks**

### 🚀 **Interactive Colab Notebooks**

<table>
<tr>
<td width="50%">

**📘 [Quick Start Notebook](notebooks/LEAF_YOLO_Quick_Start.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/LEAF-YOLO/blob/main/examples/notebooks/LEAF_YOLO_Quick_Start.ipynb)

Perfect for beginners:
- ✅ 5-minute setup
- ✅ Pre-trained model testing  
- ✅ Upload your images
- ✅ Try different variants
- ✅ Export models

</td>
<td width="50%">

**📙 [Complete Training Tutorial](notebooks/LEAF_YOLO_Complete_Training.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/LEAF-YOLO/blob/main/examples/notebooks/LEAF_YOLO_Complete_Training.ipynb)

Full training pipeline:
- ✅ Dataset preparation
- ✅ Model training
- ✅ Progress monitoring
- ✅ Model evaluation
- ✅ Export & deployment

</td>
</tr>
</table>

### 📚 **Local Jupyter Examples**

Browse the `notebooks/` directory for:
- 🎯 **Object Detection** examples
- 🏋️ **Custom Training** tutorials  
- 📊 **Data Analysis** notebooks
- 🔧 **Configuration** examples
- 📈 **Performance Analysis** tools

---

## 🐍 **Python Scripts**

### 🛠️ **Setup & Utilities**

```bash
examples/scripts/
├── colab_setup.py          # Google Colab environment setup
├── colab_train_example.py  # Complete training example
├── data_preparation.py     # Dataset preparation helper
├── benchmark.py            # Performance benchmarking
└── export_examples.py      # Model export examples
```

### 🚀 **Usage Examples**

```bash
# Quick setup for any environment
python examples/scripts/colab_setup.py

# Run complete training pipeline  
python examples/scripts/colab_train_example.py

# Benchmark model performance
python examples/scripts/benchmark.py --model leafyolo_s

# Export model to different formats
python examples/scripts/export_examples.py --model best.pt --formats onnx tflite
```

---

## 🚀 **Quick Start**

### **Option 1: Google Colab (Recommended)**
1. Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/LEAF-YOLO/blob/main/examples/notebooks/LEAF_YOLO_Quick_Start.ipynb)
2. Run all cells
3. Upload your image
4. See results in 5 minutes!

### **Option 2: Local Jupyter**
```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook examples/notebooks/

# Open LEAF_YOLO_Quick_Start.ipynb
```

### **Option 3: Python Scripts**
```bash
# Setup environment
python examples/scripts/colab_setup.py

# Run your first detection
python -c "
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect', variant='leafyolo_s')
results = model.predict('your_image.jpg')
print('✅ Detection complete!')
"
```

---

## 🎓 **Learning Path**

<details open>
<summary><strong>👶 Beginner Path (2-3 hours)</strong></summary>

1. **🚀 Start**: [Quick Start Notebook](notebooks/LEAF_YOLO_Quick_Start.ipynb)
   - Learn basic object detection
   - Try different model variants
   - Upload and test your images

2. **📊 Understand**: Basic concepts
   - What is object detection?
   - How do confidence scores work?
   - Understanding bounding boxes

3. **🎯 Practice**: Upload different images
   - Try various image types
   - Adjust confidence thresholds
   - Compare model variants

</details>

<details>
<summary><strong>🧑‍💻 Intermediate Path (1-2 days)</strong></summary>

1. **🏋️ Complete**: [Training Tutorial](notebooks/LEAF_YOLO_Complete_Training.ipynb)
   - Prepare custom dataset
   - Train your own model
   - Monitor training progress

2. **📈 Analyze**: Training results
   - Understand loss curves
   - Evaluate model performance
   - Compare different configurations

3. **🚀 Deploy**: Your trained model
   - Export to different formats
   - Test deployment scenarios
   - Optimize for your use case

</details>

<details>
<summary><strong>🚀 Advanced Path (1-2 weeks)</strong></summary>

1. **🔧 Customize**: Model architecture
   - Modify neural network components
   - Implement custom loss functions
   - Experiment with new ideas

2. **📊 Benchmark**: Performance
   - Compare with other models
   - Optimize for specific hardware
   - Measure real-world performance

3. **🤝 Contribute**: To the project
   - Add new features
   - Improve documentation
   - Share your improvements

</details>

---

## 💡 **Tips & Tricks**

### 🎯 **For Better Results**

```python
# Tip 1: Adjust confidence threshold
results = model.predict('image.jpg', conf_thres=0.3)  # Lower = more detections

# Tip 2: Use appropriate model variant
model_mobile = LEAFYOLO('detect', variant='leafyolo_n')    # For mobile
model_accuracy = LEAFYOLO('detect', variant='leafyolo_l')  # For accuracy

# Tip 3: Batch processing for speed
results = model.predict(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### ⚡ **Performance Optimization**

```python
# Tip 4: Use GPU when available
model = LEAFYOLO('detect', device='cuda')

# Tip 5: Optimize image size
results = model.predict('image.jpg', img_size=416)  # Smaller = faster

# Tip 6: Export for deployment
model.export(format='onnx')  # Universal format
```

### 🐛 **Troubleshooting**

```python
# Common Issue 1: No objects detected
# Solution: Lower confidence threshold
results = model.predict('image.jpg', conf_thres=0.1)

# Common Issue 2: Too many false detections  
# Solution: Higher confidence threshold
results = model.predict('image.jpg', conf_thres=0.5)

# Common Issue 3: Out of memory
# Solution: Smaller batch size or image size
results = model.predict('image.jpg', img_size=320)
```

---

## 📚 **Additional Resources**

### 🔗 **External Links**
- [Google Colab Tutorial](https://colab.research.google.com/notebooks/intro.ipynb) - Learn Colab basics
- [Jupyter Documentation](https://jupyter.readthedocs.io/) - Jupyter notebook guide
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Deep learning basics

### 📖 **Documentation**
- [📚 Full Documentation](../docs/README.md) - Complete guides
- [⚙️ Configuration](../configs/README.md) - Dataset and model config
- [🧪 Testing](../tests/README.md) - Testing and validation

### 🤝 **Community**
- [GitHub Discussions](https://github.com/your_username/LEAF-YOLO/discussions) - Ask questions
- [Issues](https://github.com/your_username/LEAF-YOLO/issues) - Report bugs
- [Contributing](../docs/contributing.md) - How to contribute

---

<div align="center">

**🎉 Happy Learning!**

*Ready to build amazing AI applications with LEAF-YOLO?*

[![Quick Start](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/examples/notebooks/LEAF_YOLO_Quick_Start.ipynb)

</div>
