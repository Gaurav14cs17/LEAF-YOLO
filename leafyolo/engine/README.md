# ⚙️ LEAF-YOLO Engine

<div align="center">

**🔥 The Powerhouse of LEAF-YOLO** 

*Where the real work happens - training, predicting, and evaluating*

[![Back to Package](https://img.shields.io/badge/←%20Back%20to-Package%20README-blue?style=for-the-badge)](../README.md)
[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../../README.md)

</div>

---

## 📋 Table of Contents

- [🎯 What's This All About?](#-whats-this-all-about)
- [🏋️ Trainer - Your AI Teacher](#️-trainer---your-ai-teacher)
- [🔍 Predictor - Your AI Detective](#-predictor---your-ai-detective)  
- [📊 Validator - Your AI Examiner](#-validator---your-ai-examiner)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [🔧 Configuration Tips](#-configuration-tips)
- [📈 Monitoring Progress](#-monitoring-progress)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's This All About?

The engine is like having three super-smart assistants:
- 🏋️ **Trainer**: Teaches your AI to recognize objects
- 🔍 **Predictor**: Uses your trained AI to find objects in new images  
- 📊 **Validator**: Tests how good your AI actually is

```
engine/
├── 🏋️ trainer/         # The AI teacher - trains your models
├── 🔍 predictor/       # The AI detective - finds objects in images
├── 📊 validator/       # The AI examiner - tests model performance
└── 📄 __init__.py      # Makes everything work together
```

## 🏋️ Trainer - Your AI Teacher

### What Does It Do?
The trainer is like a patient teacher that shows your AI thousands of images and says "See this car? This is a car. See this person? This is a person." Over time, your AI gets really good at recognizing these objects on its own.

### When Do You Use It?
- 🎯 Training models on your own data
- 🔄 Fine-tuning existing models for better performance  
- 📈 Improving model accuracy on specific tasks

### Simple Example
```python
from leafyolo.engine.trainer import LeafTrainer
from leafyolo import LEAFYOLO

# Create your model
model = LEAFYOLO('detect', variant='leafyolo_s')

# Create the trainer
trainer = LeafTrainer(model=model)

# Train on your dataset (this takes a while, grab some coffee! ☕)
results = trainer.train(
    data='my_dataset.yaml',    # Your dataset configuration
    epochs=100,                # How many times to go through the data
    batch_size=16,             # How many images to process at once
    device='0'                 # Use GPU if you have one
)

print(f"Training complete! Best model saved at: {results['best']}")
```

### Advanced Training Options
```python
# Fine-tune an existing model
trainer.train(
    data='dataset.yaml',
    epochs=50,
    resume='path/to/existing/model.pt',  # Start from existing model
    freeze=[0, 1, 2],                    # Freeze some layers
    patience=10,                         # Stop early if no improvement
    save_period=5                        # Save checkpoint every 5 epochs
)
```

## 🔍 Predictor - Your AI Detective

### What Does It Do?
The predictor takes your trained AI and puts it to work! Give it any image (or video, or webcam feed) and it will find all the objects it has learned to recognize.

### When Do You Use It?
- 📷 Detecting objects in photos
- 🎥 Processing videos frame by frame
- 📱 Real-time detection from cameras
- 📁 Batch processing lots of images

### Simple Example
```python
from leafyolo.engine.predictor import LeafPredictor
from leafyolo import LEAFYOLO

# Load your trained model
model = LEAFYOLO('path/to/your/trained/model.pt')

# Create the predictor
predictor = LeafPredictor(model=model)

# Predict on a single image
results = predictor.predict('photo.jpg')

# Predict on multiple images
results = predictor.predict('folder_of_photos/')

# Predict on video
results = predictor.predict('drone_video.mp4', save=True)

# Real-time webcam (press 'q' to quit)
results = predictor.predict(0)  # 0 = default camera
```

### Advanced Prediction Options
```python
# Fine-tune prediction settings
results = predictor.predict(
    source='images/',
    conf_thres=0.25,      # Only show detections above 25% confidence
    iou_thres=0.45,       # Remove overlapping detections
    max_det=1000,         # Maximum detections per image
    classes=[0, 1, 2],    # Only detect specific classes
    save=True,            # Save results with bounding boxes
    save_txt=True,        # Save detection coordinates to files
    save_conf=True        # Include confidence scores in saved files
)
```

## 📊 Validator - Your AI Examiner

### What Does It Do?
The validator is like a strict teacher giving your AI a final exam. It tests your model on images it has never seen before and gives you detailed grades on how well it performs.

### When Do You Use It?
- 🎯 Testing how accurate your model is
- 📈 Comparing different models
- 🔍 Finding where your model makes mistakes
- 📊 Getting scientific metrics for research papers

### Simple Example
```python
from leafyolo.engine.validator import LeafValidator
from leafyolo import LEAFYOLO

# Load your model
model = LEAFYOLO('path/to/your/model.pt')

# Create the validator
validator = LeafValidator(model=model)

# Test your model
results = validator.validate('test_dataset.yaml')

# See how well it performed
print(f"Overall Accuracy (mAP@0.5): {results['map50']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1-Score: {results['f1']:.3f}")
```

### Understanding the Results
```python
# Get detailed per-class results
results = validator.validate('dataset.yaml', verbose=True)

# The results dictionary contains:
# - map50: Overall accuracy (higher is better)
# - precision: How many detections were correct
# - recall: How many actual objects were found  
# - f1: Balance between precision and recall
# - confusion_matrix: Detailed breakdown of mistakes
```

## 🚀 Quick Start Guide

### 1. **Just Want to Detect Objects?**
```python
from leafyolo import LEAFYOLO

model = LEAFYOLO('detect')           # Uses pre-trained model
results = model('your_image.jpg')    # Automatic prediction
```

### 2. **Want to Train Your Own Model?**
```python
from leafyolo import LEAFYOLO

model = LEAFYOLO('detect', variant='leafyolo_s')
results = model.train(data='dataset.yaml', epochs=100)
```

### 3. **Want to Test Model Performance?**
```python
from leafyolo import LEAFYOLO

model = LEAFYOLO('your_model.pt')
metrics = model.val(data='test_data.yaml')
```

## 🔧 Configuration Tips

### Training Configuration
```python
# Fast training (for testing)
trainer.train(epochs=10, batch_size=8, img_size=416)

# High quality training (for production)
trainer.train(epochs=300, batch_size=32, img_size=640, patience=50)

# Memory-efficient training
trainer.train(batch_size=4, img_size=320, workers=2)
```

### Prediction Configuration
```python
# Fast prediction (lower quality)
predictor.predict(conf_thres=0.5, img_size=416)

# High quality prediction (slower)
predictor.predict(conf_thres=0.1, img_size=1280, augment=True)

# Batch processing optimization
predictor.predict(batch_size=16, device='0')  # Use GPU
```

## 📈 Monitoring Progress

### Training Progress
The trainer shows you beautiful progress bars and live metrics:
```
Epoch 1/100: 100%|██████████| 1000/1000 [02:15<00:00,  7.41it/s]
                 Class     Images  Instances      P      R     mAP50
                   all       1000      5000   0.612  0.584     0.598
```

### Prediction Progress  
The predictor shows you what it's working on:
```
Processing images: 100%|██████████| 500/500 [01:23<00:00,  6.02it/s]
Results saved to: runs/detect/exp/
```

## 🐛 Troubleshooting

### **Training Issues**
```python
# Out of memory?
trainer.train(batch_size=4, img_size=320)

# Training too slow?
trainer.train(workers=8, device='0')  # Use more CPU cores + GPU

# Model not improving?
trainer.train(patience=20, lr0=0.001)  # More patience + lower learning rate
```

### **Prediction Issues**
```python
# No detections found?
results = predictor.predict(conf_thres=0.1)  # Lower confidence threshold

# Too many false detections?
results = predictor.predict(conf_thres=0.5)  # Higher confidence threshold

# Prediction too slow?
results = predictor.predict(img_size=416, device='0')  # Smaller size + GPU
```

## 💡 Pro Tips

1. **Start Small**: Use `leafyolo_n` for quick experiments, then scale up
2. **Use GPU**: Always set `device='0'` if you have a GPU - it's 10x faster!
3. **Save Everything**: Use `save=True` to keep your results
4. **Monitor Training**: Watch the loss curves - they should go down over time
5. **Test Early**: Validate your model every few epochs to catch problems

## 🎓 Next Steps

1. **Try Training**: Start with a small dataset to understand the process
2. **Experiment with Settings**: Change batch sizes, learning rates, etc.
3. **Analyze Results**: Use the validator to understand your model's strengths and weaknesses
4. **Deploy Your Model**: Use the predictor for real-world applications

---

**Remember**: The engine is the heart of LEAF-YOLO. Master these three components, and you'll be able to solve almost any object detection problem! 🚀
