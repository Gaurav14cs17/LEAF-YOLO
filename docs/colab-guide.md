# 🍃 LEAF-YOLO Google Colab Guide

Complete guide for using LEAF-YOLO in Google Colab - no local installation required!

## 🎯 Why Use Google Colab?

- ✅ **Free GPU Access** - Train models 10x faster than CPU
- ✅ **No Installation** - Everything runs in your browser
- ✅ **Pre-configured Environment** - Python, PyTorch, and common libraries ready
- ✅ **Cloud Storage** - Save and share your work easily
- ✅ **Collaboration** - Share notebooks with teammates
- ✅ **Perfect for Learning** - Interactive tutorials and examples

## 🚀 Quick Start Options

### Option 1: Ready-to-Use Notebooks (Recommended)

#### 🎯 **Instant Object Detection** (5 minutes)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb)

**Perfect for**: First-time users, testing pre-trained models, quick experiments

**What you'll do**:
- Run object detection on your images in 5 minutes
- Try different model variants (nano, small, medium)
- Test with webcam (if available)
- Export models for deployment

#### 🏋️ **Complete Training Tutorial** (30-60 minutes)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)

**Perfect for**: Learning AI training, custom datasets, research projects

**What you'll learn**:
- Prepare datasets for training
- Train your own custom AI model
- Monitor training progress with beautiful visualizations
- Evaluate model performance
- Export trained models
- Download everything for local use

### Option 2: Manual Setup (For Custom Projects)

#### Step 1: Create New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "New Notebook"
3. Make sure GPU is enabled: `Runtime → Change runtime type → Hardware accelerator → GPU`

#### Step 2: Quick Setup
```python
# Install LEAF-YOLO in one cell
!git clone https://github.com/Gaurav14cs17/LEAF-YOLO.git
%cd LEAF-YOLO
!pip install -r requirements.txt

# Test installation
from leafyolo import LEAFYOLO
print("✅ LEAF-YOLO ready!")
```

#### Step 3: Start Using
```python
# Load pre-trained model
model = LEAFYOLO('detect', variant='leafyolo_s')

# Upload and detect objects in your image
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    results = model.predict(filename)
    print(f"Found {len(results[0].boxes)} objects!")
```

## 📚 Step-by-Step Tutorials

### 🎯 **Tutorial 1: Object Detection**

```python
# 1. Setup
!git clone https://github.com/Gaurav14cs17/LEAF-YOLO.git
%cd LEAF-YOLO
!pip install -r requirements.txt

# 2. Load Model
from leafyolo import LEAFYOLO
model = LEAFYOLO('detect', variant='leafyolo_m')

# 3. Upload Image
from google.colab import files
uploaded = files.upload()

# 4. Run Detection
import matplotlib.pyplot as plt
for filename in uploaded.keys():
    results = model.predict(filename, save=False)
    
    # Show results
    if len(results) > 0:
        img_with_boxes = results[0].plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.title(f'LEAF-YOLO Detection Results')
        plt.show()
```

### 🏋️ **Tutorial 2: Model Training**

```python
# 1. Setup Environment
!git clone https://github.com/Gaurav14cs17/LEAF-YOLO.git
%cd LEAF-YOLO
!pip install -r requirements.txt

# 2. Prepare Dataset
from pathlib import Path
import os

# Create dataset structure
dataset_path = Path('/content/my_dataset')
for split in ['train', 'val']:
    (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Create dataset config
config_content = f"""
path: {dataset_path}
train: images/train
val: images/val
nc: 3
names: ['car', 'truck', 'bus']
"""
with open(dataset_path / 'dataset.yaml', 'w') as f:
    f.write(config_content)

print("✅ Dataset structure created")
print("📁 Upload your images to images/train and images/val")
print("🏷️ Upload corresponding labels to labels/train and labels/val")

# 3. Train Model (after uploading data)
from leafyolo import LEAFYOLO

model = LEAFYOLO('detect', variant='leafyolo_s', nc=3)
results = model.train(
    data=str(dataset_path / 'dataset.yaml'),
    epochs=50,
    batch_size=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='/content/runs',
    name='my_training'
)

print(f"✅ Training complete! Best model: {results['best']}")
```

### 📊 **Tutorial 3: Model Evaluation**

```python
# 1. Load Trained Model
from leafyolo import LEAFYOLO
model = LEAFYOLO('/content/runs/detect/my_training/weights/best.pt')

# 2. Run Evaluation
metrics = model.val(data='/content/my_dataset/dataset.yaml')

print(f"📊 Model Performance:")
print(f"   mAP@0.5: {metrics['map50']:.3f}")
print(f"   Precision: {metrics['precision']:.3f}")
print(f"   Recall: {metrics['recall']:.3f}")

# 3. Visualize Training Progress
import pandas as pd
import matplotlib.pyplot as plt

results_file = '/content/runs/detect/my_training/results.csv'
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    plt.title('Training Losses')
    plt.legend()
    
    plt.subplot(1, 2, 2) 
    plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## 🎛️ Configuration & Tips

### ⚡ **GPU Optimization for Colab**

```python
import torch

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
else:
    print("⚠️ No GPU - enable in Runtime → Change runtime type")

# Optimal settings for Colab GPU
training_config = {
    'batch_size': 16,        # Good for T4 GPU (15GB memory)
    'img_size': 640,         # Standard size
    'workers': 2,            # Don't overload CPU
    'cache': True,           # Cache images in memory
    'amp': True,             # Mixed precision training
}
```

### 💾 **Managing Colab Storage**

```python
# Check available space
import shutil
total, used, free = shutil.disk_usage("/content/")
print(f"💽 Disk Space: {free/1e9:.1f}GB free of {total/1e9:.1f}GB total")

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Save important files to Drive
!cp -r /content/runs /content/drive/MyDrive/LEAF-YOLO-Results/
print("✅ Results saved to Google Drive")
```

### 📥 **Downloading Results**

```python
# Download trained models and results
from google.colab import files
import zipfile

# Create download package
with zipfile.ZipFile('leafyolo_results.zip', 'w') as zipf:
    # Add trained model
    zipf.write('/content/runs/detect/my_training/weights/best.pt', 'best_model.pt')
    
    # Add training results
    zipf.write('/content/runs/detect/my_training/results.csv', 'training_results.csv')
    
    # Add sample usage code
    usage_code = '''
from leafyolo import LEAFYOLO

# Load your trained model
model = LEAFYOLO('best_model.pt')

# Run inference
results = model.predict('your_image.jpg')
    '''
    zipf.writestr('usage_example.py', usage_code)

# Download
files.download('leafyolo_results.zip')
print("✅ Download complete!")
```

## 📊 **Dataset Preparation in Colab**

### Option 1: Upload Your Own Dataset

```python
# Upload dataset ZIP file
from google.colab import files
import zipfile

print("📁 Upload your dataset as a ZIP file:")
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/datasets/')
    print(f"✅ Extracted {filename}")
```

### Option 2: Use Public Datasets

```python
# Example: Download COCO subset
!mkdir -p /content/datasets/coco
!wget http://images.cocodataset.org/zips/val2017.zip
!unzip val2017.zip -d /content/datasets/coco/

# Example: Use Roboflow dataset
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")  
dataset = project.version(1).download("yolov5")
```

### Option 3: Create Synthetic Dataset

```python
# Create sample dataset for testing
import cv2
import numpy as np
import os

def create_sample_dataset():
    dataset_path = '/content/sample_dataset'
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(f'{dataset_path}/images/{split}', exist_ok=True)
        os.makedirs(f'{dataset_path}/labels/{split}', exist_ok=True)
    
    # Generate sample images and labels
    for i in range(100):  # 100 sample images
        # Create random image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some random rectangles as "objects"
        for _ in range(3):  # 3 objects per image
            x1, y1 = np.random.randint(0, 400, 2)
            x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Create YOLO format label
            cx, cy = (x1 + x2) / 2 / 640, (y1 + y2) / 2 / 640
            w, h = (x2 - x1) / 640, (y2 - y1) / 640
            
            split = 'train' if i < 80 else 'val'
            
            # Save image
            cv2.imwrite(f'{dataset_path}/images/{split}/img_{i:03d}.jpg', img)
            
            # Save label
            with open(f'{dataset_path}/labels/{split}/img_{i:03d}.txt', 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    # Create config file
    config = f"""
path: {dataset_path}
train: images/train
val: images/val
nc: 1
names: ['object']
"""
    with open(f'{dataset_path}/dataset.yaml', 'w') as f:
        f.write(config)
    
    print(f"✅ Sample dataset created at {dataset_path}")
    return f'{dataset_path}/dataset.yaml'

# Create and use sample dataset
dataset_config = create_sample_dataset()
```

## 🐛 **Common Issues & Solutions**

### Issue 1: "Runtime disconnected"
**Solution**: Colab has usage limits. Save work regularly to Google Drive.
```python
# Auto-save to Drive every few epochs
from google.colab import drive
drive.mount('/content/drive')

# In training callback
def save_callback(epoch):
    if epoch % 10 == 0:  # Save every 10 epochs
        !cp -r /content/runs /content/drive/MyDrive/LEAF-YOLO-backup/
```

### Issue 2: "Out of memory"
**Solution**: Reduce batch size and image size.
```python
# Use smaller settings
model.train(
    data='dataset.yaml',
    batch_size=8,     # Reduced from 16
    img_size=416,     # Reduced from 640
    cache=False       # Don't cache images
)
```

### Issue 3: "No module named 'leafyolo'"
**Solution**: Make sure you're in the right directory and installed properly.
```python
# Check current directory
import os
print(f"Current directory: {os.getcwd()}")

# Reinstall if needed
%cd LEAF-YOLO
!pip install -r requirements.txt --force-reinstall
```

### Issue 4: "Dataset not found"
**Solution**: Check paths and file structure.
```python
# Debug dataset
import os
dataset_path = '/content/my_dataset'
print(f"Dataset exists: {os.path.exists(dataset_path)}")
print(f"Contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Not found'}")
```

## 🎓 **Learning Resources**

### **Beginner Path**
1. Start with [Quick Start Notebook](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb)
2. Try uploading your own images
3. Experiment with different model variants
4. Learn about confidence thresholds

### **Intermediate Path**  
1. Use [Complete Training Tutorial](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)
2. Prepare your own dataset
3. Train a custom model
4. Understand training metrics

### **Advanced Path**
1. Experiment with hyperparameters
2. Try data augmentation techniques
3. Implement custom loss functions
4. Deploy models to production

## 🤝 **Getting Help**

- 📖 **Documentation**: Check README files in each directory
- 🐛 **Issues**: [GitHub Issues](https://github.com/Gaurav14cs17/LEAF-YOLO/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Gaurav14cs17/LEAF-YOLO/discussions)
- 🎓 **Tutorials**: Start with the Colab notebooks above

## 💡 **Pro Tips**

1. **Use GPU**: Always enable GPU in Colab (Runtime → Change runtime type)
2. **Save Regularly**: Copy important files to Google Drive
3. **Start Small**: Begin with small datasets and short training runs
4. **Monitor Progress**: Use the training visualization code
5. **Experiment**: Try different model variants and hyperparameters

---

**Ready to start?** Click on one of the Colab notebook links above and begin your LEAF-YOLO journey! 🚀

[![Quick Start](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Quick_Start.ipynb) [![Complete Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LEAF-YOLO/blob/main/LEAF_YOLO_Complete_Training.ipynb)
