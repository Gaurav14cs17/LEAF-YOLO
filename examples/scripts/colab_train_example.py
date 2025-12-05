#!/usr/bin/env python3
"""
LEAF-YOLO Training Example for Google Colab
Complete training pipeline with visualization
"""

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def setup_training_environment():
    """Setup environment for training"""
    print("🏋️ LEAF-YOLO Training Setup")
    print("=" * 40)
    
    # Check GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (training will be slower)")
    
    return device

def create_sample_dataset():
    """Create a sample dataset structure"""
    print("\n📊 Creating sample dataset...")
    
    dataset_path = Path('/content/sample_dataset')
    
    # Create directory structure
    (dataset_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True) 
    (dataset_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Create dataset config
    config_content = f"""
# Sample Dataset Configuration
path: {dataset_path}
train: images/train
val: images/val

# Classes
nc: 5
names: ['person', 'car', 'truck', 'bus', 'motorcycle']
"""
    
    with open(dataset_path / 'dataset.yaml', 'w') as f:
        f.write(config_content)
    
    print(f"✅ Dataset structure created at: {dataset_path}")
    print("💡 Add your images and labels to the created directories")
    
    return dataset_path / 'dataset.yaml'

def train_model(dataset_config, device):
    """Train LEAF-YOLO model"""
    print("\n🚀 Starting model training...")
    
    from leafyolo import LEAFYOLO
    
    # Create model
    model = LEAFYOLO('detect', variant='leafyolo_s', nc=5)
    
    # Training configuration
    training_config = {
        'data': str(dataset_config),
        'epochs': 10,  # Small for demo
        'batch_size': 8,
        'img_size': 640,
        'device': device,
        'project': '/content/runs',
        'name': 'leafyolo_demo',
        'patience': 5,
        'save_period': 5,
        'cache': True,
        'workers': 2
    }
    
    print("📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print("\n🎉 Training completed!")
        print(f"Best model: {results['best']}")
        print(f"Last model: {results['last']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def visualize_training_results():
    """Visualize training results"""
    print("\n📈 Visualizing training results...")
    
    results_file = '/content/runs/detect/leafyolo_demo/results.csv'
    
    if os.path.exists(results_file):
        # Load results
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🏋️ LEAF-YOLO Training Progress', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/obj_loss'], label='Object Loss') 
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
        axes[0, 0].set_title('📉 Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP metrics
        axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
        axes[0, 1].set_title('📊 Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision & Recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall'], label='Recall')
        axes[1, 0].set_title('🎯 Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
        axes[1, 1].set_title('📚 Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_metrics = df.iloc[-1]
        print("\n🏆 Final Training Metrics:")
        print(f"   mAP@0.5: {final_metrics['metrics/mAP_0.5']:.3f}")
        print(f"   Precision: {final_metrics['metrics/precision']:.3f}")
        print(f"   Recall: {final_metrics['metrics/recall']:.3f}")
        
    else:
        print("📊 No training results found")

def test_trained_model():
    """Test the trained model"""
    print("\n🎯 Testing trained model...")
    
    best_model_path = '/content/runs/detect/leafyolo_demo/weights/best.pt'
    
    if os.path.exists(best_model_path):
        from leafyolo import LEAFYOLO
        
        # Load trained model
        model = LEAFYOLO(best_model_path)
        
        # Create test image
        test_image = torch.randn(640, 640, 3).numpy()
        
        # Run inference
        results = model.predict(test_image)
        
        print("✅ Model inference successful!")
        print(f"🎯 Model classes: {list(model.names.values())}")
        
    else:
        print("❌ No trained model found")

def export_model():
    """Export trained model"""
    print("\n📤 Exporting trained model...")
    
    best_model_path = '/content/runs/detect/leafyolo_demo/weights/best.pt'
    
    if os.path.exists(best_model_path):
        from leafyolo import LEAFYOLO
        
        model = LEAFYOLO(best_model_path)
        
        try:
            # Export to ONNX
            onnx_path = model.export(format='onnx', img_size=640)
            print(f"✅ ONNX model exported: {onnx_path}")
            
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
    
    else:
        print("❌ No trained model to export")

def download_results():
    """Download training results"""
    print("\n📥 Preparing results for download...")
    
    try:
        from google.colab import files
        import zipfile
        
        # Create results package
        with zipfile.ZipFile('leafyolo_training_results.zip', 'w') as zipf:
            
            # Add trained models
            results_dir = Path('/content/runs/detect/leafyolo_demo')
            if results_dir.exists():
                for file_path in results_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(results_dir.parent))
            
            # Add sample code
            sample_code = """
# Using your trained LEAF-YOLO model

from leafyolo import LEAFYOLO

# Load your trained model
model = LEAFYOLO('path/to/weights/best.pt')

# Run inference
results = model.predict('your_image.jpg')

# Process results
for r in results:
    for detection in r.boxes:
        class_name = model.names[int(detection.cls)]
        confidence = float(detection.conf)
        print(f"Found {class_name} with {confidence:.2f} confidence")
"""
            
            zipf.writestr('usage_example.py', sample_code)
        
        # Download
        files.download('leafyolo_training_results.zip')
        print("✅ Results downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")

def main():
    """Main training pipeline"""
    
    # Setup environment
    device = setup_training_environment()
    
    # Create sample dataset
    dataset_config = create_sample_dataset()
    
    # Train model
    results = train_model(dataset_config, device)
    
    if results:
        # Visualize results
        visualize_training_results()
        
        # Test model
        test_trained_model()
        
        # Export model
        export_model()
        
        # Download results
        download_results()
        
        print("\n🎉 Training pipeline completed!")
        print("\n📚 Next Steps:")
        print("1. Add your own dataset to the created directories")
        print("2. Increase epochs for better results")
        print("3. Experiment with different model variants")
        print("4. Try different hyperparameters")
        
    else:
        print("❌ Training pipeline failed")

if __name__ == "__main__":
    main()
