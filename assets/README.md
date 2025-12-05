# 🖼️ LEAF-YOLO Assets

<div align="center">

**🎨 Visual Assets & Media**

*Images, figures, logos, and other visual content*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)

</div>

---

## 📋 Table of Contents

- [📊 Performance Figures](#-performance-figures)
- [🖼️ Example Images](#️-example-images) 
- [🎨 Brand Assets](#-brand-assets)
- [📈 Charts & Graphs](#-charts--graphs)
- [🔧 Usage Guidelines](#-usage-guidelines)

---

## 📊 **Performance Figures**

### **Model Performance Charts**

<table>
<tr>
<td width="50%">

**📈 [Performance Comparison](figures/params.png)**
- Model size vs accuracy comparison
- Speed benchmarks across variants
- Memory usage analysis

</td>
<td width="50%">

**🔥 [Training Curves](figures/training_curves.png)**
- Loss progression during training
- Validation accuracy over epochs
- Learning rate schedules

</td>
</tr>
</table>

### **Architecture Diagrams**

```
figures/
├── model_architecture.png      # LEAF-YOLO architecture overview
├── attention_mechanism.png     # Attention mechanism visualization
├── feature_pyramid.png         # Feature pyramid network
└── comparison_table.png        # Performance comparison table
```

---

## 🖼️ **Example Images**

### **Detection Results**

<table>
<tr>
<td align="center">

**🚁 Aerial Detection**
<br>
<img src="images/aerial_detection_example.jpg" width="200px" alt="Aerial Detection"/>
<br>
*Drone footage object detection*

</td>
<td align="center">

**🚗 Vehicle Detection**  
<br>
<img src="images/vehicle_detection_example.jpg" width="200px" alt="Vehicle Detection"/>
<br>
*Traffic monitoring results*

</td>
<td align="center">

**👥 Crowd Detection**
<br>
<img src="images/crowd_detection_example.jpg" width="200px" alt="Crowd Detection"/>
<br>
*People counting in crowds*

</td>
</tr>
</table>

### **Before/After Comparisons**

```
images/comparisons/
├── before_after_detection.png   # Raw vs detected images
├── model_comparison.png         # Different model results
└── confidence_levels.png        # Different confidence thresholds
```

---

## 🎨 **Brand Assets**

### **LEAF-YOLO Logos**

<table>
<tr>
<td align="center" width="33%">

**🍃 Main Logo**
<br>
<img src="logos/leafyolo_logo.png" width="120px" alt="LEAF-YOLO Logo"/>
<br>
*Primary brand logo*

</td>
<td align="center" width="33%">

**🌿 Icon Version**
<br>
<img src="logos/leafyolo_icon.png" width="120px" alt="LEAF-YOLO Icon"/>
<br>
*Square icon format*

</td>
<td align="center" width="33%">

**📱 Monochrome**
<br>
<img src="logos/leafyolo_mono.png" width="120px" alt="LEAF-YOLO Monochrome"/>
<br>
*Black & white version*

</td>
</tr>
</table>

### **Logo Variants**

```
logos/
├── leafyolo_logo.png           # Main color logo (PNG)
├── leafyolo_logo.svg           # Vector format (SVG)
├── leafyolo_icon.png           # Square icon version
├── leafyolo_mono.png           # Monochrome version
├── leafyolo_white.png          # White version (for dark backgrounds)
└── leafyolo_banner.png         # Wide banner format
```

### **Usage Guidelines**

- **✅ DO**: Use official logos for presentations and documentation
- **✅ DO**: Maintain aspect ratio when resizing
- **✅ DO**: Use appropriate version for background (color/mono/white)
- **❌ DON'T**: Modify, distort, or change colors of logos
- **❌ DON'T**: Use low-resolution versions for print materials

---

## 📈 **Charts & Graphs**

### **Performance Analysis**

#### **Speed vs Accuracy**
```
figures/performance/
├── speed_accuracy_plot.png     # Speed vs accuracy scatter plot
├── memory_usage_chart.png      # Memory consumption by variant
├── fps_comparison.png          # FPS across different hardware
└── latency_analysis.png        # Inference latency breakdown
```

#### **Training Analytics**
```
figures/training/
├── loss_curves.png             # Training and validation loss
├── learning_curves.png         # Learning rate schedules
├── augmentation_examples.png   # Data augmentation samples
└── convergence_analysis.png    # Training convergence patterns
```

### **Dataset Visualizations**

```
figures/datasets/
├── class_distribution.png      # Class frequency distribution
├── bbox_size_analysis.png      # Bounding box size distribution
├── dataset_samples.png         # Representative dataset samples
└── annotation_quality.png      # Annotation quality analysis
```

---

## 🔧 **Usage Guidelines**

### **In Documentation**

```markdown
# Reference images in markdown
![Performance Chart](assets/figures/params.png)

# With specific sizing
<img src="assets/figures/model_architecture.png" width="60%" alt="Model Architecture"/>

# Center aligned
<div align="center">
<img src="assets/logos/leafyolo_logo.png" width="200px" alt="LEAF-YOLO"/>
</div>
```

### **In Code**

```python
# Load example images for testing
import cv2
import os

# Load example image
example_img = cv2.imread('assets/images/example_detection.jpg')

# Use for demos
def demo_detection():
    model = LEAFYOLO('detect')
    results = model.predict('assets/images/example_aerial.jpg')
    return results
```

### **In Presentations**

- Use **PNG format** for screenshots and photos
- Use **SVG format** for logos and scalable graphics
- **High resolution** versions available in `high_res/` subdirectories
- Follow **brand guidelines** for color and placement

---

## 📁 **Directory Structure**

```
assets/
├── README.md                   # This file
│
├── 📊 figures/                 # Performance charts and graphs
│   ├── params.png              # Main performance chart
│   ├── model_architecture.png  # Architecture diagram
│   ├── performance/            # Performance analysis charts
│   ├── training/               # Training visualization
│   └── datasets/               # Dataset analysis plots
│
├── 🖼️ images/                  # Example images and screenshots
│   ├── example_detection.jpg   # Detection result examples
│   ├── aerial_samples/         # Aerial imagery samples
│   ├── comparisons/            # Before/after comparisons
│   └── high_res/               # High resolution versions
│
├── 🎨 logos/                   # Brand assets and logos
│   ├── leafyolo_logo.png       # Main logo
│   ├── leafyolo_logo.svg       # Vector version
│   ├── variants/               # Logo variants
│   └── brand_guidelines.md     # Brand usage guidelines
│
└── 📐 templates/               # Templates for creating assets
    ├── presentation.pptx       # PowerPoint template
    ├── poster_template.pdf     # Academic poster template
    └── social_media/           # Social media templates
```

---

## 🎯 **Contributing Assets**

### **Adding New Assets**

1. **Place in appropriate subdirectory**
2. **Use descriptive filenames**
3. **Include both standard and high-res versions**
4. **Update this README with descriptions**

### **Image Guidelines**

- **Format**: PNG for images, SVG for scalable graphics
- **Size**: Optimize for web (< 500KB for standard images)
- **Quality**: High enough for print and presentations
- **Naming**: Use descriptive, consistent naming convention

### **Documentation Images**

```bash
# Standard size for documentation
width: 600-800px for main images
width: 200-400px for thumbnails
format: PNG with transparency when needed

# High resolution for presentations  
width: 1200px+ for detailed charts
format: PNG or SVG for scalability
```

---

## 📚 **Asset Usage Examples**

### **In Main README**
```markdown
<img src="assets/figures/params.png" width="60%" alt="LEAF-YOLO Performance"/>
```

### **In Documentation**
```markdown  
![Model Architecture](assets/figures/model_architecture.png)
```

### **In Presentations**
- Use `assets/logos/leafyolo_logo.svg` for scalable logo
- Use `assets/figures/high_res/` versions for large displays
- Follow brand guidelines for colors and spacing

---

<div align="center">

**🎨 Visual Assets Ready!**

*Professional quality assets for all your LEAF-YOLO needs*

[![View Figures](https://img.shields.io/badge/📊%20View-Performance%20Figures-blue?style=for-the-badge)](figures/)
[![Brand Assets](https://img.shields.io/badge/🎨%20Download-Brand%20Assets-green?style=for-the-badge)](logos/)

</div>
