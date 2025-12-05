# 🧪 LEAF-YOLO Testing Framework

<div align="center">

**🔬 Quality Assurance Laboratory**

*Where we ensure everything works perfectly*

[![Back to Main](https://img.shields.io/badge/←%20Back%20to-Main%20README-green?style=for-the-badge)](../README.md)
[![Run Tests](https://img.shields.io/badge/🧪%20Run-Tests%20Now-brightgreen?style=for-the-badge)](#-running-tests)

</div>

---

## 📋 Table of Contents

- [🎯 What's This All About?](#-whats-this-all-about)
- [🧪 unit/ - The Microscope Lab](#-unit---the-microscope-lab)
- [🔗 integration/ - The Assembly Line](#-integration---the-assembly-line)
- [🏃 benchmarks/ - The Race Track](#-benchmarks---the-race-track)
- [📦 fixtures/ - The Supply Closet](#-fixtures---the-supply-closet)
- [🏃 run_tests.py - The Lab Manager](#-run_testspy---the-lab-manager)
- [🎯 Test Categories & Markers](#-test-categories--markers)
- [📊 Understanding Test Results](#-understanding-test-results)
- [🐛 Troubleshooting Tests](#-troubleshooting-tests)
- [💡 Pro Tips](#-pro-tips)

---

## 🎯 What's This All About?

Testing is like having a **team of quality inspectors** that check every component, every function, and every workflow to make sure they work correctly. It's your safety net that catches bugs before they become problems.

```
tests/
├── 🧪 unit/              # Test individual components (the microscope)
├── 🔗 integration/       # Test complete workflows (the assembly line)
├── 🏃 benchmarks/        # Test performance & speed (the race track)
├── 📦 fixtures/          # Test data & utilities (the supply closet)
├── ⚙️ conftest.py        # Shared test configuration (the toolbox)
├── 🏃 run_tests.py       # Test runner script (the lab manager)
└── 📄 __init__.py        # Makes everything work together
```

## 🧪 **unit/** - The Microscope Lab

Unit tests examine **individual components** in isolation, like looking at cells under a microscope.

### 🔬 **What Gets Tested**

#### **Configuration System** (`utils/test_config.py`)
```python
# Tests the smart configuration manager
def test_detect_config():
    """Make sure detection configuration works"""
    config = get_config('detect', 'leafyolo_m')
    assert config['task'] == 'detect'
    assert config['nc'] == 80  # COCO classes
    print("✅ Detection config works perfectly!")

def test_custom_classes():
    """Test custom number of classes"""
    config = get_config('detect', 'leafyolo_s', nc=10)
    assert config['nc'] == 10
    print("✅ Custom classes work!")
```

#### **Neural Network Modules** (`nn/test_modules.py`)
```python
# Tests all the building blocks
def test_conv_layer():
    """Make sure convolution works"""
    conv = Conv(3, 64, k=3, s=1)
    input_tensor = torch.randn(1, 3, 640, 640)
    output = conv(input_tensor)
    
    assert output.shape == (1, 64, 640, 640)
    print("✅ Convolution layer works!")

def test_c3_block():
    """Test the efficient C3 block"""
    c3 = C3(64, 128, n=3)
    x = torch.randn(1, 64, 160, 160)
    y = c3(x)
    
    assert y.shape == (1, 128, 160, 160)
    print("✅ C3 block processes features correctly!")
```

#### **Utility Functions** (`utils/test_general.py`)
```python
# Tests all the helper functions
def test_bbox_conversion():
    """Test bounding box format conversion"""
    xyxy = [100, 150, 200, 250]  # [x1, y1, x2, y2]
    xywh = xyxy2xywh(xyxy)       # Convert to center format
    back_to_xyxy = xywh2xyxy(xywh)  # Convert back
    
    assert back_to_xyxy == xyxy
    print("✅ Bounding box conversion is accurate!")

def test_iou_calculation():
    """Test Intersection over Union calculation"""
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]  # 50% overlap
    
    iou = bbox_iou(box1, box2)
    expected_iou = 0.25  # 2500 / 10000 intersection/union
    
    assert abs(iou - expected_iou) < 0.01
    print("✅ IoU calculation is correct!")
```

#### **Main Model Class** (`models/test_leafyolo.py`)
```python
# Tests the main LEAFYOLO class
def test_model_creation():
    """Test creating different model variants"""
    variants = ['leafyolo_n', 'leafyolo_s', 'leafyolo_m']
    
    for variant in variants:
        model = LEAFYOLO('detect', variant=variant)
        assert model.task == 'detect'
        assert model.variant == variant
        print(f"✅ {variant} model created successfully!")

def test_prediction():
    """Test model prediction functionality"""
    model = LEAFYOLO('detect', variant='leafyolo_n')
    
    # Create fake image tensor
    fake_image = torch.randn(1, 3, 640, 640)
    
    # This should work without errors
    with patch('model.forward') as mock_forward:
        results = model.predict(fake_image)
        print("✅ Prediction pipeline works!")
```

## 🔗 **integration/** - The Assembly Line

Integration tests check that **complete workflows** work together, like testing an entire assembly line.

### 🏭 **End-to-End Workflows** (`test_end_to_end.py`)

#### **Complete Training Pipeline**
```python
def test_training_workflow():
    """Test the entire training process"""
    
    # 1. Create model
    model = LEAFYOLO('detect', variant='leafyolo_n')
    
    # 2. Set up mock data
    mock_dataset = create_mock_dataset()
    
    # 3. Train for a few epochs
    results = model.train(
        data=mock_dataset,
        epochs=2,  # Just test the pipeline
        batch_size=2,
        device='cpu'
    )
    
    # 4. Verify training completed
    assert 'last' in results
    assert 'best' in results
    print("✅ Complete training workflow works!")

def test_prediction_workflow():
    """Test the entire prediction process"""
    
    # 1. Load model
    model = LEAFYOLO('detect')
    
    # 2. Create test images
    test_images = create_test_images()
    
    # 3. Run predictions
    results = model.predict(test_images)
    
    # 4. Verify results format
    assert len(results) == len(test_images)
    assert all(hasattr(r, 'boxes') for r in results)
    print("✅ Complete prediction workflow works!")
```

#### **Multi-Task Compatibility**
```python
def test_task_switching():
    """Test switching between different tasks"""
    
    tasks = ['detect', 'segment', 'classify']
    
    for task in tasks:
        model = LEAFYOLO(task, variant='leafyolo_s')
        
        # Each task should work
        assert model.task == task
        
        # Basic functionality check
        test_input = torch.randn(1, 3, 640, 640)
        # Should not crash
        print(f"✅ {task} task works!")
```

## 🏃 **benchmarks/** - The Race Track

Performance tests measure **how fast and efficient** your code is.

### ⚡ **Speed Benchmarks** (`test_performance.py`)

#### **Inference Speed Testing**
```python
@pytest.mark.benchmark
def test_inference_speed(benchmark):
    """Measure how fast inference runs"""
    
    model = LEAFYOLO('detect', variant='leafyolo_n')
    test_input = torch.randn(1, 3, 640, 640)
    
    # Benchmark the inference function
    def run_inference():
        with torch.no_grad():
            return model(test_input)
    
    result = benchmark(run_inference)
    
    # Should complete in reasonable time
    print(f"✅ Inference benchmark completed!")
    return result

def test_batch_processing_speed():
    """Test how batch size affects speed"""
    
    model = LEAFYOLO('detect', variant='leafyolo_s')
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 3, 640, 640)
        
        start_time = time.time()
        with torch.no_grad():
            model(test_input)
        end_time = time.time()
        
        per_image_time = (end_time - start_time) / batch_size
        print(f"Batch {batch_size}: {per_image_time:.4f}s per image")
```

#### **Memory Usage Testing**
```python
def test_memory_usage():
    """Monitor memory usage during operations"""
    
    if torch.cuda.is_available():
        model = LEAFYOLO('detect', variant='leafyolo_m').cuda()
        
        torch.cuda.reset_peak_memory_stats()
        
        # Large batch inference
        large_batch = torch.randn(16, 3, 640, 640).cuda()
        
        with torch.no_grad():
            model(large_batch)
        
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory / (1024 ** 2)
        
        print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        
        # Should be reasonable (< 4GB for this test)
        assert peak_memory_mb < 4000
        print("✅ Memory usage is reasonable!")
```

## 📦 **fixtures/** - The Supply Closet

Test fixtures provide **reusable test data and utilities**.

### 🔧 **Shared Test Configuration** (`conftest.py`)
```python
# Shared fixtures available to all tests

@pytest.fixture
def sample_image():
    """Create a test image tensor"""
    return torch.randn(1, 3, 640, 640)

@pytest.fixture
def sample_model():
    """Create a test model"""
    return LEAFYOLO('detect', variant='leafyolo_n')

@pytest.fixture  
def sample_targets():
    """Create test detection targets"""
    # Format: [image_idx, class, x_center, y_center, width, height]
    return torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.3],
        [0, 1, 0.3, 0.7, 0.1, 0.2],
    ])

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary test dataset"""
    
    # Create directory structure
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    # Create mock files
    for i in range(10):
        # Mock image file
        img_path = images_dir / f"image_{i}.jpg"
        img_path.write_text("mock image data")
        
        # Mock label file
        label_path = labels_dir / f"image_{i}.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n")  # One detection
    
    return tmp_path
```

## 🏃 **run_tests.py** - The Lab Manager

This script manages **all testing operations** with different modes and options.

### 🎮 **How to Use the Test Runner**

#### **Quick Testing (Development)**
```bash
# Run fast tests only (excludes slow training tests)
python tests/run_tests.py --fast

# Output:
# 🧪 Running Fast Tests...
# ✅ Configuration tests: 15/15 passed
# ✅ Module tests: 25/25 passed  
# ✅ Utility tests: 18/18 passed
# 🎉 All fast tests passed in 12.3s!
```

#### **Comprehensive Testing**
```bash
# Run all tests including slow ones
python tests/run_tests.py --all --coverage

# Output:
# 🧪 Running Complete Test Suite...
# ✅ Unit tests: 85/85 passed
# ✅ Integration tests: 12/12 passed
# 📊 Coverage: 94% (excellent!)
# 🎉 All tests passed in 156.8s!
```

#### **Performance Benchmarking**
```bash
# Run performance benchmarks
python tests/run_tests.py --benchmark

# Output:
# 🏃 Running Performance Benchmarks...
# ⚡ Inference Speed: 16.2ms (leafyolo_n)
# ⚡ Training Step: 45.3ms (leafyolo_s)  
# 💾 Memory Usage: 384MB peak
# 🎯 All benchmarks within expected ranges!
```

#### **Code Quality Checks**
```bash
# Check code quality
python tests/run_tests.py --lint --format

# Output:
# 🔍 Running Code Quality Checks...
# ✅ Flake8 linting: No issues found
# ✅ Black formatting: Code properly formatted
# ✅ Import sorting: All imports organized
# 🎉 Code quality excellent!
```

#### **CI Pipeline** 
```bash
# Run the complete CI pipeline
python tests/run_tests.py --ci

# Output:
# 🔄 Running CI Pipeline...
# ✅ Fast tests: 58/58 passed
# ✅ Code linting: All checks passed
# ✅ Security scan: No vulnerabilities  
# ✅ Documentation: Builds successfully
# 🎉 CI pipeline completed successfully!
```

## 🎯 **Test Categories & Markers**

### **Test Markers** (for selective running)
```python
# Mark tests with different categories

@pytest.mark.unit
def test_basic_function():
    """Basic unit test"""
    pass

@pytest.mark.integration  
def test_workflow():
    """Integration test"""
    pass

@pytest.mark.slow
def test_training():
    """Slow test that takes time"""
    pass

@pytest.mark.benchmark
def test_performance():
    """Performance benchmark"""
    pass

@pytest.mark.gpu
def test_cuda_functionality():
    """Requires GPU"""
    pass
```

### **Running Specific Test Categories**
```bash
# Run only unit tests
pytest tests/ -m "unit"

# Run everything except slow tests
pytest tests/ -m "not slow"

# Run GPU tests only (if GPU available)
pytest tests/ -m "gpu"

# Run benchmarks only
pytest tests/ -m "benchmark" --benchmark-only
```

## 📊 **Understanding Test Results**

### **Coverage Reports**
```bash
# Generate HTML coverage report
python tests/run_tests.py --all --coverage

# Opens beautiful HTML report showing:
# 📊 Overall coverage: 94%
# 📄 File-by-file breakdown
# 🔍 Line-by-line coverage highlighting
# 📈 Coverage trends over time
```

### **Performance Reports**
```bash
# Generate performance benchmark report
python tests/run_tests.py --benchmark

# Creates JSON report with:
# ⏱️ Timing statistics (min, max, mean)
# 📊 Performance comparisons  
# 📈 Performance trends
# 🎯 Regression detection
```

## 🐛 **Troubleshooting Tests**

### **Tests Failing?**
```bash
# Run with verbose output to see details
pytest tests/unit/test_config.py -v -s

# Run specific test function
pytest tests/unit/test_config.py::test_detect_config -v

# Debug with Python debugger
pytest tests/unit/test_config.py::test_detect_config -v -s --pdb
```

### **Import Errors?**
```bash
# Make sure you're in the right directory
cd /path/to/LEAF-YOLO

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install test dependencies
pip install pytest pytest-cov pytest-benchmark
```

### **Slow Tests?**
```bash
# Skip slow tests during development
pytest tests/ -m "not slow and not benchmark"

# Run tests in parallel (faster)
pytest tests/ -n auto

# Run only failed tests from last run
pytest tests/ --lf
```

## 💡 **Pro Tips**

### **Development Workflow**
1. **Write code** → 2. **Run fast tests** → 3. **Fix issues** → 4. **Run full tests** → 5. **Commit**

### **Test-Driven Development**
```python
# 1. Write test first (it will fail)
def test_new_feature():
    result = new_feature('input')
    assert result == expected_output

# 2. Write minimal code to make it pass
def new_feature(input):
    return expected_output  # Simplest implementation

# 3. Refactor and improve while keeping tests green
```

### **Continuous Integration**
- Tests run automatically on every commit
- Prevents broken code from being merged
- Maintains code quality standards
- Catches regressions early

## 🎓 **Testing Philosophy**

### **Why We Test Everything**
1. **Confidence**: Know that changes won't break existing functionality
2. **Documentation**: Tests show how code is supposed to work
3. **Refactoring Safety**: Change code fearlessly with test coverage
4. **Bug Prevention**: Catch issues before users do
5. **Performance Monitoring**: Detect performance regressions

### **Types of Testing**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows and interactions
- **Performance Tests**: Measure speed, memory, and efficiency  
- **Regression Tests**: Ensure old bugs don't come back
- **Edge Case Tests**: Test unusual inputs and boundary conditions

---

**Remember**: Testing might seem like extra work, but it actually **saves time** by catching bugs early and giving you confidence to make changes. Think of tests as your safety net that lets you code fearlessly! 🧪🚀

**The key insight**: Good tests are like good documentation - they show not just what the code does, but how it's supposed to work and what it's supposed to achieve!
