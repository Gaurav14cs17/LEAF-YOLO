"""
pytest configuration and shared fixtures for LEAF-YOLO tests
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock

# Test configuration
torch.manual_seed(42)
np.random.seed(42)

# Test data paths
TEST_ROOT = Path(__file__).parent
FIXTURES_DIR = TEST_ROOT / "fixtures"
TEST_DATA_DIR = FIXTURES_DIR / "data"


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_image_tensor():
    """Create test image tensor."""
    return torch.randn(1, 3, 640, 640)


@pytest.fixture(scope="session")
def test_batch_tensor():
    """Create test batch tensor."""
    return torch.randn(4, 3, 640, 640)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'task': 'detect',
        'nc': 80,
        'ch': 3,
        'depth_multiple': 1.0,
        'width_multiple': 1.0,
        'anchors': [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]],
        'backbone': [
            [-1, 1, 'Conv', [64, 6, 2, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
        ],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]],
        ]
    }


@pytest.fixture
def sample_targets():
    """Sample targets for detection training."""
    # Format: [image_idx, class, x_center, y_center, width, height]
    return torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.3],
        [0, 1, 0.3, 0.7, 0.1, 0.2],
        [1, 0, 0.6, 0.4, 0.15, 0.25],
    ])


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(return_value=(
        torch.randn(3, 640, 640),  # image
        torch.tensor([[0, 0.5, 0.5, 0.2, 0.3]]),  # targets
        "test_path.jpg",  # path
        (640, 640)  # shapes
    ))
    return dataset


@pytest.fixture
def mock_model():
    """Mock LEAF-YOLO model for testing."""
    from leafyolo.nn.tasks.detect import DetectionModel
    
    model = MagicMock(spec=DetectionModel)
    model.names = {i: f'class_{i}' for i in range(80)}
    model.nc = 80
    model.stride = torch.tensor([8, 16, 32])
    
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0]
        outputs = []
        for stride in [8, 16, 32]:
            h, w = 640 // stride, 640 // stride
            output = torch.randn(batch_size, 3, 85, h, w)  # 3 anchors, 85 = 80 classes + 5
            outputs.append(output)
        return outputs
    
    model.forward = mock_forward
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    
    return model


@pytest.fixture
def sample_detection_output():
    """Sample detection output for testing."""
    # Format: [x1, y1, x2, y2, confidence, class]
    return torch.tensor([
        [100, 100, 200, 200, 0.9, 0],
        [300, 150, 400, 250, 0.8, 1],
        [50, 300, 150, 400, 0.7, 0],
    ])


@pytest.fixture(scope="session")
def test_weights_path():
    """Path to test model weights."""
    weights_path = FIXTURES_DIR / "test_model.pt"
    if not weights_path.exists():
        # Create dummy weights file
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': MagicMock(),
            'epoch': 100,
            'best_fitness': 0.5
        }, weights_path)
    return weights_path


@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for training."""
    return {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 0.05,
        'cls': 0.5,
        'obj': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
    }


# Test utility functions
def assert_tensor_equal(a, b, rtol=1e-5, atol=1e-8):
    """Assert two tensors are equal within tolerance."""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"Tensors not equal: {a} vs {b}"


def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def create_test_image(size=(640, 640), channels=3, batch_size=1):
    """Create test image tensor."""
    if batch_size == 1:
        return torch.randn(channels, size[0], size[1])
    else:
        return torch.randn(batch_size, channels, size[0], size[1])


def create_test_targets(num_objects=3, num_classes=80):
    """Create test detection targets."""
    targets = []
    for i in range(num_objects):
        # [image_idx, class, x_center, y_center, width, height]
        target = [
            0,  # image index
            torch.randint(0, num_classes, (1,)).item(),  # class
            torch.rand(1).item(),  # x_center
            torch.rand(1).item(),  # y_center
            torch.rand(1).item() * 0.3 + 0.1,  # width
            torch.rand(1).item() * 0.3 + 0.1,  # height
        ]
        targets.append(target)
    return torch.tensor(targets)
