"""
Test cases for general utility functions
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from leafyolo.utils.general import (
    make_divisible, check_img_size, non_max_suppression,
    scale_coords, xyxy2xywh, xywh2xyxy, clip_coords,
    bbox_iou, box_iou, colorstr, increment_path
)


class TestMathUtils:
    """Test mathematical utility functions."""
    
    @pytest.mark.parametrize("x,divisor,expected", [
        (32, 8, 32),
        (30, 8, 32),
        (25, 8, 32),
        (64, 16, 64),
        (60, 16, 64),
        (100, 32, 96),
    ])
    def test_make_divisible(self, x, divisor, expected):
        """Test make_divisible function."""
        result = make_divisible(x, divisor)
        assert result == expected
        assert result % divisor == 0
    
    def test_make_divisible_edge_cases(self):
        """Test make_divisible edge cases."""
        # Zero input
        assert make_divisible(0, 8) == 0
        
        # Negative input should still work
        assert make_divisible(-10, 8) == -8
        
        # Divisor of 1
        assert make_divisible(37, 1) == 37


class TestImageUtils:
    """Test image utility functions."""
    
    def test_check_img_size_valid(self):
        """Test check_img_size with valid sizes."""
        # Single size
        result = check_img_size(640, 32)
        assert result == 640
        
        # Already divisible
        result = check_img_size(608, 32)
        assert result == 608
    
    def test_check_img_size_adjustment(self):
        """Test check_img_size with size adjustment needed."""
        # Size that needs adjustment
        result = check_img_size(650, 32)
        assert result == 672  # Next multiple of 32
        assert result % 32 == 0
    
    def test_check_img_size_warning(self):
        """Test check_img_size warning for adjustments."""
        with patch('leafyolo.utils.general.print') as mock_print:
            result = check_img_size(650, 32)
            mock_print.assert_called()  # Should print warning


class TestBboxUtils:
    """Test bounding box utility functions."""
    
    def test_xyxy2xywh_single_box(self):
        """Test xyxy2xywh conversion for single box."""
        xyxy = torch.tensor([100, 150, 200, 250])  # x1, y1, x2, y2
        xywh = xyxy2xywh(xyxy)
        
        expected = torch.tensor([150, 200, 100, 100])  # x_center, y_center, width, height
        assert torch.allclose(xywh, expected)
    
    def test_xyxy2xywh_multiple_boxes(self):
        """Test xyxy2xywh conversion for multiple boxes."""
        xyxy = torch.tensor([
            [100, 150, 200, 250],
            [0, 0, 50, 100],
            [300, 400, 500, 600]
        ])
        xywh = xyxy2xywh(xyxy)
        
        expected = torch.tensor([
            [150, 200, 100, 100],
            [25, 50, 50, 100], 
            [400, 500, 200, 200]
        ])
        assert torch.allclose(xywh, expected)
    
    def test_xywh2xyxy_single_box(self):
        """Test xywh2xyxy conversion for single box."""
        xywh = torch.tensor([150, 200, 100, 100])  # x_center, y_center, width, height
        xyxy = xywh2xyxy(xywh)
        
        expected = torch.tensor([100, 150, 200, 250])  # x1, y1, x2, y2
        assert torch.allclose(xyxy, expected)
    
    def test_xywh2xyxy_multiple_boxes(self):
        """Test xywh2xyxy conversion for multiple boxes."""
        xywh = torch.tensor([
            [150, 200, 100, 100],
            [25, 50, 50, 100],
            [400, 500, 200, 200]
        ])
        xyxy = xywh2xyxy(xywh)
        
        expected = torch.tensor([
            [100, 150, 200, 250],
            [0, 0, 50, 100],
            [300, 400, 500, 600]
        ])
        assert torch.allclose(xyxy, expected)
    
    def test_bbox_conversion_consistency(self):
        """Test that bbox conversions are consistent."""
        original_xyxy = torch.tensor([
            [100, 150, 200, 250],
            [0, 0, 50, 100]
        ])
        
        # Convert xyxy -> xywh -> xyxy
        xywh = xyxy2xywh(original_xyxy)
        converted_xyxy = xywh2xyxy(xywh)
        
        assert torch.allclose(original_xyxy, converted_xyxy)
    
    def test_clip_coords(self):
        """Test coordinate clipping function."""
        boxes = torch.tensor([
            [-10, -20, 50, 100],   # Negative coordinates
            [600, 700, 800, 900], # Coordinates beyond image
            [100, 150, 200, 250]  # Valid coordinates
        ])
        
        img_shape = (640, 640)  # height, width
        clipped = clip_coords(boxes, img_shape)
        
        expected = torch.tensor([
            [0, 0, 50, 100],       # Clipped negatives
            [600, 640, 640, 640], # Clipped to image bounds
            [100, 150, 200, 250]  # Unchanged valid coords
        ])
        
        assert torch.allclose(clipped, expected)
    
    def test_scale_coords(self):
        """Test coordinate scaling function."""
        coords = torch.tensor([
            [32, 48, 64, 96],    # Coordinates in 128x128 image
            [16, 24, 48, 72]
        ])
        
        img1_shape = (128, 128)  # Original image shape
        img0_shape = (640, 640)  # Target image shape
        
        scaled = scale_coords(img1_shape, coords, img0_shape)
        
        # Should scale by factor of 5 (640/128)
        expected = torch.tensor([
            [160, 240, 320, 480],
            [80, 120, 240, 360]
        ])
        
        assert torch.allclose(scaled, expected)


class TestIoUFunctions:
    """Test IoU calculation functions."""
    
    def test_bbox_iou_perfect_overlap(self):
        """Test IoU calculation for perfect overlap."""
        box1 = torch.tensor([[100, 100, 200, 200]])
        box2 = torch.tensor([[100, 100, 200, 200]])
        
        iou = bbox_iou(box1, box2)
        assert torch.allclose(iou, torch.tensor([[1.0]]))
    
    def test_bbox_iou_no_overlap(self):
        """Test IoU calculation for no overlap."""
        box1 = torch.tensor([[0, 0, 50, 50]])
        box2 = torch.tensor([[100, 100, 150, 150]])
        
        iou = bbox_iou(box1, box2)
        assert torch.allclose(iou, torch.tensor([[0.0]]))
    
    def test_bbox_iou_partial_overlap(self):
        """Test IoU calculation for partial overlap."""
        box1 = torch.tensor([[0, 0, 100, 100]])
        box2 = torch.tensor([[50, 50, 150, 150]])
        
        iou = bbox_iou(box1, box2)
        
        # Intersection: 50x50 = 2500
        # Union: 100x100 + 100x100 - 2500 = 17500
        # IoU: 2500 / 17500 = 1/7 ≈ 0.1429
        expected_iou = 2500.0 / 17500.0
        assert torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-4)
    
    def test_box_iou_multiple_boxes(self):
        """Test box_iou with multiple boxes."""
        boxes1 = torch.tensor([
            [0, 0, 50, 50],
            [100, 100, 150, 150]
        ])
        boxes2 = torch.tensor([
            [25, 25, 75, 75],
            [125, 125, 175, 175]
        ])
        
        iou_matrix = box_iou(boxes1, boxes2)
        
        # Should return 2x2 matrix
        assert iou_matrix.shape == (2, 2)
        
        # Check diagonal elements (overlapping boxes)
        assert iou_matrix[0, 0] > 0  # Box1[0] overlaps with Box2[0]
        assert iou_matrix[1, 1] > 0  # Box1[1] overlaps with Box2[1]
        
        # Check off-diagonal elements (non-overlapping)
        assert iou_matrix[0, 1] == 0  # Box1[0] doesn't overlap Box2[1]
        assert iou_matrix[1, 0] == 0  # Box1[1] doesn't overlap Box2[0]


class TestNMS:
    """Test Non-Maximum Suppression."""
    
    def test_nms_basic(self):
        """Test basic NMS functionality."""
        predictions = torch.tensor([
            [100, 100, 200, 200, 0.9, 0],  # High confidence box
            [110, 110, 210, 210, 0.8, 0],  # Overlapping lower confidence
            [300, 300, 400, 400, 0.7, 1],  # Different class
        ])
        
        # Reshape to match expected format [batch, detections, 6]
        predictions = predictions.unsqueeze(0)
        
        result = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.5)
        
        assert len(result) == 1  # One batch
        assert len(result[0]) <= 2  # Should remove overlapping box
    
    def test_nms_confidence_filtering(self):
        """Test NMS confidence threshold filtering."""
        predictions = torch.tensor([
            [100, 100, 200, 200, 0.9, 0],  # Above threshold
            [300, 300, 400, 400, 0.3, 1],  # Below threshold
            [500, 500, 600, 600, 0.7, 2],  # Above threshold
        ])
        
        predictions = predictions.unsqueeze(0)
        result = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.5)
        
        assert len(result[0]) == 2  # Only 2 boxes above confidence threshold
    
    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        empty_predictions = torch.empty(1, 0, 6)
        result = non_max_suppression(empty_predictions, conf_thres=0.5, iou_thres=0.5)
        
        assert len(result) == 1
        assert len(result[0]) == 0


class TestStringUtils:
    """Test string utility functions."""
    
    def test_colorstr_basic(self):
        """Test basic colorstr functionality."""
        # Test different colors
        red_str = colorstr('red', 'test message')
        assert isinstance(red_str, str)
        assert 'test message' in red_str
        
        blue_str = colorstr('blue', 'another message') 
        assert isinstance(blue_str, str)
        assert 'another message' in blue_str
    
    def test_colorstr_invalid_color(self):
        """Test colorstr with invalid color."""
        # Should handle invalid colors gracefully
        result = colorstr('invalid_color', 'test')
        assert 'test' in result


class TestPathUtils:
    """Test path utility functions."""
    
    def test_increment_path_new_dir(self, temp_dir):
        """Test increment_path with new directory."""
        base_path = temp_dir / "exp"
        result = increment_path(base_path)
        
        # Should return original path if it doesn't exist
        assert result == base_path
    
    def test_increment_path_existing_dir(self, temp_dir):
        """Test increment_path with existing directories."""
        base_path = temp_dir / "exp"
        
        # Create existing directories
        base_path.mkdir()
        (temp_dir / "exp1").mkdir()
        (temp_dir / "exp2").mkdir()
        
        result = increment_path(base_path)
        
        # Should increment to next available number
        assert result == temp_dir / "exp3"
    
    def test_increment_path_exist_ok(self, temp_dir):
        """Test increment_path with exist_ok=True."""
        base_path = temp_dir / "exp"
        base_path.mkdir()
        
        result = increment_path(base_path, exist_ok=True)
        
        # Should return original path when exist_ok=True
        assert result == base_path


@pytest.mark.parametrize("input_val,expected", [
    (32, True),
    (64, True),
    (33, False),
    (0, True),
    (-32, True),
])
def test_is_power_of_two(input_val, expected):
    """Parametrized test for power of two checking."""
    # This would test a hypothetical is_power_of_two function
    # Replace with actual function if it exists
    pass
