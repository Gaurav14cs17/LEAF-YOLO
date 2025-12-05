"""
Test cases for LEAFYOLO main model class
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

from leafyolo.models.leafyolo import LEAFYOLO, Model, attempt_load


class TestLEAFYOLOInit:
    """Test LEAFYOLO model initialization."""
    
    def test_init_with_task_name(self):
        """Test initialization with task name."""
        with patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
            model = LEAFYOLO('detect', variant='leafyolo_n')
            
            assert model.task == 'detect'
            assert model.variant == 'leafyolo_n'
            mock_model.assert_called_once()
    
    def test_init_with_config_file(self, temp_dir):
        """Test initialization with config file."""
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text("""
        nc: 80
        ch: 3
        backbone: []
        head: []
        """)
        
        with patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
            model = LEAFYOLO(str(config_file), task='detect')
            
            mock_model.assert_called_once()
            assert model.task == 'detect'
    
    def test_init_with_weights(self, test_weights_path):
        """Test initialization with pretrained weights."""
        with patch('torch.load') as mock_load:
            mock_load.return_value = {'model': MagicMock()}
            
            model = LEAFYOLO(str(test_weights_path))
            
            mock_load.assert_called_once()
            assert model.model is not None
    
    def test_init_with_detection_model(self, mock_model):
        """Test initialization with DetectionModel instance."""
        model = LEAFYOLO(mock_model)
        
        assert model.model == mock_model
        assert model.task == 'detect'  # Default task
    
    def test_task_detection_from_model_spec(self):
        """Test automatic task detection from model specification."""
        test_cases = [
            ('detect', 'detect'),
            ('segment', 'segment'),
            ('classify', 'classify'),
            ('model_segment.yaml', 'segment'),
            ('model_cls.yaml', 'classify'),
            ('custom_model.yaml', 'detect'),  # default
        ]
        
        for model_spec, expected_task in test_cases:
            with patch('leafyolo.models.leafyolo.DetectionModel'):
                model = LEAFYOLO(model_spec)
                assert model.task == expected_task


class TestLEAFYOLOTraining:
    """Test LEAFYOLO training functionality."""
    
    def test_train_method(self, mock_model):
        """Test train method."""
        with patch('leafyolo.models.leafyolo.LeafTrainer') as mock_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {'best': 'path/to/best.pt'}
            mock_trainer.return_value = mock_trainer_instance
            
            model = LEAFYOLO(mock_model)
            result = model.train(data='coco', epochs=100)
            
            mock_trainer.assert_called_once_with(model=mock_model)
            mock_trainer_instance.train.assert_called_once_with(
                data='coco', epochs=100
            )
            assert 'best' in result
    
    def test_train_with_custom_params(self, mock_model):
        """Test training with custom parameters."""
        with patch('leafyolo.models.leafyolo.LeafTrainer') as mock_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            
            model = LEAFYOLO(mock_model)
            model.train(
                data='custom.yaml',
                epochs=300,
                batch_size=32,
                device='cuda:0',
                lr0=0.01
            )
            
            mock_trainer_instance.train.assert_called_once_with(
                data='custom.yaml',
                epochs=300,
                batch_size=32,
                device='cuda:0',
                lr0=0.01
            )


class TestLEAFYOLOPrediction:
    """Test LEAFYOLO prediction functionality."""
    
    def test_predict_method(self, mock_model):
        """Test predict method."""
        with patch('leafyolo.models.leafyolo.LeafPredictor') as mock_predictor:
            mock_predictor_instance = MagicMock()
            mock_predictor_instance.predict.return_value = [
                {'path': 'image1.jpg', 'detections': []},
                {'path': 'image2.jpg', 'detections': []}
            ]
            mock_predictor.return_value = mock_predictor_instance
            
            model = LEAFYOLO(mock_model)
            results = model.predict('path/to/images')
            
            mock_predictor.assert_called_once_with(model=mock_model)
            mock_predictor_instance.predict.assert_called_once_with(
                source='path/to/images'
            )
            assert len(results) == 2
    
    def test_predict_with_params(self, mock_model):
        """Test prediction with custom parameters."""
        with patch('leafyolo.models.leafyolo.LeafPredictor') as mock_predictor:
            mock_predictor_instance = MagicMock()
            mock_predictor.return_value = mock_predictor_instance
            
            model = LEAFYOLO(mock_model)
            model.predict(
                source='test.jpg',
                conf_thres=0.3,
                iou_thres=0.5,
                img_size=1280,
                save=True
            )
            
            mock_predictor_instance.predict.assert_called_once_with(
                source='test.jpg',
                conf_thres=0.3,
                iou_thres=0.5,
                img_size=1280,
                save=True
            )
    
    def test_predict_call_shortcut(self, mock_model):
        """Test __call__ method as predict shortcut."""
        with patch('leafyolo.models.leafyolo.LeafPredictor') as mock_predictor:
            mock_predictor_instance = MagicMock()
            mock_predictor.return_value = mock_predictor_instance
            
            model = LEAFYOLO(mock_model)
            model('test_image.jpg')  # Using __call__
            
            mock_predictor_instance.predict.assert_called_once_with(
                source='test_image.jpg'
            )


class TestLEAFYOLOValidation:
    """Test LEAFYOLO validation functionality."""
    
    def test_val_method(self, mock_model):
        """Test val method."""
        with patch('leafyolo.models.leafyolo.LeafValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate.return_value = {
                'map50': 0.65,
                'map': 0.45,
                'precision': 0.7,
                'recall': 0.6
            }
            mock_validator.return_value = mock_validator_instance
            
            model = LEAFYOLO(mock_model)
            results = model.val(data='coco')
            
            mock_validator.assert_called_once_with(model=mock_model)
            mock_validator_instance.validate.assert_called_once_with(data='coco')
            
            assert results['map50'] == 0.65
            assert results['map'] == 0.45
    
    def test_val_with_params(self, mock_model):
        """Test validation with custom parameters."""
        with patch('leafyolo.models.leafyolo.LeafValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator.return_value = mock_validator_instance
            
            model = LEAFYOLO(mock_model)
            model.val(
                data='custom.yaml',
                batch_size=64,
                img_size=1280,
                conf_thres=0.001,
                iou_thres=0.6
            )
            
            mock_validator_instance.validate.assert_called_once_with(
                data='custom.yaml',
                batch_size=64,
                img_size=1280,
                conf_thres=0.001,
                iou_thres=0.6
            )


class TestLEAFYOLOExport:
    """Test LEAFYOLO export functionality."""
    
    def test_export_method(self, mock_model):
        """Test export method."""
        with patch('leafyolo.engine.exporter.LeafExporter') as mock_exporter:
            mock_exporter_instance = MagicMock()
            mock_exporter_instance.export.return_value = 'model.onnx'
            mock_exporter.return_value = mock_exporter_instance
            
            model = LEAFYOLO(mock_model)
            result = model.export(format='onnx')
            
            mock_exporter.assert_called_once_with(model=mock_model)
            mock_exporter_instance.export.assert_called_once_with(format='onnx')
            assert result == 'model.onnx'
    
    def test_export_with_params(self, mock_model):
        """Test export with custom parameters."""
        with patch('leafyolo.engine.exporter.LeafExporter') as mock_exporter:
            mock_exporter_instance = MagicMock()
            mock_exporter.return_value = mock_exporter_instance
            
            model = LEAFYOLO(mock_model)
            model.export(
                format='tensorrt',
                img_size=1280,
                dynamic=True,
                simplify=True
            )
            
            mock_exporter_instance.export.assert_called_once_with(
                format='tensorrt',
                img_size=1280,
                dynamic=True,
                simplify=True
            )


class TestLEAFYOLOUtilities:
    """Test LEAFYOLO utility methods."""
    
    def test_info_method(self, mock_model):
        """Test info method."""
        mock_model.info = MagicMock()
        
        model = LEAFYOLO(mock_model)
        model.info(detailed=True)
        
        mock_model.info.assert_called_once_with(detailed=True)
    
    def test_fuse_method(self, mock_model):
        """Test fuse method."""
        mock_model.fuse = MagicMock(return_value=mock_model)
        
        model = LEAFYOLO(mock_model)
        result = model.fuse()
        
        mock_model.fuse.assert_called_once()
        assert result == model  # Should return self
    
    def test_getattr_delegation(self, mock_model):
        """Test attribute delegation to underlying model."""
        mock_model.some_attribute = 'test_value'
        mock_model.some_method = MagicMock(return_value='method_result')
        
        model = LEAFYOLO(mock_model)
        
        # Test attribute access
        assert model.some_attribute == 'test_value'
        
        # Test method access
        result = model.some_method('arg1', 'arg2')
        mock_model.some_method.assert_called_once_with('arg1', 'arg2')
        assert result == 'method_result'
    
    def test_getattr_attribute_error(self, mock_model):
        """Test AttributeError for non-existent attributes."""
        model = LEAFYOLO(mock_model)
        
        with pytest.raises(AttributeError, match="'LEAFYOLO' object has no attribute 'non_existent'"):
            _ = model.non_existent


class TestLEAFYOLOConfigIntegration:
    """Test LEAFYOLO integration with config system."""
    
    def test_adaptive_config_detect(self):
        """Test adaptive configuration for detection."""
        with patch('leafyolo.models.leafyolo.get_config') as mock_get_config, \
             patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
            
            mock_get_config.return_value = {'task': 'detect', 'nc': 80}
            
            model = LEAFYOLO('detect', variant='leafyolo_m', nc=91)
            
            mock_get_config.assert_called_once_with(task='detect', variant='leafyolo_m', nc=91)
            mock_model.assert_called_once()
    
    def test_adaptive_config_segment(self):
        """Test adaptive configuration for segmentation."""
        with patch('leafyolo.models.leafyolo.get_config') as mock_get_config, \
             patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
            
            mock_get_config.return_value = {'task': 'segment', 'nc': 91}
            
            model = LEAFYOLO('segment', variant='leafyolo_s')
            
            mock_get_config.assert_called_once_with(task='segment', variant='leafyolo_s')
            assert model.task == 'segment'
    
    def test_adaptive_config_classify(self):
        """Test adaptive configuration for classification."""
        with patch('leafyolo.models.leafyolo.get_config') as mock_get_config, \
             patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
            
            mock_get_config.return_value = {'task': 'classify', 'nc': 1000}
            
            model = LEAFYOLO('classify', variant='leafyolo_n')
            
            mock_get_config.assert_called_once_with(task='classify', variant='leafyolo_n')
            assert model.task == 'classify'


class TestLegacyCompatibility:
    """Test legacy compatibility functions."""
    
    def test_model_alias(self, mock_model):
        """Test Model alias for LEAFYOLO."""
        model = Model(mock_model)
        assert isinstance(model, LEAFYOLO)
    
    def test_attempt_load_function(self, test_weights_path):
        """Test attempt_load function."""
        with patch('torch.load') as mock_load:
            mock_load.return_value = {'model': MagicMock()}
            
            model = attempt_load(str(test_weights_path))
            
            assert isinstance(model, LEAFYOLO)
            mock_load.assert_called_once()


@pytest.mark.parametrize("task,variant,expected_task", [
    ('detect', 'leafyolo_n', 'detect'),
    ('segment', 'leafyolo_s', 'segment'),
    ('classify', 'leafyolo_m', 'classify'),
    ('detect', 'leafyolo_l', 'detect'),
    ('segment', 'leafyolo_x', 'segment'),
])
def test_leafyolo_parametrized(task, variant, expected_task):
    """Parametrized test for different LEAFYOLO configurations."""
    with patch('leafyolo.models.leafyolo.DetectionModel') as mock_model:
        model = LEAFYOLO(task, variant=variant)
        
        assert model.task == expected_task
        assert model.variant == variant
        mock_model.assert_called_once()


class TestLEAFYOLOErrorHandling:
    """Test LEAFYOLO error handling."""
    
    def test_invalid_task(self):
        """Test initialization with invalid task."""
        with patch('leafyolo.models.leafyolo.get_config') as mock_get_config:
            mock_get_config.side_effect = ValueError("Invalid task")
            
            with pytest.raises(ValueError):
                LEAFYOLO('invalid_task')
    
    def test_missing_weights_file(self):
        """Test loading non-existent weights file."""
        with patch('torch.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                LEAFYOLO('non_existent.pt')
    
    def test_corrupt_weights_file(self):
        """Test loading corrupt weights file."""
        with patch('torch.load') as mock_load:
            mock_load.side_effect = RuntimeError("Corrupt file")
            
            with pytest.raises(RuntimeError):
                LEAFYOLO('corrupt.pt')
