"""
Test cases for configuration system
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from leafyolo.utils.config import ConfigManager, get_config, get_dataset_config


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_init(self, temp_dir):
        """Test ConfigManager initialization."""
        config_path = temp_dir / "test_config.yaml"
        
        # Create test config
        test_config = {
            'nc': 80,
            'ch': 3,
            'task': 'detect',
            'variants': {
                'test_variant': {'depth_multiple': 1.0, 'width_multiple': 1.0}
            },
            'tasks': {
                'detect': {'head': 'head_detect', 'loss': 'DetectionLoss'}
            },
            'datasets': {
                'test_dataset': {'nc': 10, 'names': ['class1', 'class2']}
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Test initialization
        config_manager = ConfigManager(config_path)
        assert config_manager.base_config == test_config
        assert config_manager.config_path == config_path
    
    def test_get_config_detect(self, sample_config):
        """Test get_config for detection task."""
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            config = config_manager.get_config('detect', 'leafyolo_m')
            
            assert config['task'] == 'detect'
            assert config['nc'] == 80
            assert 'head' in config
            assert 'loss_type' in config
    
    def test_get_config_segment(self, sample_config):
        """Test get_config for segmentation task."""
        sample_config['tasks'] = {
            'segment': {'head': 'head_segment', 'loss': 'SegmentationLoss'}
        }
        sample_config['head_segment'] = [[-1, 1, 'Conv', [256, 3, 1]]]
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            config = config_manager.get_config('segment', 'leafyolo_s')
            
            assert config['task'] == 'segment'
            assert config['loss_type'] == 'SegmentationLoss'
    
    def test_get_config_classify(self, sample_config):
        """Test get_config for classification task."""
        sample_config['tasks'] = {
            'classify': {'head': 'head_classify', 'loss': 'CrossEntropyLoss'}
        }
        sample_config['head_classify'] = [[-1, 1, 'nn.AdaptiveAvgPool2d', [1]]]
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            config = config_manager.get_config('classify', 'leafyolo_n')
            
            assert config['task'] == 'classify'
            assert config['loss_type'] == 'CrossEntropyLoss'
    
    def test_get_config_with_overrides(self, sample_config):
        """Test get_config with parameter overrides."""
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            config = config_manager.get_config('detect', 'leafyolo_m', nc=91, custom_param='test')
            
            assert config['nc'] == 91
            assert config['custom_param'] == 'test'
    
    def test_get_dataset_config(self, sample_config):
        """Test get_dataset_config method."""
        sample_config['datasets'] = {
            'test_dataset': {
                'nc': 10,
                'names': ['class1', 'class2'],
                'train': 'train.txt',
                'val': 'val.txt'
            }
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            dataset_config = config_manager.get_dataset_config('test_dataset')
            
            assert dataset_config['nc'] == 10
            assert len(dataset_config['names']) == 2
            assert dataset_config['train'] == 'train.txt'
    
    def test_get_hyperparameters_detect(self, sample_config):
        """Test get_hyperparameters for detection."""
        sample_config['hyp'] = {
            'lr0': 0.01,
            'box': 0.05,
            'cls': 0.5,
            'obj': 1.0,
            'anchor_t': 4.0
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            hyp = config_manager.get_hyperparameters('detect')
            
            assert 'box' in hyp
            assert 'cls' in hyp
            assert 'obj' in hyp
            assert 'anchor_t' in hyp
    
    def test_get_hyperparameters_classify(self, sample_config):
        """Test get_hyperparameters for classification."""
        sample_config['hyp'] = {
            'lr0': 0.01,
            'box': 0.05,  # Should be removed for classification
            'cls': 0.5,   # Should be removed for classification
            'obj': 1.0,   # Should be removed for classification
            'anchor_t': 4.0  # Should be removed for classification
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            hyp = config_manager.get_hyperparameters('classify')
            
            assert 'lr0' in hyp
            assert 'box' not in hyp
            assert 'cls' not in hyp
            assert 'obj' not in hyp
            assert 'anchor_t' not in hyp
    
    def test_validate_config_missing_keys(self, sample_config):
        """Test config validation with missing keys."""
        del sample_config['nc']  # Remove required key
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            
            with pytest.raises(ValueError, match="Missing required configuration key"):
                config_manager.get_config('detect', 'leafyolo_m')
    
    def test_validate_config_detection_no_anchors(self, sample_config):
        """Test config validation for detection without anchors."""
        del sample_config['anchors']  # Remove anchors
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            
            with pytest.raises(ValueError, match="Detection task requires 'anchors'"):
                config_manager.get_config('detect', 'leafyolo_m')
    
    def test_list_variants(self, sample_config):
        """Test list_variants method."""
        sample_config['variants'] = {
            'variant1': {'depth_multiple': 0.5},
            'variant2': {'depth_multiple': 1.0},
            'variant3': {'depth_multiple': 1.5}
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            variants = config_manager.list_variants()
            
            assert len(variants) == 3
            assert 'variant1' in variants
            assert 'variant2' in variants
            assert 'variant3' in variants
    
    def test_list_datasets(self, sample_config):
        """Test list_datasets method."""
        sample_config['datasets'] = {
            'dataset1': {'nc': 10},
            'dataset2': {'nc': 20},
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            datasets = config_manager.list_datasets()
            
            assert len(datasets) == 2
            assert 'dataset1' in datasets
            assert 'dataset2' in datasets
    
    def test_list_tasks(self, sample_config):
        """Test list_tasks method."""
        sample_config['tasks'] = {
            'detect': {},
            'segment': {},
            'classify': {}
        }
        
        with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
            config_manager = ConfigManager()
            tasks = config_manager.list_tasks()
            
            assert len(tasks) == 3
            assert 'detect' in tasks
            assert 'segment' in tasks
            assert 'classify' in tasks


class TestConfigFunctions:
    """Test module-level config functions."""
    
    def test_get_config_function(self, sample_config):
        """Test get_config convenience function."""
        with patch('leafyolo.utils.config.config_manager') as mock_manager:
            mock_manager.get_config.return_value = sample_config
            
            result = get_config('detect', 'leafyolo_m', nc=80)
            
            mock_manager.get_config.assert_called_once_with('detect', 'leafyolo_m', nc=80)
            assert result == sample_config
    
    def test_get_dataset_config_function(self):
        """Test get_dataset_config convenience function."""
        expected_dataset = {'nc': 10, 'names': ['class1', 'class2']}
        
        with patch('leafyolo.utils.config.config_manager') as mock_manager:
            mock_manager.get_dataset_config.return_value = expected_dataset
            
            result = get_dataset_config('test_dataset')
            
            mock_manager.get_dataset_config.assert_called_once_with('test_dataset')
            assert result == expected_dataset


@pytest.mark.parametrize("task,variant,expected_task", [
    ('detect', 'leafyolo_n', 'detect'),
    ('segment', 'leafyolo_s', 'segment'), 
    ('classify', 'leafyolo_m', 'classify'),
])
def test_config_parametrized(task, variant, expected_task, sample_config):
    """Parametrized test for different task/variant combinations."""
    sample_config['tasks'] = {
        'detect': {'head': 'head_detect'},
        'segment': {'head': 'head_segment'},
        'classify': {'head': 'head_classify'}
    }
    
    with patch('leafyolo.utils.config.ConfigManager._load_config', return_value=sample_config):
        config_manager = ConfigManager()
        config = config_manager.get_config(task, variant)
        
        assert config['task'] == expected_task
