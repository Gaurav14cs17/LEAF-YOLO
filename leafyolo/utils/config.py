"""
LEAF-YOLO Configuration Manager
Handles adaptive configuration for different tasks
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy


class ConfigManager:
    """
    Adaptive configuration manager for LEAF-YOLO.
    Automatically adjusts configuration based on task type.
    """
    
    def __init__(self, config_path: Union[str, Path] = 'leafyolo.yaml'):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.base_config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML file."""
        if not self.config_path.exists():
            # Try to find config in project root or current directory
            possible_paths = [
                Path.cwd() / 'leafyolo.yaml',
                Path(__file__).parent.parent.parent / 'leafyolo.yaml',
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.config_path = path
                    break
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_config(self, task: str = 'detect', variant: str = 'leafyolo_m', **overrides) -> Dict[str, Any]:
        """
        Get task-specific configuration.
        
        Args:
            task: Task type ('detect', 'segment', 'classify')
            variant: Model variant ('leafyolo_n', 'leafyolo_s', etc.)
            **overrides: Additional configuration overrides
            
        Returns:
            Complete configuration dictionary for the task
        """
        config = deepcopy(self.base_config)
        
        # Set task
        config['task'] = task
        
        # Apply model variant
        if variant in config.get('variants', {}):
            variant_config = config['variants'][variant]
            config.update(variant_config)
        
        # Get task-specific configuration
        task_config = config.get('tasks', {}).get(task, {})
        
        # Set appropriate head architecture
        head_key = task_config.get('head', f'head_{task}')
        if head_key in config:
            config['head'] = config[head_key]
        else:
            # Default to detection head
            config['head'] = config.get('head_detect', [])
        
        # Set task-specific parameters
        config['loss_type'] = task_config.get('loss', 'DetectionLoss')
        config['metrics'] = task_config.get('metrics', ['mAP50'])
        config['export_formats'] = task_config.get('export_formats', ['onnx'])
        
        # Apply overrides
        config.update(overrides)
        
        # Validate configuration
        self._validate_config(config, task)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any], task: str):
        """Validate configuration for specific task."""
        required_keys = ['nc', 'ch', 'backbone', 'head']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Task-specific validation
        if task == 'detect':
            if 'anchors' not in config:
                raise ValueError("Detection task requires 'anchors' configuration")
        elif task == 'classify':
            if config['nc'] < 2:
                raise ValueError("Classification requires at least 2 classes")
        
        # Validate head architecture
        if not isinstance(config['head'], list):
            raise ValueError("Head configuration must be a list")
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get dataset-specific configuration.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'coco', 'visdrone')
            
        Returns:
            Dataset configuration dictionary
        """
        datasets = self.base_config.get('datasets', {})
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        return datasets[dataset_name]
    
    def get_hyperparameters(self, task: str = 'detect') -> Dict[str, Any]:
        """
        Get task-specific hyperparameters.
        
        Args:
            task: Task type
            
        Returns:
            Hyperparameters dictionary
        """
        hyp = deepcopy(self.base_config.get('hyp', {}))
        
        # Task-specific hyperparameter adjustments
        if task == 'classify':
            # Remove detection-specific hyperparameters
            detection_keys = ['box', 'cls', 'obj', 'anchor_t', 'iou_t']
            for key in detection_keys:
                hyp.pop(key, None)
        elif task == 'segment':
            # Add segmentation-specific hyperparameters
            hyp['mask'] = hyp.get('mask', 1.0)  # mask loss gain
        
        return hyp
    
    def create_model_config(self, task: str = 'detect', variant: str = 'leafyolo_m', 
                           nc: Optional[int] = None, **kwargs) -> str:
        """
        Create a temporary model configuration file for specific task.
        
        Args:
            task: Task type
            variant: Model variant
            nc: Number of classes (overrides config)
            **kwargs: Additional overrides
            
        Returns:
            Path to temporary configuration file
        """
        config = self.get_config(task, variant, **kwargs)
        
        if nc is not None:
            config['nc'] = nc
        
        # Create temporary config file
        temp_config_path = Path.cwd() / f'temp_{task}_{variant}.yaml'
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(temp_config_path)
    
    def list_variants(self) -> list:
        """List available model variants."""
        return list(self.base_config.get('variants', {}).keys())
    
    def list_datasets(self) -> list:
        """List available datasets."""
        return list(self.base_config.get('datasets', {}).keys())
    
    def list_tasks(self) -> list:
        """List available tasks."""
        return list(self.base_config.get('tasks', {}).keys())


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(task: str = 'detect', variant: str = 'leafyolo_m', **kwargs) -> Dict[str, Any]:
    """
    Convenience function to get configuration.
    
    Args:
        task: Task type ('detect', 'segment', 'classify')
        variant: Model variant ('leafyolo_n', 'leafyolo_s', etc.)
        **kwargs: Additional configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    return config_manager.get_config(task, variant, **kwargs)


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Convenience function to get dataset configuration.
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        Dataset configuration
    """
    return config_manager.get_dataset_config(dataset_name)


def get_hyperparameters(task: str = 'detect') -> Dict[str, Any]:
    """
    Convenience function to get hyperparameters.
    
    Args:
        task: Task type
        
    Returns:
        Hyperparameters dictionary
    """
    return config_manager.get_hyperparameters(task)


# Example usage:
if __name__ == '__main__':
    # Get detection configuration
    detect_config = get_config('detect', 'leafyolo_s', nc=80)
    print("Detection config loaded successfully")
    
    # Get segmentation configuration
    segment_config = get_config('segment', 'leafyolo_m', nc=91)
    print("Segmentation config loaded successfully")
    
    # Get classification configuration  
    classify_config = get_config('classify', 'leafyolo_n', nc=1000)
    print("Classification config loaded successfully")
    
    # List available options
    print(f"Available variants: {config_manager.list_variants()}")
    print(f"Available datasets: {config_manager.list_datasets()}")
    print(f"Available tasks: {config_manager.list_tasks()}")
