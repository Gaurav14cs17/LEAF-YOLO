"""
LEAF-YOLO Main Model Class - Professional Implementation
"""

from leafyolo.nn.tasks.detect import DetectionModel
from leafyolo.engine.trainer import LeafTrainer
from leafyolo.engine.predictor import LeafPredictor
from leafyolo.engine.validator import LeafValidator
from leafyolo.utils.config import get_config


class LEAFYOLO:
    """
    LEAF-YOLO model class with professional API design.
    Supports adaptive configuration for different tasks.
    """
    
    def __init__(self, model='detect', task=None, variant='leafyolo_m', **kwargs):
        """
        Initialize LEAF-YOLO model.
        
        Args:
            model: Model specification - can be:
                  - Task name ('detect', 'segment', 'classify')
                  - Config file path ('model.yaml')
                  - Pretrained weights ('model.pt')
            task: Task type (auto-detected if not specified)
            variant: Model variant ('leafyolo_n', 'leafyolo_s', 'leafyolo_m', etc.)
            **kwargs: Additional configuration overrides
        """
        self.model = None
        self.task = task or self._detect_task(model)
        self.variant = variant
        
        if isinstance(model, str):
            if model.endswith('.pt') or model.endswith('.pth'):
                self._load_weights(model)
            elif model in ['detect', 'segment', 'classify']:
                self._load_adaptive_config(model, variant, **kwargs)
            else:
                self._load_config(model, **kwargs)
        elif isinstance(model, DetectionModel):
            self.model = model
    
    def _detect_task(self, model_spec):
        """Auto-detect task from model specification."""
        if isinstance(model_spec, str):
            if model_spec in ['detect', 'segment', 'classify']:
                return model_spec
            elif 'segment' in model_spec.lower():
                return 'segment'
            elif 'classify' in model_spec.lower() or 'cls' in model_spec.lower():
                return 'classify'
        return 'detect'  # default
    
    def _load_adaptive_config(self, task, variant, **kwargs):
        """Load model from adaptive configuration."""
        config = get_config(task=task, variant=variant, **kwargs)
        self.model = DetectionModel(config, task=task)
        self.task = task
    
    def _load_config(self, config_path, **kwargs):
        """Load model from config file."""
        self.model = DetectionModel(config_path, task=self.task, **kwargs)
    
    def _load_weights(self, weights_path):
        """Load pretrained weights."""
        import torch
        ckpt = torch.load(weights_path, map_location='cpu')
        self.model = ckpt.get('model') or ckpt.get('ema')
        # Try to detect task from model structure
        if hasattr(self.model, 'task'):
            self.task = self.model.task
    
    def train(self, **kwargs):
        """Train the model."""
        trainer = LeafTrainer(model=self.model, **kwargs)
        return trainer.train(**kwargs)
    
    def predict(self, source, **kwargs):
        """Run prediction."""
        predictor = LeafPredictor(model=self.model, **kwargs)
        return predictor.predict(source=source, **kwargs)
    
    def val(self, **kwargs):
        """Validate the model."""
        validator = LeafValidator(model=self.model, **kwargs)
        return validator.validate(**kwargs)
    
    def export(self, **kwargs):
        """Export model."""
        from leafyolo.engine.exporter import LeafExporter
        exporter = LeafExporter(model=self.model, **kwargs)
        return exporter.export(**kwargs)
    
    def info(self, **kwargs):
        """Model info."""
        if self.model:
            self.model.info(**kwargs)
    
    def fuse(self):
        """Fuse layers."""
        if self.model:
            self.model.fuse()
        return self
    
    def __call__(self, *args, **kwargs):
        """Call predict method."""
        return self.predict(*args, **kwargs)


# Legacy compatibility
Model = LEAFYOLO

def attempt_load(weights, map_location=None):
    """Load model (legacy compatibility)."""
    return LEAFYOLO(weights)