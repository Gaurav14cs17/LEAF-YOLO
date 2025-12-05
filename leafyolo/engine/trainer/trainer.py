"""
LEAF-YOLO Trainer - Professional Implementation
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

from leafyolo.data.datasets import create_dataloader
from leafyolo.utils.loss import ComputeLoss, ComputeLossOTA
from leafyolo.utils.torch_utils import ModelEMA, select_device
from leafyolo.utils.general import colorstr, increment_path
from leafyolo.utils.callbacks.wandb_logging.wandb_utils import WandbLogger


class LeafTrainer:
    """
    LEAF-YOLO Trainer with professional training pipeline.
    """
    
    def __init__(self, model=None, data=None, **kwargs):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            data: Dataset configuration
            **kwargs: Additional arguments
        """
        self.model = model
        self.data = data
        self.device = select_device(kwargs.get('device', ''))
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 16)
        self.img_size = kwargs.get('img_size', 640)
        self.save_dir = Path(kwargs.get('project', 'runs/train')) / kwargs.get('name', 'exp')
        self.save_dir = increment_path(self.save_dir)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.compute_loss = None
        self.dataloader = None
        self.testloader = None
        
        # Logging
        self.logger = WandbLogger(kwargs) if kwargs.get('wandb') else None
        
    def setup_model(self):
        """Setup model for training."""
        if self.model is None:
            raise ValueError("Model not provided")
            
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.compute_loss = ComputeLossOTA(self.model)
        
        # Setup EMA
        self.ema = ModelEMA(self.model)
        
    def setup_data(self, data):
        """Setup data loaders."""
        import yaml
        
        with open(data) as f:
            data_dict = yaml.safe_load(f)
            
        # Training dataloader
        self.dataloader, _ = create_dataloader(
            data_dict['train'],
            self.img_size,
            self.batch_size,
            32,  # stride
            hyp={},  # hyperparameters
            augment=True,
            cache=False,
            rect=False,
            rank=-1,
            world_size=1,
            workers=8
        )
        
        # Validation dataloader  
        self.testloader, _ = create_dataloader(
            data_dict['val'],
            self.img_size,
            self.batch_size * 2,
            32,  # stride
            hyp={},
            augment=False,
            cache=False,
            rect=True,
            rank=-1,
            world_size=1,
            workers=8
        )
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Optimizer groups
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
                
        self.optimizer = torch.optim.SGD(pg0, lr=0.01, momentum=0.937, nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})
        self.optimizer.add_param_group({'params': pg2})
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0.01 * 0.01
        )
        
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = len(self.dataloader)
        
        for i, (imgs, targets, paths, _) in enumerate(self.dataloader):
            imgs = imgs.to(self.device).float() / 255.0
            targets = targets.to(self.device)
            
            # Forward
            pred = self.model(imgs)
            loss, loss_items = self.compute_loss(pred, targets, imgs)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # EMA
            if self.ema:
                self.ema.update(self.model)
                
            total_loss += loss.item()
            
            # Logging
            if i % 100 == 0:
                print(f'Epoch {epoch}/{self.epochs}, Batch {i}/{num_batches}, Loss: {loss.item():.4f}')
                
        return total_loss / num_batches
    
    def validate(self):
        """Run validation."""
        if not hasattr(self, 'validator'):
            from leafyolo.engine.validator import LeafValidator
            self.validator = LeafValidator(model=self.ema.ema if self.ema else self.model)
            
        return self.validator.validate(dataloader=self.testloader)
    
    def save_checkpoint(self, epoch, best_fitness):
        """Save training checkpoint."""
        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'model': self.ema.ema if self.ema else self.model,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        # Save last
        torch.save(ckpt, self.save_dir / 'last.pt')
        
        # Save best
        if best_fitness == self.best_fitness:
            torch.save(ckpt, self.save_dir / 'best.pt')
    
    def train(self, data=None, epochs=None, **kwargs):
        """
        Main training loop.
        
        Args:
            data: Dataset configuration file
            epochs: Number of epochs
            **kwargs: Additional arguments
        """
        # Update parameters
        if data:
            self.data = data
        if epochs:
            self.epochs = epochs
            
        # Setup
        self.setup_model()
        self.setup_data(self.data)
        self.setup_optimizer()
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'{colorstr("train:")} Starting training for {self.epochs} epochs...')
        
        self.best_fitness = 0.0
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                results = self.validate()
                fitness = results.get('fitness', 0.0)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    
                # Save checkpoint
                self.save_checkpoint(epoch, self.best_fitness)
                
                print(f'Epoch {epoch}: Loss={train_loss:.4f}, Fitness={fitness:.4f}')
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                
        print(f'{colorstr("train:")} Training completed!')
        return {'last': self.save_dir / 'last.pt', 'best': self.save_dir / 'best.pt'}
