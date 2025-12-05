"""
Test cases for training engine
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from leafyolo.engine.trainer import LeafTrainer


class TestLeafTrainerInit:
    """Test LeafTrainer initialization."""
    
    def test_trainer_init_basic(self, mock_model):
        """Test basic trainer initialization."""
        trainer = LeafTrainer(model=mock_model, data='coco.yaml')
        
        assert trainer.model == mock_model
        assert trainer.data == 'coco.yaml'
        assert trainer.epochs == 100  # default
        assert trainer.batch_size == 16  # default
        assert trainer.img_size == 640  # default
    
    def test_trainer_init_custom_params(self, mock_model):
        """Test trainer initialization with custom parameters."""
        trainer = LeafTrainer(
            model=mock_model,
            data='custom.yaml',
            epochs=300,
            batch_size=32,
            img_size=1280,
            device='cuda:0'
        )
        
        assert trainer.epochs == 300
        assert trainer.batch_size == 32
        assert trainer.img_size == 1280
    
    def test_trainer_init_save_dir(self, mock_model, temp_dir):
        """Test trainer save directory creation."""
        with patch('leafyolo.engine.trainer.trainer.increment_path') as mock_increment:
            mock_increment.return_value = temp_dir / "exp1"
            
            trainer = LeafTrainer(
                model=mock_model,
                project=str(temp_dir),
                name='test_run'
            )
            
            mock_increment.assert_called_once()


class TestLeafTrainerSetup:
    """Test trainer setup methods."""
    
    def test_setup_model(self, mock_model):
        """Test model setup."""
        with patch('leafyolo.engine.trainer.trainer.ComputeLossOTA') as mock_loss, \
             patch('leafyolo.engine.trainer.trainer.ModelEMA') as mock_ema:
            
            mock_model.to.return_value = mock_model
            trainer = LeafTrainer(model=mock_model)
            trainer.device = torch.device('cpu')
            
            trainer.setup_model()
            
            mock_model.to.assert_called_once_with(trainer.device)
            mock_loss.assert_called_once_with(mock_model)
            mock_ema.assert_called_once_with(mock_model)
    
    def test_setup_data(self, mock_model, temp_dir):
        """Test data setup."""
        # Create mock data config
        data_config = temp_dir / "data.yaml"
        data_config.write_text("""
        train: train/images
        val: val/images
        nc: 80
        names: ['person', 'bicycle']
        """)
        
        with patch('leafyolo.engine.trainer.trainer.create_dataloader') as mock_create_dl:
            mock_create_dl.return_value = (MagicMock(), None)
            
            trainer = LeafTrainer(model=mock_model)
            trainer.img_size = 640
            trainer.batch_size = 16
            
            trainer.setup_data(str(data_config))
            
            # Should be called twice (train and val)
            assert mock_create_dl.call_count == 2
    
    def test_setup_optimizer(self, mock_model):
        """Test optimizer setup."""
        # Create mock model with named modules
        mock_module = MagicMock()
        mock_module.weight = torch.randn(64, 32, 3, 3)
        mock_module.bias = torch.randn(64)
        
        mock_bn = MagicMock(spec=torch.nn.BatchNorm2d)
        mock_bn.weight = torch.randn(64)
        
        mock_model.named_modules.return_value = [
            ('conv1', mock_module),
            ('bn1', mock_bn)
        ]
        
        trainer = LeafTrainer(model=mock_model)
        trainer.epochs = 100
        
        trainer.setup_optimizer()
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert len(trainer.optimizer.param_groups) == 3  # no_decay, decay, biases


class TestLeafTrainerTraining:
    """Test training process."""
    
    def test_train_epoch(self, mock_model, mock_dataset):
        """Test single epoch training."""
        with patch('leafyolo.engine.trainer.trainer.create_dataloader') as mock_create_dl:
            mock_dataloader = MagicMock()
            mock_dataloader.__len__.return_value = 10
            mock_dataloader.__iter__.return_value = iter([
                (torch.randn(2, 3, 640, 640), torch.randn(2, 6), ['path1', 'path2'], None)
                for _ in range(3)
            ])
            mock_create_dl.return_value = (mock_dataloader, None)
            
            trainer = LeafTrainer(model=mock_model)
            trainer.device = torch.device('cpu')
            trainer.dataloader = mock_dataloader
            
            # Setup required components
            trainer.compute_loss = MagicMock()
            trainer.compute_loss.return_value = (torch.tensor(1.5), torch.tensor([1.5]))
            trainer.optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.01)
            trainer.ema = None
            
            loss = trainer.train_epoch(1)
            
            assert isinstance(loss, float)
            assert loss > 0
    
    def test_train_epoch_with_ema(self, mock_model):
        """Test training epoch with EMA."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__.return_value = 2
        mock_dataloader.__iter__.return_value = iter([
            (torch.randn(1, 3, 640, 640), torch.randn(1, 6), ['path1'], None)
            for _ in range(2)
        ])
        
        trainer = LeafTrainer(model=mock_model)
        trainer.device = torch.device('cpu')
        trainer.dataloader = mock_dataloader
        
        # Setup required components
        trainer.compute_loss = MagicMock()
        trainer.compute_loss.return_value = (torch.tensor(1.0), torch.tensor([1.0]))
        trainer.optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.01)
        trainer.ema = MagicMock()
        
        trainer.train_epoch(1)
        
        # EMA update should be called for each batch
        assert trainer.ema.update.call_count == 2
    
    def test_validate(self, mock_model):
        """Test validation during training."""
        trainer = LeafTrainer(model=mock_model)
        trainer.ema = MagicMock()
        trainer.ema.ema = mock_model
        
        with patch('leafyolo.engine.trainer.trainer.LeafValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate.return_value = {'fitness': 0.65}
            mock_validator.return_value = mock_validator_instance
            
            result = trainer.validate()
            
            mock_validator.assert_called_once_with(model=mock_model)
            mock_validator_instance.validate.assert_called_once()
            assert result['fitness'] == 0.65
    
    def test_save_checkpoint(self, mock_model, temp_dir):
        """Test checkpoint saving."""
        trainer = LeafTrainer(model=mock_model)
        trainer.save_dir = temp_dir
        trainer.optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.01)
        trainer.scheduler = None
        trainer.ema = None
        trainer.best_fitness = 0.5
        
        trainer.save_checkpoint(epoch=50, best_fitness=0.7)
        
        # Check that checkpoint files are created
        assert (temp_dir / 'last.pt').exists()
        assert (temp_dir / 'best.pt').exists()
    
    def test_full_training_loop(self, mock_model, temp_dir):
        """Test complete training loop."""
        # Create mock data config
        data_config = temp_dir / "data.yaml"
        data_config.write_text("""
        train: train/images
        val: val/images
        nc: 80
        """)
        
        with patch.multiple(
            trainer := LeafTrainer(model=mock_model),
            setup_model=MagicMock(),
            setup_data=MagicMock(),
            setup_optimizer=MagicMock(),
            train_epoch=MagicMock(return_value=1.5),
            validate=MagicMock(return_value={'fitness': 0.6}),
            save_checkpoint=MagicMock()
        ):
            trainer.save_dir = temp_dir
            trainer.epochs = 5  # Short training for test
            trainer.best_fitness = 0.0
            
            result = trainer.train(data=str(data_config), epochs=5)
            
            # Verify setup methods called
            trainer.setup_model.assert_called_once()
            trainer.setup_data.assert_called_once()
            trainer.setup_optimizer.assert_called_once()
            
            # Verify training epochs
            assert trainer.train_epoch.call_count == 5
            
            # Verify validation and checkpointing
            trainer.validate.assert_called()
            trainer.save_checkpoint.assert_called()
            
            # Verify return value
            assert 'last' in result
            assert 'best' in result


class TestLeafTrainerLogging:
    """Test trainer logging functionality."""
    
    def test_wandb_logger_init(self, mock_model):
        """Test WandB logger initialization."""
        with patch('leafyolo.engine.trainer.trainer.WandbLogger') as mock_wandb:
            trainer = LeafTrainer(model=mock_model, wandb=True, project='test_project')
            
            mock_wandb.assert_called_once()
            assert trainer.logger is not None
    
    def test_no_logger_init(self, mock_model):
        """Test initialization without logger."""
        trainer = LeafTrainer(model=mock_model)
        
        assert trainer.logger is None
    
    def test_logging_during_training(self, mock_model):
        """Test logging during training process."""
        mock_logger = MagicMock()
        
        trainer = LeafTrainer(model=mock_model)
        trainer.logger = mock_logger
        trainer.device = torch.device('cpu')
        
        # Mock training components
        trainer.dataloader = MagicMock()
        trainer.dataloader.__len__.return_value = 1
        trainer.dataloader.__iter__.return_value = iter([
            (torch.randn(1, 3, 640, 640), torch.randn(1, 6), ['path1'], None)
        ])
        
        trainer.compute_loss = MagicMock()
        trainer.compute_loss.return_value = (torch.tensor(1.0), torch.tensor([1.0]))
        trainer.optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.01)
        trainer.ema = None
        
        trainer.train_epoch(1)
        
        # Verify logging occurred (implementation dependent)
        # This would depend on actual logging implementation


class TestLeafTrainerErrorHandling:
    """Test trainer error handling."""
    
    def test_missing_model_error(self):
        """Test error when model is not provided."""
        trainer = LeafTrainer(model=None)
        
        with pytest.raises(ValueError, match="Model not provided"):
            trainer.setup_model()
    
    def test_invalid_data_config(self, mock_model, temp_dir):
        """Test error with invalid data configuration."""
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:")
        
        trainer = LeafTrainer(model=mock_model)
        
        with pytest.raises(Exception):  # YAML parsing error
            trainer.setup_data(str(invalid_config))
    
    def test_missing_data_file(self, mock_model):
        """Test error with missing data file."""
        trainer = LeafTrainer(model=mock_model)
        
        with pytest.raises(FileNotFoundError):
            trainer.setup_data('non_existent.yaml')


@pytest.mark.parametrize("epochs,batch_size,expected_batches", [
    (10, 16, 10),
    (50, 32, 50),
    (100, 8, 100),
])
def test_trainer_parametrized_configs(epochs, batch_size, expected_batches, mock_model):
    """Parametrized test for different trainer configurations."""
    trainer = LeafTrainer(
        model=mock_model,
        epochs=epochs,
        batch_size=batch_size
    )
    
    assert trainer.epochs == epochs
    assert trainer.batch_size == batch_size


class TestTrainerOptimizations:
    """Test trainer optimization features."""
    
    def test_mixed_precision_training(self, mock_model):
        """Test mixed precision (AMP) training."""
        # This would test automatic mixed precision if implemented
        trainer = LeafTrainer(model=mock_model)
        
        # Mock AMP scaler
        with patch('torch.cuda.amp.GradScaler') as mock_scaler:
            mock_scaler_instance = MagicMock()
            mock_scaler.return_value = mock_scaler_instance
            
            # Test would verify AMP usage in training loop
            pass
    
    def test_gradient_accumulation(self, mock_model):
        """Test gradient accumulation."""
        trainer = LeafTrainer(model=mock_model)
        
        # Test would verify gradient accumulation implementation
        pass
    
    def test_learning_rate_scheduling(self, mock_model):
        """Test learning rate scheduling."""
        trainer = LeafTrainer(model=mock_model)
        trainer.epochs = 100
        trainer.setup_optimizer()
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Step scheduler
        if trainer.scheduler:
            trainer.scheduler.step()
            new_lr = trainer.optimizer.param_groups[0]['lr']
            
            # LR should change (cosine annealing decreases)
            assert new_lr <= initial_lr
