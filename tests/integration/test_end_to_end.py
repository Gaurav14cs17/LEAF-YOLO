"""
End-to-end integration tests for LEAF-YOLO
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from leafyolo import LEAFYOLO
from leafyolo.utils.config import get_config


class TestE2ETraining:
    """End-to-end training tests."""
    
    @pytest.mark.slow
    def test_complete_training_pipeline(self, temp_dir, sample_config):
        """Test complete training pipeline from config to trained model."""
        # Create mock dataset config
        data_config = temp_dir / "dataset.yaml"
        data_config.write_text("""
        train: train/images
        val: val/images
        nc: 10
        names: ['class0', 'class1', 'class2', 'class3', 'class4',
                'class5', 'class6', 'class7', 'class8', 'class9']
        """)
        
        # Create minimal model config
        model_config = temp_dir / "model.yaml"
        with open(model_config, 'w') as f:
            import yaml
            yaml.dump(sample_config, f)
        
        with patch('leafyolo.data.datasets.create_dataloader') as mock_create_dl, \
             patch('leafyolo.engine.trainer.trainer.LeafValidator') as mock_validator:
            
            # Mock dataloader
            mock_dl = MagicMock()
            mock_dl.__len__.return_value = 5
            mock_dl.__iter__.return_value = iter([
                (torch.randn(2, 3, 640, 640), torch.randn(2, 6), ['path1', 'path2'], None)
                for _ in range(5)
            ])
            mock_create_dl.return_value = (mock_dl, None)
            
            # Mock validator
            mock_val_instance = MagicMock()
            mock_val_instance.validate.return_value = {'fitness': 0.65}
            mock_validator.return_value = mock_val_instance
            
            # Initialize model
            model = LEAFYOLO(str(model_config))
            
            # Train for minimal epochs
            results = model.train(
                data=str(data_config),
                epochs=2,  # Minimal for testing
                batch_size=2,
                project=str(temp_dir),
                name='test_train'
            )
            
            # Verify training completed
            assert 'last' in results
            assert 'best' in results
    
    @pytest.mark.slow
    def test_adaptive_config_training(self, temp_dir):
        """Test training with adaptive configuration."""
        # Create dataset config
        data_config = temp_dir / "dataset.yaml"
        data_config.write_text("""
        train: train/images
        val: val/images
        nc: 5
        names: ['person', 'car', 'bike', 'dog', 'cat']
        """)
        
        with patch('leafyolo.data.datasets.create_dataloader') as mock_create_dl, \
             patch('leafyolo.engine.trainer.trainer.LeafValidator') as mock_validator, \
             patch('leafyolo.utils.config.get_config') as mock_get_config:
            
            # Mock config
            mock_get_config.return_value = {
                'task': 'detect',
                'nc': 5,
                'ch': 3,
                'backbone': [[-1, 1, 'Conv', [32, 6, 2, 2]]],
                'head': [[-1, 1, 'Conv', [85, 1, 1]]],
                'anchors': [[10, 13], [30, 61], [116, 90]]
            }
            
            # Mock dataloader and validator
            mock_dl = MagicMock()
            mock_dl.__len__.return_value = 3
            mock_dl.__iter__.return_value = iter([
                (torch.randn(1, 3, 640, 640), torch.randn(1, 6), ['path1'], None)
                for _ in range(3)
            ])
            mock_create_dl.return_value = (mock_dl, None)
            
            mock_val_instance = MagicMock()
            mock_val_instance.validate.return_value = {'fitness': 0.5}
            mock_validator.return_value = mock_val_instance
            
            # Test adaptive configuration
            model = LEAFYOLO('detect', variant='leafyolo_n', nc=5)
            
            results = model.train(
                data=str(data_config),
                epochs=1,
                batch_size=1,
                project=str(temp_dir),
                name='adaptive_test'
            )
            
            assert results is not None
            mock_get_config.assert_called_once()


class TestE2EPrediction:
    """End-to-end prediction tests."""
    
    def test_complete_prediction_pipeline(self, temp_dir, mock_model):
        """Test complete prediction pipeline."""
        # Create test images directory
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # Create mock image files
        for i in range(3):
            img_path = images_dir / f"test_image_{i}.jpg"
            img_path.write_text("mock image data")  # Mock file content
        
        with patch('leafyolo.data.datasets.LoadImages') as mock_load_images, \
             patch('leafyolo.utils.general.non_max_suppression') as mock_nms, \
             patch('cv2.imwrite') as mock_cv2_write:
            
            # Mock image loader
            mock_dataset = MagicMock()
            mock_dataset.__iter__.return_value = iter([
                ('path1.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None),
                ('path2.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None),
                ('path3.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None),
            ])
            mock_load_images.return_value = mock_dataset
            
            # Mock NMS output
            mock_nms.return_value = [torch.tensor([
                [100, 100, 200, 200, 0.9, 0],  # x1, y1, x2, y2, conf, class
                [300, 150, 400, 250, 0.8, 1],
            ])]
            
            # Test prediction
            model = LEAFYOLO(mock_model)
            results = model.predict(
                source=str(images_dir),
                save=True,
                project=str(temp_dir),
                name='test_predict'
            )
            
            assert len(results) == 3  # 3 images processed
            mock_load_images.assert_called_once()
            mock_nms.assert_called()
    
    def test_real_time_prediction(self, mock_model):
        """Test real-time prediction capability."""
        with patch('leafyolo.data.datasets.LoadStreams') as mock_load_streams, \
             patch('leafyolo.utils.general.non_max_suppression') as mock_nms:
            
            # Mock video stream
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter([
                ('frame_001.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None)
                for _ in range(10)  # 10 frames
            ])
            mock_load_streams.return_value = mock_stream
            
            # Mock detections
            mock_nms.return_value = [torch.tensor([[100, 100, 200, 200, 0.9, 0]])]
            
            model = LEAFYOLO(mock_model)
            results = model.predict(source='0')  # Webcam
            
            assert len(results) == 10  # 10 frames processed


class TestE2EValidation:
    """End-to-end validation tests."""
    
    def test_complete_validation_pipeline(self, temp_dir, mock_model):
        """Test complete validation pipeline."""
        # Create dataset config
        data_config = temp_dir / "val_dataset.yaml"
        data_config.write_text("""
        val: val/images
        nc: 80
        names: ['person', 'bicycle', 'car']  # Subset for testing
        """)
        
        with patch('leafyolo.data.datasets.create_dataloader') as mock_create_dl, \
             patch('leafyolo.utils.general.non_max_suppression') as mock_nms, \
             patch('leafyolo.utils.metrics.metrics.ap_per_class') as mock_ap:
            
            # Mock validation dataloader
            mock_dl = MagicMock()
            mock_dl.__len__.return_value = 10
            mock_dl.__iter__.return_value = iter([
                (
                    torch.randn(2, 3, 640, 640),  # images
                    torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.3],  # targets
                                 [1, 1, 0.3, 0.7, 0.1, 0.2]]),
                    ['path1.jpg', 'path2.jpg'],  # paths
                    [(640, 640), (640, 640)]  # shapes
                )
                for _ in range(10)
            ])
            mock_create_dl.return_value = (mock_dl, None)
            
            # Mock NMS and metrics
            mock_nms.return_value = [torch.tensor([
                [100, 100, 200, 200, 0.9, 0],
                [300, 150, 400, 250, 0.8, 1],
            ])]
            
            mock_ap.return_value = (
                torch.tensor([0.8, 0.7]),  # tp
                torch.tensor([0.2, 0.3]),  # fp
                torch.tensor([0.85, 0.75]), # precision
                torch.tensor([0.80, 0.70]), # recall
                torch.tensor([0.82, 0.72]), # f1
                torch.tensor([[0.65, 0.55], [0.60, 0.50]]), # ap
                torch.tensor([0, 1])  # ap_class
            )
            
            model = LEAFYOLO(mock_model)
            results = model.val(data=str(data_config))
            
            assert 'map50' in results
            assert 'map' in results
            assert 'precision' in results
            assert 'recall' in results


class TestE2EExport:
    """End-to-end export tests."""
    
    def test_onnx_export_pipeline(self, temp_dir, mock_model):
        """Test ONNX export pipeline."""
        with patch('leafyolo.engine.exporter.LeafExporter') as mock_exporter:
            mock_exporter_instance = MagicMock()
            mock_exporter_instance.export.return_value = str(temp_dir / 'model.onnx')
            mock_exporter.return_value = mock_exporter_instance
            
            model = LEAFYOLO(mock_model)
            result = model.export(
                format='onnx',
                img_size=640,
                dynamic=True,
                simplify=True
            )
            
            mock_exporter.assert_called_once_with(model=mock_model)
            mock_exporter_instance.export.assert_called_once()
            assert result.endswith('.onnx')
    
    def test_multiple_format_export(self, mock_model):
        """Test exporting to multiple formats."""
        formats = ['onnx', 'tensorrt', 'coreml']
        results = {}
        
        with patch('leafyolo.engine.exporter.LeafExporter') as mock_exporter:
            mock_exporter_instance = MagicMock()
            mock_exporter.return_value = mock_exporter_instance
            
            model = LEAFYOLO(mock_model)
            
            for fmt in formats:
                mock_exporter_instance.export.return_value = f'model.{fmt}'
                results[fmt] = model.export(format=fmt)
            
            assert len(results) == 3
            assert mock_exporter_instance.export.call_count == 3


class TestE2EWorkflows:
    """Test complete workflows."""
    
    @pytest.mark.slow
    def test_train_validate_predict_workflow(self, temp_dir):
        """Test complete train -> validate -> predict workflow."""
        # Setup directories and configs
        data_config = temp_dir / "workflow_data.yaml"
        data_config.write_text("""
        train: train/images
        val: val/images
        nc: 3
        names: ['cat', 'dog', 'bird']
        """)
        
        with patch('leafyolo.data.datasets.create_dataloader') as mock_create_dl, \
             patch('leafyolo.engine.trainer.trainer.LeafValidator') as mock_validator, \
             patch('leafyolo.data.datasets.LoadImages') as mock_load_images, \
             patch('leafyolo.utils.general.non_max_suppression') as mock_nms:
            
            # Mock training dataloader
            mock_train_dl = MagicMock()
            mock_train_dl.__len__.return_value = 2
            mock_train_dl.__iter__.return_value = iter([
                (torch.randn(1, 3, 640, 640), torch.randn(1, 6), ['path1'], None)
                for _ in range(2)
            ])
            mock_create_dl.return_value = (mock_train_dl, None)
            
            # Mock validator
            mock_val_instance = MagicMock()
            mock_val_instance.validate.return_value = {'fitness': 0.7, 'map50': 0.65}
            mock_validator.return_value = mock_val_instance
            
            # Mock prediction
            mock_pred_dataset = MagicMock()
            mock_pred_dataset.__iter__.return_value = iter([
                ('test.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None)
            ])
            mock_load_images.return_value = mock_pred_dataset
            mock_nms.return_value = [torch.tensor([[100, 100, 200, 200, 0.9, 0]])]
            
            # 1. Train model
            model = LEAFYOLO('detect', variant='leafyolo_n', nc=3)
            train_results = model.train(
                data=str(data_config),
                epochs=1,
                batch_size=1,
                project=str(temp_dir),
                name='workflow_train'
            )
            
            # 2. Validate model
            val_results = model.val(data=str(data_config))
            
            # 3. Run predictions
            pred_results = model.predict(
                source=str(temp_dir / "test_image.jpg"),
                project=str(temp_dir),
                name='workflow_predict'
            )
            
            # Verify all steps completed
            assert train_results is not None
            assert val_results['map50'] == 0.65
            assert len(pred_results) == 1
    
    def test_config_variant_comparison(self, temp_dir):
        """Test different model variants with same configuration."""
        variants = ['leafyolo_n', 'leafyolo_s', 'leafyolo_m']
        results = {}
        
        data_config = temp_dir / "comparison_data.yaml"
        data_config.write_text("""
        val: val/images
        nc: 10
        names: ['class0', 'class1', 'class2', 'class3', 'class4',
                'class5', 'class6', 'class7', 'class8', 'class9']
        """)
        
        with patch('leafyolo.data.datasets.create_dataloader') as mock_create_dl, \
             patch('leafyolo.utils.metrics.metrics.ap_per_class') as mock_ap:
            
            # Mock validation setup
            mock_dl = MagicMock()
            mock_dl.__len__.return_value = 5
            mock_dl.__iter__.return_value = iter([
                (torch.randn(1, 3, 640, 640), torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.3]]),
                 ['path1.jpg'], [(640, 640)])
                for _ in range(5)
            ])
            mock_create_dl.return_value = (mock_dl, None)
            
            mock_ap.return_value = (
                torch.tensor([0.8]), torch.tensor([0.2]), torch.tensor([0.85]),
                torch.tensor([0.80]), torch.tensor([0.82]), 
                torch.tensor([[0.65]]), torch.tensor([0])
            )
            
            # Test each variant
            for variant in variants:
                model = LEAFYOLO('detect', variant=variant, nc=10)
                val_results = model.val(data=str(data_config))
                results[variant] = val_results['map50']
            
            # Verify all variants ran
            assert len(results) == 3
            for variant in variants:
                assert variant in results


@pytest.mark.integration
class TestSystemIntegration:
    """Test system-level integration."""
    
    def test_memory_management(self, mock_model):
        """Test memory management during operations."""
        import gc
        
        model = LEAFYOLO(mock_model)
        
        # Track memory usage
        initial_objects = len(gc.get_objects())
        
        # Perform operations
        with patch('leafyolo.data.datasets.LoadImages') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__.return_value = iter([
                ('test.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None)
                for _ in range(100)  # Many images
            ])
            mock_load.return_value = mock_dataset
            
            results = model.predict(source='test_dir/')
        
        # Force garbage collection
        del results
        del model
        gc.collect()
        
        # Memory should be cleaned up (allow some tolerance)
        final_objects = len(gc.get_objects())
        assert final_objects < initial_objects + 1000  # Reasonable tolerance
    
    def test_device_compatibility(self, mock_model):
        """Test device compatibility (CPU/GPU)."""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            mock_model.to.return_value = mock_model
            
            model = LEAFYOLO(mock_model)
            
            # Test prediction on device
            with patch('leafyolo.data.datasets.LoadImages') as mock_load:
                mock_dataset = MagicMock()
                mock_dataset.__iter__.return_value = iter([
                    ('test.jpg', torch.randn(3, 640, 640), torch.randn(480, 640, 3), None)
                ])
                mock_load.return_value = mock_dataset
                
                results = model.predict(source='test.jpg', device=device)
                
                assert len(results) == 1
    
    def test_concurrent_operations(self, mock_model):
        """Test concurrent model operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def predict_worker(model, image_id):
            with patch('leafyolo.data.datasets.LoadImages') as mock_load:
                mock_dataset = MagicMock()
                mock_dataset.__iter__.return_value = iter([
                    (f'test_{image_id}.jpg', torch.randn(3, 640, 640), 
                     torch.randn(480, 640, 3), None)
                ])
                mock_load.return_value = mock_dataset
                
                result = model.predict(source=f'test_{image_id}.jpg')
                results_queue.put((image_id, len(result)))
        
        model = LEAFYOLO(mock_model)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=predict_worker, args=(model, i))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations completed
        assert results_queue.qsize() == 3
        
        while not results_queue.empty():
            image_id, result_count = results_queue.get()
            assert result_count == 1
