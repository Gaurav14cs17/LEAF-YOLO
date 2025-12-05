"""
LEAF-YOLO Predictor - Professional Implementation
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List

from leafyolo.data.datasets import LoadImages, LoadStreams
from leafyolo.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from leafyolo.utils.plots import plot_one_box
from leafyolo.utils.torch_utils import select_device


class LeafPredictor:
    """
    LEAF-YOLO Predictor with professional inference pipeline.
    """
    
    def __init__(self, model=None, **kwargs):
        """
        Initialize predictor.
        
        Args:
            model: Model for prediction
            **kwargs: Additional arguments
        """
        self.model = model
        self.device = select_device(kwargs.get('device', ''))
        self.conf_thres = kwargs.get('conf_thres', 0.25)
        self.iou_thres = kwargs.get('iou_thres', 0.45)
        self.img_size = kwargs.get('img_size', 640)
        self.save_dir = Path(kwargs.get('project', 'runs/predict')) / kwargs.get('name', 'exp')
        
        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def preprocess(self, img):
        """Preprocess image for inference."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(self.device)
        
        img = img.float() / 255.0  # 0-255 to 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def postprocess(self, pred, img, orig_img):
        """Postprocess predictions."""
        # Apply NMS
        pred = non_max_suppression(
            pred, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres
        )
        
        results = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img.shape).round()
                results.append(det)
            else:
                results.append(torch.empty((0, 6)))
                
        return results
    
    def predict(self, source, **kwargs):
        """
        Run prediction.
        
        Args:
            source: Input source
            **kwargs: Additional arguments
        """
        # Setup data loader
        if str(source).isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            dataset = LoadStreams(source, img_size=self.img_size, stride=32)
        else:
            dataset = LoadImages(source, img_size=self.img_size, stride=32)
        
        # Create save directory
        save_img = kwargs.get('save', True)
        if save_img:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for path, img, orig_img, _ in dataset:
            # Preprocess
            img = self.preprocess(img)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img)[0]
            
            # Postprocess
            detections = self.postprocess(pred, img, orig_img)
            
            # Save results
            if save_img and len(detections[0]):
                for *xyxy, conf, cls in detections[0]:
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, orig_img, label=label, color=(255, 0, 0), line_thickness=2)
                
                # Save image
                save_path = self.save_dir / Path(path).name
                cv2.imwrite(str(save_path), orig_img)
            
            results.append({
                'path': path,
                'detections': detections[0],
                'orig_shape': orig_img.shape,
                'img_shape': img.shape
            })
        
        return results
