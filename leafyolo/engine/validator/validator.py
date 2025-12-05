"""
LEAF-YOLO Validator - Professional Implementation
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from leafyolo.data.datasets import create_dataloader
from leafyolo.utils.general import non_max_suppression, scale_coords, box_iou, xywh2xyxy
from leafyolo.utils.metrics.metrics import ap_per_class, ConfusionMatrix
from leafyolo.utils.torch_utils import select_device


class LeafValidator:
    """
    LEAF-YOLO Validator with professional evaluation pipeline.
    """
    
    def __init__(self, model=None, data=None, **kwargs):
        """
        Initialize validator.
        
        Args:
            model: Model to validate
            data: Dataset configuration
            **kwargs: Additional arguments
        """
        self.model = model
        self.data = data
        self.device = select_device(kwargs.get('device', ''))
        self.conf_thres = kwargs.get('conf_thres', 0.001)
        self.iou_thres = kwargs.get('iou_thres', 0.6)
        self.img_size = kwargs.get('img_size', 640)
        self.batch_size = kwargs.get('batch_size', 32)
        self.save_dir = Path(kwargs.get('save_dir', 'runs/val'))
        
        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def validate(self, dataloader=None, data=None, **kwargs):
        """
        Run validation.
        
        Args:
            dataloader: Validation dataloader
            data: Dataset configuration
            **kwargs: Additional arguments
        """
        if dataloader is None and data:
            # Create dataloader from data config
            import yaml
            with open(data) as f:
                data_dict = yaml.safe_load(f)
                
            dataloader, _ = create_dataloader(
                data_dict['val'],
                self.img_size,
                self.batch_size,
                32,  # stride
                hyp={},
                augment=False,
                cache=False,
                rect=True,
                rank=-1,
                world_size=1,
                workers=8
            )
        
        if dataloader is None:
            raise ValueError("No dataloader provided")
            
        # Validation metrics
        seen = 0
        stats = []
        nc = self.model.nc if hasattr(self.model, 'nc') else 80
        confusion_matrix = ConfusionMatrix(nc=nc)
        names = getattr(self.model, 'names', [f'class{i}' for i in range(nc)])
        
        # Validation loop
        pbar = tqdm(dataloader, desc='Validating')
        for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
            img = img.to(self.device).float() / 255.0
            targets = targets.to(self.device)
            
            # Inference
            with torch.no_grad():
                out, _ = self.model(img)
            
            # Apply NMS
            out = non_max_suppression(
                out, 
                conf_thres=self.conf_thres, 
                iou_thres=self.iou_thres
            )
            
            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1
                
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                
                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                
                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = self._process_batch(predn, labelsn)
                    confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool)
                    
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        
        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=self.save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        else:
            mp = mr = map50 = map = 0.0
            
        # Print results
        print(f'Precision: {mp:.3f}, Recall: {mr:.3f}, mAP@0.5: {map50:.3f}, mAP@0.5:0.95: {map:.3f}')
        
        return {
            'fitness': 0.1 * map50 + 0.9 * map,  # fitness function
            'precision': mp,
            'recall': mr,
            'map50': map50,
            'map': map
        }
    
    def _process_batch(self, detections, labels):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        
        Args:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
            
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        iouv = torch.linspace(0.5, 0.95, 10).to(detections.device)  # iou vector for mAP@0.5:0.95
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=detections.device)
        
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
        
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(detections.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
            
        return correct
