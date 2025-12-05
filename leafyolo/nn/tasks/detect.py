"""
Detection Task Model
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from copy import deepcopy

from leafyolo.nn.modules import *
from leafyolo.utils.general import make_divisible
from leafyolo.utils.torch_utils import initialize_weights, model_info
from leafyolo.utils.autoanchor import check_anchor_order
from leafyolo.utils.config import get_config


class DetectionModel(nn.Module):
    """
    LEAF-YOLO Detection Model with professional architecture.
    """
    
    def __init__(self, cfg='leafyolo.yaml', ch=3, nc=None, anchors=None, task='detect'):
        """
        Initialize detection model.
        
        Args:
            cfg: Model configuration file, dict, or task name
            ch: Input channels
            nc: Number of classes
            anchors: Anchor configuration
            task: Task type for adaptive configuration
        """
        super().__init__()
        self.traced = False
        
        if isinstance(cfg, dict):
            self.yaml = cfg
        elif cfg in ['detect', 'segment', 'classify']:
            # Use adaptive configuration
            self.yaml = get_config(task=cfg, nc=nc or 80)
        else:
            # Load from file or use adaptive config
            try:
                if Path(cfg).exists():
                    with open(cfg) as f:
                        self.yaml = yaml.safe_load(f)
                else:
                    raise FileNotFoundError
            except:
                # Fallback to adaptive configuration
                self.yaml = get_config(task=task, nc=nc or 80)
        
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = anchors
            
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, IDetect)):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            
        # Init weights
        initialize_weights(self)
        self.info()
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        """Forward pass."""
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)
    
    def _forward_augment(self, x):
        """Augmented forward pass."""
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        
        for si, fi in zip(s, f):
            xi = self._scale_img(x.flip(fi) if fi else x, si)
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
            
        y = torch.cat(y, 1)
        return y, None
    
    def _forward_once(self, x, profile=False, visualize=False):
        """Single forward pass."""
        y, dt = [], []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            
        return x
    
    def _scale_img(self, img, ratio=1.0, same_shape=False):
        """Scale image."""
        if ratio == 1.0:
            return img
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x / 32) * 32 for x in s)
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
    
    def _descale_pred(self, p, flips, scale, img_size):
        """De-scale predictions."""
        p[:, :4] /= scale  # de-scale
        if flips == 2:
            p[:, 1] = img_size[0] - p[:, 1]  # de-flip ud
        elif flips == 3:
            p[:, 0] = img_size[1] - p[:, 0]  # de-flip lr
        return p
    
    def info(self, verbose=False, img_size=640):
        """Print model information."""
        model_info(self, verbose, img_size)
    
    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers."""
        from leafyolo.utils.torch_utils import fuse_conv_and_bn
        
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        self.info()
        return self


def parse_model(d, ch):
    """Parse model configuration."""
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, IDetect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
            
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)
