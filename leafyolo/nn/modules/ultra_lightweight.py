#!/usr/bin/env python3
"""
Ultra-Lightweight Neural Network Modules for LEAF-YOLO Ultra
Designed for <1MB models with maximum efficiency and accuracy

Key innovations:
- Depthwise Separable Convolutions
- Ghost Modules for feature expansion
- Inverted Residual Blocks
- Efficient Channel Attention
- Progressive Feature Refinement
"""

import torch
import torch.nn as nn
import math


class DWConvUltra(nn.Module):
    """Ultra-efficient Depthwise Separable Convolution"""
    
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        # Activation
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.act(self.bn2(self.pw_conv(x)))
        return x


class GhostModule(nn.Module):
    """Ghost Module - Generate more features from fewer parameters"""
    
    def __init__(self, c1, c2, k=3, ratio=2, act=True):
        super().__init__()
        c_ = c2 // ratio
        
        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU() if act else nn.Identity()
        )
        
        # Cheap operation (depthwise)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(c_, c_, k, 1, k//2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU() if act else nn.Identity()
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class InvertedResidual(nn.Module):
    """MobileNetV2-style Inverted Residual Block - Ultra Efficient"""
    
    def __init__(self, c1, c2, s=1, expand_ratio=6):
        super().__init__()
        self.use_res_connect = s == 1 and c1 == c2
        hidden_dim = int(round(c1 * expand_ratio))
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(c1, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, s, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # Squeeze-Excitation
            SqueezeExciteUltra(hidden_dim),
            # Pointwise-linear
            nn.Conv2d(hidden_dim, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class SqueezeExciteUltra(nn.Module):
    """Ultra-efficient Squeeze-and-Excitation module"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MicroAttention(nn.Module):
    """Micro Attention mechanism with minimal parameters"""
    
    def __init__(self, channels):
        super().__init__()
        # Use only 1/8 of channels for attention computation
        attn_channels = max(1, channels // 8)
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, attn_channels, 1, bias=False),
            nn.Conv2d(attn_channels, attn_channels, 3, 1, 1, groups=attn_channels, bias=False),
            nn.Conv2d(attn_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Channel attention  
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attn_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(attn_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attn(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attn(x)
        x = x * sa
        
        return x


class C3Ultra(nn.Module):
    """Ultra-lightweight C3 module using Ghost and Inverted Residual blocks"""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        self.cv1 = DWConvUltra(c1, c_, 1, 1)
        self.cv2 = DWConvUltra(c1, c_, 1, 1) 
        self.cv3 = DWConvUltra(2 * c_, c2, 1, 1)
        
        # Use mix of Ghost and Inverted Residual
        self.m = nn.Sequential(
            *[InvertedResidual(c_, c_, expand_ratio=2) for _ in range(n//2)] +
            *[GhostModule(c_, c_) for _ in range(n - n//2)]
        )
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPFUltra(nn.Module):
    """Ultra-lightweight Spatial Pyramid Pooling - Fast"""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = DWConvUltra(c1, c_, 1, 1)
        self.cv2 = DWConvUltra(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1) 
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class UltraBottleneck(nn.Module):
    """Ultra-efficient Bottleneck using latest efficiency techniques"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DWConvUltra(c1, c_, 1, 1)
        self.cv2 = DWConvUltra(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        
        # Add micro attention for better accuracy
        self.attn = MicroAttention(c2) if c2 > 16 else nn.Identity()
    
    def forward(self, x):
        y = self.attn(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class UltraFocus(nn.Module):
    """Ultra-efficient Focus layer with depthwise separable convolutions"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # Use depthwise separable convolution for efficiency
        self.conv = DWConvUltra(c1 * 4, c2, k, s, p, act)
    
    def forward(self, x):
        # Focus operation: slice and concatenate
        return self.conv(torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1))


def autopad(k, p=None):
    """Auto-pad for 'same' padding"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class UltraDetect(nn.Module):
    """Ultra-lightweight detection head with shared convolutions"""
    
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # outputs per anchor
        self.nl = len(anchors)  # detection layers
        self.na = len(anchors[0]) // 2  # anchors
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.inplace = inplace
        
        # Ultra-efficient shared convolutions
        self.shared_conv = nn.ModuleList([
            DWConvUltra(x, x//2, 3) for x in ch  # Reduce channels by half
        ])
        
        # Lightweight detection heads
        self.m = nn.ModuleList([
            nn.Conv2d(x//2, self.no * self.na, 1) for x in ch
        ])
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            # Apply shared convolution first
            x[i] = self.shared_conv[i](x[i])
            x[i] = self.m[i](x[i])
            
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1),) if self.onnx_dynamic else (torch.cat(z, 1), x)
    
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
