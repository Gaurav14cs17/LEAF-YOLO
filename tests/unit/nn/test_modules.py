"""
Test cases for neural network modules
"""

import pytest
import torch
import torch.nn as nn

from leafyolo.nn.modules.common import (
    Conv, DWConv, Bottleneck, C3, SPP, SPPF, Focus, 
    autopad, MP, ReOrg, Concat
)
from leafyolo.nn.modules.heads import Detect, Segment, Classify
from leafyolo.nn.modules.activations import SiLU, Hardswish, MemoryEfficientSwish


class TestBasicModules:
    """Test basic neural network modules."""
    
    def test_autopad_function(self):
        """Test autopad function."""
        # Single kernel
        assert autopad(3) == 1
        assert autopad(5) == 2
        assert autopad(7) == 3
        
        # Multiple kernels
        assert autopad([3, 5]) == [1, 2]
        assert autopad([1, 3, 5]) == [0, 1, 2]
        
        # Custom padding
        assert autopad(3, 2) == 2
        assert autopad([3, 5], [2, 3]) == [2, 3]
    
    def test_mp_layer(self):
        """Test MaxPool layer."""
        mp = MP(k=2)
        x = torch.randn(1, 64, 32, 32)
        output = mp(x)
        
        assert output.shape == (1, 64, 16, 16)
    
    def test_reorg_layer(self):
        """Test ReOrg layer."""
        reorg = ReOrg()
        x = torch.randn(1, 64, 32, 32)
        output = reorg(x)
        
        # Should increase channels by 4x and decrease spatial by 2x
        assert output.shape == (1, 256, 16, 16)
    
    def test_concat_layer(self):
        """Test Concat layer."""
        concat = Concat(dimension=1)
        
        x1 = torch.randn(1, 32, 16, 16)
        x2 = torch.randn(1, 64, 16, 16)
        x3 = torch.randn(1, 16, 16, 16)
        
        output = concat([x1, x2, x3])
        
        # Should concatenate along channel dimension
        assert output.shape == (1, 112, 16, 16)  # 32 + 64 + 16 = 112


class TestConvolutionModules:
    """Test convolution modules."""
    
    def test_conv_basic(self):
        """Test basic Conv module."""
        conv = Conv(64, 128, k=3, s=1, p=1)
        x = torch.randn(1, 64, 32, 32)
        output = conv(x)
        
        assert output.shape == (1, 128, 32, 32)
        assert isinstance(conv.conv, nn.Conv2d)
        assert isinstance(conv.bn, nn.BatchNorm2d)
        assert isinstance(conv.act, nn.SiLU)
    
    def test_conv_stride(self):
        """Test Conv module with stride."""
        conv = Conv(64, 128, k=3, s=2)
        x = torch.randn(1, 64, 32, 32)
        output = conv(x)
        
        # Should reduce spatial dimensions by factor of stride
        assert output.shape == (1, 128, 16, 16)
    
    def test_conv_no_activation(self):
        """Test Conv module without activation."""
        conv = Conv(64, 128, k=3, s=1, act=False)
        x = torch.randn(1, 64, 32, 32)
        output = conv(x)
        
        assert output.shape == (1, 128, 32, 32)
        assert isinstance(conv.act, nn.Identity)
    
    def test_dwconv(self):
        """Test Depthwise Convolution."""
        dwconv = DWConv(64, 128, k=3, s=1)
        x = torch.randn(1, 64, 32, 32)
        output = dwconv(x)
        
        assert output.shape == (1, 128, 32, 32)
    
    def test_conv_fuse_forward(self):
        """Test Conv module fused forward."""
        conv = Conv(64, 128, k=3, s=1)
        x = torch.randn(1, 64, 32, 32)
        
        # Test both forward methods
        output1 = conv(x)
        output2 = conv.fuseforward(x)
        
        assert output1.shape == output2.shape
    
    @pytest.mark.parametrize("c1,c2,k,s", [
        (32, 64, 1, 1),
        (64, 128, 3, 1),
        (128, 256, 3, 2),
        (256, 512, 5, 1),
    ])
    def test_conv_parametrized(self, c1, c2, k, s):
        """Parametrized test for different Conv configurations."""
        conv = Conv(c1, c2, k=k, s=s)
        x = torch.randn(1, c1, 32, 32)
        output = conv(x)
        
        expected_h = 32 // s
        expected_w = 32 // s
        assert output.shape == (1, c2, expected_h, expected_w)


class TestBottleneckModules:
    """Test bottleneck and CSP modules."""
    
    def test_bottleneck_basic(self):
        """Test basic Bottleneck module."""
        bottleneck = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 32, 32)
        output = bottleneck(x)
        
        assert output.shape == (1, 64, 32, 32)
    
    def test_bottleneck_no_shortcut(self):
        """Test Bottleneck without shortcut."""
        bottleneck = Bottleneck(64, 128, shortcut=False)
        x = torch.randn(1, 64, 32, 32)
        output = bottleneck(x)
        
        assert output.shape == (1, 128, 32, 32)
    
    def test_bottleneck_different_channels(self):
        """Test Bottleneck with different input/output channels."""
        bottleneck = Bottleneck(64, 128, shortcut=True)  # Different channels
        x = torch.randn(1, 64, 32, 32)
        output = bottleneck(x)
        
        # Should not add shortcut when channels differ
        assert output.shape == (1, 128, 32, 32)
    
    def test_c3_module(self):
        """Test C3 CSP module."""
        c3 = C3(64, 128, n=2)  # 2 bottleneck layers
        x = torch.randn(1, 64, 32, 32)
        output = c3(x)
        
        assert output.shape == (1, 128, 32, 32)
    
    def test_c3_different_n(self):
        """Test C3 module with different number of layers."""
        for n in [1, 2, 3, 5]:
            c3 = C3(64, 128, n=n)
            x = torch.randn(1, 64, 32, 32)
            output = c3(x)
            
            assert output.shape == (1, 128, 32, 32)


class TestPoolingModules:
    """Test pooling modules."""
    
    def test_spp_module(self):
        """Test SPP (Spatial Pyramid Pooling) module."""
        spp = SPP(256, 512, k=(5, 9, 13))
        x = torch.randn(1, 256, 32, 32)
        output = spp(x)
        
        assert output.shape == (1, 512, 32, 32)
    
    def test_sppf_module(self):
        """Test SPPF (SPP-Fast) module."""
        sppf = SPPF(256, 512, k=5)
        x = torch.randn(1, 256, 32, 32)
        output = sppf(x)
        
        assert output.shape == (1, 512, 32, 32)
    
    def test_spp_different_kernels(self):
        """Test SPP with different kernel sizes."""
        spp1 = SPP(128, 256, k=(3, 5, 7))
        spp2 = SPP(128, 256, k=(5, 9, 13))
        
        x = torch.randn(1, 128, 16, 16)
        
        output1 = spp1(x)
        output2 = spp2(x)
        
        assert output1.shape == output2.shape == (1, 256, 16, 16)


class TestFocusModule:
    """Test Focus module."""
    
    def test_focus_basic(self):
        """Test basic Focus module."""
        focus = Focus(3, 32, k=3)
        x = torch.randn(1, 3, 640, 640)
        output = focus(x)
        
        # Should reduce spatial dimensions by 2x and increase channels
        assert output.shape == (1, 32, 320, 320)
    
    def test_focus_different_input_sizes(self):
        """Test Focus with different input sizes."""
        focus = Focus(3, 32, k=3)
        
        # Test different input sizes
        for size in [320, 640, 1280]:
            x = torch.randn(1, 3, size, size)
            output = focus(x)
            
            expected_size = size // 2
            assert output.shape == (1, 32, expected_size, expected_size)


class TestActivationFunctions:
    """Test activation functions."""
    
    def test_silu_activation(self):
        """Test SiLU activation function."""
        if hasattr(nn, 'SiLU'):
            silu = nn.SiLU()
        else:
            silu = SiLU()  # Custom implementation
        
        x = torch.randn(1, 64, 32, 32)
        output = silu(x)
        
        assert output.shape == x.shape
        # Test that it's different from input (unless input is specific values)
        assert not torch.equal(output, x)
    
    def test_hardswish_activation(self):
        """Test Hardswish activation function."""
        try:
            hardswish = Hardswish()
            x = torch.randn(1, 64, 32, 32)
            output = hardswish(x)
            
            assert output.shape == x.shape
        except NameError:
            pytest.skip("Hardswish not available")
    
    def test_memory_efficient_swish(self):
        """Test MemoryEfficientSwish activation."""
        try:
            me_swish = MemoryEfficientSwish()
            x = torch.randn(1, 64, 32, 32)
            output = me_swish(x)
            
            assert output.shape == x.shape
        except NameError:
            pytest.skip("MemoryEfficientSwish not available")


class TestDetectionHeads:
    """Test detection head modules."""
    
    def test_detect_head_init(self):
        """Test Detect head initialization."""
        nc = 80  # COCO classes
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        ch = [256, 512, 1024]  # Input channels for 3 detection layers
        
        detect = Detect(nc=nc, anchors=anchors, ch=ch)
        
        assert detect.nc == nc
        assert detect.no == nc + 5  # classes + box + objectness
        assert detect.nl == len(anchors)  # number of detection layers
        assert detect.na == len(anchors[0]) // 2  # number of anchors per layer
    
    def test_detect_head_forward_training(self):
        """Test Detect head forward pass in training mode."""
        nc = 80
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        ch = [256, 512, 1024]
        
        detect = Detect(nc=nc, anchors=anchors, ch=ch)
        detect.training = True
        
        # Create inputs for 3 detection layers
        inputs = [
            torch.randn(1, 256, 80, 80),   # P3
            torch.randn(1, 512, 40, 40),   # P4  
            torch.randn(1, 1024, 20, 20),  # P5
        ]
        
        outputs = detect(inputs)
        
        # In training mode, should return raw outputs
        assert len(outputs) == 3
        assert outputs[0].shape == (1, 3, 85, 80, 80)  # 3 anchors, 85 outputs
        assert outputs[1].shape == (1, 3, 85, 40, 40)
        assert outputs[2].shape == (1, 3, 85, 20, 20)
    
    def test_detect_head_forward_inference(self):
        """Test Detect head forward pass in inference mode."""
        nc = 80
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        ch = [256, 512, 1024]
        
        detect = Detect(nc=nc, anchors=anchors, ch=ch)
        detect.training = False
        detect.stride = torch.tensor([8., 16., 32.])  # Set stride for inference
        
        inputs = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 1024, 20, 20),
        ]
        
        outputs = detect(inputs)
        
        # In inference mode, should return processed outputs
        assert len(outputs) == 2  # (inference_output, raw_outputs)
        inference_out, raw_out = outputs
        
        # Check inference output shape
        assert inference_out.shape[0] == 1  # batch size
        assert inference_out.shape[2] == 85  # 80 classes + 5 box params
    
    def test_segment_head_init(self):
        """Test Segment head initialization."""
        nc = 91  # COCO + stuff classes
        anchors = [[10, 13], [30, 61], [116, 90]]
        nm = 32  # number of masks
        ch = [256, 512, 1024]
        
        segment = Segment(nc=nc, anchors=anchors, nm=nm, ch=ch)
        
        assert segment.nc == nc
        assert segment.nm == nm
        assert segment.no == nc + 5  # classes + box + objectness
        assert len(segment.m) == len(ch)  # output convs
        assert len(segment.mn) == len(ch)  # mask convs
    
    def test_classify_head(self):
        """Test Classify head."""
        c1, c2 = 1024, 1000  # Input channels, output classes
        
        classify = Classify(c1, c2)
        
        # Test with single feature map
        x = torch.randn(1, c1, 7, 7)
        output = classify(x)
        
        assert output.shape == (1, c2)
        
        # Test with list of feature maps
        x_list = [torch.randn(1, c1//2, 7, 7), torch.randn(1, c1//2, 7, 7)]
        output = classify(x_list)
        
        assert output.shape == (1, c2)


# Integration tests
class TestModuleIntegration:
    """Test integration between different modules."""
    
    def test_backbone_integration(self):
        """Test integration of backbone modules."""
        # Simulate a simple backbone
        x = torch.randn(1, 3, 640, 640)
        
        # Focus -> Conv -> C3 -> Conv
        focus = Focus(3, 32, k=3)
        conv1 = Conv(32, 64, k=3, s=2)
        c3_1 = C3(64, 64, n=1)
        conv2 = Conv(64, 128, k=3, s=2)
        
        x = focus(x)      # 1, 32, 320, 320
        x = conv1(x)      # 1, 64, 160, 160  
        x = c3_1(x)       # 1, 64, 160, 160
        x = conv2(x)      # 1, 128, 80, 80
        
        assert x.shape == (1, 128, 80, 80)
    
    def test_fpn_integration(self):
        """Test Feature Pyramid Network integration."""
        # Simulate FPN-like structure
        p5 = torch.randn(1, 1024, 20, 20)
        p4 = torch.randn(1, 512, 40, 40)  
        p3 = torch.randn(1, 256, 80, 80)
        
        # Upsampling and concatenation
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        concat = Concat(dimension=1)
        
        # P5 -> P4
        p5_up = upsample(Conv(1024, 512, k=1)(p5))
        p4_enhanced = concat([p4, p5_up])
        
        # P4 -> P3  
        p4_up = upsample(Conv(1024, 256, k=1)(p4_enhanced))
        p3_enhanced = concat([p3, p4_up])
        
        assert p4_enhanced.shape == (1, 1024, 40, 40)  # 512 + 512
        assert p3_enhanced.shape == (1, 512, 80, 80)   # 256 + 256


@pytest.mark.slow
class TestModulePerformance:
    """Test module performance characteristics."""
    
    def test_conv_memory_efficiency(self):
        """Test Conv module memory usage."""
        conv = Conv(64, 128, k=3, s=1)
        
        # Test with large input
        x = torch.randn(4, 64, 512, 512)  # Large batch/image
        
        with torch.no_grad():
            output = conv(x)
            
        assert output.shape == (4, 128, 512, 512)
    
    def test_spp_computation_efficiency(self):
        """Test SPP vs SPPF efficiency."""
        x = torch.randn(1, 256, 32, 32)
        
        spp = SPP(256, 512, k=(5, 9, 13))
        sppf = SPPF(256, 512, k=5)
        
        # Both should produce similar output shapes
        spp_out = spp(x)
        sppf_out = sppf(x)
        
        assert spp_out.shape == sppf_out.shape
