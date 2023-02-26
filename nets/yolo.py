import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv, SiLU, autopad
from utils.utils_bbox import make_anchors

class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        # 80, 80, 128 => 40, 40, 256
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 40, 40, 256 => 20, 20, 512
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 20, 20, 1024 => 20, 20, 512
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        #------------------------加强特征提取网络------------------------# 
        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes
        
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    # def fuse(self):
    #     print('Fusing layers... ')
    #     for m in self.modules():
    #         if isinstance(m, RepConv):
    #             m.fuse_repvgg_block()
    #         elif type(m) is Conv and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)
    #             delattr(m, 'bn')
    #             m.forward = m.fuseforward
    #     return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        #------------------------加强特征提取网络------------------------# 
        # 20, 20, 256 => 40, 40, 256
        P5_upsample = self.upsample(feat3)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 512 => 40, 40, 256
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 128 => 80, 80, 128
        P4_upsample = self.upsample(P4)
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 256 => 80, 80, 128
        P3          = self.conv3_for_upsample2(P3)

        # 80, 80, 128 => 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 256 => 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 => 20, 20, 1024
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------# 
        # P3 80, 80, 128 
        # P4 40, 40, 256
        # P5 20, 20, 512
        shape = P3.shape  # BCHW
        
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)