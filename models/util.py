from __future__ import print_function

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .general import GeneralNet

class SepConv1(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv1, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class FcLayer(nn.Module):

    def __init__(self,  num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(FcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            SepConv1(
                channel_in=64,
                channel_out=128
            ),
            SepConv1(
                channel_in=128,
                channel_out=256
            ),
            SepConv1(
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            SepConv1(
                channel_in=64,
                channel_out=128
            ),
            SepConv1(
                channel_in=128,
                channel_out=256
            ),
            SepConv1(
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala3 = nn.Sequential(
            SepConv1(
                channel_in=128,
                channel_out=256,
            ),
            SepConv1(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.Sequential(
            SepConv1(
                channel_in=128,
                channel_out=256,
            ),
            SepConv1(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala5 = nn.Sequential(
            SepConv1(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(
            SepConv1(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.scala8 = nn.AvgPool2d(4, 4)


        self.fc1 = nn.Linear(512 , num_classes)
        self.fc2 = nn.Linear(512 , num_classes)
        self.fc3 = nn.Linear(512 , num_classes)
        self.fc4 = nn.Linear(512 , num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        self.fc6 = nn.Linear(512, num_classes)
        self.fc7 = nn.Linear(512, num_classes)
        self.fc8 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x
        out1_feature = self.scala1(feature_list[0])
        out2_feature = self.scala2(feature_list[1])
        out3_feature = self.scala3(feature_list[2])
        out4_feature = self.scala4(feature_list[3])
        out5_feature = self.scala5(feature_list[4])
        out6_feature = self.scala6(feature_list[5])
        out7_feature = self.scala7(feature_list[6])
        out8_feature = self.scala8(feature_list[7])

        out1 = self.fc1(out1_feature.view(x[0].size(0), -1))
        out2 = self.fc2(out2_feature.view(x[0].size(0), -1))
        out3 = self.fc3(out3_feature.view(x[0].size(0), -1))
        out4 = self.fc4(out4_feature.view(x[0].size(0), -1))
        out5 = self.fc5(out5_feature.view(x[0].size(0), -1))
        out6 = self.fc6(out6_feature.view(x[0].size(0), -1))
        out7 = self.fc7(out7_feature.view(x[0].size(0), -1))
        out8 = self.fc8(out8_feature.view(x[0].size(0), -1))

        return [out8,out7, out6, out5,out4, out3, out2, out1], [out8_feature, out7_feature, out6_feature, out5_feature,out4_feature, out3_feature, out2_feature, out1_feature]