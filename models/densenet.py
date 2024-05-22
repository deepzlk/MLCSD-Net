"""dense net in pytorch
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn
from torchstat import stat
from .general import GeneralNet



#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out=torch.cat([x, self.bottle_neck(x)], 1)
        out = self.relu(out)
        return out

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(GeneralNet):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        # 1st block
        self.layer1=self._make_dense_layers(block, inner_channels, nblocks[0])
        inner_channels += growth_rate * nblocks[0]
        out_channels = int(reduction * inner_channels)
        self.trans1 = Transition(inner_channels, out_channels)
        inner_channels = out_channels
        # 2st block
        self.layer2 = self._make_dense_layers(block, inner_channels, nblocks[1])
        inner_channels += growth_rate * nblocks[1]
        out_channels = int(reduction * inner_channels)
        self.trans2 = Transition(inner_channels, out_channels)
        inner_channels = out_channels
        # 3st block
        self.layer3 = self._make_dense_layers(block, inner_channels, nblocks[2])
        inner_channels += growth_rate * nblocks[2]
        out_channels = int(reduction * inner_channels)
        self.trans3 = Transition(inner_channels, out_channels)
        inner_channels = out_channels
        # 4st block
        self.layer4 = self._make_dense_layers(block, inner_channels, nblocks[3])
        inner_channels += growth_rate * nblocks[3]
        out_channels = int(reduction * inner_channels)
        self.trans4 = Transition(inner_channels, out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.layer1(out)
        f1 = out
        out = self.trans1(out)
        out = self.layer2(out)
        f2 = out
        out = self.trans2(out)
        out = self.layer3(out)
        f3 = out
        out = self.trans3(out)
        out = self.layer4(out)
        f4 = out

        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
            else:
                return [f1, f2, f3, f4], out
        else:
            return out

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

if __name__ == '__main__':
    net = densenet121()
    stat(net, (3, 32, 32))

    x = torch.randn(2, 3, 32, 32)
    print(net)
    logit = net(x)

