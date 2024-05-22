'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
from torchvision.models import resnet50
from thop import profile

class ReductionConv1(nn.Module):

    def __init__(self, in_planes, planes):
        super(ReductionConv1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out
class SepConv1(nn.Module):

    def __init__(self, channel_input,channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv1, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_input, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
class SepConv(nn.Module):

    def __init__(self, channel_input,channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_input, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),

        )

    def forward(self, x):
        return self.op(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)


        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = self.relu(out)

        return out
class BottleneckConv(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BottleneckConv, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = self.relu(out)

        return out

class Transform():
    def __init__(self, in_planes, planes):
        super(Transform, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.transform(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv5=SepConv1(512,128,512,stride=1)
        self.conv6 = SepConv1(1024, 256, 1024, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv11 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.scala1 = nn.Sequential(

            BottleneckConv(
                in_planes=64,
                planes=32,
                stride=2
            ),

        )
        self.scala2 = nn.Sequential(

            BottleneckConv(
                in_planes=128,
                planes=64,
                stride=2
            ),

        )
        self.scala3 = nn.Sequential(

            BottleneckConv(
                in_planes=256,
                planes=128,
                stride=2
            ),
            nn.AvgPool2d(4, 4)

        )

        self.scala4 = nn.Sequential(

            SepConv(
                channel_input=64,
                channel_in=64,
                channel_out=128
            ),

        )
        self.scala5 = nn.Sequential(

            SepConv(
                channel_input=128,
                channel_in=128,
                channel_out=256
            ),

        )
        self.scala6 = nn.Sequential(

            SepConv(
                channel_input=256,
                channel_in=256,
                channel_out=512
            ),

        )
        self.scala7 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),

        )
        self.scala8 = nn.Sequential(

            BasicBlock(
                in_planes=512,
                planes=1024,
                stride=2
            ),

        )
        self.scala9 = nn.Sequential(

            BasicBlock(
                in_planes=1024,
                planes=2048,
                stride=2
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala10 = nn.Sequential(

            BasicBlock(
                in_planes=1024,
                planes=1024,
            ),
        )
        self.scala11 = nn.Sequential(
            ReductionConv1(256, 2048),
        )
        self.scala12 = nn.Sequential(
            ReductionConv1(512, 2048),
        )
        self.scala13 = nn.Sequential(
            ReductionConv1(1024, 2048),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 1024 // 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024 // 32, 1024, kernel_size=1),
            nn.Sigmoid()
        )
        self.eca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.scala3(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)



        #out = self.conv3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
            else:
                return [f0, f1, f2, f3, f4, f5], out
        else:
            return out


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    net = ResNet34(num_classes=100)
    print(net)
    stat(net, (3, 32, 32))


    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, (input,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))




    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
