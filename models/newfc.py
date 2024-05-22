'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict
from models.util import FcLayer

class Transform(nn.Module):
    def __init__(self, in_planes, planes):
        super(Transform, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SimilarityNet(nn.Module):
    """ This class is responsible for inserting a linear transformation
        layer between two networks with identical architecture. The user has
        a choice to index at which layer should the transformation happen.
        If the indexed layer is a convolution, the first model is transferred
        to the second one by a conv1x1 layer. If the layer is not a convolution
        but a linear layer, the transformation will be a linear layer too.
    """
    # ==============================================================
    # CONSTRUCTORS
    # ==============================================================
    def __init__(self,
                 backbone_path,
                 fc_path,
                 front_layer_name,
                 end_layer_name,
                 dataset_name: str,
                 n_cls
):
        super().__init__()
        # Init variables
        self.backbone_path = backbone_path
        self.fc_path = fc_path
        self.front_layer_name = front_layer_name
        self.end_layer_name = end_layer_name
        self.dataset_name = dataset_name
        self.n_cls = n_cls


        # Derived variables
        self.fc = self.load_fc(self.fc_path)

        self.transform = self.load_transform()


    def load_fc(self,model_path):
        print('==> loading fc model')
        fclayer = FcLayer()
        fclayer.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return fclayer

    def load_transform(self):
        print('==> loading fc model')
        transform = Transform(512,512)
        transform.load_state_dict(torch.load('./save/models/selfdis/ResNet18_cifar100_lr_0.05_decay_0.0005_trial_0/tansform_best.pth')['model'])
        print('==> done')
        return transform

