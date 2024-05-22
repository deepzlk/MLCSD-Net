from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosSimilarity(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """

    def __init__(self):
        super(CosSimilarity, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(feature1, feature2):
        # Normalize each vector by its norm
        # 假设feature1为N*C*W*H， feature2也为N*C*W*H（基本网络中的tensor都是这样）
        feature1 = feature1.view(feature1.shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = feature2.view(feature2.shape[0], -1)
        feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
        feature2 = F.normalize(feature2)
        distance = torch.mm(feature1, feature2.transpose(0, 1))
        loss = torch.mean(distance)
        return loss
    @staticmethod
    def lance_loss(feature1, feature2, eps=0.0000001):
        # Normalize each vector by its norm
        # 假设feature1为N*C*W*H， feature2也为N*C*W*H（基本网络中的tensor都是这样）
        feature1 = feature1.view(feature1.shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = feature2.view(feature2.shape[0], -1)
        distance1=feature1-feature2
        distance1=torch.abs(distance1)
        distance2 = feature1 + feature2
        distance=distance1/(distance2+eps)
        loss = 1-torch.mean(distance)
        return loss

    @staticmethod
    def euclidean_loss(feature1, feature2, eps=0.0000001):
        # Normalize each vector by its norm
        # 假设feature1为N*C*W*H， feature2也为N*C*W*H（基本网络中的tensor都是这样）
        feature1 = feature1.view(feature1.shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = feature2.view(feature2.shape[0], -1)

        distance = feature1 - feature2
        distance = distance.pow(2).sum()

        distance=distance**0.5
        loss = 1 - 1/(1+distance)
        return loss