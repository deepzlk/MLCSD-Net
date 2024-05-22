import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaEmbedding(nn.Module):
    
    def __init__(self, feat_dim=512, num_domain=8):
        super(MetaEmbedding, self).__init__()
        self.num_domain = num_domain
        self.hallucinator1 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator2 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator3 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator4 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator5 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator6 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator7 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.hallucinator8 = nn.Sequential(
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )


    def forward(self, x):
        # storing direct feature
        direct_feature = x
        hal_scale=[]
        hal_scale7 = self.hallucinator1(x[7])
        hal_scale6 = self.hallucinator1(x[6])
        hal_scale5 = self.hallucinator1(x[5])
        hal_scale4 = self.hallucinator1(x[4])
        hal_scale3 = self.hallucinator1(x[3])
        hal_scale2 = self.hallucinator1(x[2])
        hal_scale1 = self.hallucinator1(x[1])
        hal_scale0 = self.hallucinator1(x[0])

        hal_scale.append(hal_scale0)
        hal_scale.append(hal_scale1)
        hal_scale.append(hal_scale2)
        hal_scale.append(hal_scale3)
        hal_scale.append(hal_scale4)
        hal_scale.append(hal_scale5)
        hal_scale.append(hal_scale6)
        hal_scale.append(hal_scale7)
        # hal_scale = torch.softmax(domain_code, -1)
        return  hal_scale

def build_MetaEmbedding(feat_dim, num_domain):
    return MetaEmbedding(feat_dim, num_domain)
