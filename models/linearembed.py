import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEmbed(nn.Module):
    
    def __init__(self, dim_in=512, dim_out=128):
        super(LinearEmbed, self).__init__()

        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)
        self.linear3 = nn.Linear(dim_in, dim_out)
        self.linear4 = nn.Linear(dim_in, dim_out)
        self.linear5 = nn.Linear(dim_in, dim_out)
        self.linear6 = nn.Linear(dim_in, dim_out)
        self.linear7 = nn.Linear(dim_in, dim_out)
        self.linear8 = nn.Linear(dim_in, dim_out)


    def forward(self, x):
        # storing direct feature
        direct_feature = x
        linear=[]
        linear7 = self.linear1(x[7].view(x[7].shape[0], -1))
        linear6 = self.linear1(x[6].view(x[6].shape[0], -1))
        linear5 = self.linear1(x[5].view(x[5].shape[0], -1))
        linear4 = self.linear1(x[4].view(x[4].shape[0], -1))
        linear3 = self.linear1(x[3].view(x[3].shape[0], -1))
        linear2 = self.linear1(x[2].view(x[2].shape[0], -1))
        linear1 = self.linear1(x[1].view(x[1].shape[0], -1))
        linear0 = self.linear1(x[0].view(x[0].shape[0], -1))

        linear.append(linear0)
        linear.append(linear1)
        linear.append(linear2)
        linear.append(linear3)
        linear.append(linear4)
        linear.append(linear5)
        linear.append(linear6)
        linear.append(linear7)
        # hal_scale = torch.softmax(domain_code, -1)
        return  linear


