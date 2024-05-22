from __future__ import print_function

import torch
import torch.nn as nn

class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = compute_pearson(f_s, f_t)
        return loss

def compute_pearson(input1,input2):
    """
    input: tensor [b,c,n] or [b,c,h,w]
    return: pearson correlation coefficient [c,c]
    """
    batch1, channel1,_ ,_= input1.shape
    input1 = input1.view(batch1, channel1,-1)
    mean1 = torch.mean(input1,dim=0).unsqueeze(0)
    batch2, channel2, _ ,_= input2.shape
    input2 = input2.view(batch2, channel2, -1)
    mean2 = torch.mean(input2, dim=0).unsqueeze(0)

    cov1 = torch.matmul(input1-mean1,(input1-mean1).permute(0,2,1))
    cov2 = torch.matmul(input2 - mean2, (input2 - mean2).permute(0, 2, 1))
    cov = torch.matmul(input1 - mean1, (input2 - mean2).permute(0, 2, 1))


    diag1 = torch.sum(torch.eye(channel1).unsqueeze(0).cuda() * cov1,dim=2).view(batch1,channel1,-1)
    diag2 = torch.sum(torch.eye(channel2).unsqueeze(0).cuda() * cov2, dim=2).view(batch2, channel2, -1)
    stddev = torch.sqrt(torch.matmul(diag1,diag2.permute(0,2,1)))
    pearson = torch.div(cov,stddev)
    pearson = torch.abs(pearson)
    pearson=pearson.mean()
    return pearson


def pearson_correlation( x, y, eps=1e-8):
    '''
    Arguments
    ---------
    x1 : 3D torch.Tensor
    x2 : 3D torch.Tensor
    batch dim first
    '''
    batch, channel, _, _ = x.shape
    x=x.view(batch, channel,-1)
    y=y.view(batch, channel,-1)
    mean_x = torch.mean(x, dim=2, keepdim=True)
    mean_y = torch.mean(y, dim=2, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    # dot product
    r_num = torch.sum(torch.mul(xm, ym), dim=2, keepdim=True)
    r_den = torch.norm(xm, 2, dim=2, keepdim=True) * torch.norm(ym, 2, dim=2, keepdim=True)
    r_den[torch.where(r_den == 0)] = 1.000  # avoid division by zero
    r_val = r_num / r_den
    r_val = r_val.mean()
    return r_val
