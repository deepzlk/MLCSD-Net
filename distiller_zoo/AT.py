from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        f_s,_ = torch.max(f_s, dim=1, keepdim=True)
        f_t,_ = torch.max(f_t, dim=1, keepdim=True)

        return (f_s - f_t).pow(2).mean()


    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


