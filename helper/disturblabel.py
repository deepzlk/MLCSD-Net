import torch
import torch.nn as nn


class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, C):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = C
        # Multinoulli distribution
        self.p_c = (1 - ((C - 1)/C) * (alpha/100))
        self.p_i = (1 / C) * (alpha / 100)

    def forward(self, y):
        # convert classes to index
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)

        # create disturbed labels
        depth = self.C
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))

        # sample from Multinoulli distribution
        distribution = torch.distributions.OneHotCategorical(y_one_hot)
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed


import torch
import torch.nn.functional as F
class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss