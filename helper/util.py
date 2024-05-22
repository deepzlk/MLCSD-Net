from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class SPLLoss(nn.NLLLoss):
    def __init__(self, n_samples=0):
        super(SPLLoss, self).__init__()
        self.n_samples = n_samples
        self.SamplesPercentage = 50
        self.Increase = 10
        self.steplength = 40
        self.v = torch.ones(n_samples).int()

    def forward(self, input, target):
        loss = nn.functional.cross_entropy(input, target, reduction="none")
        sample_loss=loss
        sorted_samples = np.argsort(sample_loss.cpu().detach().numpy())

        n_samples = int(np.round(self.SamplesPercentage * self.n_samples / 100))

        # Select the classes (number given by ClassesForLoss) to modify the final loss
        in_samples = sorted_samples[0:n_samples]
        out_samples = sorted_samples[n_samples:]

        assert (len(in_samples) + len(out_samples) == len(sorted_samples))

        # Update v
        self.v[out_samples] = 0
        self.v[in_samples] = 1

        self.v=self.v.int()

        w_loss = loss * self.v[target].cuda()

        return w_loss.mean()

    def increase_classes(self, epoch):
        if (self.SamplesPercentage < 100) and (np.mod(epoch, self.steplength) == 0) and (epoch != 0):
            self.SamplesPercentage += self.Increase

    def update_weigths(self, sample_loss):
        sorted_samples = np.argsort(sample_loss)

        n_samples = int(np.round(self.SamplesPercentage * self.n_samples / 100))

        # Select the classes (number given by ClassesForLoss) to modify the final loss
        in_samples = sorted_samples[0:n_samples]
        out_samples = sorted_samples[n_samples:]

        assert (len(in_samples) + len(out_samples) == len(sorted_samples))

        # Update v
        self.v[out_samples] = 0
        self.v[in_samples] = 1

        return self.v.int()





# class SPLLoss(nn.NLLLoss):
#     def __init__(self, *args, alpha, beta, n_samples=0, **kwargs):
#         super(SPLLoss, self).__init__(*args, **kwargs)
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.threshold = alpha
#         self.growing_factor = beta
#         self.v = torch.zeros(n_samples).int().to(self.device)
#
#     def forward(self, input, target) :
#         super_loss = nn.functional.cross_entropy(input, target, reduction="none")
#         print("super_loss",super_loss)
#         v = self.spl_loss(super_loss)
#         self.v[target] = v
#         return (super_loss * v).mean()
#
#     def increase_threshold(self):
#         self.threshold *= self.growing_factor
#
#     def spl_loss(self, super_loss):
#         v = super_loss < self.threshold
#         return v.int()

class SCELoss(torch.nn.Module):
    '''
    2019 - iccv - Symmetric cross entropy for robust learning with noisy labels
    '''
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, target):
        pred = self.softmax(pred)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        ce = (-1*torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        loss = self.alpha * ce + self.beta * rce

        return loss

class DisturbLabel(torch.nn.Module):
    '''
    2016 - cvpr - DisturbLabel: Regularizing CNN on the Loss Layer
    '''
    def __init__(self, num_classes):
        super(DisturbLabel, self).__init__()
        self.noisy_rate = 0.1
        self.num_classes = num_classes
        self.bound = (num_classes - 1.) / float(num_classes) * self.noisy_rate
    def forward(self, output, target):
        batchsize = output.shape[0]
        new_target = target.clone()
        for kk in range(batchsize):
            r = torch.rand(1)
            if r < self.bound:
                dlabel = torch.randint(low=0, high=self.num_classes, size=(1,))
                while new_target[kk] == dlabel[0]:
                    dlabel = torch.randint(low=0, high=self.num_classes, size=(1,))
                new_target[kk] = dlabel[0]
        return torch.nn.functional.cross_entropy(output, new_target)

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def similarity(output, target):
    """Computes the error@k for the specified values of k"""
    output = F.softmax(output, dim=1)
    batch_size = target.size(0)

    confidence, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_confidence_sum = 0.0
    incorrect_confidence_sum = 0.0
    res = []

    correct_k = correct.contiguous().view(-1)
    # Calculate confidence for correctly classified samples
    correct_confidence = torch.sum(confidence[correct_k])
    # Calculate confidence for incorrectly classified samples
    incorrect_confidence = torch.sum(confidence[~correct_k])

    correct_confidence_sum += correct_confidence
    incorrect_confidence_sum += incorrect_confidence

    accuracy_k = correct_k.float().sum(0)
    res.append(accuracy_k)
    res.append(correct_confidence_sum/accuracy_k)
    res.append(incorrect_confidence_sum/accuracy_k)

    return res

def increase(output1, output2,target):
    """Computes the error@k for the specified values of k"""
    output1 = F.softmax(output1, dim=1)
    output2 = F.softmax(output2, dim=1)
    batch_size = target.size(0)
    res = []

    misclassified_samples = (output2.argmax(1) != target) & (output1.argmax(1) == target)
    # Filter out the misclassified samples and calculate their prediction confidence in Group 2
    total_samples = misclassified_samples.sum().item()
    if(total_samples!=0):

        misclassified_samples_output_group1 = output1[misclassified_samples]
        misclassified_samples_output_group2 = output2[misclassified_samples]

        # Calculate the prediction confidence for each of these samples in Group 2
        confidence_misclassified_samples_group1 = misclassified_samples_output_group1.max(1).values.mean()
        confidence_misclassified_samples_group2 = misclassified_samples_output_group2.max(1).values.mean()


        res.append(total_samples)
        res.append(confidence_misclassified_samples_group1)
        res.append(confidence_misclassified_samples_group2)

        return res
    else:
        return None,None,None

if __name__ == '__main__':

    pass
