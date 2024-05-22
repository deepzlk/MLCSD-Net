from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy,similarity,increase
import torch.nn.functional as F
import torch.nn as nn

from distiller_zoo import cka_torch


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        
        output = model(input, is_feat=False)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg
def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def train_menm(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """vanilla training"""
    for module in module_list:
        module.train()
    model_feature = module_list[0]
    model_fc = module_list[1]

    criterion_cls = criterion_list[0]

    correct1 = [0 for _ in range(9)]
    correct5 = [0 for _ in range(9)]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        loss = torch.FloatTensor([0.]).cuda()
        loss_sum = torch.FloatTensor([0.]).cuda()
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
        outputs, outputs_feature=model_fc(feat_f)
        #hal_scale = embedding(outputs_feature)


        loss += criterion_cls(outputs[0], target)

        for index in range(1, len(outputs)):
            #   logits distillation
            loss += criterion_cls(outputs[index], target) * opt.gamma
            #   logits distillation


        for classifier_index in range(len(outputs)):
            acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
            correct1[classifier_index] = acc1[0]
            correct5[classifier_index] = acc5[0]

        losses.update(loss.item(), input.size(0))

        top1_8.update(correct1[7], input.size(0))
        top2_8.update(correct1[6], input.size(0))
        top3_8.update(correct1[5], input.size(0))
        top4_8.update(correct1[4], input.size(0))
        top5_8.update(correct1[3], input.size(0))
        top6_8.update(correct1[2], input.size(0))
        top7_8.update(correct1[1], input.size(0))
        top8_8.update(correct1[0], input.size(0))

        top8.update(correct5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8, top8=top8))
            sys.stdout.flush()

    print(' * Acc@1 {top8_8.avg:.3f} Acc@5 {top8.avg:.3f}'
          .format(top8_8=top8_8, top8=top8))

    return top8_8.avg, losses.avg
def validate_menm(val_loader, module_list, criterion_list, opt):
    """validation"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()
    loss = torch.FloatTensor([0.]).cuda()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_feature = module_list[0]
    model_fc = module_list[1]

    criterion_cls = criterion_list[0]

    correct1 = [0.0 for _ in range(9)]
    correct5 = [0.0 for _ in range(9)]

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
            outputs, outputs_feature = model_fc(feat_f)


            loss += criterion_cls(outputs[0], target)

            # measure accuracy and record loss

            for classifier_index in range(len(outputs)):
                acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
                correct1[classifier_index] = acc1[0]
                correct5[classifier_index] = acc5[0]

            losses.update(loss.item(), input.size(0))
            top1_8.update(correct1[7], input.size(0))
            top2_8.update(correct1[6], input.size(0))
            top3_8.update(correct1[5], input.size(0))
            top4_8.update(correct1[4], input.size(0))
            top5_8.update(correct1[3], input.size(0))
            top6_8.update(correct1[2], input.size(0))
            top7_8.update(correct1[1], input.size(0))
            top8_8.update(correct1[0], input.size(0))

            top8.update(correct5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8,top8=top8))
                sys.stdout.flush()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top8.avg:.3f}'
                  .format(top1=top8_8, top8=top8))

    return top1_8.avg,top2_8.avg,top3_8.avg,top4_8.avg,top5_8.avg,top6_8.avg,top7_8.avg,top8_8.avg, top8.avg,losses.avg


def train_self(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """vanilla training"""
    for module in module_list:
        module.train()
    model_feature = module_list[0]
    model_fc = module_list[1]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    correct1 = [0 for _ in range(9)]
    correct5 = [0 for _ in range(9)]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        loss = torch.FloatTensor([0.]).cuda()
        loss_sum = torch.FloatTensor([0.]).cuda()
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
        outputs, outputs_feature=model_fc(feat_f)

        teacher_output = outputs[0].detach()

        loss += criterion_cls(outputs[0], target)

        for index in range(1, len(outputs)):
            loss += criterion_cls(outputs[index], target) * opt.gamma

        for index in range(1, len(outputs)):
            #   logits distillation
            loss += criterion_div(outputs[index], teacher_output) * opt.alpha


        for classifier_index in range(len(outputs)):
            acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
            correct1[classifier_index] = acc1[0]
            correct5[classifier_index] = acc5[0]

        losses.update(loss.item(), input.size(0))

        top1_8.update(correct1[7], input.size(0))
        top2_8.update(correct1[6], input.size(0))
        top3_8.update(correct1[5], input.size(0))
        top4_8.update(correct1[4], input.size(0))
        top5_8.update(correct1[3], input.size(0))
        top6_8.update(correct1[2], input.size(0))
        top7_8.update(correct1[1], input.size(0))
        top8_8.update(correct1[0], input.size(0))

        top8.update(correct5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8, top8=top8))
            sys.stdout.flush()

    print(' * Acc@1 {top8_8.avg:.3f} Acc@5 {top8.avg:.3f}'
          .format(top8_8=top8_8, top8=top8))

    return top8_8.avg, losses.avg
def validate_self(val_loader, module_list, criterion_list, opt):
    """validation"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()
    loss = torch.FloatTensor([0.]).cuda()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_feature = module_list[0]
    model_fc = module_list[1]

    criterion_cls = criterion_list[0]

    correct1 = [0.0 for _ in range(9)]
    correct5 = [0.0 for _ in range(9)]

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
            outputs, outputs_feature = model_fc(feat_f)


            loss += criterion_cls(outputs[0], target)

            # measure accuracy and record loss

            for classifier_index in range(len(outputs)):
                acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
                correct1[classifier_index] = acc1[0]
                correct5[classifier_index] = acc5[0]

            losses.update(loss.item(), input.size(0))
            top1_8.update(correct1[7], input.size(0))
            top2_8.update(correct1[6], input.size(0))
            top3_8.update(correct1[5], input.size(0))
            top4_8.update(correct1[4], input.size(0))
            top5_8.update(correct1[3], input.size(0))
            top6_8.update(correct1[2], input.size(0))
            top7_8.update(correct1[1], input.size(0))
            top8_8.update(correct1[0], input.size(0))

            top8.update(correct5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8,top8=top8))
                sys.stdout.flush()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top8.avg:.3f}'
                  .format(top1=top8_8, top8=top8))

    return top1_8.avg,top2_8.avg,top3_8.avg,top4_8.avg,top5_8.avg,top6_8.avg,top7_8.avg,top8_8.avg, top8.avg,losses.avg


def train_mlcsd(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    for module in module_list:
        module.train()
    model_feature = module_list[0]
    model_fc = module_list[1]
    embedding = module_list[2]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_hint = criterion_list[2]

    correct1 = [0 for _ in range(9)]
    correct5 = [0 for _ in range(9)]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        loss = torch.FloatTensor([0.]).cuda()
        loss_sum = torch.FloatTensor([0.]).cuda()
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
        outputs, outputs_feature=model_fc(feat_f)
        hal_scale = embedding(outputs_feature)

        teacher_output = outputs[0].detach()

        loss += criterion_cls(outputs[0], target)

        for index in range(1, len(outputs)):
            #   logits distillation
            loss += criterion_cls(outputs[index], target) * opt.gamma
            #   logits distillation

        #   mlcsd teachers
        logit_t_list = []
        logit_st_list = []
        logit_f_list = []
        logit_sf_list = []
        for index in range(0, 8):
            logit_t_list.append(outputs[index])
            logit_f_list.append(outputs_feature[index].squeeze())
        logit_t_list = torch.stack(logit_t_list, dim=1)
        logit_f_list = torch.stack(logit_f_list, dim=1)

        for index in range(0, 8):
            logit_st = hal_scale[index].squeeze(3) * logit_t_list
            logit_st = torch.sum(logit_st, dim=1)
            logit_st_list.append(logit_st)
        for index in range(0, 8):
            logit_sf = hal_scale[index].squeeze(3) * logit_f_list
            logit_sf = torch.sum(logit_sf, dim=1)
            logit_sf_list.append(logit_sf)
        for index in range(0, 8):
            #   logits distillation
            loss += criterion_div(outputs[index], logit_st_list[index].detach()) * opt.alpha
            loss += criterion_hint(outputs_feature[index], logit_sf_list[index].detach()) * opt.beta


        for classifier_index in range(len(outputs)):
            acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
            correct1[classifier_index] = acc1[0]
            correct5[classifier_index] = acc5[0]

        losses.update(loss.item(), input.size(0))

        top1_8.update(correct1[7], input.size(0))
        top2_8.update(correct1[6], input.size(0))
        top3_8.update(correct1[5], input.size(0))
        top4_8.update(correct1[4], input.size(0))
        top5_8.update(correct1[3], input.size(0))
        top6_8.update(correct1[2], input.size(0))
        top7_8.update(correct1[1], input.size(0))
        top8_8.update(correct1[0], input.size(0))

        top8.update(correct5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8, top8=top8))
            sys.stdout.flush()

    print(' * Acc@1 {top8_8.avg:.3f} Acc@5 {top8.avg:.3f}'
          .format(top8_8=top8_8, top8=top8))

    return top8_8.avg, losses.avg
def validate_mlcsd(val_loader, module_list, criterion_list, opt):
    """validation"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_8 = AverageMeter()
    top2_8 = AverageMeter()
    top3_8 = AverageMeter()
    top4_8 = AverageMeter()
    top5_8 = AverageMeter()
    top6_8 = AverageMeter()
    top7_8 = AverageMeter()
    top8_8 = AverageMeter()
    top8 = AverageMeter()
    loss = torch.FloatTensor([0.]).cuda()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_feature = module_list[0]
    model_fc = module_list[1]

    criterion_cls = criterion_list[0]

    correct1 = [0.0 for _ in range(9)]
    correct5 = [0.0 for _ in range(9)]

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            feat_f, logit_f = model_feature(input, is_feat=True, preact=False)
            outputs, outputs_feature = model_fc(feat_f)


            loss += criterion_cls(outputs[0], target)

            # measure accuracy and record loss

            for classifier_index in range(len(outputs)):
                acc1, acc5 = accuracy(outputs[classifier_index], target, topk=(1, 5))
                correct1[classifier_index] = acc1[0]
                correct5[classifier_index] = acc5[0]

            losses.update(loss.item(), input.size(0))
            top1_8.update(correct1[7], input.size(0))
            top2_8.update(correct1[6], input.size(0))
            top3_8.update(correct1[5], input.size(0))
            top4_8.update(correct1[4], input.size(0))
            top5_8.update(correct1[3], input.size(0))
            top6_8.update(correct1[2], input.size(0))
            top7_8.update(correct1[1], input.size(0))
            top8_8.update(correct1[0], input.size(0))

            top8.update(correct5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1/8 {top1_8.val:.3f} ({top1_8.avg:.3f})\t'
                  'Acc@2/8 {top2_8.val:.3f} ({top2_8.avg:.3f})\t'
                  'Acc@3/8 {top3_8.val:.3f} ({top3_8.avg:.3f})\t'
                  'Acc@4/8 {top4_8.val:.3f} ({top4_8.avg:.3f})\t'
                  'Acc@5/8 {top5_8.val:.3f} ({top5_8.avg:.3f})\t'
                  'Acc@6/8 {top6_8.val:.3f} ({top6_8.avg:.3f})\t'
                  'Acc@7/8 {top7_8.val:.3f} ({top7_8.avg:.3f})\t'
                  'Acc@8/8 {top8_8.val:.3f} ({top8_8.avg:.3f})\t'
                  'Acc@5 {top8.val:.3f} ({top8.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    top1_8=top1_8,top2_8=top2_8,top3_8=top3_8,top4_8=top4_8,top5_8=top5_8,top6_8=top6_8,top7_8=top7_8,top8_8=top8_8,top8=top8))
                sys.stdout.flush()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top8.avg:.3f}'
                  .format(top1=top8_8, top8=top8))

    return top1_8.avg,top2_8.avg,top3_8.avg,top4_8.avg,top5_8.avg,top6_8.avg,top7_8.avg,top8_8.avg, top8.avg,losses.avg

