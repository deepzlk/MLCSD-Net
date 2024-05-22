from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.tiny_imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_menm as train, validate_menm as validate
from models.util import FcLayer
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=9.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=100.0, help='weight balance for other losses')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # dataset
    parser.add_argument('--model', type=str, default='ShuffleV2',
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
                                 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'MobileNetV1',])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','cifar10','tiny-imagenet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--path_t', type=str, default='./save/models/teacher/ResNet18_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet18_best.pth',
                        help='teacher model snapshot')

    opt = parser.parse_args()
    opt.teachertest_path = './result'

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2','MobileNetV1',]:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models/menm'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    vallog_headers = [
        'model',
        'trial',
        'bestepoch',
        'bestacc',
        'bestacc1_8',
        'bestacc2_8',
        'bestacc3_8',
        'bestacc4_8',
        'bestacc5_8',
        'bestacc6_8',
        'bestacc7_8',
        'bestacc8_8',
    ]
    if not os.path.exists(opt.teachertest_path):
        os.makedirs(opt.teachertest_path)
    if not os.path.exists(os.path.join(opt.teachertest_path, 'menm_test.csv')):
        with open(os.path.join(opt.teachertest_path, 'menm_test.csv'), 'w') as f:
            f.write(','.join(vallog_headers) + '\n')

    return opt


def get_backbone_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_backbone(model_path, n_cls):
    print('==> loading backbone model')
    model_t = get_backbone_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return model


def main():
    best_acc = 0
    best_acc_1_8 = 0
    best_acc_2_8 = 0
    best_acc_3_8 = 0
    best_acc_4_8 = 0
    best_acc_5_8 = 0
    best_acc_6_8 = 0
    best_acc_7_8 = 0
    best_acc_8_8 = 0
    best_epoch = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    elif opt.dataset == 'tiny-imagenet':
        train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = load_backbone(opt.path_t, n_cls)
    print("model_t", model)
    fclayer = FcLayer(num_classes=n_cls)
    print("fclayer", fclayer)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(fclayer)

    # loss
    criterion_cls = nn.CrossEntropyLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        trainable_list.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, trainable_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        top1_8_acc, top2_8_acc, top3_8_acc, top4_8_acc, top5_8_acc, top6_8_acc, top7_8_acc, top8_8_acc,  test_acc_top5, test_loss = \
            validate(val_loader, trainable_list, criterion_list, opt)

        logger.log_value('test_acc_top8_8', top8_8_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if top8_8_acc > best_acc:
            best_acc = top8_8_acc
            best_epoch = epoch
            best_acc_1_8 = top1_8_acc
            best_acc_2_8 = top2_8_acc
            best_acc_3_8 = top3_8_acc
            best_acc_4_8 = top4_8_acc
            best_acc_5_8 = top5_8_acc
            best_acc_6_8 = top6_8_acc
            best_acc_7_8 = top7_8_acc
            best_acc_8_8 = top8_8_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    with open(os.path.join(opt.teachertest_path, 'menm_test.csv'), 'a') as f:
        log = [opt.model, opt.trial, best_epoch, best_acc.item(),
               best_acc_1_8.item(), best_acc_2_8.item(), best_acc_3_8.item(), best_acc_4_8.item(), best_acc_5_8.item(), best_acc_6_8.item(), best_acc_7_8.item(),best_acc_8_8.item()]
        log = map(str, log)
        f.write(','.join(log) + '\n')

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
