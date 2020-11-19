from __future__ import print_function
import scipy.io as sio
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import numpy as np
import torchvision
import models
from torch.autograd import Variable
from torchvision import transforms

import reader
import utils


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = extracted_layers
#
#     def forward(self, x):
#         outputs = []
#         for name, module in self.submodule._modules.items():
#             x = module(x)
#             if name in self.extracted_layers:
#                 outputs += [x]
#         return outputs + [x]


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)

        output = nn.functional.sigmoid(output)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


feat_test = []
def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)
            output = model(image)
            output = output.tolist()
            feat_test.append(output)

    #         loss = criterion(output, target)
    #
    #         acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    #         # FIXME need to take into account that the datasets
    #         # could have been padded in distributed setup
    #         batch_size = image.shape[0]
    #         metric_logger.update(loss=loss.item())
    #         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    #
    # print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return feat_test


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    # traindir = os.path.join(args.data_path, 'train.txt')
    # valdir = os.path.join(args.data_path, 'val.txt')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    traindir = r'./datasets/Corel5k/train.txt'
    valdir = r'./datasets/Corel5k/val.txt'

    print("Creating data loaders")

    data_loader, data_loader_test = reader.read_data(traindir=traindir, valdir=valdir,
                                                     batch_size=args.batch_size, num_works=args.workers)

    print("Creating model")
    model = models.__dict__[args.model](pretrained=True)
    model.to(device)
    if args.distributed:
        model = torch.nn.utils.convert_sync_batchnorm(model)

