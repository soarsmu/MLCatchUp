# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import errno
import random
import numpy as np
from colorama import Fore
import pdb

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from modelbase.dataloader import lydataloader
from modelbase.args import parser
from modelbase.model import Autoencoder, weights_init
from modelbase.utils import save_checkpoint

# Parse args
args = parser.parse_args()

# Make dirs
try:
    os.makedirs(args.outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Fix seed for randomization
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# CUDA, CUDNN
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# cudnn.benchmark = True
cudnn.fastest = True

# Init model
if args.resume:  # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(Fore.YELLOW + args.resume + Fore.RESET))
        checkpoint = torch.load(args.resume)

        print("=> creating model")
        netAE = Autoencoder().cuda()

        netAE.load_state_dict(checkpoint['netAE_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})".
              format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".
              format(Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else:
    print("=> creating model")
    netAE = Autoencoder().cuda()
    print(netAE)
    netAE.apply(weights_init)

# Prepare data
task = args.task
featdirs = ('../timit_opensmile_feat', '../interrogation_opensmile_feat')
trainlists = ('./timit_train_featlist.ctl', './interrogation_train_featlist.ctl')
testlists = ('./timit_test_featlist.ctl', './interrogation_test_featlist.ctl')
print('=> loading data for task ' + Fore.GREEN + '{}'.format(task) + Fore.RESET)
loader_args = {'batch': True, 'batch_size': args.batchSize, 'shuffle': True, 'num_workers': 32}
train_loader, test_loader = lydataloader(featdirs, trainlists, testlists, loader_args)

# Eval
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    # test_feat = np.loadtxt(args.testFn, delimiter=';', skiprows=1, usecols=range(1, 6374 + 1))
    # x = torch.from_numpy(test_feat).float().view(1, -1)
    # x = Variable(x.cuda(), volatile=True)
    # z, xr, yp = netAE(x)
    # pred = yp.ge(0.5)
    # np.savetxt(os.path.join(args.outf, 'eval', 'z_feature.csv'), z.data.cpu().numpy(),
               # delimiter=',', header='dim {}'.format(z.size(1)), comments='# ')

    pred_acc = 0  # prediction accuracy
    zs = []
    for x, y in test_loader:
        x = Variable(x.cuda(), volatile=True)
        y = Variable(y.float().cuda())
        z, xr, yp = netAE(x)
        loss_r = torch.nn.functional.mse_loss(x, xr) + z.norm()
        loss_p = torch.nn.functional.binary_cross_entropy(yp, y)
        loss = loss_r + loss_p
        pred = yp.ge(0.5).float()
        pred_acc += (pred.eq(y).sum().float() / float(y.size(0))).data[0]
        zs.append(z)
    pred_acc = pred_acc / float(len(test_loader))
    for i in range(len(zs)):
        np.savetxt(os.path.join(args.outf, 'eval', 'z_feature_{:d}.csv'.format(i)), zs[i].data.cpu().numpy(),
                   delimiter=',', header='dim {}'.format(zs[i].size(1)), comments='# ')

    print('Prediction accuracy: {:.4f}'.format(pred_acc))
    print('Saved z feature to {}'.format(os.path.join(args.outf, 'eval')))
    sys.exit(0)

# Setup optimizer
optimizerAE = optim.Adam(netAE.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# Training settings
old_record_fn = 'youll_never_find_me'  # old record filename
best_test_acc = 0.
best_epoch = 0

# Train model
print("=> traning")
for epoch in range(args.nepoch):
    i = 0
    for x, y in train_loader:
        i += 1
        netAE.zero_grad()
        x = Variable(x.cuda())
        y = Variable(y.float().cuda())
        z, xr, yp = netAE(x)
        loss_r = torch.nn.functional.mse_loss(x, xr) + z.norm()  # reconstruction loss
        loss_p = torch.nn.functional.binary_cross_entropy(yp, y)  # prediction loss
        loss = loss_r + loss_p
        loss.backward()
        optimizerAE.step()
        print('[{:d}/{:d}][{:d}/{:d}] '.format(epoch, args.nepoch, i, len(train_loader)) +
              'loss_r: {:.4f} loss_p: {:.4f}'.format(loss_r.data[0], loss_p.data[0]))

    # Test
    test_loss = 0  # average test loss
    pred_acc = 0  # prediction accuracy
    for x, y in test_loader:
        x = Variable(x.cuda(), volatile=True)
        y = Variable(y.float().cuda())
        z, xr, yp = netAE(x)
        loss_r = torch.nn.functional.mse_loss(x, xr) + z.norm()
        loss_p = torch.nn.functional.binary_cross_entropy(yp, y)
        loss = loss_r + loss_p
        pred = yp.ge(0.5).float()
        test_loss += loss.data[0]
        pred_acc += (pred.eq(y).sum().float() / float(y.size(0))).data[0]
    test_loss = test_loss / float(len(test_loader))
    pred_acc = pred_acc / float(len(test_loader))
    print(Fore.RED + 'Test loss: {:.4f} Pred acc: {:.4f}'.
          format(test_loss, pred_acc) + Fore.RESET)

    # Save best
    if not os.path.exists(os.path.join(args.outf, 'checkpoints')):
        os.makedirs(os.path.join(args.outf, 'checkpoints'))
    is_best = pred_acc > best_test_acc
    if is_best:
        best_test_acc = pred_acc
        best_epoch = epoch
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_test_acc': best_test_acc,
            'netAE_state_dict': netAE.state_dict()
        }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST_lie.pth.tar')
        print(Fore.GREEN + 'Saved checkpoint for best test accuracy {:.4f} at epoch {:d}'.
              format(best_test_acc, best_epoch) + Fore.RESET)

    # Checkpointing
    save_checkpoint({
        'args': args,
        'epoch': epoch,
        'netAE_state_dict': netAE.state_dict()
    }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))

    # Delete old checkpoint to save space
    new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))
    if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
        os.remove(old_record_fn)
    old_record_fn = new_record_fn
