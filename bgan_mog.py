from __future__ import print_function
import os, pickle
import numpy as np
import random, math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import copy
import seaborn
from mog import data_generator
import math
# Default Parameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=104, help='input batch size')
parser.add_argument('--nz', type=int, default=20, help='size of the latent z vector')
parser.add_argument('-iterations', type=int, default=30000, help='number of epochs to train (default: 100)')
parser.add_argument('-interval', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--dim_h', type=int, default=32, help='number of GPUs to use')
parser.add_argument('--outf', default='modelfiles/MNIST_dense_unsupervised_mog_4/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1, help='The number of set of z to marginalize over.')
parser.add_argument('--num_mcmc', type=int, default= 4, help='The number of MCMC chains to run in parallel')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')

import sys;

sys.argv = [''];
del sys
opt = parser.parse_args()
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

plt.style.use('ggplot')
dset = data_generator()
dset.uniform_distribution()


from models.discriminators import dense_D_mog
from models.generators import dense_G_mog

netGs = []
for _idxz in range(opt.numz):
    for _idxm in range(opt.num_mcmc):
        netG = dense_G_mog(opt.dim_h, nz=opt.nz)
        # netG.apply(weights_init)
        netGs.append(netG)
##### Discriminator ######
# We will use 1 chain of MCMCs for the discriminator
# The number of classes for semi-supervised case is 11; that is,
# index 0 for fake data and 0-9 for the 10 classes of MNIST.
num_classes = 1
netD = dense_D_mog(opt.dim_h, num_classes=num_classes)

bcewl = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

from models.distributions import Normal
from models.bayes import NoiseLoss, PriorLoss

# Finally, initialize the ``optimizers''
# Since we keep track of a set of parameters, we also need a set of
# ``optimizers''
if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr * 0.5, betas=(0.5, 0.999))
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr,
                                 momentum=0.9,
                                 nesterov=True,
                                 weight_decay=1e-4)
optimizerGs = []
for netG in netGs:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerGs.append(optimizerG)

# since the log posterior is the average per sample, we also scale down the prior and the noise
gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netGs[0].parameters(), scale=math.sqrt(2 * opt.gnoise_alpha / opt.lr),
                             observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(2 * opt.dnoise_alpha * opt.lr), observed=50000.)

# Fixed noise for data generation
fixed_noise = torch.FloatTensor(opt.batchSize // opt.num_mcmc, opt.nz).normal_(0, 1).cuda()
fixed_noise = Variable(fixed_noise)

# initialize input variables and use CUDA (optional)
input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    for netG in netGs:
        netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()


def generator_sample():
    with torch.no_grad():
        for _zid in range(opt.numz):
            for _mid in range(opt.num_mcmc):
                idx = _zid * opt.num_mcmc + _mid
                netG = netGs[idx]
                _fake = netG(fixed_noise)
                fakes.append(_fake)
                fake = torch.cat(fakes)
        return fake.cpu().numpy()


def plot(points, title):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(opt.outf + title + '.png')
    plt.clf()

samples = []
for iteration in range(opt.iterations):
    #######
    data = torch.from_numpy(dset.sample(opt.batchSize))
    # 1. real input
    netD.zero_grad()
    _input = data
    batch_size = _input.size(0)
    if opt.cuda:
        _input = _input.cuda()
    input.resize_as_(_input).copy_(_input)
    label.resize_(batch_size, 1).fill_(real_label)
    inputv = Variable(input).view(-1, opt.imageSize)

    output = netD(inputv)
    errD_real = bcewl(output, label)
    errD_real.backward()

    #######
    # 2. Generated input
    fakes = []
    for _idxz in range(opt.numz):
        noise.resize_(batch_size, opt.nz).normal_(0, 1)
        noisev = Variable(noise)
        for _idxm in range(opt.num_mcmc):
            idx = _idxz * opt.num_mcmc + _idxm
            netG = netGs[idx]
            _fake = netG(noisev)
            fakes.append(_fake)
    fake = torch.cat(fakes)
    output = netD(fake.detach())
    # print("output", output.size())
    labelv = torch.Tensor(fake.data.shape[0], 1).cuda().fill_(fake_label)
    # print("labelv", labelv.size())
    errD_fake = bcewl(output, labelv)
    errD_fake.backward()

    #######
    if opt.bayes:
        errD_prior = dprior_criterion(netD.parameters())
        errD_prior.backward()
        errD_noise = dnoise_criterion(netD.parameters())
        errD_noise.backward()
        errD = errD_real + errD_fake + errD_prior + errD_noise
    else:
        errD = errD_real + errD_fake
    optimizerD.step()

    # 4. Generator
    for netG in netGs:
        netG.zero_grad()
    labelv = torch.FloatTensor(fake.data.shape[0], 1).cuda().fill_(real_label)
    output = netD(fake)
    errG = bcewl(output, labelv)
    if opt.bayes:
        for netG in netGs:
            errG += gprior_criterion(netG.parameters())
            errG += gnoise_criterion(netG.parameters())
    errG.backward()
    for optimizerG in optimizerGs:
        optimizerG.step()

    # 6. get test accuracy after every interval
    if (iteration+1) % opt.interval == 0:
        x_sam = generator_sample()
        plot(x_sam, title='Iteration {}'.format(iteration+1))
        print('[%d/%d] Loss_D: %.2f Loss_G: %.2f ' % ((iteration+1), opt.iterations, errD.data.item(), errG.data.item()))

