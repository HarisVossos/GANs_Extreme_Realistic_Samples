from tensorboardX import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import torch.optim as optim
from torch import LongTensor, FloatTensor
from scipy.stats import skewnorm, genpareto
from torchvision.utils import save_image
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
import sys
sns.set()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def convTBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=1):
    # nn.Sequential: applies modules in the order that they are passed
    # nn.ConvTranspose1d: Applies a 1d transpose convolutional layer
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        # Normalize by computing the mean and std across each individual channel for a single example
        # Alternatives: group normalization, layer normalization, or batch normalization
        # https://wandb.ai/wandb_fc/Normalization-Series/reports/Instance-Normalization-in-PyTorch-With-Examples---VmlldzoxNDIyNTQx
        # Documentation: torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
        nn.LeakyReLU(0.2, True),
    )


def convBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels, 512, 8, 4, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose1d(64, out_channels, 2, 1, 1)

    def forward(self, inp):
        # print(f'Input size: {inp.size}')
        out = self.block1(inp)
        # print(f'Block 1 Generator: {out.shape}')
        out = self.block2(out)
        # print(f'Block 2 Generator: {out.shape}')
        out = self.block3(out)
        # print(f'Block 3 Generator: {out.shape}')
        out = self.block4(out)
        # print(f'Block 4 Generator: {out.shape}')
        return self.block5(out)


class Discriminator(nn.Module):
    def __init__(self, in_channels=None):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv1d(512, 64, 6, 4, 0)
        self.source = nn.Linear(64, 1)

    def forward(self, inp):
        # print(f'Input size: {inp.shape}')
        # print(f'Weight {self.weight}')
        out = self.block1(inp)
        # print(f'Block 1 Dicsriminator: {out.shape}')
        out = self.block2(out)
        # print(f'Block 2 Dicsriminator: {out.shape}')
        out = self.block3(out)
        # print(f'Block 3 Dicsriminator: {out.shape}')
        out = self.block4(out)
        # print(f'Block 4 Dicsriminator: {out.shape}')
        out = self.block5(out)
        # print(f'Block 5 Dicsriminator: {out.shape}')
        size = out.shape[0]
        out = out.view(size, -1)
        # print(f'Block 6 Dicsriminator: {out.shape}')
        return torch.sigmoid(self.source(out))

class DCGAN(nn.Module):
    def __init__(self, data, batch_size, epochs, Gloss_function = nn.L1Loss, Dloss_function = nn.L1Loss):
        super(DCGAN, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.Gloss_function = Gloss_function
        self.DLoss_function = Dloss_function
        self.data_list = []
        self.prepare_data()

    def prepare_data(self):
        number_of_splits = self.data.shape[0] // self.batch_size
        loc = 0
        for i in range(0, number_of_splits+1):
            self.data_list.append(self.data[loc:loc+self.batch_size])
            loc += self.batch_size

    def prepare_new_data(self, data):
        self.data = data
        self.data_list = []
        self.prepare_data()

    def train(self):
        self.latentdim = 1
        criterionSource = nn.L1Loss()
        criterionContinuous = nn.L1Loss()
        criterionValG = self.Gloss_function
        criterionValD = self.DLoss_function

        self.G = Generator(in_channels=self.latentdim, out_channels=1).cuda()
        self.D = Discriminator(in_channels=1).cuda()
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        optimizerG = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerD = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))

        step = 0
        for epoch in range(self.epochs):
            for data in self.data_list:
                noise = 1e-5 * max(1 - (epoch / 500.0), 0)
                step += 1
                batch_size = data.shape[0]
                trueTensor = 0.7 + 0.5 * torch.rand(batch_size)
                falseTensor = 0.3 * torch.rand(batch_size)
                probFlip = torch.rand(batch_size) < 0.05
                probFlip = probFlip.float()
                trueTensor, falseTensor = (
                    probFlip * falseTensor + (1 - probFlip) * trueTensor,
                    probFlip * trueTensor + (1 - probFlip) * falseTensor,
                )
                trueTensor = trueTensor.view(-1, 1).cuda()
                falseTensor = falseTensor.view(-1, 1).cuda()
                losses = torch.tensor(np.array(data)).cuda().view(data.shape[0], 1, 1)
                losses = losses.float()
                realSource = self.D(losses + noise)
                realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
                latent = Variable(torch.randn(batch_size, self.latentdim, 1)).cuda()
                fakeData = self.G(latent)
                fakeSource = self.D(fakeData.detach())
                fakeLoss = criterionSource(fakeSource, falseTensor.expand_as(fakeSource))
                lossD = realLoss + fakeLoss
                optimizerD.zero_grad()
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 20)
                optimizerD.step()
                fakeSource = self.D(fakeData)
                trueTensor = 0.9 * torch.ones(batch_size).view(-1, 1).cuda()
                lossG = criterionSource(fakeSource, trueTensor.expand_as(fakeSource))
                optimizerG.zero_grad()
                lossG.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 20)
                optimizerG.step()
            # if epoch % 10 == 0:
            #     print(f'Epoch: {epoch} , Loss G: {lossG} , Loss D: {lossD}')

    def generate_samples(self,number_of_samples):
        self.G.eval()
        torch.cuda.empty_cache()
        fakeSamples = self.G(Variable(torch.randn(number_of_samples, self.latentdim, 1)).cuda())
        sums = fakeSamples.sum(dim=(1, 2)).detach().cpu().numpy().argsort()[::-1].copy()
        return pd.Series(fakeSamples[sums].cpu().data.numpy().ravel())


