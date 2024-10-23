import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import numpy as np
import pywt

def softshrinkImg(x, lambd):
    sgn = torch.sign(x)
    Lambda = lambd
    tmp = torch.abs(x)-Lambda
    out = sgn*(tmp + torch.abs(tmp))/2
    return out

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class wavelet(nn.Module):
    def __init__(self, stride=2):
        super(wavelet, self).__init__()

        wavelet = pywt.Wavelet('haar')  # haar
        dec_hi = torch.tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.tensor(wavelet.dec_lo[::-1])
        self.filters_dec = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                    dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_dec = self.filters_dec.unsqueeze(1)

        rec_hi = torch.tensor(wavelet.rec_hi[::-1])
        rec_lo = torch.tensor(wavelet.rec_lo[::-1])
        self.filters_rec = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_rec = self.filters_rec.unsqueeze(1)

        self.stride = stride

    def forward(self, x):
        chals = x.size(1)
        out = []
        for i in range(chals):
            coeff = torch.nn.functional.conv2d(x[:, i, :, :].unsqueeze(1), self.filters_dec, stride=self.stride,
                                         bias=None, padding=0)
            if i == 0:
                out = coeff
            else:
                out = torch.cat((out, coeff), 1)

        return out

    def inverse(self, x):
        chals = x.size(1)
        out = []
        for i in range(int(chals / 4)):
            coeff = torch.nn.functional.conv_transpose2d(x[:, i*4:(i+1)*4, :, :], self.filters_rec, stride=self.stride,
                                                   bias=None, padding=0)
            if i == 0:
                out = coeff
            else:
                out = torch.cat((out, coeff), 1)

        return (out * self.stride ** 2) / 4


class unfoldSepLayer(nn.Module):
    def __init__(self, fgin, fxin, fsz, chals):
        super(unfoldSepLayer, self).__init__()
        self.chals = chals

        self.conv_Phi1 = nn.Conv2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.conv_Phi2 = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.conv_Theta1 = nn.Conv2d(fgin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.conv_Theta2 = nn.Conv2d(fxin, fgin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.conv_Gamma1 = nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.conv_Gamma2 = nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)

        self.thre = Parameter(0.1 * torch.ones(fxin)*4)#4 *
        self.l1 = Parameter(0.1 * torch.ones(chals))
        self.l2 = Parameter(0.1 * torch.ones(chals))

        self.softplus = nn.Softplus(beta=20)
        self.wavelet = wavelet(stride=1)

    def forward(self, g, x, y1, y2, z1, z2):
        z1 = z1 + self.conv_Phi1(y1 - self.conv_Phi2(z1))
        z2 = z2 + self.conv_Phi1(y2 - self.conv_Phi2(z2))

        l1 = self.l1.repeat(z2.size(0), z2.size(2), z2.size(3), 1).permute(0, 3, 1, 2).contiguous()
        l2 = self.l2.repeat(z2.size(0), z2.size(2), z2.size(3), 1).permute(0, 3, 1, 2).contiguous()
        z1 = softshrinkImg(z1, self.softplus(l1))
        z2 = softshrinkImg(z2, self.softplus(l2))

        w1 = self.wavelet.forward(y1)
        thre = self.thre.repeat(w1.size(0), w1.size(2), w1.size(3), 1).permute(0, 3, 1, 2).contiguous()

        y2 = y2 + self.conv_Gamma1(x - self.conv_Gamma2(y1) - self.conv_Gamma2(y2)) + (y2 - self.conv_Phi2(z2))
        w2 = self.wavelet.forward(y2)
        w2 = softshrinkImg(w2, torch.abs(self.softplus(thre) * w1))
        y2 = self.wavelet.inverse(w2)

        y1 = y1 + self.conv_Gamma1(x - self.conv_Gamma2(y1) - self.conv_Gamma2(y2)) \
             + self.conv_Theta1(g - self.conv_Theta2(y1)) + (y1 - self.conv_Phi2(z1))

        w1 = self.wavelet.forward(y1)
        w1 = softshrinkImg(w1, torch.abs(self.softplus(thre) * w2))
        y1 = self.wavelet.inverse(w1)

        return g, x, y1, y2, z1, z2


class Net(nn.Module):
    def __init__(self, layers=5, fsz=5):
        super(Net, self).__init__()
        fgin = 1
        fxin = 1

        self.chals = 64
        self.rfeats = 64
        self.rlayers = 3
        if self.rlayers == 0:
            self.rfeats = fxin#self.chals
        '''
        Note if rfeats = self.chals and rlayers = 0, then the reconstruction layers are linear
        '''
        layers_net = []
        for ii in range(layers):
            layers_net.append(unfoldSepLayer(fgin=fgin, fxin=fxin, fsz=fsz, chals=self.chals))
        self.UnfoldSepNet = mySequential(*layers_net)

        layers_xr = []
        layers_xr.append(nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2)))
        self.recon_xr = nn.Sequential(*layers_xr)

        layers_r1x1 = []
        for ii in range(self.rlayers):
            if ii == 0:
                layers_r1x1.append(
                    nn.Conv2d(fxin, self.rfeats, fsz, stride=1, padding=math.floor(fsz / 2), bias=True))
                layers_r1x1.append(nn.ReLU(True))
            else:
                layers_r1x1.append(
                    nn.Conv2d(self.rfeats, self.rfeats, fsz, stride=1, padding=math.floor(fsz / 2), bias=True))
                layers_r1x1.append(nn.ReLU(True))
        layers_r1x1.append(nn.Conv2d(self.rfeats, fgin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False))
        self.recon_r1x1 = nn.Sequential(*layers_r1x1)

    def forward(self, x, g, r):

        z1 = torch.zeros(x.shape[0], self.chals, x.shape[2], x.shape[3]).cuda()
        z2 = torch.zeros(x.shape[0], self.chals, x.shape[2], x.shape[3]).cuda()

        y1 = r
        y2 = x - r

        g, x, y1, y2, z1, z2 = self.UnfoldSepNet(g, x, y1, y2, z1, z2)

        x1 = self.recon_xr(y1)
        x2 = self.recon_xr(y2)
        xr = x1 + x2

        r1 = self.recon_r1x1(x1)

        return r1, x, xr, x1, x2
