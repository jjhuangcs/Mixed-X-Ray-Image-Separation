from __future__ import print_function
import argparse
import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from XraySepNet_model import Net

import numpy as np
import visdom
from torch.autograd import Variable
from PIL import Image, ImageFilter
import pywt
import math

viz = visdom.Visdom()

#########################################################################################
# Functions
#########################################################################################

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 100 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('Learning Rate: ', param_group['lr'])

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

class ExclusionLoss(nn.Module):
    def __init__(self, level=1):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))

            gradx1_s = (self.sigmoid((gradx1)) * 2) - 1
            grady1_s = (self.sigmoid((grady1)) * 2) - 1
            gradx2_s = (self.sigmoid((gradx2 * alphax)) * 2) - 1
            grady2_s = (self.sigmoid((grady2 * alphay)) * 2) - 1

            gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            # here is the l2 norm (original version)

            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 3) + sum(grady_loss) / (self.level * 3)
        return loss_gradxy

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

#########################################################################################
# Training settings
#########################################################################################

parser = argparse.ArgumentParser(description='PyTorch Image Separation')
parser.add_argument('--nEpochs', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=12345, help='random seed to use. Default=123')
parser.add_argument("--step", type=int, default=40, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
opt = parser.parse_args()

print(opt)

if not os.path.exists('XraySepUnfold'):
    os.makedirs('XraySepUnfold')

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

criterion = nn.MSELoss()
exclusion_loss = ExclusionLoss(level=3)


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

#########################################################################################
# Load images
#########################################################################################
patchsize = 64
stride = 16
# cut images into patches with patch size as patchsize = 64 and with stride as stride  = 16

image_data = Image.open("Goya_r1.jpg")  # visible image
rgb1 = np.asarray(image_data, dtype="float32")

image_data = Image.open("Goya_x.jpg")  # Xray image
xray_data = np.asarray(image_data, dtype="float32")
image_data = Image.open("Goya_g1.jpg")  # gray scale image
gray = np.asarray(image_data, dtype="float32")
xray = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")
xray[:, :, 0] = xray_data
xray = xray / 255
rgb1 = rgb1 / 255
gray = gray / 255

rgb1_data = rgb2gray(rgb1)  # gray version of the visible image
rgb1 = np.empty((rgb1.shape[0], xray_data.shape[1], 1), dtype="float32")
rgb1[:, :, 0] = rgb1_data

gray1_data = rgb2gray(gray)  # gray version of the visible image
gray1 = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")
gray1[:, :, 0] = gray1_data
m = xray_data.shape[0]
n = xray_data.shape[1]

xrayTorch = np_to_torch(xray)
xrayTorch = xrayTorch.permute(0, 3, 1, 2)
xray_patches = xrayTorch.squeeze(0)
xray_patches = xray_patches.data.unfold(0, 1, 1).unfold(1, patchsize, stride).unfold(2, patchsize, stride)
xray_patches = xray_patches.reshape(-1, 1, patchsize, patchsize)

rndIdx = torch.randperm(xray_patches.size(0))
xray_patches = xray_patches[rndIdx, :, :, :]

rgb1dim = 1
rgb1Torch = np_to_torch(rgb1)
rgb1Torch = rgb1Torch.permute(0, 3, 1, 2)
rgb1_patches = rgb1Torch.squeeze(0)
rgb1_patches = rgb1_patches.data.unfold(0, rgb1dim, rgb1dim).unfold(1, patchsize, stride).unfold(2, patchsize, stride)
rgb1_patches = rgb1_patches.reshape(-1, rgb1dim, patchsize, patchsize)
rgb1_patches = rgb1_patches[rndIdx, :, :, :]

gray1Torch = np_to_torch(gray1)
gray1Torch = gray1Torch.permute(0, 3, 1, 2)
gray1_patches = gray1Torch.squeeze(0)
gray1_patches = gray1_patches.data.unfold(0, 1, 1).unfold(1, patchsize, stride).unfold(2, patchsize, stride)
gray1_patches = gray1_patches.reshape(-1, 1, patchsize, patchsize)
gray1_patches = gray1_patches[rndIdx, :, :, :]

#########################################################################################
# Training
#########################################################################################
def train(epoch, reg):
    epoch_loss = 0
    visible_loss = 0
    reco_loss = 0
    excl_loss = 0
    batchszize = 32

    for sz in range(1, int(xray_patches.size(0)/batchszize)):
        # print("Batch ===> {:d}".format(sz))
        optimizer.zero_grad()

        xray_batch = xray_patches[batchszize * (sz - 1): batchszize * sz, :, :, :]
        xray_batch = (xray_batch).cuda()
        rgb1_batch = rgb1_patches[batchszize * (sz - 1): batchszize * sz, :, :, :]
        rgb1_batch = (rgb1_batch).cuda()
        gray1_batch = gray1_patches[batchszize * (sz - 1): batchszize * sz, :, :, :]
        gray1_batch = (gray1_batch).cuda()

        r1, x, xr, x1, x2 = model(xray_batch, rgb1_batch, gray1_batch)

        loss1 = 0.5*criterion(r1, rgb1_batch)
        loss2 = 1*criterion(xr, xray_batch)
        loss3 = 0.1 * exclusion_loss(x2, x1)
        loss = loss1 + loss2 + loss3

        visible_loss += loss1.item()
        reco_loss += loss2.item()
        excl_loss += loss3.item()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("===> Epoch {} Complete: Avg. Total Loss: {:.4f}, Visible Loss: {:.4f}, Recon. Loss:"
          " {:.4f}, Excl. Loss: {:.4f}".format(epoch, epoch_loss, visible_loss, reco_loss, excl_loss))

    return epoch_loss, reco_loss, excl_loss


def checkpoint(epoch):
    model_out_path = "XraySepUnfold/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


reg = 0
for epoch in range(1, opt.nEpochs + 1):

    epoch_Trainloss, recloss, excloss = train(epoch, reg)
    if epoch % 10 == 0:
        checkpoint(epoch)

    if epoch % opt.step == 0:
        adjust_learning_rate(optimizer, epoch)

    if epoch % 10 == 0:

        xrayTorch = xrayTorch.cuda()
        rgb1Torch = rgb1Torch.cuda()
        gray1Torch = gray1Torch.cuda()
        with torch.no_grad():
            r1, x, xr, x1, x2 = model(xrayTorch, rgb1Torch, gray1Torch)

        out1 = x1.cpu()
        out_img_y1 = out1[0].detach().numpy()
        out_img_y1 *= 255.0
        out_img_y1 = out_img_y1.clip(0, 255)
        out_img_y1 = Image.fromarray(np.uint8(out_img_y1[0]), mode='L')
        save_out_path = "XraySepUnfold/Xray_surface_epoch_{}.jpg".format(epoch)
        out_img_y1.save(save_out_path)

        out2 = x2.cpu()
        out_img_y2 = out2[0].detach().numpy()
        out_img_y2 *= 255.0
        out_img_y2 = out_img_y2.clip(0, 255)
        out_img_y2 = Image.fromarray(np.uint8(out_img_y2[0]), mode='L')
        save_out_path = "XraySepUnfold/Xray_hidden_epoch_{}.jpg".format(epoch)
        out_img_y2.save(save_out_path)

        save_out_path = "XraySepUnfold/Xray_combined_epoch_{}.jpg".format(epoch)
        out12 = (x1 + x2).cpu()
        out_img_y1y2 = out12[0].detach().numpy()
        out_img_y1y2 *= 255.0
        out_img_y1y2 = out_img_y1y2.clip(0, 255)
        out_img_y1y2 = Image.fromarray(np.uint8(out_img_y1y2[0]), mode='L')
        out_img_y1y2.save(save_out_path)

        save_out_path = "XraySepUnfold/Xray_color_epoch_{}.jpg".format(epoch)
        out12 = (r1).cpu()
        out_img_r1 = out12[0].detach().numpy()
        out_img_r1 *= 255.0
        out_img_r1 = out_img_r1.clip(0, 255)
        out_img_r1 = Image.fromarray(np.uint8(out_img_r1[0]), mode='L')
        out_img_r1.save(save_out_path)

        save_out_path = "XraySepUnfold/Xray_residual_epoch_{}.jpg".format(epoch)
        out12 = abs(xrayTorch - x1 - x2).cpu()
        out_img_r1 = out12[0].detach().numpy()
        out_img_r1 *= 255.0
        out_img_r1 = out_img_r1.clip(0, 255)
        out_img_r1 = Image.fromarray(np.uint8(out_img_r1[0]), mode='L')
        out_img_r1.save(save_out_path)
