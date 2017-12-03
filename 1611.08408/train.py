import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f


import cv2



from models import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)
    else:
        print ('Error!')

def Interp(src, zoom=1, shrink=1):
    # height_in =
    # return cv2.resize(src, )
    dst = ''
    return dst

def product(input, label_map):
    b = input[:, 0, :, :].repeat(1, label_map.size()[1], 1, 1)
    g = input[:, 1, :, :].repeat(1, label_map.size()[1], 1, 1)
    r = input[:, 2, :, :].repeat(1, label_map.size()[1], 1, 1)

    product_b = label_map * b
    product_g = label_map * g
    product_r = label_map * r

    output = torch.cat((product_b, product_g, product_r), dim=1)

def main():
    G = Generator(21)
    G.apply(weights_init)

    D = Discriminator(21)
    D.apply(weights_init)

    bceLoss = nn.BCEWithLogitsLoss()
    mceLoss = nn.CrossEntropyLoss()

    input = torch.FloatTensor(opt.batchSize, opt.imageSize, opt.imageSize)
    ground_truth = torch.FloatTensor(opt.batchSize, opt.imageSize, opt.imageSize)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        G.cuda()
        D.cuda()
        bceLoss.cuda()
        mceLoss.cuda()
        input, ground_truth, label = input.cuda(), ground_truth.cuda(), label.cuda()


    optimizerG = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.niter):
        inputv = Variable(input)
        ground_truthv = Variable(ground_truth)
        ground_truth_Interpv = Variable(Interp(ground_truth))

        D.zero_grad()
        

        G.zero_grad()
        labelv = Variable(label.fill_(real_label))
        outputG = G(inputv)
        outputD = D(product(inputv,  f.softmax(outputG)))
        GLoss = mceLoss(outputG, ground_truth_Interpv) + bceLoss(outputD, labelv)
        GLoss.bachward()
        optimizerG.step()


