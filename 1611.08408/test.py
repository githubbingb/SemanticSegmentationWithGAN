import numpy as np
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from DataFolder import MyDataFolder
from torch.utils.data import DataLoader
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')

opt = parser.parse_args()

mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

dataFolder = MyDataFolder(data_root='/media/Disk/wangfuyu/data/cxr/801/',
                          txt='/media/Disk/wangfuyu/data/cxr/801/trainJM.txt',
                          input_transform=transforms.Compose([
                              transforms.Lambda(lambda  x: x.mul_(255)),
                              transforms.ToTensor(),
                              transforms.Normalize(*mean_std),]),
                          target_transform=transforms.ToTensor()
                          )

dataloader = DataLoader(dataset=dataFolder, batch_size=opt.batchsize, shuffle=True, num_workers=2)

for i, data in enumerate(dataloader, 0):
    img, label = data
    label.squeeze_(1)
    print label




if __name__ == '__main__':
    main()

