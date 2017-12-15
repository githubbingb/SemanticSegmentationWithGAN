import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from DataFolder import MyDataFolder
from torch.utils.data import DataLoader
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')

opt = parser.parse_args()

mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

dataFolder = MyDataFolder(data_root='/media/Disk/wangfuyu/data/cxr/801/',
                          txt='/media/Disk/wangfuyu/data/cxr/801/trainJM.txt',
                          input_transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.mul_(255)),
                              transforms.Normalize(*mean_std),]),
                          target_transform=MaskToTensor()
                          )

dataloader = DataLoader(dataset=dataFolder, batch_size=opt.batchsize, shuffle=True, num_workers=2)

class Deeplab(nn.Module):
    def __init__(self, n_classes):
        super(Deeplab, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )

        self.classifiers1 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers2 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=18, dilation=18),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers3 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers4 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

    def forward(self, inputs):
        features = self.features(inputs)
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)
        cl1 = self.classifiers1(fc1)
        cl2 = self.classifiers2(fc2)
        cl3 = self.classifiers2(fc3)
        cl4 = self.classifiers2(fc4)
        outputs = cl1 + cl2 + cl3 + cl4
        return outputs


def adjust_learning_rate(optimizer, power=0.9, epoch=0):
    for index, param_group in enumerate(optimizer.param_groups):
        if index %2 == 0:
            param_group['lr'] = param_group['lr'] * ((1 - 1.0* epoch / opt.niter) ** (power))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print m.weight.data
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.constant(m.bias.data, 0)
        # nn.init.normal(m.weight.data, mean=0, std=0.01)
        # nn.init.constant(m.bias.data, 0)


def main():
    for epoch in range(opt.niter):
        for index, data in enumerate(dataloader, 0):
            images, ground_truths = data
            ground_truths = interp(ground_truths, shrink=8)
            print ground_truths.size()
            print images, ground_truths




if __name__ == '__main__':
    main()


