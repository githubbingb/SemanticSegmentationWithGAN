import numpy as np
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f
import torchvision.transforms as transforms
from DataFolder import MyDataFolder
from torch.utils.data import DataLoader
from reader import Reader
import torch.nn.functional as f
from collections import OrderedDict


class Deeplab(nn.Module):
    def __init__(self, n_classes):
        super(Deeplab, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),

        )
        # self.classifiers1 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, self.n_classes, kernel_size=1),
        # )
        # self.classifiers2 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, self.n_classes, kernel_size=1),
        # )
        # self.classifiers3 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=18, dilation=18),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, self.n_classes, kernel_size=1),
        # )
        # self.classifiers4 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5, True),
        #
        #     nn.Conv2d(1024, self.n_classes, kernel_size=1),
        # )

    def forward(self, x):
        x = self.features(x)
        #outputs = self.classifiers1(features) + self.classifiers2(features) + self.classifiers3(features) + self.classifiers4(features)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def main():
    reader = Reader('/media/Disk/wangfuyu/data/cxr/801/',
                    '/media/Disk/wangfuyu/data/cxr/801/trainJM.txt')

    model = Deeplab(2)
    model.apply(weights_init)

    keys = model.state_dict().keys()

    print model, keys

    pretrain_dict = np.load('/media/Disk/wangfuyu/voc12.npy').item()
    print pretrain_dict.keys()

    for key in keys:
        if key in pretrain_dict.keys():
            print key
            model.state_dict()[key] = torch.from_numpy(pretrain_dict[key]).float()

    mceLoss = nn.CrossEntropyLoss()

    model.cuda()
    mceLoss.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    for step in range(10000):
        images, ground_truths = reader.next()
        # label_onehot = torch.FloatTensor([onehot_encoder(label1.numpy()) for label1 in label])

        imgs = Variable(torch.from_numpy(images).float().cuda())
        gts = Variable(torch.from_numpy(ground_truths).long().cuda())

        model.zero_grad()
        pred_map = model(imgs)
        loss = mceLoss(pred_map, gts)
        loss.backward()
        optimizer.step()

        print loss
        
        if step % 1000 == 0:
            torch.save(model.state_dict(), 'step_%d.pth' % step)

if __name__ == '__main__':
    main()


