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
        self.features = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('relu1_1', nn.ReLU(False)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('relu1_2', nn.ReLU(False)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(False)),
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('relu2_2', nn.ReLU(False)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('relu3_1', nn.ReLU(False)),
            ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3_2', nn.ReLU(False)),
            ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3_3', nn.ReLU(False)),
            ('pool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ('relu4_1', nn.ReLU(False)),
            ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu4_2', nn.ReLU(False)),
            ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu4_3', nn.ReLU(False)),
            ('pool4', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

            ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
            ('relu5_1', nn.ReLU(False)),
            ('conv5_2', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
            ('relu5_2', nn.ReLU(False)),
            ('conv5_3', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
            ('relu5_3', nn.ReLU(False)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
            ])
        )

        self.classifiers1 = nn.Sequential(OrderedDict(
            [('fc6_1', nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)),
             ('relu6_1', nn.ReLU(False)),
             ('dropout6_1', nn.Dropout(0.5, False)),

             ('fc7_1', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_1', nn.ReLU(False)),
             ('dropout7_1', nn.Dropout(0.5, False)),

             ('fc8_cxr_1', nn.Conv2d(1024, self.n_classes, kernel_size=1)),
             ])
        )

        self.classifiers2 = nn.Sequential(OrderedDict(
            [('fc6_2', nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12)),
             ('relu6_2', nn.ReLU(False)),
             ('dropout6_2', nn.Dropout(0.5, False)),

             ('fc7_2', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_2', nn.ReLU(False)),
             ('dropout7_2', nn.Dropout(0.5, False)),

             ('fc8_cxr_2', nn.Conv2d(1024, self.n_classes, kernel_size=1)),
             ])
        )

        self.classifiers3 = nn.Sequential(OrderedDict(
            [('fc6_3', nn.Conv2d(512, 1024, kernel_size=3, padding=16, dilation=16)),
             ('relu6_3', nn.ReLU(False)),
             ('dropout6_3', nn.Dropout(0.5, False)),

             ('fc7_3', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_3', nn.ReLU(False)),
             ('dropout7_3', nn.Dropout(0.5, False)),

             ('fc8_cxr_3', nn.Conv2d(1024, self.n_classes, kernel_size=1)),
             ])
        )

        self.classifiers4 = nn.Sequential(OrderedDict(
            [('fc6_4', nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24)),
             ('relu6_4', nn.ReLU(False)),
             ('dropout6_4', nn.Dropout(0.5, False)),

             ('fc7_4', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_4', nn.ReLU(False)),
             ('dropout7_4', nn.Dropout(0.5, False)),

             ('fc8_cxr_4', nn.Conv2d(1024, self.n_classes, kernel_size=1)),
             ])
        )

    def forward(self, inputs):
        features = self.features(inputs)
        outputs = self.classifiers1(features) + self.classifiers2(features) + self.classifiers3(features) + self.classifiers4(features)
        return outputs


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


