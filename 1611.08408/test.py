# import numpy as np
# import os
# import argparse
# import random
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# from torch.autograd import Variable
# import torch.nn.functional as f
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import torchvision.models as models
# import torch.nn.functional as f
# from collections import OrderedDict

# lr = 1e-3

# class Deeplab(nn.Module):
#     def __init__(self, n_classes):
#         super(Deeplab, self).__init__()
#         self.n_classes = n_classes
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
#             ('relu1_1', nn.ReLU(False)),
#             ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
#             ('relu1_2', nn.ReLU(False)),
#             ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

#             ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
#             ('relu2_1', nn.ReLU(False)),
#             ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
#             ('relu2_2', nn.ReLU(False)),
#             ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

#             ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
#             ('relu3_1', nn.ReLU(False)),
#             ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
#             ('relu3_2', nn.ReLU(False)),
#             ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
#             ('relu3_3', nn.ReLU(False)),
#             ('pool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

#             ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
#             ('relu4_1', nn.ReLU(False)),
#             ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
#             ('relu4_2', nn.ReLU(False)),
#             ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
#             ('relu4_3', nn.ReLU(False)),
#             ('pool4', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

#             ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
#             ('relu5_1', nn.ReLU(False)),
#             ('conv5_2', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
#             ('relu5_2', nn.ReLU(False)),
#             ('conv5_3', nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)),
#             ('relu5_3', nn.ReLU(False)),
#             ('pool5', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
#             ])
#         )


#         self.fc1 = nn.Sequential(OrderedDict(
#             [('fc6_1', nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)),
#              ('relu6_1', nn.ReLU(False)),
#              ('dropout6_1', nn.Dropout(0.5, False)),

#              ('fc7_1', nn.Conv2d(1024, 1024, kernel_size=1)),
#              ('relu7_1', nn.ReLU(False)),
#              ('dropout7_1', nn.Dropout(0.5, False)),
#              ])
#         )

#         self.classifiers1 = nn.Sequential(OrderedDict(
#             [('fc8_cxr_2', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
#         )

#         self.fc2 = nn.Sequential(OrderedDict(
#             [('fc6_2', nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12)),
#              ('relu6_2', nn.ReLU(False)),
#              ('dropout6_2', nn.Dropout(0.5, False)),

#              ('fc7_2', nn.Conv2d(1024, 1024, kernel_size=1)),
#              ('relu7_2', nn.ReLU(False)),
#              ('dropout7_2', nn.Dropout(0.5, False)),
#              ])
#         )

#         self.classifiers2 = nn.Sequential(OrderedDict(
#             [('fc8_cxr_2', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
#         )

#         self.fc3 = nn.Sequential(OrderedDict(
#             [('fc6_3', nn.Conv2d(512, 1024, kernel_size=3, padding=16, dilation=16)),
#              ('relu6_3', nn.ReLU(False)),
#              ('dropout6_3', nn.Dropout(0.5, False)),

#              ('fc7_3', nn.Conv2d(1024, 1024, kernel_size=1)),
#              ('relu7_3', nn.ReLU(False)),
#              ('dropout7_3', nn.Dropout(0.5, False)),
#              ])
#         )

#         self.classifiers3 = nn.Sequential(OrderedDict(
#             [('fc8_cxr_3', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
#         )

#         self.fc4 = nn.Sequential(OrderedDict(
#             [('fc6_4', nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24)),
#              ('relu6_4', nn.ReLU(False)),
#              ('dropout6_4', nn.Dropout(0.5, False)),

#              ('fc7_4', nn.Conv2d(1024, 1024, kernel_size=1)),
#              ('relu7_4', nn.ReLU(False)),
#              ('dropout7_4', nn.Dropout(0.5, False)),
#              ])
#         )

#         self.classifiers4 = nn.Sequential(OrderedDict(
#             [('fc8_cxr_4', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
#         )

#     def forward(self, inputs):
#         features = self.features(inputs)
#         fc1 = self.fc1(features)
#         fc2 = self.fc2(features)
#         fc3 = self.fc3(features)
#         fc4 = self.fc4(features)
#         outputs = self.classifiers1(fc1) + self.classifiers2(fc2) + self.classifiers3(fc3) + self.classifiers4(fc4)
#         return outputs

# model = Deeplab(2)
# print model.state_dict().keys()
# optimizer = optim.SGD([{'params': model.features.parameters()},
#                            {'params': model.fc1.parameters()},
#                            {'params': model.fc2.parameters()},
#                            {'params': model.fc3.parameters()},
#                            {'params': model.fc4.parameters()},
#                            {'params': model.classifiers1.parameters(), 'lr': 1e-2},
#                            {'params': model.classifiers2.parameters(), 'lr': 1e-2},
#                            {'params': model.classifiers3.parameters(), 'lr': 1e-2},
#                            {'params': model.classifiers4.parameters(), 'lr': 1e-2},
#                            ], lr=1e-3, momentum=0.9, weight_decay=5e-4)

# for param_group in optimizer.param_groups:
#     print param_group.keys(), param_group['lr']

from reader import Reader
import torch

batsize = 1
reader = Reader('/media/Disk/wangfuyu/data/cxr/801',
                '/media/Disk/wangfuyu/data/cxr/801/trainJM.txt', batchsize=batsize)

for step in range(3000):
    # adjust_learning_rate(optimizer, decay_rate=0.9, step=step)
    images, ground_truths = reader.next()
    print step
    imgs = torch.from_numpy(images).float()
    gts = torch.from_numpy(ground_truths).long()
    print images, ground_truths, ground_truths >= 1


