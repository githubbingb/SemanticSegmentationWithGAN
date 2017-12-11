import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict


class Generator(nn.Module):
    def __init__(self, n_classes):
        super(Generator, self).__init__()
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
        outputs = self.classifiers1(features) + self.classifiers2(features) + self.classifiers3(
            features) + self.classifiers4(features)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(3*self.n_classes, 96, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        outputs = self.conv3_2(inputs)
        outputs = f.avg_pool2d(outputs, kernel_size=outputs.size()[1])

        return outputs.view(-1,1).squeeze(1)



