import torch
import torch.nn as nn
import torch.nn.functional as f


class Generator(nn.Module):
    def __init__(self, n_classes):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.conv5_3 = nn.Sequential(
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
        )
        self.fc8_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),
        ),
        self.fc8_2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),
        )
        self.fc8_3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=18, dilation=18),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),
        )
        self.fc8_4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),
        )

        self.fc8 = self.fc8_1 + self.fc8_2 + self.fc8_3 + self.fc8_4

    def forward(self, inputs):
        outputs = self.fc8(inputs)
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



