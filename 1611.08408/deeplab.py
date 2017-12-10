import torch
import torch.nn as nn
import torch.nn.functional as f


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
        )
        self.fc8_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(0.5, True),

            nn.Conv2d(1024, self.n_classes, kernel_size=1),
        )
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

    def forward(self, inputs):
        outputs = self.fc8_1(inputs) + self.fc8_2(inputs) + self.fc8_3(inputs) + self.fc8_4(inputs)
        return outputs


model = Deeplab(21)
print model, model.state_dict().keys()