import os
import re
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


def default_loader(path, gray=False):
    if gray:
        return Image.open(path).convert('L')
    else:
        return Image.open(path).convert('RGB')


class MyDataFolder(data.Dataset):
    def __init__(self, data_root, txt, input_transform=None, target_transform=None, loader=default_loader):
        self.img_list = []
        self.label_list = []

        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = re.split(' ', line)
                self.img_list.append(data_root + tmp[0])
                self.label_list.append(data_root + tmp[1][0:-1])

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img = self.loader(self.img_list[index])
        # print img.size[1]
        label = self.loader(self.label_list[index], gray=True)

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.img_list)


