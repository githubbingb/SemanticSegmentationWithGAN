import os
import re
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


def default_loader(path, gray=False):
    if gray:
        return Image.open(path)
    else:
        return Image.open(path).convert('RGB')


class MyDataFolder(data.Dataset):
    def __init__(self, data_root, txt, transform=None, loader=default_loader, n_classes=2):
        self.img_list = []
        self.label_list = []

        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = re.split(' ', line)
                self.img_list.append(data_root + tmp[0])
                self.label_list.append(data_root + tmp[1][0:-1])

        self.transform = transform
        self.loader = loader
        self.n_classes = n_classes

    def __getitem__(self, index):
        img = self.loader(self.img_list[index])
        #print np.array(img - [104.008, 116.669, 122.675])
        label = self.loader(self.label_list[index], gray=True)
        label_interp = label.resize((label.size[0], label.size[1]))
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
            label_interp = self.transform(label_interp)
        return img, label

    def __len__(self):
        return len(self.img_list)


data =