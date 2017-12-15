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
import numpy as np
from models import Generator, Discriminator
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')


opt = parser.parse_args()

cudnn.benchmark = True


mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

dataFolder = MyDataFolder(data_root='/media/Disk/work', txt='/media/Disk/work/train.txt',
                         input_transform=transforms.Compose([
                             transforms.ToTensor(),
                              transforms.Normalize(*mean_std),
                         ]))

dataloader = DataLoader(dataset=dataFolder, batch_size=opt.batchsize, shuffle=True, num_workers=2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.constant(m.bias.data, 0)
        nn.init.normal(m.weight.data, mean=0, std=0.01)
        nn.init.constant(m.bias.data, 0)


def adjust_learning_rate(optimizer, power=0.9, step=0):
    for index, param_group in optimizer.param_groups:
        if index %2 == 0:
            param_group['lr'] = param_group['lr'] * ((1 - 1.0* step / opt.niter) ** (power))

def main():
    reader = Reader('/media/Disk/wangfuyu/data/cxr/801/',
                '/media/Disk/wangfuyu/data/cxr/801/trainJM.txt', batchsize=opt.batchsize)

    D = Discriminator(n_classes=opt.nclasses)
    D.apply(weights_init)

    G = Generator(n_classes=opt.nclasses)
    G.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/init.pth'))
    # G.apply(weights_init)

    print D, G

    mceLoss = nn.CrossEntropyLoss(ignore_index=255)


    G.cuda()
    D.cuda()
    mceLoss.cuda()

    optimizerG = optim.SGD([{'params': [param for name, param in model.features.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in model.features.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in model.fc1.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in model.fc1.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in model.fc2.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in model.fc2.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in model.fc3.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in model.fc3.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in model.fc4.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in model.fc4.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in model.classifiers1.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in model.classifiers1.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in model.classifiers2.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in model.classifiers2.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in model.classifiers3.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in model.classifiers3.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in model.classifiers4.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in model.classifiers4.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},
                           ], momentum=0.9, weight_decay=5e-4)

    optimizerD = optim.Adam(D.parameters(), lr=1e-3)


    for epoch in xrange(opt.niter):
        for index, data in enumerate(dataloader, 0):


    for step in range(20000):
        adjust_learning_rate(optimizerG)
        images, ground_truths = reader.next()
        # label_onehot = torch.FloatTensor([onehot_encoder(label1.numpy()) for label1 in label])

        imgs = Variable(torch.from_numpy(images).float()).cuda()
        gts = Variable(torch.from_numpy(ground_truths).long()).cuda()
        real_label = Variable(torch.ones(1).long()).cuda()
        fake_label = Variable(torch.zeros(1).long()).cuda()

        # train Discriminator
        D.zero_grad()
        pred_map = G(imgs)
        # x_fake = product(Interp(inputv), f.softmax(pred_map))
        x_fake = f.softmax(pred_map)
        y_fake = D(x_fake.detach())
        DLoss_fake = mceLoss(y_fake, fake_label)
        DLoss_fake.backward()

        #x_real = product(Interp(inputv), one_hot(ground_truthv))
        x_real = Variable(torch.from_numpy(onehot_encoder(ground_truths)).float()).cuda()
        y_real = D(x_real)
        DLoss_real = mceLoss(y_real, real_label)
        DLoss_real.backward()

        optimizerD.step()

        # train Generator
        G.zero_grad()
        #pred_map = G(imgs)
        #x_fake = product(inputv,  f.softmax(pred_map))
        #x_fake = f.softmax(pred_map)
        y_fake = D(x_fake)
        GLoss = mceLoss(pred_map, gts) + mceLoss(y_fake, real_label)
        GLoss.backward()
        optimizerG.step()

        if step % 10 == 0:
            print 'DLoss_fake: ', DLoss_fake, 'DLoss_real: ', DLoss_real, 'GLoss: ', GLoss

            preds = pred_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            masks = gts.data.squeeze_(0).cpu().numpy()
            acc, acc_class = evaluate(preds, masks, 2)
            print acc, acc_class

        if step % 1000 == 0:
            torch.save(D.state_dict(), 'D_step_%d.pth' % step)
            torch.save(G.state_dict(), 'G_step_%d.pth' % step)


if __name__ == '__main__':
    main()

