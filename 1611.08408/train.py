import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f
from reader import *
from models import Generator, Discriminator
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')
parser.add_argument('--niter', type=int, default=20001, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')


opt = parser.parse_args()

cudnn.benchmark = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data, 0)
        # nn.init.normal(m.weight.data, mean=0, std=0.01)
        # nn.init.constant(m.bias.data, 0)


def adjust_learning_rate(optimizer, power=0.9, step=0):
    for index, param_group in enumerate(optimizer.param_groups):
        if index %2 == 0:
            param_group['lr'] = param_group['lr'] * ((1 - 1.0* step / opt.niter) ** (power))
    # return optimizer


def main():
    dataReader = DataReader(data_root='/media/Disk/wangfuyu/data/cxr/801/',
                            txt='/media/Disk/wangfuyu/data/cxr/801/trainJM_id.txt',
                            batchsize=opt.batchsize)

    D = Discriminator(n_classes=opt.nclasses, product=False)
    D.apply(weights_init)

    G = Generator(n_classes=opt.nclasses)
    G.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/init.pth'))
    # G.apply(weights_init)
    print D, G

    mceLoss = nn.CrossEntropyLoss(ignore_index=255)

    G.cuda()
    D.cuda()
    mceLoss.cuda()

    optimizerG = optim.SGD([{'params': [param for name, param in G.features.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in G.features.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in G.fc1.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in G.fc1.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in G.fc2.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in G.fc2.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in G.fc3.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in G.fc3.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in G.fc4.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': opt.lr},
                           {'params': [param for name, param in G.fc4.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 2 * opt.lr},

                           {'params': [param for name, param in G.classifiers1.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in G.classifiers1.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in G.classifiers2.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in G.classifiers2.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in G.classifiers3.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in G.classifiers3.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},

                           {'params': [param for name, param in G.classifiers4.named_parameters() if
                                       name[-4] != 'bias'],
                            'lr': 10 * opt.lr},
                           {'params': [param for name, param in G.classifiers4.named_parameters() if
                                       name[-4] == 'bias'],
                            'lr': 20 * opt.lr},
                           ], momentum=0.9, weight_decay=5e-4)

    # optimizerD = optim.SGD([{'params': [param for name, param in D.features.named_parameters() if
    #                                    name[-4] != 'bias'],
    #                         'lr': 1e-4},
    #                        {'params': [param for name, param in D.features.named_parameters() if
    #                                    name[-4] == 'bias'],
    #                         'lr': 2*1e-4},
    #                        ], momentum=0.9, weight_decay=5e-4)
    optimizerD = optim.SGD(params=D.parameters(), lr=1e-2, momentum=0.99, weight_decay=5e-4)

    for step in xrange(0, opt.niter):
        adjust_learning_rate(optimizerG, step=step)
        # adjust_learning_rate(optimizerD, step=step)

        images, images_down, _, ground_truths_down = dataReader.next()

        imgs = Variable(torch.from_numpy(images).float()).cuda()
        gts_down = Variable(torch.from_numpy(ground_truths_down).long()).cuda()

        real_label = Variable(torch.ones(1).long()).cuda()
        fake_label = Variable(torch.zeros(1).long()).cuda()

        # train Discriminator
        D.zero_grad()
        pred_map = G(imgs)
        # x_fake = Variable(product(torch.from_numpy(images_down).float(), f.softmax(pred_map).cpu().data)).cuda()
        x_fake = f.softmax(pred_map)
        # x_fake = Variable(torch.from_numpy(np.concatenate((f.softmax(pred_map).cpu().data.numpy(), images_down), axis=1)).float()).cuda()
        y_fake = D(x_fake.detach())
        DLoss_fake = mceLoss(y_fake, fake_label)
        DLoss_fake.backward()

        # x_real = Variable(product(torch.from_numpy(images_down).float(), onehot_encoder(ground_truths_down, n_classes=opt.nclasses))).cuda()
        #x_real = Variable(torch.from_numpy(np.concatenate((onehot, images_down), axis=1)).float()).cuda()
        onehot = onehot_encoder(ground_truths_down, n_classes=opt.nclasses)
        x_real = Variable(torch.from_numpy(onehot).float()).cuda()
        y_real = D(x_real)
        DLoss_real = mceLoss(y_real, real_label)
        DLoss_real.backward()
        optimizerD.step()

        # train Generator
        G.zero_grad()
        y_fake = D(x_fake)
        GLoss = mceLoss(pred_map, gts_down) + mceLoss(y_fake, real_label)
        GLoss.backward()
        optimizerG.step()

        if step % 20 == 0:
            print 'DLoss_fake: ', DLoss_fake, 'DLoss_real: ', DLoss_real, 'GLoss: ', GLoss

            preds = pred_map.data.max(1)[1].squeeze_(1).cpu().numpy()
            masks = gts_down.data.cpu().numpy()
            acc, acc_class, _, _, _ = evaluate(preds, masks, 2)
            print acc, acc_class

        if step % 1000 == 0:
            torch.save(D.state_dict(), 'D_step_%d.pth' % step)
            torch.save(G.state_dict(), 'G_step_%d.pth' % step)


if __name__ == '__main__':
    main()

