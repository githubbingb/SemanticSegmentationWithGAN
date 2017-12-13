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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'



# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
# parser.add_argument('--nclasses', type=int, default=21, help='number of classes')
# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
#
# opt = parser.parse_args()
# print(opt)
#
# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass
#
# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# if opt.cuda:
#     torch.cuda.manual_seed_all(opt.manualSeed)
#
# cudnn.benchmark = True
#
# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# dataFolder = MyDataFolder(data_root='/media/Disk/work', txt='/media/Disk/work/train.txt',
#                           transform=transforms.Compose([
#                               transforms.RandomCrop(321),
#                               transforms.RandomHorizontalFlip(),
#                               transforms.ToTensor()]))
# dataloader = DataLoader(dataset=dataFolder, batch_size=2, shuffle=False, num_workers=1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # print m.weight.data
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.constant(m.bias.data, 0)
        nn.init.normal(m.weight.data, mean=0, std=0.01)
        nn.init.constant(m.bias.data, 0)

def Interp(src, zoom=1, shrink=1):
    # height_in =
    # return cv2.resize(src, )
    pass

def product(input, label_map):
    b = input[:, 0, :, :].repeat(1, label_map.size()[1], 1, 1)
    g = input[:, 1, :, :].repeat(1, label_map.size()[1], 1, 1)
    r = input[:, 2, :, :].repeat(1, label_map.size()[1], 1, 1)

    product_b = label_map * b
    product_g = label_map * g
    product_r = label_map * r

    return torch.cat((product_b, product_g, product_r), dim=1)


def onehot_encoder(ground_truth):
    outputs = np.zeros((ground_truth.shape[0], 2, ground_truth.shape[1], ground_truth.shape[2]))
    for i in xrange(ground_truth.shape[0]):
        gt = ground_truth[i, :, :]
        for index, c in enumerate(range(0, 2)):
            mask = (gt == c)
            mask = np.expand_dims(mask, 0)
            # print mask.shape
            # mask.view(1, mask.shape[0], mask.shape[1])
            if index == 0:
                onehot = mask
            else:
                # print onehot.shape
                onehot = np.concatenate((onehot, mask), axis=0)

        outputs[i, :, :, :] = onehot

    return outputs


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    return acc, acc_cls

def main():
    batsize = 1
    reader = Reader('/media/Disk/wangfuyu/data/cxr/801/',
                '/media/Disk/wangfuyu/data/cxr/801/trainJM.txt', batchsize=batsize)

    D = Discriminator(n_classes=2)
    D.apply(weights_init)

    G = Generator(n_classes=2)
    G.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/init.pth'))
    # G.apply(weights_init)

    print D, G

    # bceLoss = nn.BCELoss()
    mceLoss = nn.CrossEntropyLoss(ignore_index=255)

    # input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    # ground_truth = torch.FloatTensor(opt.batchSize, opt.imageSize, opt.imageSize)
    # label = torch.FloatTensor(opt.batchSize)
    # real_label = 1
    # fake_label = 0

    G.cuda()
    D.cuda()
    mceLoss.cuda()
        # input, ground_truth, label = input.cuda(), ground_truth.cuda(), label.cuda()

    optimizerG = optim.SGD([{'params': G.features.parameters()},
                           {'params': G.fc1.parameters()},
                           {'params': G.fc2.parameters()},
                           {'params': G.fc3.parameters()},
                           {'params': G.fc4.parameters()},
                           {'params': G.classifiers1.parameters(), 'lr': 1e-2},
                           {'params': G.classifiers2.parameters(), 'lr': 1e-2},
                           {'params': G.classifiers3.parameters(), 'lr': 1e-2},
                           {'params': G.classifiers4.parameters(), 'lr': 1e-2},
                           ], lr=1e-3, momentum=0.9, weight_decay=5e-4)

    optimizerD = optim.Adam(D.parameters(), lr=1e-3)

    for step in range(20000):
        images, ground_truths = reader.next()
        # label_onehot = torch.FloatTensor([onehot_encoder(label1.numpy()) for label1 in label])

        imgs = Variable(torch.from_numpy(images).float().cuda())
        gts = Variable(torch.from_numpy(ground_truths).long().cuda())
        real_label = Variable(torch.ones(1).long().cuda())
        fake_label = Variable(torch.zeros(1).long().cuda())

        # train Discriminator
        D.zero_grad()
        pred_map = G(imgs)
        # x_fake = product(Interp(inputv), f.softmax(pred_map))
        x_fake = f.softmax(pred_map)
        y_fake = D(x_fake.detach())
        DLoss_fake = mceLoss(y_fake, fake_label)
        DLoss_fake.backward()

        #x_real = product(Interp(inputv), one_hot(ground_truthv))
        x_real = Variable(torch.from_numpy(onehot_encoder(ground_truths)).float().cuda())
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

