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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')

opt = parser.parse_args()

cudnn.benchmark = True


def main():
    dataReader = DataReader(data_root='/media/Disk/wangfuyu/data/cxr/801/',
                    txt='/media/Disk/wangfuyu/data/cxr/801/trainJM_id.txt', is_train=False)

    G = Generator(n_classes=opt.nclasses)
    G.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/G_step_20000.pth'))

    G.cuda()

    gts_all, predictions_all = [], [], []

    for step in xrange(0, len(dataReader.length())):
        images, _, ground_truths, _ = dataReader.next()

        imgs = Variable(torch.from_numpy(images).float()).cuda()
        gts = Variable(torch.from_numpy(ground_truths).long()).cuda()

        pred_map = G(imgs)
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())
        predictions_all.append(pred_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy())


    acc, acc_class, miou, _ = evaluate(preds, masks, 2)
    print acc, acc_class, miou



if __name__ == '__main__':
    main()

