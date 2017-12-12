import numpy as np
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
import torch.nn.functional as f
import torchvision
from reader import Reader
import torch.nn.functional as f
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cudnn.benchmark = True

max_step = 20000


class Deeplab(nn.Module):
    def __init__(self, n_classes):
        super(Deeplab, self).__init__()
        self.n_classes = n_classes
        '''
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


        self.fc1 = nn.Sequential(OrderedDict(
            [('fc6_1', nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)),
             ('relu6_1', nn.ReLU(False)),
             ('dropout6_1', nn.Dropout(0.5, False)),

             ('fc7_1', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_1', nn.ReLU(False)),
             ('dropout7_1', nn.Dropout(0.5, False)),
             ])
        )

        self.classifiers1 = nn.Sequential(OrderedDict(
            [('fc8_cxr_2', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
        )

        self.fc2 = nn.Sequential(OrderedDict(
            [('fc6_2', nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12)),
             ('relu6_2', nn.ReLU(False)),
             ('dropout6_2', nn.Dropout(0.5, False)),

             ('fc7_2', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_2', nn.ReLU(False)),
             ('dropout7_2', nn.Dropout(0.5, False)),
             ])
        )

        self.classifiers2 = nn.Sequential(OrderedDict(
            [('fc8_cxr_2', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
        )

        self.fc3 = nn.Sequential(OrderedDict(
            [('fc6_3', nn.Conv2d(512, 1024, kernel_size=3, padding=16, dilation=16)),
             ('relu6_3', nn.ReLU(False)),
             ('dropout6_3', nn.Dropout(0.5, False)),

             ('fc7_3', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_3', nn.ReLU(False)),
             ('dropout7_3', nn.Dropout(0.5, False)),
             ])
        )

        self.classifiers3 = nn.Sequential(OrderedDict(
            [('fc8_cxr_3', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
        )

        self.fc4 = nn.Sequential(OrderedDict(
            [('fc6_4', nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24)),
             ('relu6_4', nn.ReLU(False)),
             ('dropout6_4', nn.Dropout(0.5, False)),

             ('fc7_4', nn.Conv2d(1024, 1024, kernel_size=1)),
             ('relu7_4', nn.ReLU(False)),
             ('dropout7_4', nn.Dropout(0.5, False)),
             ])
        )

        self.classifiers4 = nn.Sequential(OrderedDict(
            [('fc8_cxr_4', nn.Conv2d(1024, self.n_classes, kernel_size=1)),])
        )
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )

        self.classifiers1 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers2 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=18, dilation=18),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers3 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

        self.fc4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=24, dilation=24),
            nn.ReLU(False),
            nn.Dropout(0.5, False),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(False),
            nn.Dropout(0.5, False)
        )
        self.classifiers4 = nn.Conv2d(1024, self.n_classes, kernel_size=1)

    def forward(self, inputs):
        features = self.features(inputs)
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)
        cl1 = self.classifiers1(fc1)
        cl2 = self.classifiers2(fc2)
        cl3 = self.classifiers2(fc3)
        cl4 = self.classifiers2(fc4)
        outputs = cl1 + cl2 + cl3 + cl4

        print 'fc1', fc1
        print 'fc2', fc2
        print 'fc3', fc3
        print 'fc4', fc4
        print 'cl1', cl1
        print 'cl2', cl2
        print 'cl3', cl3
        print 'cl4', cl4

        return outputs


def adjust_learning_rate(optimizer, decay_rate=0.9, step=0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * ((1 - 1.0*step/max_step)**decay_rate)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.constant(m.bias.data, 0)
        nn.init.normal(m.weight.data, mean=0, std=0.01)
        nn.init.constant(m.bias.data, 0)


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

    model = Deeplab(n_classes=2)
    # model.apply(weights_init)
    model_dict = model.state_dict()
    keys = model_dict.keys()
    print model, keys

    # pretrain_dict = {}
    # voc12_dict = np.load('/media/Disk/wangfuyu/voc12.npy').item()
    # for key in voc12_dict:
    #     if key in model_dict:
    #         pretrain_dict[key] = torch.from_numpy(voc12_dict[key]).float()

    # pretrained_dict = torchvision.models.vgg16(pretrained=True).state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)

    model.load_state_dict(torch.load('/media/Disk/wangfuyu/init.pth'))
    print model_dict, model_dict['features.0.weight'].data

    mceLoss = nn.CrossEntropyLoss(ignore_index=255)

    model.cuda()
    mceLoss.cuda()

    optimizer = optim.SGD([{'params': model.features.parameters()},
                           {'params': model.fc1.parameters()},
                           {'params': model.fc2.parameters()},
                           {'params': model.fc3.parameters()},
                           {'params': model.fc4.parameters()},
                           {'params': model.classifiers1.parameters(), 'lr': 1e-2},
                           {'params': model.classifiers2.parameters(), 'lr': 1e-2},
                           {'params': model.classifiers3.parameters(), 'lr': 1e-2},
                           {'params': model.classifiers4.parameters(), 'lr': 1e-2},
                           ], lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # lambda1 = lambda step: (1 - 1.0 * step / max_step)
    # lambda2 = lambda step: step ** 0.95
    # scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    for step in range(max_step):
        print 'parameters: '
        print model_dict['classifiers1.weight'].data, model_dict['classifiers1.bias'].data
        # adjust_learning_rate(optimizer, decay_rate=0.9, step=step)
        images, ground_truths = reader.next()

        imgs = Variable(torch.from_numpy(images).float()).cuda()
        gts = Variable(torch.from_numpy(ground_truths).long()).cuda()

        print gts.data.size(), imgs.data.size()

        model.zero_grad()
        pred_map = model(imgs)
        loss = mceLoss(pred_map, gts)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print 'loss: ', loss, 'acc: '

            # preds = pred_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            # masks = gts.data.squeeze_(0).cpu().numpy()
            # acc, acc_class = evaluate(preds, masks, 2)
            # print  acc, acc_class

        if step % 1000 == 0:
            torch.save(model.state_dict(), 'step_%d.pth' % step)


        # adjust_learning_rate(optimizer, step=step)

if __name__ == '__main__':
    main()


