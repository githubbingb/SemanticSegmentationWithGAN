import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from DataFolder import MyDataFolder
from torch.utils.data import DataLoader
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')

opt = parser.parse_args()

mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

dataFolder = MyDataFolder(data_root='/media/Disk/wangfuyu/data/cxr/801',
                          txt='/media/Disk/wangfuyu/data/cxr/801/testJM.txt',
                          input_transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.mul_(255)),
                              transforms.Normalize(*mean_std), ]),
                          target_transform=MaskToTensor()
                          )

dataloader = DataLoader(dataset=dataFolder, batch_size=opt.batchsize, shuffle=False, num_workers=2)

class Deeplab(nn.Module):
    def __init__(self, n_classes):
        super(Deeplab, self).__init__()
        self.n_classes = n_classes
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
        return outputs


def adjust_learning_rate(optimizer, power=0.9, epoch=0):
    for index, param_group in enumerate(optimizer.param_groups):
        if index %2 == 0:
            param_group['lr'] = param_group['lr'] * ((1 - 1.0* epoch / opt.niter) ** (power))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print m.weight.data
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.constant(m.bias.data, 0)
        # nn.init.normal(m.weight.data, mean=0, std=0.01)
        # nn.init.constant(m.bias.data, 0)


def train():
    model = Deeplab(n_classes=2)
    model.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/init.pth'))
    # model.apply(weights_init)

    mceLoss = nn.CrossEntropyLoss(ignore_index=255)


    model.cuda()
    mceLoss.cuda()

    optimizer = optim.SGD([{'params': [param for name, param in model.features.named_parameters() if
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
    # optimizer = optim.SGD([{'params': model.features.parameters()},
    #                        {'params': model.fc1.parameters()},
    #                        {'params': model.fc2.parameters()},
    #                        {'params': model.fc3.parameters()},
    #                        {'params': model.fc4.parameters()},
    #                        {'params': model.classifiers1.parameters(), 'lr': 1e-2},
    #                        {'params': model.classifiers2.parameters(), 'lr': 1e-2},
    #                        {'params': model.classifiers3.parameters(), 'lr': 1e-2},
    #                        {'params': model.classifiers4.parameters(), 'lr': 1e-2},
    #                        ], lr=1e-3, momentum=0.9, weight_decay=5e-4)

    for epoch in range(opt.niter):
        # adjust_learning_rate(optimizer, power=0.9, epoch=epoch)
        for index, data in enumerate(dataloader, 0):
            images, ground_truths = data
            ground_truths = torch.from_numpy(interp(ground_truths.numpy(), shrink=8)).float()

            imgs = Variable(images.float()).cuda()
            gts = Variable(ground_truths.long()).cuda()

            model.zero_grad()
            pred_map = model(imgs)
            loss = mceLoss(pred_map, gts)
            loss.backward()
            optimizer.step()

            preds = pred_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            acc, acc_class, _, _, _ = evaluate(preds, ground_truths.squeeze_(0).numpy(), 2)

        print 'loss: ', loss
        print acc, acc_class

        torch.save(model.state_dict(), 'deeplab_epoch_%d.pth' % epoch)


def eval():
    model = Deeplab(n_classes=2)
    model.load_state_dict(torch.load('/media/Disk/wangfuyu/SemanticSegmentationWithGAN/1611.08408/deeplab_epoch_60.pth'))
    model.cuda()

    gts_all, predictions_all = [], []

    for index, data in enumerate(dataloader, 0):
        images, ground_truths = data

        imgs = Variable(images.float()).cuda()
        gts = Variable(ground_truths.long()).cuda()

        pred_map = model(imgs)
        pred_map_interp = interp(pred_map.data.max(1)[1].squeeze_(1).cpu().numpy(), zoom=8)
        predictions_all.append(np.squeeze(pred_map_interp.astype(long), axis=0))
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())

    acc, acc_class, iou, _, dice = evaluate(predictions_all, gts_all, 2)
    print acc, acc_class, iou, dice


if __name__ == '__main__':
    eval()


