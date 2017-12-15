import os
import numpy as np
import cv2


class DataReader():
    def __init__(self, data_root, txt, batchsize=1, scale=801):
        self.data_root = data_root
        self.filelist = []
        self.batchsize = batchsize
        self.scale = scale
        self.index = 0

        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.filelist.append(line[0:-1])

        np.random.shuffle(self.filelist)

    def next(self):
        images = np.zeros((self.scale, self.scale, 3, self.batchsize))
        images_interp = np.zeros(((self.scale - 1) / 8 + 1, (self.scale - 1) / 8 + 1, 3, self.batchsize))
        ground_truths = np.zeros((self.scale, self.scale, self.batchsize))
        ground_truths_interp = np.zeros(((self.scale - 1) / 8 + 1, (self.scale - 1) / 8 + 1, self.batchsize))

        if self.index + self.batchsize > len(self.filelist):
            self.index = 0
            np.random.shuffle(self.filelist)

        for i in xrange(self.batchsize):
            img = cv2.imread(os.path.join(self.data_root, 'images', self.filelist[self.index + i] + '.jpg'),
                             cv2.IMREAD_COLOR).astype(float)
            img_interp = cv2.resize(img, ((img.shape[0] - 1) / 8 + 1, (img.shape[1] - 1) / 8 + 1), cv2.INTER_LINEAR)

            img[:, :, 0] = img[:, :, 0] - 104.008
            img[:, :, 1] = img[:, :, 1] - 116.669
            img[:, :, 2] = img[:, :, 2] - 122.675
            images[:, :, :, i] = img

            img_interp[:, :, 0] = img_interp[:, :, 0] - 104.008
            img_interp[:, :, 1] = img_interp[:, :, 1] - 116.669
            img_interp[:, :, 2] = img_interp[:, :, 2] - 122.675
            images_interp[:, :, :, i] = img_interp

            gt = cv2.imread(os.path.join(self.data_root, 'masks', self.filelist[self.index + i] + '.png'),
                            cv2.IMREAD_GRAYSCALE)
            gt_interp = cv2.resize(gt, ((gt.shape[0] - 1) / 8 + 1, (gt.shape[1] - 1) / 8 + 1), cv2.INTER_LINEAR)
            ground_truths[:, :, i] = gt
            ground_truths_interp[:, :, i] = gt_interp

            # print self.filelist[self.index + i]

        self.index += self.batchsize
        images = images.transpose((3, 2, 0, 1))
        images_interp = images_interp.transpose((3, 2, 0, 1))
        ground_truths_interp = ground_truths_interp.transpose((2, 0, 1))
        ground_truths = ground_truths.transpose((2, 0, 1))

        return images, images_interp, ground_truths, ground_truths
