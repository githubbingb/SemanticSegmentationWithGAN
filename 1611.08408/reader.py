import os
import numpy as np
import cv2


class Reader():
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
        ground_truths_interp = np.zeros(((self.scale - 1) / 8 + 1, (self.scale - 1) / 8 + 1, self.batchsize))

        if self.index + self.batchsize > len(self.filelist):
            self.index = 0
            np.random.shuffle(self.filelist)

        for i in xrange(self.batchsize):
            img = cv2.imread(os.path.join(self.data_root, 'images', self.filelist[self.index + i] + '.jpg')).astype(
                float)
            img[:, :, 0] = img[:, :, 0] - 104.008
            img[:, :, 1] = img[:, :, 1] - 116.669
            img[:, :, 2] = img[:, :, 2] - 122.675
            images[:, :, :, i] = img

            gt = cv2.imread(os.path.join(self.data_root, 'masks', self.filelist[self.index + i] + '.png'),
                            cv2.IMREAD_GRAYSCALE)
            print gt.shape
            gt_interp = cv2.resize(gt, ((img.shape[0]-1)/8+1, (img.shape[1]-1)/8+1), cv2.INTER_LINEAR)
            ground_truths_interp[:, :, i] = gt_interp

            # print self.filelist[self.index + i]

        self.index += self.batchsize
        images = images.transpose((3, 2, 0, 1))
        ground_truths_interp = ground_truths_interp.transpose((2,0,1))
        # print self.index

        return images, ground_truths_interp
