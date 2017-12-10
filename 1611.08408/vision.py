import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

num_classes = 2
GT_root = '/media/Disk/work/vision/800/JM/masks/'
pred_root = '/media/Disk/work/vision/deeplab_vgg16/'


def set_conf(y_true, y_pred, conf):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    for i in xrange(len(y_true_f)):
        conf[y_true_f[i], y_pred_f[i]] += 1


def mIoU(conf):
    res = np.zeros(num_classes)
    for i in xrange(num_classes):
        tp = conf[i, i]
        fp = conf[i, :].sum()
        fn = conf[:, i].sum()
        res[i] = 1.0 * tp / (fp + fn - tp)

    # return np.mean(accuracy)
    return res


for root, dirnames, filenames in os.walk(pred_root):
    for filename in filenames:
        gt = cv2.imread(os.path.join(GT_root, filename), flags=cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_root, filename), flags=cv2.IMREAD_GRAYSCALE)
        _, pred = cv2.threshold(pred, 1, 1, 0)

        missing = (gt > pred).astype(float)
        redundant = (gt < pred).astype(float)
        diff = cv2.bitwise_xor(pred, gt)

        conf = np.zeros((num_classes, num_classes))
        set_conf(gt, pred, conf)
        iou = mIoU(conf)

        print filename, iou
        plt.figure(figsize=(20, 20))
        plt.subplot(3, 2, 1)
        plt.title('gt')
        plt.imshow(gt)
        plt.subplot(3, 2, 2)
        plt.title('pred')
        plt.imshow(pred)
        plt.subplot(3, 2, 3)
        plt.title('missing')
        plt.imshow(missing)
        plt.subplot(3, 2, 4)
        plt.title('redundant')
        plt.imshow(redundant)
        plt.subplot(3, 2, 5)
        plt.title('diff')
        plt.imshow(diff)
        plt.show()
