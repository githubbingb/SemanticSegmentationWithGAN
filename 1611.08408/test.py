import torch
import numpy as np
import torchvision

#
# input = torch.FloatTensor(1,3,5,5)
# label_map = torch.FloatTensor(1,8,5,5)
#
# b = input[:, 0, :, :].repeat(1,label_map.size()[1], 1, 1)
# g = input[:, 1, :, :].repeat(1, label_map.size()[1], 1, 1)
# r = input[:, 2, :, :].repeat(1, label_map.size()[1], 1, 1)
#
# product_b = label_map * b
# product_g = label_map * g
# product_r = label_map * r
#
# output = torch.cat((product_b, product_g, product_r), dim=1)
#
# print output.size()
#
# for index, c in enumerate(range(0, 2)):
#     print index
#
# a = np.array([1,2,3])
# b= np.array([2,1,2])
# print a>b
#
# import numpy as np
# import cv2
# import os
# from matplotlib import pyplot as plt
#
# GT_root = '/media/Disk/work/vision/800/JM/masks/'
# pred_root = '/media/Disk/work/vision/deeplab_vgg16/'
#
#
# def set_conf(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#
#     for i in xrange(len(y_true_f)):
#         conf[y_true_f[i], y_pred_f[i]] += 1
#
#
# for root, dirnames, filenames in os.walk(pred_root):
#     for filename in filenames:
#         gt = cv2.imread(os.path.join(GT_root, filename), flags=cv2.IMREAD_GRAYSCALE)
#         pred = cv2.imread(os.path.join(pred_root, filename), flags=cv2.IMREAD_GRAYSCALE)
#         _, pred = cv2.threshold(pred, 1, 1, 0)
#
#         missing = (gt > pred).astype(float)
#         redundant = (gt < pred).astype(float)
#
#         diff = cv2.bitwise_xor(pred, gt)
#
#         print filename
#         plt.figure(figsize=(20, 20))
#         plt.subplot(3, 2, 1)
#         plt.title('gt')
#         plt.imshow(gt)
#         plt.subplot(3, 2, 2)
#         plt.title('pred')
#         plt.imshow(pred)
#         plt.subplot(3, 2, 3)
#         plt.title('missing')
#         plt.imshow(missing)
#         plt.subplot(3, 2, 4)
#         plt.title('redundant')
#         plt.imshow(redundant)
#         plt.subplot(3, 2, 5)
#         plt.title('diff')
#         plt.imshow(diff)
#         plt.show()

a = np.array([2,1,3])
b = np.array([2,1,2])

print (a==b).sum()