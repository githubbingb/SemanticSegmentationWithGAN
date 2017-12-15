import numpy as np
import torch
import cv2


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long().squeeze_(1)


def interp(src, zoom=None, shrink=None):
    shape = src.size()

    if zoom is not None:
        height_out = (shape[1] - 1) * zoom + 1
        width_out = (shape[2] - 1) * zoom + 1
    if shrink is not None:
        height_out = (shape[1] - 1) / shrink + 1
        width_out = (shape[2] - 1) / shrink + 1

    dst = np.zeros(shape=(shape[0], height_out, width_out))

    for index in xrange(shape[0]):
        single = src[index, :, :].astype(np.uint8)
        dst[index, :, :] = cv2.resize(single, (height_out, width_out), cv2.INTER_LINEAR)
    return dst


def product(input, label_map):
    print input.size(), label_map.size()
    b = input[:, 0, :, :].unsqueeze(1).repeat(1, label_map.size()[1], 1, 1)
    g = input[:, 1, :, :].unsqueeze(1).repeat(1, label_map.size()[1], 1, 1)
    r = input[:, 2, :, :].unsqueeze(1).repeat(1, label_map.size()[1], 1, 1)

    product_b = label_map * b
    product_g = label_map * g
    product_r = label_map * r

    return torch.cat((product_b, product_g, product_r), dim=1).float()


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
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc


def onehot_encoder(ground_truth, n_classes):
    outputs = np.zeros((ground_truth.shape[0], n_classes, ground_truth.shape[1], ground_truth.shape[2]))
    for i in xrange(ground_truth.shape[0]):
        gt = ground_truth[i, :, :]
        for index, c in enumerate(range(0, n_classes)):
            mask = (gt == c)
            mask = np.expand_dims(mask, 0)
            if index == 0:
                onehot = mask
            else:
                onehot = np.concatenate((onehot, mask), axis=0)
        outputs[i, :, :, :] = onehot

    return torch.from_numpy(outputs).float()
