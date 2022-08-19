import torch
from random import random
# import albumentations as A
import matplotlib.pyplot as plt
import cv2

from fastmri.data.transforms import *


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


def minmaxScale(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min), min, max


def random_augment(input, target, img_mask):

    if img_mask is None:
        img_mask = torch.zeros(target.shape)

    r_flip = random()
    if r_flip > 0.5:
        for i in range(len(input)):
            input[i] = torch.fliplr(input[i])
        for i in range(len(target)):
            target[i] = torch.fliplr(target[i])
        for i in range(len(img_mask)):
            img_mask[i] = torch.fliplr(img_mask[i])

    r_rotate = random()
    if r_rotate > 0.25:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])
        for i in range(len(img_mask)):
            img_mask[i] = torch.rot90(img_mask[i])
    if r_rotate > 0.5:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])
        for i in range(len(img_mask)):
            img_mask[i] = torch.rot90(img_mask[i])
    if r_rotate > 0.75:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])
        for i in range(len(img_mask)):
            img_mask[i] = torch.rot90(img_mask[i])

    r_rollud = int(random() * 19) - 9
    for i in range(len(input)):
        input[i] = torch.roll(input[i], shifts=r_rollud, dims=0)
    for i in range(len(target)):
        target[i] = torch.roll(target[i], shifts=r_rollud, dims=0)
    for i in range(len(img_mask)):
        img_mask[i] = torch.roll(img_mask[i], shifts=r_rollud, dims=0)

    r_rolllr = int(random() * 19) - 9
    for i in range(len(input)):
        input[i] = torch.roll(input[i], shifts=r_rolllr, dims=1)
    for i in range(len(target)):
        target[i] = torch.roll(target[i], shifts=r_rolllr, dims=1)
    for i in range(len(img_mask)):
        img_mask[i] = torch.roll(img_mask[i], shifts=r_rollud, dims=1)

    return input, target, img_mask


class VarnetDataTransform:
    def __init__(self, max_key):
        self.max_key = max_key

    def __call__(self, mask, input, target, attrs, fname, slice):
        maximum = np.max(target)

        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()

        img_mask = np.zeros(target.shape)
        img_mask[target > 5e-5] = 1
        kernel = np.ones((3, 3), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        img_mask = cv2.dilate(img_mask, kernel, iterations=15)
        img_mask = cv2.erode(img_mask, kernel, iterations=14)

        target = to_tensor(target)
        img_mask = (to_tensor(img_mask)).type(torch.FloatTensor)

        return mask, kspace, target, maximum, fname, slice, img_mask


class UnetDataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input).type(torch.FloatTensor)
        input = input[None, ...]
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]

            # target = center_crop(target, crop_size)
            target = target[None, ...]
        else:
            target = -1
            maximum = -1

        return input, target, mean, std, fname, slice, maximum


class MultichannelDataTransform:
    def __init__(self, max_key, use_augment):
        self.max_key = max_key
        self.use_augment = use_augment

    def __call__(self, input, input_num, target, attrs, fname, slice):
        input = to_tensor(input).type(torch.FloatTensor)
        input = input[-input_num:]
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)

        maximum = np.max(target)

        img_mask = np.zeros(target.shape)
        img_mask[target > 5e-5] = 1
        kernel = np.ones((3, 3), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        img_mask = cv2.dilate(img_mask, kernel, iterations=15)
        img_mask = cv2.erode(img_mask, kernel, iterations=14)

        target = to_tensor(target)
        img_mask = (to_tensor(img_mask)).type(torch.FloatTensor)

        target = target[None, ...]
        img_mask = img_mask[None, ...]

        if self.use_augment:
            input, target, img_mask = random_augment(input, target, img_mask)

        return input, target, mean, std, fname, slice, maximum, img_mask


class ADLDataTransform:
    def __init__(self, max_key, use_augment):
        self.max_key = max_key
        self.use_augment = use_augment

    def __call__(self, input, input_num, target, attrs, fname, slice):
        input = to_tensor(input).type(torch.FloatTensor)
        input = input[-input_num:]

        # normalize input
        input, min, max = minmaxScale(input)

        target = to_tensor(target)
        maximum = torch.max(target)
        # maximum = attrs[self.max_key]
        target = target[None, ...]
        target = (target - min) / (max - min)

        if self.use_augment:
            input, target, _ = random_augment(input, target, None)

        data = dict()
        data['x'] = target  # gt
        data['y'] = input  # data with noise
        data['filename'] = fname
        data['min'] = min
        data['max'] = max
        data['maximum'] = maximum  # max of target, use for SSIM loss
        data['slice'] = slice

        return data

class MultichannelDataTransform_with_cutmix:
    def __init__(self, max_key, use_augment):
        self.max_key = max_key
        self.use_augment = use_augment

    def __call__(self, input, input_num, target, attrs, fname, slice):
        input = to_tensor(input).type(torch.FloatTensor)
        input = input[-input_num:]
        # normalize input
        # input, mean, std = normalize_instance(input, eps=1e-11)
        # input = input.clamp(-6, 6)

        img_mask = np.zeros(target.shape)
        img_mask[target > 5e-5] = 1
        kernel = np.ones((3, 3), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        img_mask = cv2.dilate(img_mask, kernel, iterations=15)
        img_mask = cv2.erode(img_mask, kernel, iterations=14)

        target = to_tensor(target)
        img_mask = (to_tensor(img_mask)).type(torch.FloatTensor)

        target = target[None, ...]
        img_mask = img_mask[None, ...]

        if self.use_augment:
            input, target, img_mask = random_augment(input, target, img_mask)

        return input, target, fname, slice, img_mask