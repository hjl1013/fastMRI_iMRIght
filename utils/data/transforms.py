import torch
from random import random
# import albumentations as A
import matplotlib.pyplot as plt

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
    return (data-min)/(max-min), min, max

def random_augment(input, target):
    r_flip = random()
    if r_flip > 0.5:
        for i in range(len(input)):
            input[i] = torch.fliplr(input[i])
        for i in range(len(target)):
            target[i] = torch.fliplr(target[i])

    r_rotate = random()
    if r_rotate > 0.25:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])
    if r_rotate > 0.5:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])
    if r_rotate > 0.75:
        for i in range(len(input)):
            input[i] = torch.rot90(input[i])
        for i in range(len(target)):
            target[i] = torch.rot90(target[i])

    r_rollud = int(random() * 19) - 9
    for i in range(len(input)):
        input[i] = torch.roll(input[i], shifts=r_rollud, dims=0)
    for i in range(len(target)):
        target[i] = torch.roll(target[i], shifts=r_rollud, dims=0)

    r_rolllr = int(random() * 19) - 9
    for i in range(len(input)):
        input[i] = torch.roll(input[i], shifts=r_rolllr, dims=1)
    for i in range(len(target)):
        target[i] = torch.roll(target[i], shifts=r_rolllr, dims=1)

    return input, target

class VarnetDataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()

        return mask, kspace, target, maximum, fname, slice

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
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = -1
            maximum = -1

        return input, target, mean, std, fname, slice, maximum

class ResUnetDataTransform:
    def __init__(self, max_key, use_augment):
        self.max_key = max_key
        self.use_augment = use_augment
    def __call__(self, input, target, attrs, fname, slice):

        input = to_tensor(input).type(torch.FloatTensor)
        # input = input[3:]
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)

        maximum = np.max(target)

        target = to_tensor(target)
        target = target[None, ...]
        target = normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)

        if self.use_augment:
            input, target = random_augment(input, target)

        return input, target, maximum, mean, std, fname, slice

class ADLDataTransform:
    def __init__(self, max_key, use_augment):
        self.max_key = max_key
        self.use_augment = use_augment

    def __call__(self, input, target, attrs, fname, slice):

        input = to_tensor(input).type(torch.FloatTensor)
        input = input[3:]

        # normalize input
        input, min, max = minmaxScale(input)

        target = to_tensor(target)
        maximum = torch.max(target)
        #maximum = attrs[self.max_key]
        target = target[None, ...]
        target = (target-min)/(max-min)

        if self.use_augment:
            input, target = random_augment(input, target)

        data = dict()
        data['x'] = target # gt
        data['y'] = input # data with noise
        data['filename'] = fname
        data['min'] = min
        data['max'] = max
        data['maximum'] = maximum # max of target, use for SSIM loss
        data['slice'] = slice

        return data