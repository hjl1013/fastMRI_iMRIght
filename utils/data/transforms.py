import numpy as np
import torch
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import tensorflow as tf

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
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):

        input = to_tensor(input).type(torch.FloatTensor)
        input = input[2:]
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)

        if not self.isforward:
            target = to_tensor(target)
            target = target[None, ...]
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)

            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        return input, target, maximum, mean, std, fname, slice