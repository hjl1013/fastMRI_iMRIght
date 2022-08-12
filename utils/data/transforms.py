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
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]

            # target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = -1
            maximum = -1

        return input, target, mean, std, fname, slice, maximum

class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]

class VarNetDataTransform_pretrained:
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

        return VarNetSample(
            masked_kspace=kspace,
            mask=mask,
            num_low_frequencies=0,
            target=target,
            fname=fname,
            slice_num=slice,
            max_value=maximum,
            crop_size=(384, 384)
        )

class XPDNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]

class XPDNetDataTransform_pretrained:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = tf.convert_to_tensor(target, dtype=tf.float32)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        kspace = tf.convert_to_tensor(input, dtype=tf.complex64) * 1e6
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        masked_kspace = tf.cast(mask, dtype=kspace.dtype) * kspace
        mask_expanded = mask[None, None, :]
        fourier_mask = tf.dtypes.cast(mask_expanded, tf.uint8)
        kspace_channeled = masked_kspace[..., None]

        return XPDNetSample(
            masked_kspace=kspace_channeled.numpy(),
            mask=fourier_mask.numpy(),
            num_low_frequencies=0,
            target=target.numpy(),
            fname=fname,
            slice_num=slice,
            max_value=maximum,
            crop_size=(384, 384)
        )

class UnetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    input_image: torch.Tensor
    target: torch.Tensor
    mean: float
    std: float
    fname: str
    slice_num: int
    max_value: float

class UnetDataTransform_pretrained:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input).type(torch.FloatTensor)
        # normalize input
        input, mean, std = normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]

            # target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = -1
            maximum = -1

        return UnetSample(
            input_image=input,
            target=target,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice,
            max_value=maximum
        )