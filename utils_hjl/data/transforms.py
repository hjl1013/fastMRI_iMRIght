import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, input_mode):
        self.isforward = isforward
        self.max_key = max_key
        self.input_mode = input_mode
    def __call__(self, input, mask, target, attrs, fname_image, fname_kspace, slice): #TODO
        if self.input_mode == 'kspace':
            input = input.astype(np.complex64)
        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # mask transform TODO
        if mask is not None:
            shape = np.array(input.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask = mask.byte()
            # print(mask.shape)

        return input, mask, target, maximum, fname_image, fname_kspace, slice
