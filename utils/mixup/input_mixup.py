import numpy as np
import torch

def input_mixup(input_batch, target_batch, img_mask_batch, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    permute = torch.randperm(input_batch.shape[0])

    target_batch = target_batch * img_mask_batch

    mixup_input = lam * input_batch + (1 - lam) * input_batch[permute]
    mixup_target = lam * target_batch + (1 - lam) * target_batch[permute]
    mixup_img_mask = (img_mask_batch.type(torch.IntTensor) | img_mask_batch[permute].type(torch.IntTensor)).type(torch.FloatTensor)

    return mixup_input, mixup_target, mixup_img_mask, lam

if __name__ == '__main__':
    input_batch = torch.zeros([2, 1, 3, 3])
    target_batch = torch.zeros([2, 1, 3, 3])
    img_mask_batch = torch.Tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]])

    a = torch.tensor([[1, 1], [0, 0]])
    b = torch.tensor([[1, 0], [1, 0]])
    print(a | b)