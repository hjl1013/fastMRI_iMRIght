import numpy as np
import torch


def rand_bbox(width, height, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def cutmix(input_batch, target_batch, img_mask_batch, alpha=1.0):
    batch_size, _, width, height = input_batch.shape

    lam = np.random.beta(alpha, alpha)
    permute = torch.randperm(batch_size)
    bbx1, bby1, bbx2, bby2 = rand_bbox(width, height, lam)
    mask = torch.ones((width, height))
    mask[bbx1:bbx2, bby1:bby2] = 0
    mask = mask[None, None, ...]

    mixup_input = mask * input_batch + (1 - mask) * input_batch[permute]
    mixup_target = mask * target_batch + (1 - mask) * target_batch[permute]
    mixup_img_mask = mask * img_mask_batch + (1 - mask) * img_mask_batch[permute]

    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (width * height)

    return mixup_input, mixup_target, mixup_img_mask, lam


if __name__ == '__main__':
    input_batch = torch.zeros([32, 4, 384, 384])
    target_batch = torch.zeros([32, 1, 384, 384])
    img_mask_batch = torch.zeros([32, 1, 384, 384])

    mixup_data, mixup_target, mixup_img_mask, lam = apply_cutmix_to_batch(input_batch, target_batch, img_mask_batch)

    print(mixup_data.shape)
    print(mixup_target.shape)
    print(mixup_img_mask.shape)
    print(lam)
    print(torch.std(input_batch, dim=(1, 2, 3)).shape)
    print(torch.mean(input_batch, dim=(1, 2, 3)).shape)
    print(input_batch.max(dim=1).values.max(dim=1).values.max(dim=1).values.shape)

    a = torch.tensor([[1., 2.], [2., 3.]])
    b = torch.tensor([1., 2., 2., 3.])
    print(torch.std(a))
    print(torch.std(b))
