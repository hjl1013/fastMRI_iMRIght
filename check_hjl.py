import h5py
import matplotlib.pyplot as plt
import xml.etree.ElementTree as etree
import numpy as np
import torch
from fastmri.data.transforms import to_tensor
import fastmri
from pathlib import Path

# "../../../input/train/kspace/multicoil_train/brain103.h5"
# "../../singlecoil_challenge/multicoil_train/file1000282.h5"

# with h5py.File("../../singlecoil_challenge/multicoil_train/file1000282.h5", "r") as hf:
#     print(hf.keys())
#     # print(hf['kspace'][0].shape)
#     # print(hf['mask'].shape)
#

def check_target():
    with h5py.File("../../../input/kspace/multicoil_val/brain100.h5", "r") as hf, h5py.File("/root/input/image/multicoil_val/brain100.h5", "r") as hf2:
        print(hf2.keys())
        print(hf['kspace'])
        kspace = np.array(hf['kspace'])
        print(kspace.shape)
        kspace_torch = to_tensor(kspace)
        image = fastmri.ifft2c(kspace_torch)
        image = complex_center_crop(image, (384, 384))
        image = fastmri.complex_abs(image)
        print(image.shape)
        image = fastmri.rss(image, dim = 1)
        print(image.shape)
        print(image.max())
        print(hf2.attrs['max'])
        target = hf2['image_label']
        print(target.shape)

        i = 3
        plt.figure()
        plt.subplot(121)
        plt.imshow(target[i])
        plt.subplot(122)
        plt.imshow(image[i])
        plt.show()
    # print(float(hf['mask'].shape[0])/np.sum(hf['mask']))
    # mask = np.array(hf['mask'])
    # kspace = hf['kspace'][0]
    # kspace = kspace.astype(np.complex64)
    # kspace = to_tensor(kspace)
    #
    # shape = np.array(kspace.shape)
    # num_cols = shape[-2]
    # shape[:-3] = 1
    # mask_shape = [1] * len(shape)
    # mask_shape[-2] = num_cols
    # mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
    # mask = mask.reshape(*mask_shape)
    # mask = mask.byte()
    #
    # shape = mask.shape
    # shape = [4] * (len(shape) + 1)
    # shape[1:] = mask.shape
    # mask_tmp = torch.zeros(shape)
    # for i in range(4):
    #     mask_tmp[i] = mask
    #
    # mask = mask_tmp
    #
    # cent = mask.shape[-2] // 2
    # print(cent)
    # print(mask.squeeze()[:, :cent].shape)
    # left = torch.nonzero(mask.squeeze()[:cent] == 0)[-1]
    # right = torch.nonzero(mask.squeeze()[cent:] == 0)[0] + cent
#
def print_recon():
    with h5py.File("/root/input/leaderboard/image/brain_test1.h5", "r") as hf:
        print(hf.keys())
        recon_image = np.array(hf['image_label'])
    with h5py.File("/root/leaderboard_recon/kspace/brain_test1.h5", "r") as hf:
        target_image = np.array(hf['image_label'])
        max = hf.attrs['max']

    # print(ssim(recon_image, target_image, max))
    print(recon_image.shape)
    print(target_image.shape)

    plt.figure()
    plt.subplot(121)
    plt.imshow(recon_image[0])
    plt.subplot(122)
    plt.imshow(target_image[0])
    plt.show()

# for fname in Path("/root/fastMRI/result/VarNet_pretrained/reconstructions").iterdir():
#     with h5py.File(str(fname), "r") as hf:
#         print(hf['reconstruction'])
#         plt.figure()
#         plt.subplot(111)
#         plt.imshow(hf['reconstruction'][0])
#         plt.show()
#         data_path_image = Path("/root/input/image/multicoil_val")
#         f = h5py.File(str(data_path_image / str(fname).split('/')[-1]), "r")
#         print(ssim(np.array(hf['reconstruction']), np.array(f['image_label']), f.attrs['max']))

def data_reconstruct():
    cnt = 1
    tot = len(list(Path("/root/input/leaderboard/kspace").iterdir()))
    for fname in Path("/root/input/leaderboard/kspace").iterdir():
        print(cnt, " / ", tot)
        with h5py.File(str(fname), "r") as hf_k, h5py.File(str(Path("/root/input/leaderboard/image") / fname.name), "r") as hf_i, h5py.File(str(Path("/root/leaderboard_recon/kspace")/ fname.name), "w") as whf:
            for key in hf_k:
                whf.create_dataset(key, data = hf_k[key])
            print(fname.name + " created kspace dataset")
            for key in hf_i:
                whf.create_dataset(key, data = hf_i[key])
            print(fname.name + " created image dataset")
            for key in hf_i.attrs:
                whf.attrs[key] = hf_i.attrs[key]
            for key in hf_k.attrs:
                whf.attrs[key] = hf_k.attrs[key]
            print(fname.name + " created attributes")

        cnt += 1
        print()

def check_file(file):
    with h5py.File(file, "r") as hf:
        print(hf.keys())
        plt.figure()
        plt.subplot(111)
        plt.imshow(hf['reconstruction'][0])
        plt.show()


# check_file("/root/result/VarNet_SNU/reconstructions/brain_test1.h5")
print_recon()
# data_reconstruct()