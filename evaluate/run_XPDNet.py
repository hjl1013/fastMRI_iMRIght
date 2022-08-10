import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os
import sys
sys.path.append('/root/fastMRI_hjl')

import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import fastmri
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps
from fastmri_recon.models.utils.fourier import FFTBase

from utils.data.load_data import create_data_loaders_for_pretrained


def getXPDNet():
    n_primal = 5
    model_fun, model_kwargs, n_scales, res = [
        (model_fun, kwargs, n_scales, res)
        for m_name, m_size, model_fun, kwargs, _, n_scales, res in get_model_specs(n_primal=n_primal, force_res=False)
        if m_name == 'MWCNN' and m_size == 'medium'
    ][0]
    model_kwargs['use_bias'] = False
    run_params = dict(
        n_primal=n_primal,
        multicoil=True,
        n_scales=n_scales,
        refine_smaps=True,
        refine_big=True,
        res=res,
        output_shape_spec=True,
        n_iter=25,
    )
    model = XPDNet(model_fun, model_kwargs, **run_params)
    kspace_size = [1, 16, 768, 396]
    inputs = [
        tf.zeros(kspace_size + [1], dtype=tf.complex64),  # kspace
        tf.zeros(kspace_size, dtype=tf.complex64),  # mask
        tf.zeros(kspace_size, dtype=tf.complex64),  # smaps
        tf.constant([[384, 384]]),  # shape
    ]
    model(inputs)
    model.load_weights('/root/models/XPDNet_pretrained/model_weights.h5')

    return model


def transform(kspace, mask):
    kspace = tf.convert_to_tensor(kspace, dtype=tf.complex64)[None, :, :, :] * 1e6
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    masked_kspace = tf.cast(mask, dtype=kspace.dtype) * kspace

    shape = tf.shape(masked_kspace)
    mask_expanded = mask[None, None, None, :]

    fourier_mask = tf.tile(mask_expanded, [shape[0], 1, 1, 1])
    fourier_mask = tf.dtypes.cast(fourier_mask, tf.uint8)

    smaps = extract_smaps(kspace, low_freq_percentage=8)
    kspaces_channeled = masked_kspace[..., None]

    return kspaces_channeled, smaps, fourier_mask

def forward_file(model, kspace_fpath, image_fpath):
    with h5py.File(kspace_fpath, "r") as hf:
        kspaces = np.array(hf['kspace'])
        mask = np.array(hf['mask'])
    with h5py.File(image_fpath, "r") as hf:
        crop_size = np.array(hf['image_label']).shape[-2:]

    output = []
    for kspace in kspaces:
        kspace_input, smaps_input, mask_input = transform(kspace, mask)

        output_slice = model([
            kspace_input,
            mask_input,
            smaps_input,
            [[crop_size[0], crop_size[1]]]
        ])[0, :, :, 0]

        output.append(output_slice.numpy().tolist())

    return output


def save_XPDNet_reconstructions_leaderboard(output, recon_dir, fname):
    # assert (image_input_dir / fname).exists(), f"no file named {fname} in {image_input_dir}"
    with h5py.File(recon_dir / fname, "w") as hf:
        hf.create_dataset("reconstruction", data=output)


def save_XPDNet_output():
    data_dir = Path('/root/input/leaderboard')
    output_dir = Path('/root/result/XPDNet_pretrained')

    image_input_dir = data_dir / 'image'
    kspace_input_dir = data_dir / 'kspace'
    recon_dir = output_dir / 'reconstructions'
    recon_dir.mkdir(exist_ok=True, parents=True)

    model = getXPDNet()

    i = 1
    tot = len(list(kspace_input_dir.iterdir()))

    for path in kspace_input_dir.iterdir():
        fname = path.name
        kspace_data_path = path
        image_data_path = image_input_dir / fname

        print(f"[{i} / {tot}] Saving file {fname}")

        output = forward_file(model, kspace_data_path, image_data_path)
        save_XPDNet_reconstructions_leaderboard(output, recon_dir, fname)

        print(f"Successfully saved {fname}")
        print()
        i += 1


if __name__ == '__main__':
    save_XPDNet_output()
    # model = getXPDNet()
    # data_loader = create_data_loaders_for_pretrained(data_path=Path("/root/input/leaderboard"), model_name='XPDNet_pretrained', isforward=False)
    #
    # outputs = {}
    # # j = 0
    # for batch in tqdm(data_loader, desc="Running inference"):
    #     with tf.device('/device:GPU:0'):
    #         start = time.perf_counter()
    #         target = tf.convert_to_tensor(batch.target)
    #         masked_kspace = tf.convert_to_tensor(batch.masked_kspace)
    #
    #         mask = tf.convert_to_tensor(batch.mask)
    #
    #         print("convert time")
    #         print(time.perf_counter() - start)
    #         print()
    #
    #         start = time.perf_counter()
    #
    #         smaps = extract_smaps(masked_kspace[:,:,:,:,0], low_freq_percentage=8)
    #
    #         print("smap extract time")
    #         print(time.perf_counter() - start)
    #         print()
    #
    #         start = time.perf_counter()
    #
    #         output = model([
    #             masked_kspace,
    #             mask,
    #             smaps,
    #             [[384, 384]]
    #         ])
    #
    #         print("forward time")
    #         print(time.perf_counter() - start)
    #         print()
    #
    #         # if j == 0:
    #         #     j += 1
    #         #     print(output.shape)
    #         #     print(j)
    #         #     plt.figure()
    #         #     plt.subplot(111)
    #         #     plt.imshow(output[0,:,:,0])
    #         #     plt.show()
    #
    #     for i, f in enumerate(batch.fname):
    #         if f not in outputs:
    #             outputs[f] = []
    #
    #         outputs[f].append(output[i].numpy().tolist())
    #
    # fastmri.save_reconstructions(outputs, Path('/root/result/XPDNet_pretrained/reconstructions'))

    # with h5py.File('/root/input_recon/kspace/multicoil_val/brain1.h5', "r") as hf:
    #     kspace = np.array(hf['kspace'])
    #     mask = np.array(hf['mask'])
    #     target = np.array(hf['image_label'])
    #
        # kspace = tf.convert_to_tensor(kspace, dtype=tf.complex64)[0:1, :, :, :] * 1e6
        # mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        # masked_kspace = tf.cast(mask, dtype=kspace.dtype) * kspace
        #
        # shape = tf.shape(masked_kspace)
        # mask_expanded = mask[None, None, None, :]
        #
        # fourier_mask = tf.tile(mask_expanded, [shape[0], 1, 1, 1])
        # fourier_mask = tf.dtypes.cast(fourier_mask, tf.uint8)
        #
        # smaps = extract_smaps(kspace, low_freq_percentage=8)
        # kspaces_channeled = masked_kspace[..., None]
    #
    #     print(kspaces_channeled.shape)
    #     print(fourier_mask.shape)
    #     print(smaps.shape)
    #
    # output = model([
    #     kspaces_channeled,
    #     fourier_mask,
    #     smaps,
    #     [[384, 384]]
    # ])
    #
    # print(output.shape)
    # plt.figure()
    # plt.subplot(111)
    # plt.imshow(target[0])
    # plt.show()

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(output[0,:,:,0])
    # plt.subplot(122)
    # plt.imshow(target[0])
    # plt.show()