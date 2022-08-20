import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append('/root/fastMRI_hjl')

import tensorflow as tf
import h5py
import numpy as np
import argparse

from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps


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
    model.load_weights('/root/fastMRI/model/XPDNet.h5')

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
        ])[0, :, :, 0] / 1e6

        output.append(output_slice.numpy().tolist())

    return output


def save_file_imtoim(output, image_input_dir, output_dir, fname):
    # assert (image_input_dir / fname).exists(), f"no file named {fname} in {image_input_dir}"
    assert (image_input_dir / fname).exists(), f"no file named {fname} in {image_input_dir}"
    with h5py.File(output_dir / fname, "w") as hf, h5py.File(image_input_dir / fname, "r") as hf_i:
        hf.create_dataset("XPDNet_recon", data=output)
        for key in hf_i:
            hf.create_dataset(key, data=hf_i[key])
        for key in hf_i.attrs:
            hf.attrs[key] = hf_i.attrs[key]


def save_file_recon(output, output_dir, fname):
    with h5py.File(output_dir / fname, "w") as hf:
        hf.create_dataset("reconstruction", data=output)


def xpdnet_eval(args):

    image_input_dir = args.data_dir / 'image'
    kspace_input_dir = args.data_dir / 'kspace'
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    model = getXPDNet()

    i = 1
    tot = len(list(kspace_input_dir.iterdir()))

    for path in kspace_input_dir.iterdir():
        fname = path.name
        kspace_data_path = path
        image_data_path = image_input_dir / fname

        print(f"[{i} / {tot}] Saving file {fname}")

        output = forward_file(model, kspace_data_path, image_data_path)
        if args.save_mode == 'imtoim_input':
            save_file_imtoim(
                output=output,
                image_input_dir=image_input_dir,
                output_dir=output_dir,
                fname=fname
            )
        elif args.save_mode == 'reconstruction':
            save_file_recon(
                output=output,
                output_dir=output_dir,
                fname=fname
            )
        else:
            raise NotImplementedError(f"{args.save_mode} mode not implemented")

        print(f"Successfully saved {fname}")
        print()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help="Where to save reconstructed images"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=Path,
        help="Directory of data"
    )
    parser.add_argument(
        "--save_mode",
        default="imtoim_input",
        type=str,
        choices=["imtoim_input", "reconstruction"],
        help="Mode of saving outputs"
    )

    args = parser.parse_args()

    xpdnet_eval(args)