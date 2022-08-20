import h5py
import torch
import numpy as np
import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(Path(__file__).parent.absolute()))
sys.path.append('/root/fastMRI_hjl')

from get_model import get_model
import Main.data.transforms as T
from utils.data.transforms import VarnetDataTransform


def forward_file(model, device, kspace_fpath, image_fpath):
    with h5py.File(kspace_fpath, "r") as hf:
        kspaces = np.array(hf['kspace'])
        mask = np.array(hf['mask'])
    with h5py.File(image_fpath, "r") as hf:
        crop_size = np.array(hf['image_label']).shape[-2:]

    transform = VarnetDataTransform(isforward=True, max_key='max')
    output = []
    for kspace in kspaces:
        mask_input, kspace_input, _, _, _, _ = transform(
            mask=mask,
            input=kspace,
            target=None,
            attrs=None,
            fname=None,
            slice=None
        )

        kspace_input = kspace_input.to(device)[None, ...]
        mask_input = mask_input.to(device)[None, ...]

        output_slice = model(kspace_input, mask_input).cpu()[0]
        output_slice = T.center_crop(output_slice, crop_size)

        output.append(output_slice.cpu().tolist())

    return output


def save_file_imtoim(output, image_input_dir, output_dir, fname):
    assert (image_input_dir / fname).exists(), f"no file named {fname} in {image_input_dir}"
    with h5py.File(output_dir / fname, "w") as hf, h5py.File(image_input_dir / fname, "r") as hf_i:
        hf.create_dataset("VarNet_recon", data=output)
        for key in hf_i:
            hf.create_dataset(key, data=hf_i[key])
        for key in hf_i.attrs:
            hf.attrs[key] = hf_i.attrs[key]


def save_file_recon(output, output_dir, fname):
    with h5py.File(output_dir / fname, "w") as hf:
        hf.create_dataset("reconstruction", data=output)


def varnet_eval(args):
    device = torch.device(args.device)

    model = get_model(
        model_name=args.model_name,
        model_path=None,  # using default
    )

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image_input_dir = args.data_dir / 'image'
        kspace_input_dir = args.data_dir / 'kspace'
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        i = 1
        tot = len(list(kspace_input_dir.iterdir()))

        for path in kspace_input_dir.iterdir():
            fname = path.name
            kspace_data_path = path
            image_data_path = image_input_dir / fname

            print(f"[{i} / {tot}] Saving file {fname}")

            output = forward_file(model, device, kspace_data_path, image_data_path)
            if args.save_mode == 'imtoim_input':
                save_file_imtoim(output, image_input_dir, output_dir, fname)
            elif args.save_mode == 'reconstruction':
                save_file_recon(output, output_dir, fname)
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
        "--model_name",
        default="VarNet_pretrained",
        type=str,
        choices=['VarNet_pretrained', 'VarNet_ours', 'VarNet_SNU', 'test_varnet'],
        help="Name of model"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/input_imtoim_VarNet/train/image",
        type=Path,
        help="Where to save reconstructed images"
    )
    parser.add_argument(
        "--data_dir",
        default="/root/input/train",
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

    varnet_eval(args)