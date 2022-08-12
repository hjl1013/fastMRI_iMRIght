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

def transform(kspace, mask):
    kspace = torch.from_numpy(kspace * mask)
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
    shape = [1] + list(kspace.shape)
    kspace = kspace.reshape(shape)
    mask_input = torch.from_numpy(mask.reshape(1, 1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()

    return kspace, mask_input

def forward_file(model, device, kspace_fpath, image_fpath):
    with h5py.File(kspace_fpath, "r") as hf:
        kspaces = np.array(hf['kspace'])
        mask = np.array(hf['mask'])
    with h5py.File(image_fpath, "r") as hf:
        crop_size = np.array(hf['image_label']).shape[-2:]

    output = []
    for kspace in kspaces:
        kspace_input, mask_input = transform(kspace, mask)

        kspace_input = kspace_input.to(device)
        mask_input = mask_input.to(device)

        output_slice = model(kspace_input, mask_input).cpu()[0]
        output_slice = T.center_crop(output_slice, crop_size)

        output.append(output_slice.cpu().tolist())

    return output


def save_file_imtoim(output, image_input_dir, output_dir, fname):
    assert (image_input_dir / fname).exists(), f"no file named {fname} in {image_input_dir}"
    with h5py.File(output_dir / fname, "w") as hf, h5py.File(image_input_dir / fname, "r") as hf_i:
        hf.create_dataset("VarNet_input", data=output)
        for key in hf_i:
            hf.create_dataset(key, data=hf_i[key])
        for key in hf_i.attrs:
            hf.attrs[key] = hf_i.attrs[key]


def save_file_recon(output, output_dir, fname):
    with h5py.File(output_dir / fname, "w") as hf:
        hf.create_dataset("reconstruction", data=output)


def varnet_eval(args):
    device = torch.device(args.device)

    model, _ = get_model(
        model_name=args.model_name,
        model_path=None,  # using default
        test_path=None,  # we don't actually need this
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
        choices=['VarNet_pretrained', 'VarNet_ours', 'VarNet_SNU'],
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
        default="/root/input_imtoim/train",
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
        default="reconstruction",
        type=str,
        choices=["imtoim_input", "reconstruction"],
        help="Mode of saving outputs"
    )

    args = parser.parse_args()

    varnet_eval(args)