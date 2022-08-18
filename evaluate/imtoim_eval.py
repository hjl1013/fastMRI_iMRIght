import argparse
import torch
import h5py
import numpy as np
from pathlib import Path

from get_model import get_model
from utils.data.transforms import UnetDataTransform, MultichannelDataTransform

def forward_file_unet(args, model, device, image_fpath):
    with h5py.File(image_fpath, "r") as hf:
        images = np.array(hf['image_input'])

    transform = UnetDataTransform(isforward=True, max_key='max')

    output = []
    for image in images:
        image_input, _, mean, std, _, _, _ = transform(
            input=image,
            target=None,
            attrs=None,
            fname=None,
            slice=None,
        )

        image_input = image_input.to(device)[None, ...]

        output_slice = model(image_input).cpu()[0][0] * std + mean

        output.append(output_slice.cpu().tolist())

    return output


def forward_file_multichannel(args, model, device, image_fpath):
    with h5py.File(image_fpath, "r") as hf:
        images = np.stack([
            hf['image_input'],
            hf['image_grappa'],
            hf['VarNet_recon'],
            hf['XPDNet_recon']
        ], axis=1)

    transform = MultichannelDataTransform(max_key='max', use_augment=False)

    output = []
    for image in images:
        image_input, _, _, mean, std, _, _ = transform(
            input=image,
            input_num=args.input_num,
            target=None,
            attrs=None,
            fname=None,
            slice=None,
        )

        image_input = image_input.to(device)[None, ...]

        output_slice = model(image_input).cpu()[0][0] * std + mean

        output.append(output_slice.cpu().tolist())

    return output


def save_file_recon(output, output_dir, fname):
    with h5py.File(output_dir / fname, "w") as hf:
        hf.create_dataset("reconstruction", data=output)


def unet_eval(args):
    device = torch.device(args.device)

    model = get_model(
        model_name=args.model_name,
        model_path=None,  # using default
    )

    model = model.to(device)
    model.eval()

    if args.model_type == 'Unet':
        forward_file = forward_file_unet
    elif args.model_type in ['ResUnet', 'MLPMixer']:
        forward_file = forward_file_multichannel

    with torch.no_grad():
        image_input_dir = args.data_dir / 'image'
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        i = 1
        tot = len(list(image_input_dir.iterdir()))

        for path in image_input_dir.iterdir():
            fname = path.name
            image_data_path = path

            print(f"[{i} / {tot}] Saving file {fname}")

            output = forward_file(args, model, device, image_data_path)
            if args.save_mode == 'reconstruction':
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
        default="Unet_finetune",
        type=str,
        choices=['Unet_finetune', 'ResUnet_with_stacking', 'test_mlpmixer'],
        help="Name of model"
    )
    parser.add_argument(
        "--model_type",
        default="Unet",
        type=str,
        choices=['Unet', 'ResUnet', 'MLPMixer'],
        help="Type of model"
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
    parser.add_argument(
        "--input_num",
        default=4,
        type=int,
        help="number of input layers"
    )

    args = parser.parse_args()

    unet_eval(args)