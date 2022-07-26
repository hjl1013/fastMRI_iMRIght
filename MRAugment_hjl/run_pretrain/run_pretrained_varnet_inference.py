"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri
import MRAugment_hjl.data.transforms as T
from MRAugment_hjl.data.fastmri_data import SliceDataset
from MRAugment_hjl.models.varnet import VarNet

from MRAugment_hjl.loss.evaluate import ssim
import h5py

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def run_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]

    return output, int(batch.slice_num[0]), batch.fname[0]


def run_inference(challenge, state_dict_file, data_path_kspace, data_path_image, output_path, device):
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    # download the state_dict if we don't have it
    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    # data loader setup
    data_transform = T.VarNetDataTransform()
    dataset = SliceDataset(
        root_kspace=data_path_kspace, root_image=data_path_image, transform=data_transform, challenge="multicoil"
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_varnet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
        f = h5py.File(str(data_path_image / fname.split('/')[-1]), "r")
        print(ssim(outputs[fname], np.asarray(f['image_label']), f.attrs['max']))

    fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="varnet_brain_mc",
        choices=(
            "varnet_knee_mc",
            "varnet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path_kspace",
        default="/root/input/kspace/multicoil_val",
        type=Path,
        help="Path to fastMRI data root for kspace",
    )
    parser.add_argument(
        "--data_path_image",
        default="/root/input/image/multicoil_val",
        type=Path,
        help="Path to fastMRI data root for image",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="/root/fastMRI/result/VarNet_pretrained",
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path_kspace,
        args.data_path_image,
        args.output_path,
        torch.device(args.device),
    )
