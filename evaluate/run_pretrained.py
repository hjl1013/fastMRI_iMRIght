import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri
from fastmri import evaluate
import Main.data.transforms as T
from Main.data.fastmri_data import SliceDataset
from Main.pl_modules.varnet_module import VarNetModule
from fastmri.models import VarNet

from fastmri.evaluate import ssim
import h5py
from Main.pl_modules.fastmri_data_module import FastMriDataModule

from evaluate.get_model import get_model
import matplotlib.pyplot as plt


def inference_step(model_name, model, batch, device, vol_info, outputs):
    if model_name == "VarNet_ours" or model_name == "VarNet_pretrained":
        crop_size = batch.crop_size
        output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()
        output = T.center_crop(output, crop_size)
    else:
        output = None

    for i, f in enumerate(batch.fname):
        if f not in vol_info:
            vol_info[f] = []
        if f not in outputs:
            outputs[f] = []

        vol_info[f].append(
            (
                output[i].cpu(),
                batch.masked_kspace[i].cpu(),
                batch.slice_num[i],
                batch.target[i].cpu(),
                batch.max_value[i],
            )
        )
        outputs[f].append(output[i].cpu().tolist())

    return vol_info, outputs


def calculate_loss(vol_info, vol_based=True):
    all_ssims, all_psnrs, all_nmses = [], [], []
    for vol_name, vol_data in vol_info.items():
        # slice_data is (output, masked_kspace, slice_num, target, max_value, prob_mask)
        output = torch.stack([slice_data[0] for slice_data in vol_data]).numpy()
        target = torch.stack([slice_data[3] for slice_data in vol_data]).numpy()

        # ----- Metrics calculation -----
        if vol_based:
            # Note that SSIMLoss computes average SSIM over the entire batch
            ssim = evaluate.ssim(target, output)
            psnr = evaluate.psnr(target, output)
            nmse = evaluate.nmse(target, output)
            all_ssims.append(ssim)
            all_psnrs.append(psnr)
            all_nmses.append(nmse)
        else:
            for gt, rec in zip(target, output):
                gt = gt[np.newaxis, :]
                rec = rec[np.newaxis, :]
                ssim = evaluate.ssim(gt, rec)
                psnr = evaluate.psnr(gt, rec)
                nmse = evaluate.nmse(gt, rec)
                all_ssims.append(ssim)
                all_psnrs.append(psnr)
                all_nmses.append(nmse)

    # --------------------------------------------------------------------------------
    # Aggregate everything
    # --------------------------------------------------------------------------------
    ssim_array = np.concatenate(np.array(all_ssims)[:, None], axis=0)
    psnr_array = np.concatenate(np.array(all_psnrs)[:, None], axis=0)
    nmse_array = np.concatenate(np.array(all_nmses)[:, None], axis=0)

    return_dict = {
        "ssim": ssim_array.mean().item(),
        "psnr": psnr_array.mean().item(),
        "nmse": nmse_array.mean().item(),
    }
    print(return_dict)


def run_inference(args):
    device = torch.device(args.device)

    # get model and corresponding data_module
    model, data_loader = get_model(
        model_name=args.model_name,
        model_path=args.state_dict_file,
        test_path=args.test_path,
        challenge=args.challenge,
    )

    model.to(device)
    model.eval()

    vol_info = {}
    outputs = {}

    torch.multiprocessing.set_sharing_strategy('file_system')
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference"):
            vol_info, outputs = inference_step(args.model_name, model, batch, args.device, vol_info, outputs)

        # evaluate and print
        calculate_loss(vol_info, args.vol_based)

        # save reconstruction
        fastmri.save_reconstructions(outputs, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="adaptive_equispaced_fraction",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        test_path="/root/input/val",  # path for test split, overwrites data_path
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--model_name",
        default="VarNet_pretrained",
        type=str,
        choices=['VarNet_pretrained', 'VarNet_ours'],
        help="Name of model"
    )
    parser.add_argument(
        "--model_file_name",
        default=None,
        type=str,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--vol_based",
        default=True,
        type=bool,
        help="Whether to do volume-based evaluation (otherwise slice-based).",
    )

    args = parser.parse_args()

    args.output_path = '/root/result' / Path(args.model_name) / 'reconstructions'
    if args.model_file_name is not None:
        args.state_dict_file = '/root/models' / Path(args.model_name) / args.model_file_name
    else:
        args.state_dict_file = None

    run_inference(args)