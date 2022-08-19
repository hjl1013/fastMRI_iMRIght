import h5py
import random

from utils.data.transforms import VarnetDataTransform, UnetDataTransform, MultichannelDataTransform, ADLDataTransform,\
    MultichannelDataTransform_with_cutmix
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


class kspace_SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


class image_SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, target, attrs, fname.name, dataslice)


class MultichannelSliceData(Dataset):
    def __init__(self, root, transform, input_key, input_num, target_key):
        self.transform = transform
        self.input_key = input_key
        self.input_num = input_num
        self.target_key = target_key
        self.examples = []

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input = np.array([
                hf['image_input'][dataslice],
                hf['image_grappa'][dataslice],
                hf['VarNet_recon'][dataslice],
                hf['XPDNet_recon'][dataslice]
            ])
            target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, self.input_num, target, attrs, fname.name, dataslice)


def create_data_loaders(data_path, args, use_augment=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    if args.model_type == 'Varnet':
        data_storage = kspace_SliceData(
            root=data_path,
            transform=VarnetDataTransform(max_key_),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward,
        )
    elif args.model_type == 'Unet':
        data_storage = image_SliceData(
            root=data_path,
            transform=UnetDataTransform(isforward, max_key_),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward
        )
    elif args.model_type in ['ResUnet', 'MLPMixer', 'NAFNet']:
        data_storage = MultichannelSliceData(
            root=data_path,
            transform=MultichannelDataTransform(max_key=max_key_, use_augment=use_augment),
            input_key=args.input_key,
            input_num=args.input_num,
            target_key=target_key_,
        )
    elif args.model_type == 'ADL':
        data_storage = MultichannelSliceData(
            root=data_path,
            transform=ADLDataTransform(max_key=max_key_, use_augment=use_augment),
            input_key=args.input_key,
            input_num=args.input_num,
            target_key=target_key_,
        )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size
    )
    return data_loader

def create_data_loaders_for_imtoim_cutmix(data_path, args, use_augment=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = MultichannelSliceData(
        root=data_path,
        transform=MultichannelDataTransform_with_cutmix(max_key=max_key_, use_augment=use_augment),
        input_key=args.input_key,
        input_num=args.input_num,
        target_key=target_key_,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size
    )

    return data_loader


def create_data_loaders_for_imtoim_cutmix_validation(data_path, args, use_augment=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = MultichannelSliceData(
        root=data_path,
        transform=MultichannelDataTransform(max_key=max_key_, use_augment=use_augment),
        input_key=args.input_key,
        input_num=args.input_num,
        target_key=target_key_,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=2
    )

    return data_loader