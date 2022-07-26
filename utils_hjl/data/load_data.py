import h5py
import random
from utils_hjl.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root_kspace, root_image, transform, input_mode, input_key, target_key, forward=False): #TODO
        self.transform = transform
        self.input_mode = input_mode
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []

        #TODO
        files_kspace = sorted(list(Path(root_kspace).iterdir()))
        files_image = sorted(list(Path(root_image).iterdir()))

        for i in range(len(files_image)):
            if self.input_mode == 'image':
                num_slices = self._get_metadata(files_image[i])
            elif self.input_mode == 'kspace':
                num_slices = self._get_metadata(files_kspace[i])

            self.examples += [
                (files_image[i], files_kspace[i], slice_ind) for slice_ind in range(num_slices)
            ]

        # for fname in sorted(files):
        #     num_slices = self._get_metadata(fname)
        #
        #     self.examples += [
        #         (fname, slice_ind) for slice_ind in range(num_slices)
        #     ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # fname, dataslice = self.examples[i]
        fname_image, fname_kspace, dataslice = self.examples[i]

        if self.input_mode == 'image':
            with h5py.File(fname_image, "r") as hf: #kspace로 바꿔주기 TODO
                input = hf[self.input_key][dataslice]
                mask = None
        elif self.input_mode == 'kspace':
            with h5py.File(fname_kspace, "r") as hf: #kspace로 바꿔주기 TODO
                input = hf[self.input_key][dataslice] * hf['mask']
                mask = np.array((hf['mask']))
        else:
            assert True, 'invalid input mode'

        with h5py.File(fname_image, "r") as hf:
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)

        # with h5py.File(fname_image, "r") as hf: #kspace로 바꿔주기 TODO
        #     input = hf[self.input_key][dataslice]
        #     if self.forward:
        #         target = -1
        #     else:
        #         target = hf[self.target_key][dataslice]
        #     attrs = dict(hf.attrs)
        # return self.transform(input, target, attrs, fname.name, dataslice)
        return self.transform(input, mask, target, attrs, fname_image.name, fname_kspace.name, dataslice)


def create_data_loaders(data_path_kspace, data_path_image, args, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData( #TODO
        root_kspace=data_path_kspace,
        root_image=data_path_image,
        transform=DataTransform(isforward, max_key_, args.input_mode),
        input_mode=args.input_mode,
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        num_workers=args.batch_size
    )
    return data_loader
