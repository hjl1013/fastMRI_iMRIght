import h5py
import argparse
from pathlib import Path

def combine_xpdnet_varnet_data(args):
    args.output_path.mkdir(exist_ok=True, parents=True)

    tot = len(list(args.data_path_xpdnet.iterdir()))
    i = 1
    for fpath in args.data_path_xpdnet.iterdir():
        print(f"[{i} / {tot}] Saving file {fpath.name}")
        fname = fpath.name
        xpdnet_fpath = fpath
        varnet_fpath = args.data_path_varnet / fname
        output_fpath = args.output_path / fname

        with h5py.File(str(xpdnet_fpath), "r") as hf_x, h5py.File(str(varnet_fpath), "r") as hf_v, h5py.File(str(output_fpath), "w") as hf_o:
            for key in hf_x:
                hf_o.create_dataset(key, data=hf_x[key])
            for key in hf_x.attrs:
                hf_o.attrs[key] = hf_x.attrs[key]
            hf_o.create_dataset("VarNet_recon", data=hf_v["VarNet_recon"])

        print(f"Successfully saved {fpath.name}")
        print()

        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path_xpdnet',
        type=Path,
        required=True,
        help='data path to xpdnet reconstruction images'
    )
    parser.add_argument(
        '--data_path_varnet',
        type=Path,
        required=True,
        help='data path to varnet reconstruction images'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='output path to save combined data'
    )

    args = parser.parse_args()

    combine_xpdnet_varnet_data(args)