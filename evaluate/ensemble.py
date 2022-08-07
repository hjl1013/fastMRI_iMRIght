from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import h5py
import numpy as np
import argparse

import fastmri


def ensemble_step(models: Dict[str, Path], fname: str, ensemble_type: str = "average"):
    """
    one step of ensembling images

    :param models: dictionary of model name and data path
    :param fname: file name to ensemble
    :param ensemble_type: ensemble strategy to ensemble
    :return output: ensembled result
    """

    output = None

    if ensemble_type == "average":
        for model_name, folder in models.items():
            if (folder / fname).exists():
                with h5py.File(str(folder / fname), "r") as hf:
                    if output is None:
                        output = np.array(hf['reconstruction'])
                    else:
                        output += np.array(hf['reconstruction'])
            else:
                print(f"failed to ensemble {fname}, {folder} has no file named {fname}")
                output = None
                break

    return output

def ensemble(args):
    """
    ensemble images and save their results
    """

    # print introduction
    print("list of models to ensemble:")
    for model, _ in args.models.items():
        print(model)
    print()

    # get list of file names to ensemble
    file_names = ["brain_test" + str(i) + ".h5" for i in range(1, 59, 1)]

    # ensembling
    outputs = {}
    for fname in file_names:
        print(f"Ensembling file {fname}...")

        output = ensemble_step(models=args.models, fname=fname, ensemble_type=args.ensemble_type)

        if output is not None:
            print(f"successfully ensembled {fname}")
            output = output / len(args.models)
            outputs[fname] = output.tolist()

        print()

    fastmri.save_reconstructions(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ensemble_type",
        default="average",
        type=str,
        choices=["average"],
        help="ensemble strategy to use"
    )

    args = parser.parse_args()

    args.models={
        "VarNet_pretrained": Path("/root/result/VarNet_pretrained/reconstructions"),
        "VarNet_SNU": Path("/root/result/VarNet_SNU/reconstructions"),
    }
    output_path = "/root/result/ensemble"
    for models, _ in args.models.items():
        output_path += "_" + models
    args.output_path = Path(output_path) / "reconstructions"

    ensemble(args)