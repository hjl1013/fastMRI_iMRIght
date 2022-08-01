from pathlib import Path
import torch
import tqdm
import requests

from fastmri.models import VarNet
from Main.pl_modules.varnet_module import VarNetModule
from Main.pl_modules.fastmri_data_module import FastMriDataModule
from Main.data.transforms import VarNetDataTransform
from utils.data.load_data import create_data_loaders


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


def get_model(model_name: str, model_path: Path, test_path: Path, challenge: str):
    """
    input: model name path
    output: model and data_module
    """
    model_dict = {}
    model_dict["VarNet_ours"] = "/root/models/VarNet_ours/epoch31-ssim0.9470.ckpt"
    model_dict["VarNet_pretrained"] = "/root/models/VarNet_pretrained/brain_leaderboard_state_dict.pt"

    print(
        f"Trying to load model {model_name}..."
    )

    if model_path is not None:
        assert model_path.exists(), f"file {model_path} doesn't exist"
        model_path = str(model_path)
    else:
        print("Using default model")
        model_path = model_dict[model_name]

    if model_name == 'VarNet_ours':
        model = VarNetModule.load_from_checkpoint(model_path)

        test_transform = VarNetDataTransform()
        data_module = FastMriDataModule(
            data_path=None,
            challenge=challenge,
            train_transform=None,
            val_transform=None,
            test_transform=test_transform,
            test_path=test_path,
            sample_rate=None,
            volume_sample_rate=1.0,
            batch_size=1,
            num_workers=4,
        )
        data_loader = data_module.test_dataloader()

    elif model_name == 'VarNet_pretrained':
        model = VarNet(num_cascades=12)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained)

        data_loader = create_data_loaders(data_path=test_path, args=None, isforward=True)

    return model, data_loader