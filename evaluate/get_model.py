from pathlib import Path
import torch
import tensorflow as tf

from fastmri.models import VarNet
from Main.pl_modules.varnet_module import VarNetModule
from Main.pl_modules.fastmri_data_module import FastMriDataModule
from Main.data.transforms import VarNetDataTransform
from utils.data.load_data import create_data_loaders, create_data_loaders_for_pretrained
from utils.model.unet import Unet


def get_model(model_name: str, model_path: Path, test_path: Path, challenge: str = "multicoil"):
    """
    input: model name path
    output: model and data loader
    """

    print(
        f"Trying to load model {model_name}..."
    )

    # preprocessing
    model_dict = {
        "VarNet_ours": "/root/models/VarNet_ours/epoch31-ssim0.9470.ckpt",
        "VarNet_pretrained": "/root/models/VarNet_pretrained/brain_leaderboard_state_dict.pt",
        "VarNet_SNU": "/root/models/VarNet_SNU/best_model.pt",
        "XPDNet_pretrained": "/root/models/XPDNet_pretrained/model_weights.h5",
        "test_unet": "/root/result/test_unet/checkpoints/best_model_ep2_train0.03348_val0.001803.pt",
        "Unet_finetune": "/root/result/Unet_finetune/checkpoints/model.pt"
    }

    test_path_dict = {
        "VarNet_ours": Path("/root/leaderboard_recon/kspace"),
        "VarNet_pretrained": Path("/root/input/leaderboard"),
        "VarNet_SNU": Path("/root/input/leaderboard"),
        "XPDNet_pretrained": Path("/root/input/leaderboard"),
        "test_unet": Path("/root/result/VarNet_pretrained/reconstructions"),
        "Unet_finetune": Path("/root/result/VarNet_pretrained/reconstructions")
    }

    if model_path is not None:
        assert model_path.exists(), f"file {model_path} doesn't exist"
        model_path = str(model_path)
    else:
        print("Using default model")
        model_path = model_dict[model_name]

    if test_path is not None:
        assert test_path.exists(), f"file {test_path} doesn't exist"
    else:
        print("Using default test path")
        test_path = test_path_dict[model_name]

    # prepare model and data loaders of model_name
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

        data_loader = create_data_loaders_for_pretrained(data_path=test_path, isforward=False, for_pretrained=True)

    elif model_name == 'VarNet_SNU':
        model = VarNet(num_cascades=3)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['model'])

        data_loader = create_data_loaders_for_pretrained(data_path=test_path, model_name=model_name)

    elif model_name == 'test_unet' or model_name == 'Unet_finetune':

        model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=3, drop_prob=0.0)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['model'])

        data_loader = create_data_loaders_for_pretrained(data_path=test_path, model_name=model_name)

    return model, data_loader