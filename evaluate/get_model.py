from pathlib import Path
import torch
import sys
sys.path.append('/root/fastMRI_hjl')

from fastmri.models import VarNet
from Main.pl_modules.varnet_module import VarNetModule
from utils.model.unet import Unet
from core.res_unet_plus import ResUnetPlusPlus


def get_model(model_name: str, model_path: Path):
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
        "Unet_finetune": "/root/result/Unet_finetune/checkpoints/model.pt",
        "ResUnet_with_stacking": "/root/result/ResUnet_with_stacking/checkpoints/model.pt",
        "test_varnet": "/root/result/test_varnet/checkpoints/best_model_ep40_train0.03726_val0.02616.pt",
        "test_mlpmixer": "/root/result/test_mlpmixer/checkpoints/model.pt",
    }

    if model_path is not None:
        assert model_path.exists(), f"file {model_path} doesn't exist"
        model_path = str(model_path)
    else:
        print("Using default model")
        model_path = model_dict[model_name]


    # prepare model of model_name
    if model_name == 'VarNet_ours':
        model = VarNetModule.load_from_checkpoint(model_path)

    elif model_name == 'VarNet_pretrained':
        model = VarNet(num_cascades=12)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained)

    elif model_name == 'VarNet_SNU' or model_name == 'test_varnet':
        model = VarNet(num_cascades=3)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['model'])

    elif model_name == 'test_unet' or model_name == 'Unet_finetune':

        model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=3, drop_prob=0.0)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['model'])

    elif model_name == 'ResUnet_with_stacking':

        model = ResUnetPlusPlus(channel=4)

        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['model'])

    return model