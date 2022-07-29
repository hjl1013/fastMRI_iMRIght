import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from pathlib import Path
from utils.learning.train_part import download_model

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # model = VarNet(num_cascades=args.cascade, pools=4, chans=18, sens_pools=4, sens_chans=8)
    # model.to(device=device)

    model = VarNet(num_cascades=12)
    model.to(device=device)

    # checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)

    # checkpoint = torch.load(args.exp_dir, map_location='cpu')
    # print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(pretrained)

    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)