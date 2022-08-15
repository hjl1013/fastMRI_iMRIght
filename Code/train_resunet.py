import argparse
import os, sys

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
sys.path.append('/root/fastMRI_hjl')

from utils.learning.train_part import resunet_train
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-w', '--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='ResUnet_with_stacking', help='Name of network')
    parser.add_argument('-i', '--input-type', type=str, default='image', help='Type of input', choices=['image', 'kspace'])
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input_imtoim_XPDNet_VarNet/train/image',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input_imtoim_XPDNet_VarNet/val/image',
                        help='Directory of validation data')
    parser.add_argument('-p', '--pretrained-file-path', type=str, default='/root/result/ResUnet_with_stacking/checkpoints/best_model_ep3_train0.03253_val0.0251.pt',
                        help='path of pretrained model to continue on training')
    parser.add_argument('-c', '--continue-training', type=str2bool, required=True,
                        help='whether to continue training or restart with the given model')

    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '/root/result' / args.net_name / 'checkpoints'
    args.val_dir = '/root/result' / args.net_name / 'reconstructions_val'
    args.main_dir = '/root/result' / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    resunet_train(args)
