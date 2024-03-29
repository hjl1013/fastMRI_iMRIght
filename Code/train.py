import argparse
import os, sys

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
sys.path.append('/root/ksunw')

from utils.learning.train_part import varnet_train, imtoim_train
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

def parse(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('-bu', '--batch-update', type=int, default=64, help='num of batch to accumulate grad')
    parser.add_argument('-cl', '--clip', type=str2bool, default=False, help='whether to use gradient clipping')
    parser.add_argument('mn', '--max-norm', type=float, default=0.01, help='max norm for gradient clipping')
    parser.add_argument('-e', '--num-epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, required=True, help='Learning rate')
    parser.add_argument('-f', '--factor', type=float, default=0.1, help='factor for ReduceLROnPlateau')
    parser.add_argument('-w', '--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='Unet_finetune', required=True, help='Name of network')
    parser.add_argument('-i', '--input-type', type=str, default='image', help='Type of input', choices=['image', 'kspace'])
    parser.add_argument('-m', '--model-type', type=str, default='', required=True, help='type of model')
    parser.add_argument('-t', '--data-path-train', type=Path, required=True, default='/root/input_imtoim/train',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, required=True, default='/root/input_imtoim/val',
                        help='Directory of validation data')
    parser.add_argument('-te', '--data-path-test', type=Path, default='/root/input_imtoim/leaderboard',
                        help='Directory of test data')
    parser.add_argument('-p', '--pretrained-file-path', type=str, default=None,
                        help='path of pretrained model to continue on training')
    parser.add_argument('-c', '--continue-training', type=str2bool, default=True, required=True,
                        help='whether to continue training or restart with the given model')

    parser.add_argument('--cascade', type=int, default=3,
                        help='Number of cascades | Should be less than 12')  ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', required=True, help='Name of input key')
    parser.add_argument('--input-num', type=int, default=1, required=True, help='number of input channels')
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

    if args.model_type == 'Varnet':
        varnet_train(args)
    elif args.model_type in ['Unet', 'ResUnet', 'MLPMixer', 'NAFNet']:
        imtoim_train(args)