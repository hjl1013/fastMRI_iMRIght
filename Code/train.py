import argparse
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
sys.path.append('/root/ksunw')

from utils.learning.train_part import varnet_train, unet_train
from pathlib import Path


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=3*1e-4, help='Learning rate')
    parser.add_argument('-w', '--weight-decay', type=float, default=5*1e-7, help='weight decay')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input/val', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=3, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    if str(args.net_name) == 'test_varnet':
        varnet_train(args)
    elif str(args.net_name) == 'test_unet':
        unet_train(args)
