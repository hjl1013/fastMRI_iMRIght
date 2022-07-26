import argparse
import shutil
from utils_hjl.learning.train_part import train
from pathlib import Path


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='VarNet', help='Name of network; Unet or VarNet')
    parser.add_argument('-tk', '--data-path-train-kspace', type=Path, default='/root/input/kspace/multicoil_train', help='Directory of train data in kspace') #TODO
    parser.add_argument('-ti', '--data-path-train-image', type=Path, default='/root/input/image/multicoil_train/',help='Directory of train data in image')
    parser.add_argument('-vk', '--data-path-val-kspace', type=Path, default='/root/input/kspace/multicoil_val', help='Directory of validation data in kspace')
    parser.add_argument('-vi', '--data-path-val-image', type=Path, default='/root/input/image/multicoil_val',help='Directory of validation data in image')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key') #TODO
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    parser.add_argument('-im', '--input-mode', type=str, default='kspace', help='image or kspace')

    parser.set_defaults(
        num_cascades=2,  # number of unrolled iterations
        pools=2,  # number of pooling layers for U-Net
        chans=2,  # number of top-level channels for U-Net
        sens_pools=2,  # number of pooling layers for sense est. U-Net
        sens_chans=2,  # number of top-level channels for sense est. U-Net
        lr=0.0003,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
