import shutil
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet
from utils.model.unet import Unet
from core.res_unet_plus import ResUnetPlusPlus
from networks import Img2Img_Mixer, ReconNet
from basicsr.models.archs.NAFNet_arch import NAFNet

##################### Load and Save ########################

def save_model(args, exp_dir, epoch, model, optimizer, scheduler, train_ssim, best_val_ssim, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_ssim': best_val_ssim,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', \
                        exp_dir / f'best_model_ep{epoch}_train{train_ssim:.4g}_val{best_val_ssim:.4g}.pt')


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

##################### kspace - Varnet ########################

def varnet_train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in tqdm(enumerate(data_loader)):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)  # default SSIM
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch

def varnet_validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    val_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return val_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start

def varnet_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade)
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)

    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs, eta_min=1e-6
    )
    best_val_loss = 1.
    start_epoch = 0

    if args.pretrained_file_path is not None:
        checkpoint = torch.load(args.pretrained_file_path)
        model.load_state_dict(checkpoint['model'])

        if args.continue_training is True:
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])


    train_loader = create_data_loaders(data_path=args.data_path_train, args=args)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print(f"learning rate: {optimizer.param_groups[0]['lr']:.5f}")

        train_loss, train_time = varnet_train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        scheduler.step()
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = varnet_validate(args, model, val_loader)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, train_loss, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

##################### IMtoIM ########################

def imtoim_validate(args, model, data_loader, loss_type):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, fnames, slices, _ = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mean = mean.cuda(non_blocking=True)
            std = std.cuda(non_blocking=True)
            output = model(input)

            std = std[:, None, None, None]
            mean = mean[:, None, None, None]
            output = output * std + mean
            target = target * std + mean

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i][0].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i][0].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    val_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return val_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def imtoim_train_epoch(args, epoch, model, data_loader, optimizer, scaler, iters_to_accumulate, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    pbar = tqdm(data_loader, desc=f"Training epoch{epoch}", bar_format='{l_bar}{bar:80}{r_bar}')

    for iter, data in enumerate(pbar):
        with autocast(enabled=False):
            input, target, mean, std, _, _, maximum = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            mean = mean.cuda(non_blocking=True)
            std = std.cuda(non_blocking=True)
            std = std[:, None, None, None]
            mean = mean[:, None, None, None]
            target = target * std + mean
            output = model(input)
            output = output * std + mean
            #print(maximum)
            loss = loss_type(output, target, maximum)  # default SSIM loss
            #print(loss)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            total_loss += loss.item()
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (iter + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # print('shape: output_{}, target_{}, std_{}, mean_{}'.format(output.shape, target.shape, std.shape, mean.shape))

        # if iter % (args.report_interval // args.batch_size) == 0:
        #     print(
        #         f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
        #         f'Iter = [{iter:4d}/{len(data_loader):4d}] '
        #         f'Loss = {loss.item():.4g} '
        #         f'Time = {time.perf_counter() - start_iter:.4f}s',
        #     )
        #     start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def imtoim_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # stack 4 input images 'image_input' 'image_grappa' 'XPDNet_recon' 'VarNet_recon'
    if args.model_type == 'Unet':
        model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=3, drop_prob=0.0)
    if args.model_type == 'ResUnet':
        model = ResUnetPlusPlus(channel=args.input_num)
    if args.model_type == 'MLPMi`xer':
        net = Img2Img_Mixer(
            img_size=320,
            img_channels=1,
            patch_size=4,
            embed_dim=128,
            num_layers=16,
            f_hidden=8,
        )
        model = ReconNet(net)
    if args.model_type == 'NAFNet':
        img_channel = args.input_num
        width = 32

        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]

        #enc_blks = [1, 1, 1, 28]
        #middle_blk_num = 1
        #dec_blks = [1, 1, 1, 1]
        model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    summary(model, input_size=(1, args.input_num, 384, 384))
    model = model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, args.num_epochs, eta_min=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.factor, patience=2, threshold=0.0001
    )
    scaler = GradScaler(enabled=False)

    best_val_ssim = 1.
    start_epoch = 0

    if args.pretrained_file_path is not None:
        print('load pretrained model')
        checkpoint = torch.load(args.pretrained_file_path)
        model.load_state_dict(checkpoint['model'])

        if args.continue_training:
            best_val_ssim = checkpoint['best_val_ssim']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args = checkpoint['args']

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, use_augment=True)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, use_augment=False)
    iters_to_accumulate = args.batch_update / args.batch_size

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print(f"learning rate: {optimizer.param_groups[0]['lr']:.5f}")

        train_loss, train_time = \
            imtoim_train_epoch(args, epoch, model, train_loader, optimizer, scaler, iters_to_accumulate,loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = \
            imtoim_validate(args, model, val_loader, loss_type)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_ssim
        best_val_ssim = min(best_val_ssim, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, train_loss, best_val_ssim, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

        scheduler.step(val_loss)