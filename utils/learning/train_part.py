import shutil
import numpy as np
import torch
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


def unet_train_epoch(args, epoch, model, data_loader, optimizer, loss_type, ssim_func):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    total_ssim = 0.

    for iter, data in tqdm(enumerate(data_loader)):
        image, target, mean, std, _, _, maximum = data
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mean = mean.cuda(non_blocking=True)
        std = std.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(image)
        std = std[:, None, None, None]
        mean = mean[:, None, None, None]
        output = output * std + mean
        target = target * std + mean
        loss = loss_type(output, target, maximum)  # default l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # print('shape: output_{}, target_{}, std_{}, mean_{}'.format(output.shape, target.shape, std.shape, mean.shape))
        ssim = ssim_func(output, target, maximum)
        total_ssim += ssim.item()

        if iter % (args.report_interval // args.batch_size) == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'SSIM = {ssim.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    total_ssim = total_ssim / len_loader
    return total_loss, total_ssim, time.perf_counter() - start_epoch


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


def unet_validate(args, model, data_loader, loss_type, ssim_func):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    total_loss = 0.
    total_ssim = 0.
    len_loader = len(data_loader)

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            image, target, mean, std, fnames, slices, maximum = data
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mean = mean.cuda(non_blocking=True)
            std = std.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            output = model(image)
            # loss = loss_type(output, target)
            # total_loss += loss.item()

            std = std[:, None, None, None]
            mean = mean[:, None, None, None]
            output = output * std + mean
            target = target * std + mean
            loss = loss_type(output, target, maximum)
            total_loss += loss.item()
            ssim = ssim_func(output, target, maximum)
            total_ssim += ssim.item()

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
    num_subjects = len(reconstructions)
    val_loss = total_loss/len_loader
    val_ssim = total_ssim/len_loader
    return val_loss, val_ssim, num_subjects, reconstructions, targets, None, time.perf_counter() - start


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


def varnet_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade)
    model.to(device=device)
    '''
    FOLDER = "/root/result/test_varnet/checkpoints/"
    url_root = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    # using pretrained parameter
    # MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    # using finetuned parameter
    MODEL_FNAMES = "best_model_ep11_0.028276184172645012.pt"
    
    if not Path(FOLDER + MODEL_FNAMES).exists():
        print('no such pretrained model')
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
        pretrained = torch.load(FOLDER + MODEL_FNAMES)
        pretrained_copy = copy.deepcopy(pretrained)
        for layer in pretrained_copy.keys():
            if layer.split('.', 2)[1].isdigit() and (args.cascade <= int(layer.split('.', 2)[1]) <= 11):
                del pretrained[layer]
        model.load_state_dict(pretrained)
    else:
        pretrained = torch.load(FOLDER + MODEL_FNAMES)
        if 'leaderboard' in MODEL_FNAMES:
            pretrained_copy = copy.deepcopy(pretrained)
            for layer in pretrained_copy.keys():
                if layer.split('.', 2)[1].isdigit() and (args.cascade <= int(layer.split('.', 2)[1]) <= 11):
                    del pretrained[layer]
            model.load_state_dict(pretrained)
        else:
            model.load_state_dict(pretrained['model'])
    '''
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


def unet_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=3, drop_prob=0.0)
    model.to(device=device)

    # FOLDER = "/root/models/Unet_pretrained/"
    # url_root = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
    # # using pretrained parameter
    # MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    # # using finetuned parameter
    # # MODEL_FNAMES = "best_model_1.pt"
    #
    # if not Path(FOLDER + MODEL_FNAMES).exists():
    #     print('no such pretrained model')
    #     download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
    #     os.replace('./'+MODEL_FNAMES, FOLDER+MODEL_FNAMES)
    #
    # pretrained = torch.load(FOLDER + MODEL_FNAMES)
    # if 'leaderboard' in MODEL_FNAMES:
    #     pretrained_copy = copy.deepcopy(pretrained)
    #     for layer in pretrained_copy.keys():
    #         split = layer.split('.')
    #         if split[0] == 'down_sample_layers' and split[1] == 3:
    #             del pretrained[layer]
    #         if split[1]!='0' and (split[0] == 'up_conv' or split[0] == 'up_transpose_conv'):
    #             split[1] = str(int(split[1]) - 1)
    #             pretrained['.'.join(split)] = pretrained_copy[layer]
    #         elif split[0] == 'conv':
    #             del pretrained[layer]
    #
    #     pretrained_copy = copy.deepcopy(pretrained)
    #     for layer in pretrained_copy.keys():
    #         if not layer in model.state_dict().keys():
    #             del pretrained[layer]
    #
    #     # print(pretrained.keys())
    #     #
    #     # for layer in model.state_dict():
    #     #     print(layer)
    #     #     print(model.state_dict()[layer].dtype)
    #     #     print()
    #     #
    #     # re = model.load_state_dict(pretrained, strict=False)
    #     #
    #     # print(re)
    # else:
    #     model.load_state_dict(pretrained['model'])

    # summary(model, (1, 1, 384, 384))
    # x = Variable(torch.randn(1, 1, 384, 384, device=device))
    # make_dot(model(x), params=dict(model.named_parameters()))#.render("graph", format="png")

    loss_type = SSIMLoss().to(device=device)
    ssim_func = SSIMLoss().to(device=device)

    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs, eta_min=1e-6
    )
    best_val_ssim = 1.
    start_epoch = 0

    if args.pretrained_file_path is not None:
        print('load pretrained model')
        checkpoint = torch.load(args.pretrained_file_path)
        model.load_state_dict(checkpoint['model'])

        if args.continue_training is True:
            best_val_ssim = checkpoint['best_val_ssim']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_ssim, train_time = unet_train_epoch(args, epoch, model, train_loader, optimizer, loss_type, ssim_func)
        scheduler.step()
        val_loss, val_ssim, num_subjects, reconstructions, targets, inputs, val_time = \
            unet_validate(args, model, val_loader, loss_type, ssim_func)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        train_ssim = torch.tensor(train_ssim).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        val_ssim = torch.tensor(val_ssim).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        is_new_best = val_ssim < best_val_ssim
        best_val_ssim = min(best_val_ssim, val_ssim)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, train_ssim, best_val_ssim, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} TrainSSIM = {train_ssim:.4g}'
            f'ValLoss = {val_loss:.4g} ValSSIM = {val_ssim:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )


################## ResUnet ########################
def resunet_validate(args, model, data_loader, loss_type):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, mean, std, fnames, slices = data
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


def resunet_train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    pbar = tqdm(data_loader)
    for iter, data in enumerate(pbar):
        image, target, maximum, mean, std, _, _ = data
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        mean = mean.cuda(non_blocking=True)
        std = std.cuda(non_blocking=True)

        output = model(image)
        std = std[:, None, None, None]
        mean = mean[:, None, None, None]
        output = output * std + mean
        target = target * std + mean
        loss = loss_type(output, target, maximum)  # default SSIM loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_description(f"Training loss = {loss.item():4f}")

        # print('shape: output_{}, target_{}, std_{}, mean_{}'.format(output.shape, target.shape, std.shape, mean.shape))
        '''
        if iter % (args.report_interval // args.batch_size) == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
        '''
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def resunet_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = ResUnetPlusPlus(channel=4) # stack 4 input images 'image_input' 'image_grappa' 'XPDNet_recon' 'VarNet_recon'
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, args.num_epochs, eta_min=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, threshold=0.0001
    )
    best_val_ssim = 1.
    start_epoch = 0

    if args.pretrained_file_path is not None:
        print('load pretrained model')
        checkpoint = torch.load(args.pretrained_file_path)
        model.load_state_dict(checkpoint['model'])

        if args.continue_training is True:
            best_val_ssim = checkpoint['best_val_ssim']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, use_augment=True)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, use_augment=False)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        train_loss, train_time = resunet_train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = \
            resunet_validate(args, model, val_loader, loss_type)

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


##################### mlpmixer ########################

def mlpmixer_validate(args, model, data_loader, loss_type):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, mean, std, fnames, slices = data
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


def mlpmixer_train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    pbar = tqdm(data_loader, desc=f"Training epoch{epoch}", bar_format='{l_bar}{bar:80}{r_bar}')

    for iter, data in enumerate(pbar):
        image, target, mean, std, _, _, maximum = data
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        mean = mean.cuda(non_blocking=True)
        std = std.cuda(non_blocking=True)

        output = model(image)
        std = std[:, None, None, None]
        mean = mean[:, None, None, None]
        output = output * std + mean
        target = target * std + mean
        loss = loss_type(output, target, maximum)  # default SSIM loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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


def mlpmixer_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    net = Img2Img_Mixer(
        img_size=320,
        img_channels=1,
        patch_size=4,
        embed_dim=128,
        num_layers=16,
        f_hidden=8,
    )    # stack 4 input images 'image_input' 'image_grappa' 'XPDNet_recon' 'VarNet_recon'
    model = ReconNet(net).to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, args.num_epochs, eta_min=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, threshold=0.0001
    )

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

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print(f"learning rate: {optimizer.param_groups[0]['lr']:.4f}")

        train_loss, train_time = mlpmixer_train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = \
            mlpmixer_validate(args, model, val_loader, loss_type)

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