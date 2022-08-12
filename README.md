# fastMRI_iMRIght
SNU fastMRI competition

<hr>

## Evaluation
There are three modes in `run_pretrained.py`

1. calculate_loss: calculates SSIM
2. save_recon: saves the outputs in reconstruction format

Terminals to run these modes

calculate_loss:
```bash
python run_pretrained.py --model_name "VarNet_pretrained" --model_file_name "brain_leaderboard_state_dict.pt" --save_recon False --calculate_loss True
```

save_recon:
```bash
python run_pretrained.py --model_name "VarNet_pretrained" --model_file_name "brain_leaderboard_state_dict.pt" --save_recon True --calculate_loss False
```

## Train
To run unet train file `train.py`:
```bash
python train.py --net-name 'Unet_finetune' --input-type 'image' --data-path-train '/root/input_imtoim/train/image' --data-path-val '/root/input_imtoim/val/image' --input-key 'image_input' --pretrained-file-path '/root/result/Unet_finetune/checkpoints/model.pt'
```