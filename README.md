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