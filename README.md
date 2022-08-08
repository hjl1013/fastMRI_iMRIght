# fastMRI_iMRIght
SNU fastMRI competition

<hr>

## Evaluation
There are three modes in `run_pretrained.py`

1. calculate_loss: calculates SSIM
2. save_recon: saves the outputs in reconstruction format
3. save_imtoim_input: save the model into imtoim input format

Terminals to run these modes

calculate_loss:
```bash
python run_pretrained.py --model_name "VarNet_pretrained" --model_file_name "brain_leaderboard_state_dict.pt" --save_recon False --save_imtoim_input False --calculate_loss True
```

save_recon:
```bash
python run_pretrained.py --model_name "VarNet_pretrained" --model_file_name "brain_leaderboard_state_dict.pt" --save_recon True --save_imtoim_input False --calculate_loss False
```

save_imtoim_input:
```bash
python run_pretrained.py --model_name "VarNet_pretrained" --model_file_name "brain_leaderboard_state_dict.pt" --save_recon False --save_imtoim_input True --calculate_loss False --imtoim_input_path "/root/input_imtoim/train" --test_path "/root/input/train"
```