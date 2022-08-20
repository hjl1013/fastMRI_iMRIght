# fastMRI_iMRIght
SNU fastMRI competition (Team iMRIght)

<hr>

0. Following codes should be executed inside the folder 'Code'
```bash
cd Code  
```
+ make sure to modify the path inside sys.path.append() to your project's location
(else it will default to `'/root/fastMRI'`)
```python
sys.path.append('PATH_TO_YOUR_PROJECT')
```
## Evaluation
1. Run kspace models on test(leaderboard) data, make image files which will be used as input image 
```bash
python varnet_eval.py --output_dir /root/eval_varnet/leaderboard/image --data_dir /root/input/leaderboard \
&& python xpdnet_eval.py --output_dir /root/eval_xpdnet/leaderboard/image --data_dir /root/input/leaderboard \
&& python combine_varnet_xpdnet.py --data_path_varnet /root/eval_varnet/leaderboard/image --data_path_xpdnet /root/eval_xpdnet/leaderboard/image --output_dir /root/eval/leaderboard/image
```

2. Run image to image model on reconstructed test(leaderboard) data and save final reconstructed images
```bash
python imtoim_eval.py --output_dir /root/eval_final/reconstructions --data_dir /root/eval/leaderboard/image
```

3. Run evaluation on final reconstructed images
```bash
python leaderboard_eval.py --your_data_path /root/eval_final/reconstructions
```

## Train
1. Make Image file used as input using kspace models
```bash
python varnet_eval.py --output_dir /root/input_varnet/train/image --data_dir /root/input/train \
&& python varnet_eval.py --output_dir /root/input_varnet/val/image --data_dir /root/input/val \
&& python xpdnet_eval.py --output_dir /root/input_xpdnet/train/image --data_dir /root/input/train \
&& python xpdnet_eval.py --output_dir /root/input_varnet/val/image --data_dir /root/input/val \
&& python combine_varnet_xpdnet.py --data_path_varnet /root/input_varnet/train/image --data_path_xpdnet /root/input_xpdnet/train/image --output_dir /root/input_varnet_xpdnet/train/image \
&& python combine_varnet_xpdnet.py --data_path_varnet /root/input_varnet/val/image --data_path_xpdnet /root/input_xpdnet/val/image --output_dir /root/input_varnet_xpdnet/val/image
```

2. Run NAFNet train file `train.py`:
```bash
python train.py --lr 0.001 --factor 0.3 --net-name 'NAFNet_stacking_lr0.001' --model-type 'NAFNet' -t /root/input_imtoim_XPDNet_VarNet/train/image -v /root/input_varnet_xpdnet/leaderboard/image -c False --input-key 'image_input' --batch-size 2 --batch-update 32 --clip True --input-num 4
```