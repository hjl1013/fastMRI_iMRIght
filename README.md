# fastMRI_iMRIght
SNU fastMRI competition (Team iMRIght)

<hr>

0. Following codes should be executed inside the folder 'Code'
```bash
cd Code  
```

## Evaluation
1. Run kspace models on test(leaderboard) data, make image files which will be used as input image 
```bash
python varnet_eval.py xpdnet_eval.py --
```

2. Run image to image model on reconstructed test(leaderboard) data and save final reconstructed images
```bash
python imtoim_eval.py --
```

3. Run evaluation on final reconstructed images
```bash
python leaderboard_eval.py --
```

## Train
1. Make Image file used as input using kspace models
```bash
python varnet_eval.py xpdnet_eval.py --
```
2. Run NAFNet train file `train.py`:
```bash
python train.py --
```