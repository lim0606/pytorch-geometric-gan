#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs


################################
# exp4: experiment script for lsun
# geometric GAN (no regul)
# tuningC = 1
# lr = 0.0002
# rmsprop
################################
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda \
  --dataset lsun --dataroot data/lsun \
  --imageSize 64 \
  --lrD 0.0002 --lrG 0.0002 \
  --Giters 10 \
  --niter 5 --nsave 1 \
  --ndisplay 500 \
  --experiment samples/lsun_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/lsun_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC}_${filename}
