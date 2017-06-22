#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs


################################
# exp3: experiment script for celeba
# geometric GAN (no regul)
# tuningC = 1
# lr = 0.0002
# rmsprop 
################################
tuningC=1
export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset folder \
  --dataroot data/celeba \
  --lrD 0.0002 --lrG 0.0002 \
  --loadSize 96 --imageSize 64 \
  --Giters 10 \
  --niter 50 --ndisplay 500 --nsave 5 \
  --experiment samples/celeba_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/celeba_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC}_${filename}
