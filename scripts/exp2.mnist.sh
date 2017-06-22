#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs

################################
# exp2: experiment script for mnist 
# geometric GAN (no regul)
# tuningC = 1
# lr = 0.0002
# rmsprop
################################
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 \
  --lrD 0.0002 --lrG 0.0002 \
  --model_G dcgan --model_D dcgan \
  --Giters 10 \
  --experiment samples/mnist_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC} \
  --niter 25 --ndisplay 100 --nsave 5 \
  2>&1 | tee logs/mnist_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC}_${filename}
