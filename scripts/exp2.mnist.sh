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

# input dim = 3 x 64 x 64 = 12288
# feature dim at \phi(x) = 8 x ndf x 4 x 4


# original/default DCGAN model
# ndf, ngf = 64: feature dim 8192 < input dim 12288 
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 \
  --lrD 0.0002 --lrG 0.0002 \
  --model_G dcgan --model_D dcgan \
  --Giters 10 \
  --niter 25 --ndisplay 100 --nsave 5 \
  --experiment samples/mnist_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/mnist_geogan_dcgan_rmsprop_lr0002_kg10_c${tuningC}_${filename}

# ndf, ngf = 96: feature dim 12288 == input dim 12288
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 \
  --lrD 0.0002 --lrG 0.0002 \
  --model_G dcgan --model_D dcgan \
  --ndf 96 --ngf 96 \
  --Giters 10 \
  --niter 25 --ndisplay 100 --nsave 5 \
  --experiment samples/mnist_geogan_dcgan96_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/mnist_geogan_dcgan96_rmsprop_lr0002_kg10_c${tuningC}_${filename}

# ndf, ngf = 128: feature dim 16384 > input dim 12288
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 \
  --lrD 0.0002 --lrG 0.0002 \
  --model_G dcgan --model_D dcgan \
  --ndf 128 --ngf 128 \
  --Giters 10 \
  --niter 25 --ndisplay 100 --nsave 5 \
  --experiment samples/mnist_geogan_dcgan128_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/mnist_geogan_dcgan128_rmsprop_lr0002_kg10_c${tuningC}_${filename}

# ndf, ngf = 192: feature dim 24576 > input dim 12288
tuningC=1
export CUDA_VISIBLE_DEVICES=1
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 \
  --lrD 0.0002 --lrG 0.0002 \
  --model_G dcgan --model_D dcgan \
  --ndf 192 --ngf 192 \
  --Giters 10 \
  --niter 25 --ndisplay 100 --nsave 5 \
  --experiment samples/mnist_geogan_dcgan192_rmsprop_lr0002_kg10_c${tuningC} \
  2>&1 | tee logs/mnist_geogan_dcgan192_rmsprop_lr0002_kg10_c${tuningC}_${filename}
