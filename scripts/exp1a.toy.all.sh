#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs

tuningC=1



################################
# experiment script for toy data 4
################################



################################
### no regularization 
################################

# exp1: GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard gan --practice \
  --cuda --dataset toy4 --dataroot '' \
  --adam --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_gan_toy4_adam_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_gan_toy4_adam_lr001_${filename}


# exp2: geometric GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_${filename}


# exp3: WGAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard wgan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_wgan_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_wgan_toy4_rmsprop_lr001_${filename}


# exp4: mean matching GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard meangan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_meangan_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_meangan_toy4_rmsprop_lr001_${filename}






################################
### weight clipping 
################################

# exp1: GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard gan --practice \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wclip \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_gan_wclip_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_gan_wclip_toy4_rmsprop_lr001_${filename}


# exp2: geometric GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wclip \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclip_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclip_toy4_rmsprop_lr001_c${tuningC}_${filename}


# exp3: WGAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard wgan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wclip \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_wgan_wclip_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_wgan_wclip_toy4_rmsprop_lr001_${filename}


# exp4: mean matching GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard meangan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wclip \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_meangan_wclip_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_meangan_wclip_toy4_rmsprop_lr001_${filename}






################################
### weight projection 
################################

# exp1: GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard gan --practice \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wproj \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_gan_wproj_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_gan_wproj_toy4_rmsprop_lr001_${filename}


# exp2: geometric GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wproj \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wproj_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wproj_toy4_rmsprop_lr001_c${tuningC}_${filename}


# exp3: WGAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard wgan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wproj \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_wgan_wproj_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_wgan_wproj_toy4_rmsprop_lr001_${filename}


# exp4: mean matching GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard meangan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --wproj \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_meangan_wproj_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_meangan_wproj_toy4_rmsprop_lr001_${filename}






################################
### weight decay 
################################

# exp1: GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard gan --practice \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --weight_decay_D 0.001 --weight_decay_G 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_gan_wdecay_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_gan_wdecay_toy4_rmsprop_lr001_${filename}


# exp2: geometric GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --weight_decay_D 0.001 --weight_decay_G 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_${filename}


# exp3: WGAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard wgan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --weight_decay_D 0.001 --weight_decay_G 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_wgan_wdecay_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_wgan_wdecay_toy4_rmsprop_lr001_${filename}


# exp4: mean matching GAN
export CUDA_VISIBLE_DEVICES=0
python main.py standard meangan \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --weight_decay_D 0.001 --weight_decay_G 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_meangan_wdecay_toy4_rmsprop_lr001 \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_meangan_wdecay_toy4_rmsprop_lr001_${filename}
