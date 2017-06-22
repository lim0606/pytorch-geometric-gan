#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs




################################
# experiment script for toy data 4
################################

tuningC=0.00001

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename













tuningC=0.0001

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename













tuningC=0.001

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename












tuningC=0.01

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename











tuningC=0.1

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename










tuningC=1

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename















tuningC=10

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename















tuningC=100

################################
### no regularization 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight clipping 
################################

export CUDA_VISIBLE_DEVICES=0
python main.py standard geogan --C ${tuningC} \
  --cuda --dataset toy4 --dataroot '' \
  --lrD 0.001 --lrG 0.001 \
  --clamp \
  --nc 2 --nz 4 --ngf 128 --ndf 128 \
  --model_G toy4 --model_D toy4 \
  --batchSize 500 \
  --experiment samples/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC} \
  --niter 500 --ndisplay 100 --nsave 50 \
  2>&1 | tee logs/toy4_geogan_wclamp_toy4_rmsprop_lr001_c${tuningC}_$filename


################################
### weight decay 
################################

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
  2>&1 | tee logs/toy4_geogan_wdecay_toy4_rmsprop_lr001_c${tuningC}_$filename

