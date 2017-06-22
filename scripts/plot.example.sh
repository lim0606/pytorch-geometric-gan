#!/bin/bash

mkdir cache


# mnist
filename=logs/mnist_geogan_dcgan_rmsprop_lr0002_kg10_c1_20170622-153736-506145308.log
cond=_kg10_c1
dataset=mnist
model=dcgan
label=geogan
name=${dataset}_${label}_${model}${cond}
rm cache/${name}*.pkl
grep -r "Loss_D_fake" ${filename} > cache/${name}.log
python plot_log.py cache/${name} --data ${label} cache/${name}.log
#eog cache/${name}_disc_medfilt_loss.png


# celeba 
filename=logs/celeba_geogan_dcgan_rmsprop_lr0002_kg10_c1_20170622-185423-782338693.log
cond=_kg10_c1
dataset=celeba
model=dcgan
label=geogan
name=${dataset}_${label}_${model}${cond}
rm cache/${name}*.pkl
grep -r "Loss_D_fake" ${filename} > cache/${name}.log
python plot_log.py cache/${name} --data ${label} cache/${name}.log
#eog cache/${name}_disc_medfilt_loss.png


# lsun 
filename=logs/lsun_geogan_dcgan_rmsprop_lr0002_kg10_c1_20170622-185428-412962928.log
cond=_kg10_c1
dataset=lsun
model=dcgan
label=geogan
name=${dataset}_${label}_${model}${cond}
rm cache/${name}*.pkl
grep -r "Loss_D_fake" ${filename} > cache/${name}.log
python plot_log.py cache/${name} --data ${label} cache/${name}.log
#eog cache/${name}_disc_medfilt_loss.png


# plot all
python plot_log.py cache/all_geogan_highdim_dcgan_kg10_c1 \
  --data "geogan-mnist"  cache/mnist_geogan_dcgan_kg10_c1.log \
  --data "geogan-celeba" cache/celeba_geogan_dcgan_kg10_c1.log \
  --data "geogan-lsun"   cache/lsun_geogan_dcgan_kg10_c1.log
eog cache/all_geogan_highdim_dcgan_kg10_c1_disc_medfilt_loss.png
