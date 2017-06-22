Geometric GAN
===============

Code accompanying the paper ["Geometric GAN"](https://arxiv.org/abs/1705.02894). \
(Ths code is modified from https://github.com/martinarjovsky/WassersteinGAN)


[Prerequisites](#prerequisites) \
[Datasets](#datasets) \
[Reproducing Experiments](#reproducing-experiments) \
[Generated Samples](#generated-samples) \
[Plot Losses](#plot-losses) 


## Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

## Datasets
### MNIST

Make empty folder at `<PATH>/<TO>/<MNIST>`.

Set symbolic link as follows;
```
mkdir data
ln -s <PATH>/<TO>/<MNIST> data/mnist
```

Note: you can leave the folder empty since `torchvision` will automatically download mnist dataset. 

### CelebA

Download Align&Cropped Images of CelebA dataset, i.e. `img_align_celeba.zip`, from https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg at `<PATH>/<TO>/<CelebA>`.

```
unzip img_align_celeba.zip
```

Then you have, 

```
<PATH>/<TO>/<CelebA>
├── img_align_celeba.zip
└── img_align_celeba
```

Set symbolic link as follows;
```
mkdir data
ln -s <PATH>/<TO>/<CelebA> data/celeba
```

### LSUN

Download LSUN bedroom dataset using https://github.com/fyu/lsun at `<PATH>/<TO>/<LSUN>`.

```
unzip bedroom_train_lmdb.zip
```

Then you have, 

```
<PATH>/<TO>/<LSUN>
├── bedroom_train_lmdb.zip
├── bedroom_train_lmdb
...
```

Set symbolic link as follows;
```
mkdir data
ln -s <PATH>/<TO>/<LSUN> data/lsun
```

## Reproducing Experiments
### Exp1: Mixture of Gaussian
```
python main.py standard geogan --cuda --dataset toy4 --dataroot '' --lrD 0.001 --lrG 0.001 --nc 2 --nz 4 --ngf 128 --ndf 128 --model_G toy4 --model_D toy4 --batchSize 500 --experiment samples/toy4_geogan_toy4_rmsprop_lr001_c1 --niter 500 --ndisplay 100 --nsave 50
```

or execute following scripts in the directory of this repo.
``` 
./scripts/exp1a.toy.all.sh
./scripts/exp1b.toy.diffC.sh
```

### Exp2: MNIST
```
python main.py standard geogan --cuda --dataset mnist --dataroot data/mnist --imageSize 64 --nc 1 --lrD 0.0002 --lrG 0.0002 --model_G dcgan --model_D dcgan --ndf 128 --ngf 128 --Giters 10 --niter 25 --ndisplay 100 --nsave 5 --experiment samples/mnist_geogan_dcgan128_rmsprop_lr0002_kg10_c1
```

or execute following scripts in the directory of this repo.
``` 
./scripts/exp2.mnist.sh
```

### Exp3: CelebA
```
python main.py standard geogan --cuda --dataset folder --dataroot data/celeba --loadSize 96 --imageSize 64 --lrD 0.0002 --lrG 0.0002 --model_G dcgan --model_D dcgan --ndf 128 --ngf 128 --Giters 10 --niter 50 --ndisplay 500 --nsave 5 --experiment samples/celeba_geogan_dcgan128_rmsprop_lr0002_kg10_c1
```

or execute following scripts in the directory of this repo.
``` 
./scripts/exp3.celeba.sh
```

### Exp4: LSUN
```
python main.py standard geogan --cuda --dataset lsun --dataroot data/lsun --imageSize 64 --lrD 0.0002 --lrG 0.0002 --model_G dcgan --model_D dcgan --ndf 128 --ngf 128 --Giters 10 --niter 5 --nsave 1 --ndisplay 500 --experiment samples/lsun_geogan_dcgan128_rmsprop_lr0002_kg10_c1
```

or execute following scripts in the directory of this repo.
``` 
./scripts/exp4.lsun.sh
```


## Generated Samples
Generated samples will be in the `samples` folder.


## Plot Losses
Logs will be in the `logs` folder (if you use the aforementioned scripts).

Use `plot_log.py`, and the example usages of it are in `scripts/plot.example.sh`
