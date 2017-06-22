'''
  This function is borrowed and modified from https://github.com/torch/demos/blob/master/train-a-digit-classifier/dataset-mnist.lua
  and from https://github.com/gcr/torch-residual-networks/blob/master/data/mnist-dataset.lua
'''

import torch
import math

def save_image(real_data, fake_data, filename):
    assert real_data.shape == fake_data.shape

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(fake_data[:,0], fake_data[:,1], color='red', label='noise (fake, sampled)')
    plt.scatter(real_data[:,0], real_data[:,1], color='blue', label='hidden (real, inferred)')
    #plt.axis('equal')
    plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.minorticks_on()
    plt.xlabel('x', fontsize=14, color='black')
    plt.ylabel('y', fontsize=14, color='black')
    plt.title('z samples (of first two dimensions)')
    plt.savefig(filename)
    plt.close()
