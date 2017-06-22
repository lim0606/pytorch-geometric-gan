'''
  This function is borrowed and modified from https://github.com/torch/demos/blob/master/train-a-digit-classifier/dataset-mnist.lua
  and from https://github.com/gcr/torch-residual-networks/blob/master/data/mnist-dataset.lua
'''

import torch
import torch.nn as nn
import math
from scipy.stats import multivariate_normal
import numpy as np
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
 
# data generating function
# exp1: mixture of 4 gaussians
def exp1(num_data=1000):
    if num_data % 4 != 0:
        raise ValueError('num_data should be multiple of 4. num_data = {}'.format(num_data)) 

    center = 8
    sigma = 1 #math.sqrt(3)

    # init data 
    d1x = torch.FloatTensor(num_data/4, 1)
    d1y = torch.FloatTensor(num_data/4, 1)
    d1x.normal_(center, sigma * 3)
    d1y.normal_(center, sigma * 1)

    d2x = torch.FloatTensor(num_data/4, 1)
    d2y = torch.FloatTensor(num_data/4, 1)
    d2x.normal_(-center, sigma * 1)
    d2y.normal_(center,  sigma * 3)

    d3x = torch.FloatTensor(num_data/4, 1)
    d3y = torch.FloatTensor(num_data/4, 1)
    d3x.normal_(center,  sigma * 3)
    d3y.normal_(-center, sigma * 2)

    d4x = torch.FloatTensor(num_data/4, 1)
    d4y = torch.FloatTensor(num_data/4, 1)
    d4x.normal_(-center, sigma * 2)
    d4y.normal_(-center, sigma * 2)

    d1 = torch.cat((d1x, d1y), 1)
    d2 = torch.cat((d2x, d2y), 1)
    d3 = torch.cat((d3x, d3y), 1)
    d4 = torch.cat((d4x, d4y), 1)

    d = torch.cat((d1, d2, d3, d4), 0) 

    # label
    label = torch.IntTensor(num_data).zero_()
    for i in range(4):
        label[i*(num_data/4):(i+1)*(num_data/4)] = i

    # shuffle
    #shuffle = torch.randperm(d.size()[0])
    #d = torch.index_select(d, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)

    # pdf
    rv1 = multivariate_normal([ center,  center], [[math.pow(sigma * 3, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])
    rv2 = multivariate_normal([-center,  center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 3, 2)]])
    rv3 = multivariate_normal([ center, -center], [[math.pow(sigma * 3, 2), 0.0], [0.0, math.pow(sigma * 2, 2)]])
    rv4 = multivariate_normal([-center, -center], [[math.pow(sigma * 2, 2), 0.0], [0.0, math.pow(sigma * 2, 2)]])

    def pdf(x):
        prob = 0.25 * rv1.pdf(x) + 0.25 * rv2.pdf(x) + 0.25 * rv3.pdf(x) + 0.25 * rv4.pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return d, label, sumloglikelihood 


# exp2: two spirals
def exp2(num_data=1000):
    '''
      This function is borrowed from http://stackoverflow.com/questions/16146599/create-artificial-data-in-matlab
    '''

    degrees = 450 #570
    start = 90
    #noise = 0 #0.2
    deg2rad = (2*math.pi)/360
    radius = 1.8
    start = start * deg2rad;
  
    N_mixtures = 100
    N  = 2 * N_mixtures
    N1 = N_mixtures #math.floor(N/2)
    N2 = N_mixtures #N-N1
    if num_data % N_mixtures != 0:
        raise ValueError('num_data should be multiple of {} (num_data = {})'.format(2*N_mixtures, num_data))
  
    n = (start + 
         torch.sqrt(torch.linspace(0.075,1,N2).view(N2,1)).mul_(degrees)
        ).mul_(deg2rad)
    mu1 = torch.cat((torch.mul(-torch.cos(n), n).mul_(radius),
                     torch.mul(torch.sin(n), n).mul_(radius)), 1)
  
    n = (start + 
         torch.sqrt(torch.linspace(0.075,1,N1).view(N1,1)).mul_(degrees)
        ).mul_(deg2rad)
    mu2 = torch.cat((torch.mul(torch.cos(n), n).mul_(radius),
                     torch.mul(-torch.sin(n), n).mul_(radius)), 1)
  
    mu = torch.cat((mu1, mu2), 0)
    num_data_per_mixture = num_data / (2*N_mixtures)
    sigma = math.sqrt(0.6)
    x = torch.zeros(num_data, 2)
    for i in range(2*N_mixtures):
        xx = x[i*num_data_per_mixture:(i+1)*num_data_per_mixture, :]
        xx.copy_(torch.cat(
                 (torch.FloatTensor(num_data_per_mixture).normal_(mu[i,0], sigma).view(num_data_per_mixture, 1),
                  torch.FloatTensor(num_data_per_mixture).normal_(mu[i,1], sigma).view(num_data_per_mixture, 1)), 1))
 
    # label
    label = torch.IntTensor(num_data).zero_()
    label[0:num_data/2] = 0
    label[num_data/2:] = 1
 
    # shuffle
    #shuffle = torch.randperm(x.size()[0])
    #x = torch.index_select(x, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)
 
    # pdf
    rv_list = []
    for i in range(2 * N_mixtures):
        rv = multivariate_normal([mu[i,0], mu[i,1]], [[math.pow(sigma, 2), 0.0], [0.0, math.pow(sigma, 2)]])
        rv_list.append(rv)

    def pdf(x):
        prob = 1 / (2*N_mixtures) * rv_list[0].pdf(x)
        for i in range(1, 2 * N_mixtures):
            prob += (1.0 / float(2*N_mixtures)) * rv_list[i].pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))
 
    return x, label, sumloglikelihood 

# exp3: mixture of 2 gaussians with high bias
def exp3(num_data=1000):
    if num_data < 2:
        raise ValueError('num_data should be larger than 2. (num_data = {})'.format(num_data))

    center = 6.2 
    sigma = 1 #math.sqrt(3)

    n1 = int(round(num_data * 0.9))
    n2 = num_data - n1

    # init data 
    d1x = torch.FloatTensor(n1, 1)
    d1y = torch.FloatTensor(n1, 1)
    d1x.normal_(center, sigma * 5)
    d1y.normal_(center, sigma * 5)

    d2x = torch.FloatTensor(n2, 1)
    d2y = torch.FloatTensor(n2, 1)
    d2x.normal_(-center, sigma * 1)
    d2y.normal_(-center, sigma * 1)

    d1 = torch.cat((d1x, d1y), 1)
    d2 = torch.cat((d2x, d2y), 1)

    d = torch.cat((d1, d2), 0) 

    # label
    label = torch.IntTensor(num_data).zero_()
    label[0:n1] = 0 
    label[n1:] = 1 

    # shuffle
    #shuffle = torch.randperm(d.size()[0])
    #d = torch.index_select(d, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)

    # pdf
    rv1 = multivariate_normal([ center,  center], [[math.pow(sigma * 5, 2), 0.0], [0.0, math.pow(sigma * 5, 2)]])
    rv2 = multivariate_normal([-center, -center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])

    def pdf(x):
        prob = (float(n1) / float(num_data)) * rv1.pdf(x) + (float(n2) / float(num_data)) * rv2.pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return d, label, sumloglikelihood

# exp4: grid shapes 
def exp4(num_data=1000):

    var = 0.1
    max_x = 21
    max_y = 21
    min_x = -max_x
    min_y = -max_y
    n = 5 

    # init
    nx, ny = (n, n)
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(x, y)
    N  = xv.size 
    if num_data % N != 0:
        raise ValueError('num_data should be multiple of {} (num_data = {})'.format(N, num_data))

    # data and label  
    mu = np.concatenate((xv.reshape(N,1), yv.reshape(N,1)), axis=1)
    mu = torch.FloatTensor(mu)
    num_data_per_mixture = num_data / N
    sigma = math.sqrt(var)
    x = torch.zeros(num_data, 2)
    label = torch.IntTensor(num_data).zero_()
    for i in range(N):
        xx = x[i*num_data_per_mixture:(i+1)*num_data_per_mixture, :]
        xx.copy_(torch.cat(
                 (torch.FloatTensor(num_data_per_mixture).normal_(mu[i,0], sigma).view(num_data_per_mixture, 1),
                  torch.FloatTensor(num_data_per_mixture).normal_(mu[i,1], sigma).view(num_data_per_mixture, 1)), 1))
        label[i*num_data_per_mixture:(i+1)*num_data_per_mixture] = i 
 
    # shuffle
    #shuffle = torch.randperm(x.size()[0])
    #x = torch.index_select(x, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)
 
    # pdf
    rv_list = []
    for i in range(N):
        rv = multivariate_normal([mu[i,0], mu[i,1]], [[math.pow(sigma, 2), 0.0], [0.0, math.pow(sigma, 2)]])
        rv_list.append(rv)

    def pdf(x):
        prob = 1 / (N) * rv_list[0].pdf(x)
        for i in range(1, N):
            prob += (1.0 / float(N)) * rv_list[i].pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return x, label, sumloglikelihood

# exp5: mixture of 2 gaussians with high bias
def exp5(num_data=1000):
    if num_data < 2:
        raise ValueError('num_data should be larger than 2. (num_data = {})'.format(num_data))

    center = -5
    sigma_x = 0.5 
    sigma_y = 7

    n1 = num_data 

    # init data 
    d1x = torch.FloatTensor(n1, 1)
    d1y = torch.FloatTensor(n1, 1)
    d1x.normal_(center, sigma_x)
    d1y.normal_(center, sigma_y)

    d1 = torch.cat((d1x, d1y), 1)

    d = d1 

    # label
    label = torch.IntTensor(num_data).zero_()
    label[:] = 0 

    # shuffle
    #shuffle = torch.randperm(d.size()[0])
    #d = torch.index_select(d, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)

    # pdf
    rv1 = multivariate_normal([ center,  center], [[math.pow(sigma_x, 2), 0.0], [0.0, math.pow(sigma_y, 2)]])

    def pdf(x):
        prob = (float(n1) / float(num_data)) * rv1.pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return d, label, sumloglikelihood

# exp6: mixture of 2 gaussians with high bias
def exp6(num_data=1000):
    if num_data < 2:
        raise ValueError('num_data should be larger than 2. (num_data = {})'.format(num_data))

    center = -5 
    sigma_x = 7 
    sigma_y = 7 

    n1 = num_data 

    # init data 
    d1x = torch.FloatTensor(n1, 1)
    d1y = torch.FloatTensor(n1, 1)
    d1x.normal_(center, sigma_x)
    d1y.normal_(center, sigma_y)

    d1 = torch.cat((d1x, d1y), 1)

    d = d1 

    # label
    label = torch.IntTensor(num_data).zero_()
    label[:] = 0 

    # shuffle
    #shuffle = torch.randperm(d.size()[0])
    #d = torch.index_select(d, 0, shuffle)
    #label = torch.index_select(label, 0, shuffle)

    # pdf
    rv1 = multivariate_normal([ center,  center], [[math.pow(sigma_x, 2), 0.0], [0.0, math.pow(sigma_y, 2)]])

    def pdf(x):
        prob = (float(n1) / float(num_data)) * rv1.pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return d, label, sumloglikelihood


def exp(exp_num='toy1', num_data=1000):
    if exp_num == 'toy1':
        return exp1(num_data)
    elif exp_num == 'toy2':
        return exp2(num_data)
    elif exp_num == 'toy3':
        return exp3(num_data)
    elif exp_num == 'toy4':
        return exp4(num_data)
    elif exp_num == 'toy5':
        return exp5(num_data)
    elif exp_num == 'toy6':
        return exp6(num_data)
    else:
        raise ValueError('unknown experiment {}'.format(exp_num))

def save_image_fake(fake_data, filename):
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    #import numpy as np
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    #plt.scatter(real_data[:,0], real_data[:,1], color='blue', label='real')
    plt.scatter(fake_data[:,0], fake_data[:,1], color='red', label='fake')
    plt.axis('equal')
    #plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
    plt.grid(True)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.minorticks_on()
    plt.xlabel('x', fontsize=14, color='black')
    plt.ylabel('y', fontsize=14, color='black')
    #plt.title('Toy dataset')
    plt.savefig(filename)
    plt.close()

def save_image_real(real_data, filename):
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    #import numpy as np
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(real_data[:,0], real_data[:,1], color='blue', label='real')
    #plt.scatter(fake_data[:,0], fake_data[:,1], color='red', label='fake')
    plt.axis('equal')
    #plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
    plt.grid(True)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.minorticks_on()
    plt.xlabel('x', fontsize=14, color='black')
    plt.ylabel('y', fontsize=14, color='black')
    #plt.title('Toy dataset')
    plt.savefig(filename)
    plt.close()

def save_image(real_data, fake_data, filename):
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    #import numpy as np
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(real_data[:,0], real_data[:,1], color='blue', label='real')
    plt.scatter(fake_data[:,0], fake_data[:,1], color='red', label='fake')
    #plt.axis('equal')
    plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
    plt.grid(True)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.minorticks_on()
    plt.xlabel('x', fontsize=14, color='black')
    plt.ylabel('y', fontsize=14, color='black')
    plt.title('Toy dataset')
    plt.savefig(filename)
    plt.close()

def save_contour(netD, filename, cuda=False):
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    #import numpy as np
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.cm as cm
    #import matplotlib.mlab as mlab
    #import matplotlib.pyplot as plt
    
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid' 

    # gen grid 
    delta = 0.1
    x = np.arange(-25.0, 25.0, delta)
    y = np.arange(-25.0, 25.0, delta)
    X, Y = np.meshgrid(x, y)

    # convert numpy array to to torch variable
    (h, w) = X.shape
    XY = np.concatenate((X.reshape((h*w, 1, 1, 1)), Y.reshape((h*w, 1, 1, 1))), axis=1)
    input = torch.Tensor(XY)
    input = Variable(input)
    if cuda:
        input = input.cuda()

    # forward
    output = netD(input)

    # convert torch variable to numpy array
    Z = output.data.cpu().view(-1).numpy().reshape(h, w)

    # plot and save 
    plt.figure()
    CS1 = plt.contourf(X, Y, Z)
    CS2 = plt.contour(X, Y, Z, alpha=.7, colors='k')
    plt.clabel(CS2, inline=1, fontsize=10, colors='k')
    plt.title('Simplest default with labels')
    plt.savefig(filename)
    plt.close()


'''
### test
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

num_data = 10000
exp_name = 'exp6'

if exp_name == 'exp1':
    data, label, sumloglikelihood = exp1(num_data)
elif exp_name == 'exp2':
    data, label, sumloglikelihood = exp2(num_data)
elif exp_name == 'exp3':
    data, label, sumloglikelihood = exp3(num_data)
elif exp_name == 'exp4':
    data, label, sumloglikelihood = exp4(num_data)
elif exp_name == 'exp5':
    data, label, sumloglikelihood = exp5(num_data)
elif exp_name == 'exp6':
    data, label, sumloglikelihood = exp6(num_data)
else:
    raise ValueError('known exp: {}'.format(exp_name))
data = data.numpy()
label = label.numpy()
colors = ['red','purple','green','blue']
#print(data)
#print(data.shape)
#print(label)
#print(label.shape)

fig, ax = plt.subplots()
#plt.scatter(data[:,0], data[:,1], c=label, alpha=0.01, label=exp_name, cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(data[:,0], data[:,1], c=label, alpha=0.1, label=exp_name, cmap=matplotlib.colors.ListedColormap(colors))
plt.axis('equal')
plt.minorticks_on()
plt.grid(True)
plt.xlabel('x', fontsize=14, color='black')
plt.ylabel('y', fontsize=14, color='black')
plt.title('Toy dataset')
plt.savefig('toy.png')
'''
