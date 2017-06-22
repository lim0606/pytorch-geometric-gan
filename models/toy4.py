from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class MLP_G(nn.Module):
    #def __init__(self, isize=1, nz=4, nc=2, ngf=128, ngpu):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        #gpu_ids = None
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #    gpu_ids = range(self.ngpu)
        #out = nn.parallel.data_parallel(self.main, input, gpu_ids)
        out = self.main(input)
        return out.view(out.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    #def __init__(self, isize=1, nz=4, nc=1, ndf=128, ngpu):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        cls = nn.Linear(ndf, 1)
        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
        )
        main.add_module('cls', cls)

        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

        # assign cls flag
        for p in self.main.parameters():
            p.cls        = False
            p.cls_weight = False
            p.cls_bias   = False
        for p in cls.parameters():
            p.cls = True
            if p.data.ndimension() == cls.weight.data.ndimension():
                p.cls_weight = True
            else:
                p.cls_bias   = True

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        #gpu_ids = None
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #    gpu_ids = range(self.ngpu)
        #return nn.parallel.data_parallel(self.main, input, gpu_ids)
        return self.main(input)
