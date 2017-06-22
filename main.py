from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import time
from scipy.stats import multivariate_normal
import numpy as np

import models.dcgan as dcgan
import models.mlp as mlp
import models.toy as toy 
import models.toy4 as toy4 
import losses.SumLoss as sumloss
import losses.HingeLoss as hingeloss
import losses.LeakyHingeLoss as leakyhingeloss
import losses.BCELoss as bceloss
import utils.plot as plt


parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | toy1~toy4')
parent_parser.add_argument('--dataroot', required=True, help='path to dataset')
parent_parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parent_parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parent_parser.add_argument('--loadSize',  type=int, default=64, help='the height / width of the input image (it will be croppred)')
parent_parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parent_parser.add_argument('--nc', type=int, default=3,   help='number of channels in input (image)')
parent_parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parent_parser.add_argument('--ngf', type=int, default=64)
parent_parser.add_argument('--ndf', type=int, default=64)
parent_parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parent_parser.add_argument('--nsave', type=int, default=1,  help='number of epochs to save models')
parent_parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parent_parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parent_parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parent_parser.add_argument('--weight_decay_D', type=float, default=0, help='weight_decay for discriminator. default=0')
parent_parser.add_argument('--weight_decay_G', type=float, default=0, help='weight_decay for generator. default=0')
parent_parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parent_parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parent_parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parent_parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parent_parser.add_argument('--Diters', type=int, default=1, help='number of D iters per loop')
parent_parser.add_argument('--Giters', type=int, default=1, help='number of G iters per loop')
parent_parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parent_parser.add_argument('--model_G', default='dcgan', help='model for G: dcgan | mlp | toy')
parent_parser.add_argument('--model_D', default='dcgan', help='model for D: dcgan | mlp | toy')
parent_parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parent_parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parent_parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

# arguments for weight clipping
parent_parser.add_argument('--wclip_lower', type=float, default=-0.01)
parent_parser.add_argument('--wclip_upper', type=float, default=0.01)
wclip_parser = parent_parser.add_mutually_exclusive_group(required=False)
wclip_parser.add_argument('--wclip', dest='wclip', action='store_true', help='flag for wclip. for wgan, it is required.')
wclip_parser.add_argument('--no-wclip', dest='wclip', action='store_false', help='flag for wclip. for wgan, it is required.')
parent_parser.set_defaults(wclip=False)

# arguments for weight projection 
parent_parser.add_argument('--wproj_upper', type=float, default=1.0)
wproj_parser = parent_parser.add_mutually_exclusive_group(required=False)
wproj_parser.add_argument('--wproj', dest='wproj', action='store_true', help='flag for wproj. for wgan, it is required.')
wproj_parser.add_argument('--no-wproj', dest='wproj', action='store_false', help='flag for wproj. for wgan, it is required.')
parent_parser.set_defaults(wproj=False)

# display setting
display_parser = parent_parser.add_mutually_exclusive_group(required=False)
display_parser.add_argument('--display', dest='display', action='store_true', help='flag for display. for toy1~toy4, it should be off.')
display_parser.add_argument('--no-display', dest='display', action='store_false', help='flag for display. for toy1~toy4, it should be off.')
parent_parser.set_defaults(display=True)
parent_parser.add_argument('--ndisplay', type=int, default=500,  help='number of epochs to display samples')

# arguments for training criterion
def add_criterion(mode_parser, parent_parser):
  criterion_subparser = mode_parser.add_subparsers(title='criterion method: gan | wgan | geogan',
                                                dest='criterion')

  # wgan
  wgan_parser = criterion_subparser.add_parser('wgan', help='train using WGAN',
                                    parents=[parent_parser])

  # meangan
  meangan_parser = criterion_subparser.add_parser('meangan', help='train using mean matching GAN',
                                    parents=[parent_parser])

  # geogan
  geogan_parser = criterion_subparser.add_parser('geogan', help='train using geoGAN',
                                    parents=[parent_parser])
  geogan_parser.add_argument('--C', type=float, default=1, help='tuning parapmeter C in 0.5 * ||w||^2 + C * hinge_loss(x)')
  geogan_parser.add_argument('--margin', type=float, default=1, help='margin size in max(0, m - c * x), hinge loss, for generator loss')
  gtrain_parser = geogan_parser.add_mutually_exclusive_group()
  gtrain_parser.add_argument('--theory', action='store_const', dest='gtrain', const='theory',
                             help='For D, real_label = 1, fake_label = -1, and minimize svm primal loss. For G, fake_label = -1, and move perpendicular to hyperplane')
  gtrain_parser.add_argument('--leaky', action='store_const', dest='gtrain', const='leaky',
                             help='For D, real_label = 1, fake_label = -1, and minimize svm primal loss. For G, fake_label = 1, and minize leaky svm primal loss with flipped labels.')
  geogan_parser.set_defaults(gtrain='theory')

  # ebgan
  ebgan_parser = criterion_subparser.add_parser('ebgan', help='train using EBGAN',
                                    parents=[parent_parser])
  ebgan_parser.add_argument('--margin', type=float, default=1, help='slack margin constant in discriminator loss for fake data.')

  # gan
  gan_parser = criterion_subparser.add_parser('gan', help='train using GAN',
                                    parents=[parent_parser])
  gtrain_parser = gan_parser.add_mutually_exclusive_group()
  gtrain_parser.add_argument('--theory', action='store_const', dest='gtrain', const='theory',
                             help='real_label = 1, fake_label = 0; thus, for D, min_D E_data[-log(D(x)] + E_gen[-log(1-D(G(z)))]. for G, min_G E_gen[log(1-D(G(z)))]')
  gtrain_parser.add_argument('--practice', action='store_const', dest='gtrain', const='practice',
                             help='for D, min_D E_data[-log(D(x)] + E_gen[-log(1-D(G(z)))]. for G, min_G E_gen[-log(D(G(z)))]')
  gtrain_parser.add_argument('--flip', action='store_const', dest='gtrain', const='flip',
                             help='real_label = 0, fake_label = 1.')
  gan_parser.set_defaults(gtrain='practice')

# main parser and training mode
main_parser = argparse.ArgumentParser()
mode_subparsers = main_parser.add_subparsers(title='training mode: standard | bigan | ali',
                                             dest='mode')
mode_standard_parser = mode_subparsers.add_parser('standard', help='train as standard implicit modeling')
add_criterion(mode_standard_parser, parent_parser)
#mode_bigan_parser = mode_subparsers.add_parser('bigan', help='train as BiGAN')
#add_criterion(mode_bigan_parser, parent_parser)
#mode_ali_parser = mode_subparsers.add_parser('ali', help='train as ALI')
#add_criterion(mode_ali_parser, parent_parser)

# parse arguments
opt = main_parser.parse_args()
print(opt)

# generate cache folder
os.system('mkdir samples')
if opt.experiment is None:
    opt.experiment = 'samples/experiment'
os.system('mkdir -p {0}'.format(opt.experiment))

# set random seed
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# apply cudnn option
cudnn.benchmark = True

# diagnose cuda option
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# load dataset
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.loadSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.loadSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Scale(opt.imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
elif 'toy' in opt.dataset: #opt.dataset in ['toy1', 'toy2', 'toy3', 'toy4', 'toy5', 'toy6']:
    if opt.nc != 2:
        raise ValueError('nc should be 2 for simulated dataset. (opt.nc = {})'.format(opt.nc))
    import datasets.toy as tdset
    num_data = 100000
    data_tensor, target_tensor, x_sumloglikelihood = tdset.exp(opt.dataset, num_data)
    data_tensor = data_tensor.view(num_data, 2, 1, 1).contiguous()
    dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# init model parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = opt.nc 
n_extra_layers = int(opt.n_extra_layers)

# custum function for weight project in l2-norm unit ball
def weight_proj_l2norm(param):
    norm = torch.norm(param.data, p=2) + 1e-8
    coeff = min(opt.wproj_upper, 1.0/norm)
    param.data.mul_(coeff)

# custom weights initialization called on netG and netD
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

def weights_init_toy(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

# model initializaton: genterator
if opt.model_G == 'dcgan':
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    netG.apply(weights_init_dcgan)
elif opt.model_G == 'mlp':
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    netG.apply(weights_init_mlp)
elif opt.model_G == 'toy':
    netG = toy.MLP_G(1, nz, 2, ngf, ngpu)
    netG.apply(weights_init_toy)
elif opt.model_G == 'toy4':
    netG = toy4.MLP_G(1, nz, 2, ngf, ngpu)
    netG.apply(weights_init_toy)
else:
    raise ValueError('unkown model: {}'.format(opt.model_G))

if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# model initializaton: discriminator
if opt.model_D == 'dcgan':
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init_dcgan)
elif opt.model_D == 'mlp':
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    netD.apply(weights_init_mlp)
elif opt.model_D == 'toy':
    netD = toy.MLP_D(1, nz, 2, ndf, ngpu)
    netD.apply(weights_init_toy)
elif opt.model_D == 'toy4':
    netD = toy4.MLP_D(1, nz, 2, ndf, ngpu)
    netD.apply(weights_init_toy)
else:
    raise ValueError('unkown model: {}'.format(opt.model_D))

if opt.criterion == 'gan':
    # add sigmoid activation function for gan
    netD.main.add_module('sigmoid',
                         nn.Sigmoid())

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# set type of adversarial training
if opt.criterion == 'gan':
    criterion_R = nn.BCELoss()
    criterion_F = nn.BCELoss()
    if opt.gtrain == 'theory' or opt.gtrain == 'flip':
        criterion_G = bceloss.BCELoss(-1)
    else: #opt.gtrain == 'practice':
        criterion_G = nn.BCELoss()
elif opt.criterion == 'wgan' or opt.criterion == 'meangan':
    criterion_R = sumloss.SumLoss()
    criterion_F = sumloss.SumLoss(-1)
    criterion_G = sumloss.SumLoss()
elif opt.criterion == 'geogan':
    criterion_R = hingeloss.HingeLoss()
    criterion_F = hingeloss.HingeLoss()
    if opt.gtrain == 'theory':
        criterion_G = sumloss.SumLoss(sign=-1.0)
    elif opt.gtrain == 'leaky':
        criterion_G = leakyhingeloss.LeakyHingeLoss(margin=opt.margin)
    else:
        raise NotImplementedError('unknown opt.gtrain: {}'.format(opt.gtrain))
elif opt.criterion == 'ebgan':
    criterion_R = sumloss.SumLoss(sign=1.0)
    criterion_F = hingeloss.HingeLoss(margin=opt.margin)
    criterion_G = sumloss.SumLoss(sign=1.0)
else:
    raise ValueError('unknown criterion: {}'.format(opt.criterion))


# init variables
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
if opt.criterion == 'gan' and opt.gtrain == 'theory':
    real_label = 1
    fake_label = 0
    gen_label = fake_label 
elif opt.criterion == 'gan' and opt.gtrain == 'flip':
    real_label = 0
    fake_label = 1
    gen_label = fake_label
elif opt.criterion == 'geogan' and opt.gtrain == 'theory':
    real_label = 1
    fake_label = -1
    gen_label = fake_label 
elif opt.criterion == 'geogan' and opt.gtrain == 'leaky':
    real_label = 1
    fake_label = -1 
    gen_label = real_label
elif opt.criterion == 'ebgan':
    real_label = -1 
    fake_label = 1 
    gen_label = fake_label
else: # opt.gtrain == 'practice'
    real_label = 1
    fake_label = 0
    gen_label = real_label 


# init cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion_R.cuda()
    criterion_F.cuda()
    criterion_G.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


# convert to autograd variable
input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


# setup optimizer
if opt.criterion == 'geogan':
    paramsD = [
        {'params': filter(lambda p: p.cls_weight, netD.parameters()), 'weight_decay': 1.0 / (float(opt.batchSize) * float(opt.C)) }, # assign weight decay for geogan to cls layer only
        {'params': filter(lambda p: p.cls_bias,   netD.parameters()) }, # no weight decay to the bias of cls layer
        {'params': filter(lambda p: not p.cls,    netD.parameters()), 'weight_decay': opt.weight_decay_D }
    ]
else:
    paramsD = [
        {'params': filter(lambda p: p.cls,        netD.parameters()) }, # no weight decay to the bias of cls layer
        {'params': filter(lambda p: not p.cls,    netD.parameters()), 'weight_decay': opt.weight_decay_D }
    ]
    #paramsD = [
    #    {'params': netD.parameters(), 'weight_decay': opt.weight_decay_D },
    #]
if opt.adam:
    optimizerD = optim.Adam(paramsD, lr=opt.lrD, betas=(opt.beta1, 0.999))#, weight_decay=opt.weight_decay_D)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay_G)
else:
    optimizerD = optim.RMSprop(paramsD, lr=opt.lrD)#, weight_decay=opt.weight_decay_D)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG, weight_decay=opt.weight_decay_G)


# training
gen_iterations = 0
disc_iterations = 0
errM_print = -float('inf')
errM_real_print = -float('inf') 
errM_fake_print = -float('inf')
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0

    while i < len(dataloader):
        tm_start = time.time()

        ############################
        # (1) Update D network
        ############################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for p in netG.parameters():
            p.requires_grad = False # to avoid computation

        # train the discriminator Diters times
        if opt.wclip and (gen_iterations < 25 or gen_iterations % 500 == 0):
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            disc_iterations += 1

            ##### weight clipping
            # wclip parameters to a cube
            if opt.wclip:
                for p in netD.parameters():
                    if not p.cls:# or opt.criterion != 'geogan':
                        p.data.clamp_(opt.wclip_lower, opt.wclip_upper)

            # wclip parameters to a cube for the last linear layer of disc if opt.criterion == 'wgan'
            if opt.criterion == 'wgan':
                for p in netD.parameters():
                    if p.cls:
                        p.data.clamp_(opt.wclip_lower, opt.wclip_upper)

            ##### weight projection 
            # weight projection to a cube for parameters
            if opt.wproj:
                for p in netD.parameters():
                    if not p.cls:# or opt.criterion != 'geogan':
                        weight_proj_l2norm(p)

            # wproj parameters to a cube for the last linear layer of disc if opt.criterion == 'meangan'
            if opt.criterion == 'meangan':
                for p in netD.parameters():
                    if p.cls:
                        weight_proj_l2norm(p)

            data_tm_start = time.time()
            data = data_iter.next()
            data_tm_end   = time.time()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)
            outD_real = netD(input)
            errD_real = criterion_R(outD_real, label)
            errD_real.backward()

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            input.data.copy_(fake.data)
            outD_fake = netD(input)
            errD_fake = criterion_F(outD_fake, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()


        ############################
        # (2) Update G network
        ############################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True # reset requires_grad

        j = 0
        while j < opt.Giters:
            j += 1
            gen_iterations += 1

            netG.zero_grad()

            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            label.data.resize_(opt.batchSize).fill_(gen_label)
            noise.data.resize_(opt.batchSize, nz, 1, 1)
            noise.data.normal_(0, 1)

            # forward G
            fake = netG(noise)

            # forward D (backward from D)
            outG = netD(fake)
            errG = criterion_G(outG, label)
            errG.backward()

            # update G
            optimizerG.step()


        ############################
        # Display results 
        ############################
        if opt.display and (gen_iterations % opt.ndisplay == 0):
            if 'toy' in opt.dataset:
                fake = netG(fixed_noise)
                tdset.save_image(real_cpu.view(-1,2).numpy(),
                                 fake.data.cpu().view(-1,2).numpy(),
                                 '{0}/real_fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
                #tdset.save_contour(netD,
                #                   '{0}/disc_contour_{1}.png'.format(opt.experiment, gen_iterations),
                #                   cuda=opt.cuda)
            else:
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment), normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations), normalize=True)

        tm_end = time.time()

        if 'toy' in opt.dataset:
            print('Epoch: [%d][%d/%d][%d]\t Time: %.3f  DataTime: %.3f    Loss_G: %f  Loss_D: %f  Loss_D_real: %f  Loss_D_fake: %f  x_real_sll: %f  x_fake_sll: %f'
                % (epoch, i, len(dataloader), gen_iterations,
                   tm_end-tm_start, data_tm_end-data_tm_start,
                errG.data[0], errD.data[0], errD_real.data[0], errD_fake.data[0], 
                x_sumloglikelihood(real_cpu.view(-1,2).numpy()), x_sumloglikelihood(fake.data.cpu().view(-1,2).numpy())))
        else:
            print('Epoch: [%d][%d/%d][%d]\t Time: %.3f  DataTime: %.3f    Loss_G: %f  Loss_D: %f  Loss_D_real: %f  Loss_D_fake: %f'
                % (epoch, i, len(dataloader), gen_iterations,
                   tm_end-tm_start, data_tm_end-data_tm_start,
                errG.data[0], errD.data[0], errD_real.data[0], errD_fake.data[0]))


        ############################
        # Detect errors 
        ############################
        if np.isnan(errG.data[0]) or np.isnan(errD.data[0]) or np.isnan(errD_real.data[0]) or np.isnan(errD_fake.data[0]):
            raise ValueError('nan detected.')
        if np.isinf(errG.data[0]) or np.isinf(errD.data[0]) or np.isinf(errD_real.data[0]) or np.isinf(errD_fake.data[0]):
            raise ValueError('inf detected.')


    # do checkpointing
    if (epoch+1) % opt.nsave == 0:
        torch.save(netG.state_dict(),       '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(optimizerG.state_dict(), '{0}/optG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(),       '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(optimizerD.state_dict(), '{0}/optD_epoch_{1}.pth'.format(opt.experiment, epoch))
