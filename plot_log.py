import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parse import *
import progressbar
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pickle
import os.path
import scipy
import scipy.signal

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_prefix", help="output prefix. output images will be <output prefix>_disc_loss.png, <output prefix>_real_loss.png, <output prefix>_fake_loss.png, <output prefix>_gen_loss.png")
parser.add_argument("-d", "--data", nargs=2, action='append', 
                    help="<label> <log_filename> pairs. multiple data are available. if it is the case, all the logs will be drawed in each corresponding plot (disc, real, fake, gen)")
parser.add_argument("-m", "--med", help="median filter size",
                    type=int,
                    default=101)
args = parser.parse_args()

def parse_logs(log_path):
  # Open log_path 
  with open(log_path, 'rt') as f:
    lines = f.readlines()
  num_data = len(lines)-1

  # Init necessary variables 
  daxis      = np.zeros(num_data)
  gaxis      = np.zeros(num_data)
  real_loss  = np.zeros(num_data)
  fake_loss  = np.zeros(num_data)
  disc_loss  = np.zeros(num_data)
  gen_loss   = np.zeros(num_data)

  # Init bar and do parsing
  print "progress: " 
  bar = progressbar.ProgressBar(maxval=num_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  for i in xrange(num_data):
    tokens = lines[i].split()

    disc_loss[i] = float(tokens[9]) 
    real_loss[i] = float(tokens[11])
    fake_loss[i] = float(tokens[13])
    gen_loss[i]  = float(tokens[7])

    buffers = parse("[{}][{}/{}][{}]", tokens[1])
    epoch = int(buffers[0])+1 
    cur_diter = int(buffers[1])
    max_diter = int(buffers[2])
    giter = int(buffers[3])

    daxis[i] = (float(epoch)-1) + float(cur_diter)/float(max_diter)
    gaxis[i] = giter 

    bar.update(i+1)
  bar.finish()

  return {'daxis':daxis, 'gaxis':gaxis, 
          'real':real_loss, 'fake':fake_loss , 'disc':disc_loss, 'gen':gen_loss }


###################################### process data
# init input arguments
num_files = len(args.data)
logs = []
output_prefix = args.output_prefix

# load logs
for i in range(0, num_files):
  log_filename = args.data[i][1] #log_filenames[i]
  log_path = log_filename
  log_cache_path = '{}.{}'.format(log_path, 'pkl')
  
  if not os.path.exists(log_cache_path):
    print 'parse log (label: {})'.format(args.data[i][0])
    logs.append(parse_logs(log_path))
    pickle.dump(logs[i], open(log_cache_path , "wb"))
  else:
    logs.append(pickle.load(open(log_cache_path, "rb")))


###################################### plot gen loss
fig, ax = plt.subplots()

for i in range(0, num_files):
  plt.plot(logs[i]['gaxis'], logs[i]['gen'], label=args.data[i][0])

plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('gen loss', fontsize=14, color='black')
plt.title('Generator Loss')
plt.savefig('{}_gen_loss'.format(output_prefix))

###################################### plot real loss
fig, ax = plt.subplots()

for i in range(0, num_files):
  plt.plot(logs[i]['gaxis'], logs[i]['real'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('real loss', fontsize=14, color='black')
plt.title('Real Loss')
plt.savefig('{}_real_loss'.format(output_prefix))

###################################### plot fake loss
fig, ax = plt.subplots()

for i in range(0, num_files):
  plt.plot(logs[i]['gaxis'], logs[i]['fake'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('fake loss', fontsize=14, color='black')
plt.title('Fake Loss')
plt.savefig('{}_fake_loss'.format(output_prefix))

###################################### plot disc loss
fig, ax = plt.subplots()
for i in range(0, num_files):
  plt.plot(logs[i]['gaxis'], logs[i]['disc'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('disc loss', fontsize=14, color='black')
plt.title('Discriminator Loss (real + fake)')
plt.savefig('{}_disc_loss'.format(output_prefix))

###################################### plot disc (medfilt) loss
fig, ax = plt.subplots()

for i in range(0, num_files):
  med_filtered_loss = scipy.signal.medfilt(logs[i]['disc'], args.med)
  plt.plot(logs[i]['gaxis'], med_filtered_loss, label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('disc loss', fontsize=14, color='black')
plt.title('Discriminator Loss (median filtered, size: {})'.format(args.med))
plt.savefig('{}_disc_medfilt_loss'.format(output_prefix))

print 'Done.'
