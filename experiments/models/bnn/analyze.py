## imports

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_base', type=str, default='../../../', help='directory of source code')
parser.add_argument('--dir_out', type=str, default='./output', help='directory of experiment output')
args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/bnn/')
#dir_data = os.path.join(args.dir_base, 'data/')

sys.path.append(dir_src)
#sys.path.append(dir_data)
#import networks
import util_bnn as util
#import data

##

samples = np.load(os.path.join(args.dir_out, 'samples.npy'), allow_pickle=True).item()
accept = np.load(os.path.join(args.dir_out, 'accept.npy'), allow_pickle=True).item()

fig, ax = util.plot_functions(samples['x_plot'], samples['y_plot_pred'], samples['x'], samples['y'], None, None, torch.mean(samples['sig2']).item())
fig.savefig(os.path.join(args.dir_out, 'posterior_functions.png'))

ax.set_ylim(-0.8,0.8)
fig.savefig(os.path.join(args.dir_out,'posterior_functions_sameaxis.png'))

fig, ax = util.plot_functions(samples['x_plot'], samples['y_plot_pred'], samples['x'], samples['y'], None, None, torch.mean(samples['sig2']).item(), plot_all=True)
fig.savefig(os.path.join(args.dir_out, 'posterior_functions_all.png'))

ax.set_ylim(-0.8,0.8)
fig.savefig(os.path.join(args.dir_out,'posterior_functions_all_sameaxis.png'))


#with open('output/output.txt', 'w') as f:

#	f.write('test_log_likelihood: %.5f\n' % util.test_log_likelihood(samples['y_test'], samples['y_test_pred'], samples['sig2']))
	
