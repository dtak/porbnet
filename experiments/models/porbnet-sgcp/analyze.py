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

dir_src = os.path.join(args.dir_base, 'src/porbnet-sgcp/')
#dir_data = os.path.join(args.dir_base, 'data/')

sys.path.append(dir_src)
#sys.path.append(dir_data)
#import networks
import util_porbnet_sgcp as util
#import data

##

samples = np.load(os.path.join(args.dir_out, 'samples.npy'), allow_pickle=True).item()
accept = np.load(os.path.join(args.dir_out, 'accept.npy'), allow_pickle=True).item()

# functions
fig, ax = util.plot_functions(samples['x_plot'], samples['y_plot_pred'], samples['x'], samples['y'], samples['x_test'], samples['y_test'], torch.mean(samples['sig2']).item())
fig.savefig(os.path.join(args.dir_out, 'posterior_functions.png'))


# sig2
fig, ax = plt.subplots()
ax.plot(samples['sig2'].numpy())
fig.savefig(os.path.join(args.dir_out, 'sig2.png'))


# intensity
fig, ax = util.plot_intensity(samples['x_plot'].numpy(), samples['gp_plot_pred'].numpy(), torch.mean(samples['rate_density_ub']).numpy(), samples['x'].numpy())
fig.savefig(os.path.join(args.dir_out, 'intensity.png'))

# traceplot of centers
fig, ax = util.plot_centers_trace(samples['c'].detach().numpy(), samples['mask'].numpy())
fig.savefig(os.path.join(args.dir_out, 'c_trace.png'))


# dim_hidden
fig, ax = plt.subplots()
ax.plot(samples['dim_hidden'].numpy())
fig.savefig(os.path.join(args.dir_out, 'dim_hidden.png'))


with open(os.path.join(args.dir_out, 'analyze_output.txt'), 'w') as f:

	f.write('test_log_likelihood: %.5f\n' % util.test_log_likelihood(samples['y_test'], samples['y_test_pred'], samples['sig2']))



## Export data
np.savetxt(os.path.join(args.dir_out, 'x.csv'), samples['x'].numpy())
