## imports

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import matplotlib.pyplot as plt
import time
import argparse

import sys

def mean_upcrossings(y, upcross_level=0):
    '''
    returns mean number of upcrossings of upcross_level

    y: (n_samples, n_gridpoints) array of function values
    '''
    u = upcross_level*np.ones(y.shape)
    return np.sum(np.logical_and(y[:,:-1]<u[:,:-1], y[:,1:]>u[:,1:])) / y.shape[0]

parser = argparse.ArgumentParser()
parser.add_argument('--dir_base', type=str, default='../../../', help='directory of source code')
parser.add_argument('--seed', type=int, default=1, help='seed for dataset')
parser.add_argument('--dir_out', type=str, default='upcross_scaling/', help='directory of output')

# prior
#parser.add_argument('--dim_hidden', type=int, default=30, help='number of hidden units')
#parser.add_argument('--prior_w1_sig2', type=float, default=150., help='layer 1 weights prior variance')
#parser.add_argument('--prior_b1_sig2', type=float, default=75., help='layer 1 bias prior variance')
#parser.add_argument('--prior_w2_sig2', type=float, default=.75, help='layer 2 weights prior variance')
#parser.add_argument('--prior_b2_sig2', type=float, default=.75, help='layer 2 bias prior variance')
parser.add_argument('--sig2', type=float, default=0.0356634620928351, help='observation noise variance')

args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/bnn/')

sys.path.append(dir_src)
sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import networks_bnn as networks
import util_bnn as util

if not os.path.exists(args.dir_out):
	os.makedirs(os.path.join(args.dir_out, 'output/'))

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

start_time = time.time()

torch.set_default_dtype(torch.float64)

## define model
torch.manual_seed(5)
np.random.seed(5) # includes one numpy part
x_plot = torch.linspace(-0.5, 0.5, 500).reshape(-1,1)

dim_hidden_list = np.array([20, 80.])
prior_w1_sig2_list = 2**np.arange(5,15)*1.0
prior_w2_sig2_list = 2.0**np.arange(0,4)

#dim_hidden_list = np.array([10, 100])
#prior_w1_sig2_list = np.array([2**5., 10000, 2**14])
#prior_w1_sig2_list = np.array([1., 2.5, 8])
#prior_w2_sig2_list = np.array([1.0, 1.5, 2.0])

upcross = np.zeros((len(dim_hidden_list), len(prior_w1_sig2_list), len(prior_w2_sig2_list)))
variance = np.zeros((len(dim_hidden_list), len(prior_w1_sig2_list), len(prior_w2_sig2_list)))

_, idx_x0 = torch.min(torch.abs(x_plot.view(-1)), dim=0)
print(x_plot[idx_x0])

for i, dim_hidden in enumerate(dim_hidden_list):
	for j, prior_w1_sig2 in enumerate(prior_w1_sig2_list):
		for k, prior_w2_sig2 in enumerate(prior_w2_sig2_list):

			print('------ %d, %d, %d ------' % (i,j,k))
			print('dim_hidden: ', dim_hidden)
			print('prior_w1_sig2: ', prior_w1_sig2)
			print('prior_w2_sig2: ', prior_w2_sig2)


			net = networks.BNN(dim_in=1, dim_hidden=int(dim_hidden), dim_out=1, \
					prior_w1_sig2 = prior_w1_sig2, prior_b1_sig2 = prior_w1_sig2, \
					prior_w2_sig2 = prior_w2_sig2, prior_b2_sig2 = 1e-8, \
                    sig2=args.sig2, prior_sig2_alpha = None, prior_sig2_beta = None, s2_0=1.0)

			#s2 = net.h(net.prior_w1_sig2)
			#print('s2 without sgcp: ', s2)
			y_samp_prior = net.sample_functions_prior(x_plot, n_samp=2000)
			upcross[i,j,k] = mean_upcrossings(y_samp_prior.detach().numpy())
			#variance[i,j,k] = torch.var(y_samp_prior[:, idx_x0]).item()
			variance[i,j,k] = torch.mean(torch.var(y_samp_prior,0)).item()

			fig, ax = util.plot_prior_predictive(x_plot.numpy().reshape(-1), y_samp_prior.detach().numpy(), plot_all_functions=True)
			fig.suptitle('dim_hidden=%.2f_prior_w1_sig2=%.2f_prior_w2_sig2=%.2f' % (dim_hidden,prior_w1_sig2,prior_w2_sig2))
			fig.savefig(os.path.join(args.dir_out, 'output/prior_samples_dim_hidden=%d_prior_w1_sig2=%d_prior_w2_sig2=%d.png' % (i,j,k)))

			plt.close('all')



fig, ax = plt.subplots(2,2, figsize=(8,6), sharex='col', sharey='row')

idx_show_j = np.array([2,5])
idx_show_k = np.array([0,-1])
#idx_show_j = np.array([0,1,2])
#idx_show_k = np.array([0,1])

palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

# y: upcrossing
# x: prior_w1_sig2
# color: prior_w2_sig2
# fixed: dim_hidden
ax[0,0].set_prop_cycle('color',palette[0:2])
ax[0,0].plot(prior_w1_sig2_list, np.squeeze(upcross[0][:,idx_show_k]),'-o')
ax[0,0].set_prop_cycle('color',palette[0:2])
ax[0,0].plot(prior_w1_sig2_list, np.squeeze(upcross[1][:,idx_show_k]),'--o')

ax[0,0].set_ylabel('Upcrossings')
#ax[0,0].set_xlabel('prior_w1_sig2')

box = ax[0,0].get_position()
ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax[0,0].legend(prior_w2_sig2_list, title='prior_w2_sig2',loc='center left', bbox_to_anchor=(1, 0.5))
ax[0,0].legend(prior_w2_sig2_list[idx_show_k], title=r'$\sigma^2_{w2}$',loc='center left', bbox_to_anchor=(1, 0.5))



# y: upcrossing
# x: prior_w2_sig2
# color: prior_w1_sig2
# fixed: dim_hidden
ax[0,1].set_prop_cycle('color',palette[2:4])
ax[0,1].plot(prior_w2_sig2_list, np.squeeze(upcross[0][idx_show_j,:]).T,'-o')
ax[0,1].set_prop_cycle('color',palette[2:4])
ax[0,1].plot(prior_w2_sig2_list, np.squeeze(upcross[1][idx_show_j,:]).T,'--o')

#ax[0,1].set_ylabel('Upcrossings')
ax[0,1].set_xlabel(r'$\sigma^2_{w2}$')

box = ax[0,1].get_position()
ax[0,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend(prior_w1_sig2_list[idx_show_j], title=r'$\sigma^2_{w1}$',loc='center left', bbox_to_anchor=(1, 0.5))




# y: var
# x: prior_w1_sig2
# color: prior_w2_sig2
# fixed: dim_hidden
ax[1,0].set_prop_cycle('color',palette[0:2])
ax[1,0].plot(prior_w1_sig2_list, np.squeeze(variance[0][:,idx_show_k]),'-o')
ax[1,0].set_prop_cycle('color',palette[0:2])
ax[1,0].plot(prior_w1_sig2_list, np.squeeze(variance[1][:,idx_show_k]),'--o')

ax[1,0].set_ylabel('Variance')
ax[1,0].set_xlabel(r'$\sigma^2_{w1}$')

box = ax[1,0].get_position()
ax[1,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,0].legend(prior_w2_sig2_list[idx_show_k], title=r'$\sigma^2_{w2}$',loc='center left', bbox_to_anchor=(1, 0.5))

# y: var
# x: prior_w2_sig2
# color: prior_w1_sig2
# fixed: dim_hidden
ax[1,1].set_prop_cycle('color',palette[2:4])
ax[1,1].plot(prior_w2_sig2_list, np.squeeze(variance[0][idx_show_j,:]).T,'-o')
ax[1,1].set_prop_cycle('color',palette[2:4])
ax[1,1].plot(prior_w2_sig2_list, np.squeeze(variance[1][idx_show_j,:]).T,'--o')

#ax[1,1].set_ylabel('Variance')
ax[1,1].set_xlabel(r'$\sigma^2_{w2}$')

box = ax[1,1].get_position()
ax[1,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,1].legend(prior_w1_sig2_list[idx_show_j], title=r'$\sigma^2_{w1}$',loc='center left', bbox_to_anchor=(1, 0.5))


fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_bnn.png'))
fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_bnn.pdf'))

