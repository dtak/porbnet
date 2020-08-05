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
parser.add_argument('--dim_hidden_initial', type=int, default=30, help='initial number of hidden units')
parser.add_argument('--dim_hidden_max', type=int, default=75, help='maximum number of hidden units')
#parser.add_argument('--s2_0', type=float, default=.175, help='inverse lengthscale when intensity = 1')
parser.add_argument('--length_scale_sgcp', type=float, default=.15, help='lengthscale of sgcp kernel')
parser.add_argument('--variance_sgcp', type=float, default=1.0, help='amplitude variance of sgcp kernel')
parser.add_argument('--proposal_std_cM', type=float, default=.2, help='perturbative proposal standard deviation for thinned locations')
parser.add_argument('--T_lb', type=float, default=-1.0, help='lower bound of PP region')
parser.add_argument('--T_ub', type=float, default=2.0, help='upper bound of PP region')


#parser.add_argument('--rate_density_ub', type=float, default=60.0, help='')
#parser.add_argument('--rate_density_ub_prior_alpha', type=float, default=None, help='alpha parameter for gamma prior on rate_density_ub') 
#parser.add_argument('--rate_density_ub_prior_beta', type=float, default=None, help='alpha parameter for gamma prior on rate_density_ub') 
#parser.add_argument('--rate_density_ub_proposal_std', type=float)
#parser.add_argument('--prior_w_sig2', type=float, default=.75, help='weights prior variance')
parser.add_argument('--prior_b_sig2', type=float, default=1e-8, help='bias prior variance')
parser.add_argument('--sig2', type=float, default=0.0356634620928351, help='observation noise variance')


parser.add_argument('--sample_rate_density_ub', action='store_true')

args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/porbnet-sgcp/')

sys.path.append(dir_src)
sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import networks_porbnet_sgcp as networks
import util_porbnet_sgcp as util

if not os.path.exists(args.dir_out):
	os.makedirs(os.path.join(args.dir_out, 'output/'))

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

start_time = time.time()

torch.set_default_dtype(torch.float64)

## define model
torch.manual_seed(5)
np.random.seed(5) # includes one numpy part
x_plot = torch.linspace(0, 1, 250).reshape(-1,1)

s2_0_list = np.array([1.])
#rate_density_ub_list = np.array([10, 25, 50])
#prior_w_sig2_list = np.array([.1, .5, 1.5])
rate_density_ub_list = 2*np.array([4, 20., 40.])
prior_w_sig2_list = np.array([0.7, 1.4, 2.8])


upcross = np.zeros((len(s2_0_list), len(rate_density_ub_list), len(prior_w_sig2_list)))
variance = np.zeros((len(s2_0_list), len(rate_density_ub_list), len(prior_w_sig2_list)))

_, idx_x0 = torch.min(torch.abs(x_plot.view(-1)), dim=0)
print(x_plot[idx_x0])

for i, s2_0 in enumerate(s2_0_list):
	for j, rate_density_ub in enumerate(rate_density_ub_list):
		for k, prior_w_sig2 in enumerate(prior_w_sig2_list):

			print('------ %d, %d, %d ------' % (i,j,k))
			print('s2_0: ', s2_0)
			print('rate_density_ub: ', rate_density_ub)
			print('prior_w_sig2: ', prior_w_sig2)

			expected_dim_hidden = int((args.T_ub - args.T_lb) * (.5*rate_density_ub))
			print('expected_dim_hidden: ', expected_dim_hidden)

			if args.sample_rate_density_ub:
				rate_density_ub_prior_beta = 1.0
				rate_density_ub_prior_alpha = rate_density_ub
			else:
				rate_density_ub_prior_beta = None
				rate_density_ub_prior_alpha = None


			net = networks.RBFN(dim_in=1, dim_hidden_initial=expected_dim_hidden, dim_hidden_max=4*expected_dim_hidden, dim_out=1, \
						s2_0 = s2_0, \
	                    T=[args.T_lb, args.T_ub], length_scale_sgcp=args.length_scale_sgcp, variance_sgcp=args.variance_sgcp, proposal_std_cM=args.proposal_std_cM, \
	                    rate_density_ub=rate_density_ub, rate_density_ub_prior_alpha = rate_density_ub_prior_alpha, rate_density_ub_prior_beta = rate_density_ub_prior_beta, \
						prior_w_sig2 = prior_w_sig2*np.sqrt(np.pi/s2_0), prior_b_sig2 = args.prior_b_sig2, \
	                    sig2=args.sig2,
	                    infer_rate_density_ub=args.sample_rate_density_ub)

			s2 = net.h(net.rate_density_ub * util.sigmoid_torch(torch.tensor([0.])))
			print('s2 with gp=0: ', s2)
			print('gK :', net.gK[net.mask])
			
			y_samp_prior = net.sample_functions_prior(x_plot, n_samp=5000, proper=True)
			upcross[i,j,k] = mean_upcrossings(y_samp_prior.detach().numpy())
			variance[i,j,k] = torch.var(y_samp_prior[:, idx_x0]).item()

			fig, ax = util.plot_prior_predictive(x_plot.numpy().reshape(-1), y_samp_prior.detach().numpy(), plot_all_functions=True)
			fig.suptitle('s2_0=%.2f_rate_density_ub=%.2f_prior_w_sig2=%.2f' % (s2_0,rate_density_ub,prior_w_sig2))
			fig.savefig(os.path.join(args.dir_out, 'output/prior_samples_s2_0=%d_rate_density_ub=%d_prior_w_sig2=%d.png' % (i,j,k)))

			plt.close('all')



fig, ax = plt.subplots(2,2, figsize=(8,6), sharex='col', sharey='row')

# y: upcrossing
# x: rate_density_ub
# color: prior_w_sig2
# fixed: s2_0
ax[0,0].plot(rate_density_ub_list, np.squeeze(upcross[0,:,:]),'-o')

ax[0,0].set_ylabel('Upcrossings')
#ax[0,0].set_xlabel('rate_density_ub')

box = ax[0,0].get_position()
ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,0].legend(prior_w_sig2_list, title=r'$\sigma^2_w$',loc='center left', bbox_to_anchor=(1, 0.5))


# y: upcrossing
# x: prior_w_sig2
# color: rate_density_ub
# fixed: s2_0
ax[0,1].plot(prior_w_sig2_list, np.squeeze(upcross[0,:,:]).T,'-o')

#ax[0,1].set_ylabel('Upcrossings')
#ax[0,1].set_xlabel(r'$\sigma^2_w$')

box = ax[0,1].get_position()
ax[0,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend(rate_density_ub_list, title=r'$\lambda^*$',loc='center left', bbox_to_anchor=(1, 0.5))

# y: var
# x: rate_density_ub
# color: prior_w_sig2
# fixed: s2_0
ax[1,0].plot(rate_density_ub_list, np.squeeze(variance[0,:,:]),'-o')

ax[1,0].set_ylabel('Variance')
ax[1,0].set_xlabel(r'$\lambda^*$')

box = ax[1,0].get_position()
ax[1,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,0].legend(prior_w_sig2_list, title=r'$\sigma^2_w$',loc='center left', bbox_to_anchor=(1, 0.5))


# y: var
# x: prior_w_sig2
# color: rate_density_ub
# fixed: s2_0
ax[1,1].plot(prior_w_sig2_list, np.squeeze(variance[0,:,:]).T,'-o')

#ax[1,1].set_ylabel('Variance')
ax[1,1].set_xlabel(r'$\sigma^2_w$')

box = ax[1,1].get_position()
ax[1,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,1].legend(rate_density_ub_list, title=r'$\lambda^*$',loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_porbnet.png'))
fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_porbnet.pdf'))

