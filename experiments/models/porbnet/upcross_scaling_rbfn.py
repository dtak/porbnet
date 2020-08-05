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
parser.add_argument('--dir_out', type=str, default='upcross_scaling_rbfn/', help='directory of output')

# prior
parser.add_argument('--dim_hidden_initial', type=int, default=30, help='initial number of hidden units')
parser.add_argument('--dim_hidden_max', type=int, default=75, help='maximum number of hidden units')
#parser.add_argument('--s2_0', type=float, default=.175, help='inverse lengthscale when intensity = 1')
parser.add_argument('--T_lb', type=float, default=-1.0, help='lower bound of PP region')
parser.add_argument('--T_ub', type=float, default=2.0, help='upper bound of PP region')
#parser.add_argument('--intensity', type=float, default=1.0)
#parser.add_argument('--prior_w_sig2', type=float, default=.75, help='weights prior variance')
parser.add_argument('--prior_b_sig2', type=float, default=1e-8, help='bias prior variance')
parser.add_argument('--sig2', type=float, default=0.0356634620928351, help='observation noise variance')

parser.add_argument('--sample_intensity', action='store_true')

#parser.add_argument('--prior_intensity_alpha', type=float, default=None, help='')
#parser.add_argument('--prior_intensity_beta', type=float, default=None, help='')

args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/porbnet/')

sys.path.append(dir_src)
sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import networks_porbnet as networks
import util_porbnet as util

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

start_time = time.time()

torch.set_default_dtype(torch.float64)

## define model
torch.manual_seed(5)
np.random.seed(5) # includes one numpy part
x_plot = torch.linspace(0, 1, 500).reshape(-1,1)

s2_0_list = np.array([25.])
#rate_density_list = np.array([4, 20., 40.])
#prior_w_sig2_list = np.array([0.7, 1.4, 2.8])

rate_density_list = 2.0**np.arange(2,7)
prior_w_sig2_list = 2.0**np.arange(0,4)

upcross = np.zeros((len(s2_0_list), len(rate_density_list), len(prior_w_sig2_list)))
variance = np.zeros((len(s2_0_list), len(rate_density_list), len(prior_w_sig2_list)))

_, idx_x0 = torch.min(torch.abs(x_plot.view(-1)), dim=0)
print(x_plot[idx_x0])

for i, s2_0 in enumerate(s2_0_list):
    for j, rate_density in enumerate(rate_density_list):
        for k, prior_w_sig2 in enumerate(prior_w_sig2_list):

            print('------ %d, %d, %d ------' % (i,j,k))
            print('s2_0: ', s2_0)
            print('rate_density: ', rate_density)
            print('prior_w_sig2: ', prior_w_sig2)

            expected_dim_hidden = int((args.T_ub - args.T_lb) * rate_density)
            print('expected_dim_hidden: ', expected_dim_hidden)

            intensity = util.Piecewise(np.array([args.T_lb,args.T_ub]),np.array([rate_density]))

            if args.sample_intensity:
                prior_intensity_beta = 1.0
                prior_intensity_alpha = util.alpha_for_sqrt_gamma(beta=prior_intensity_beta, K=rate_density)
            else:
                prior_intensity_beta = None
                prior_intensity_alpha = None

            class RegularRBFN(networks.RBFN):

                def forward(self, x):
                    h = torch.exp(-0.5 * (x - self.c[:, self.mask])**2 * s2_0)
                    return torch.nn.functional.linear(h, self.w[:, self.mask], self.b)


            net = RegularRBFN(dim_in=1, dim_hidden_initial=expected_dim_hidden, dim_hidden_max=3*expected_dim_hidden, \
                                dim_out=1, intensity=intensity, prior_intensity_alpha=prior_intensity_alpha, prior_intensity_beta=prior_intensity_beta,\
                                s2_0=s2_0, \
                                prior_w_sig2 = prior_w_sig2*np.sqrt(np.pi/s2_0), prior_b_sig2 = args.prior_b_sig2, \
                                sig2 = args.sig2, prior_sig2_alpha = None, prior_sig2_beta = None,\
                                infer_intensity=args.sample_intensity)

            #s2 = net.h(net.rate_density)
            #print('s2 without sgcp: ', s2)
            y_samp_prior = net.sample_functions_prior(x_plot, n_samp=2000, sample_K=True, sample_intensity=args.sample_intensity)
            upcross[i,j,k] = mean_upcrossings(y_samp_prior.detach().numpy())
            #variance[i,j,k] = torch.var(y_samp_prior[:, idx_x0]).item()
            variance[i,j,k] = torch.mean(torch.var(y_samp_prior,0)).item()

            fig, ax = util.plot_prior_predictive(x_plot.numpy().reshape(-1), y_samp_prior.detach().numpy(), plot_all_functions=True)
            fig.suptitle('s2_0=%.2f_rate_density=%.2f_prior_w_sig2=%.2f' % (s2_0,rate_density,prior_w_sig2))
            fig.savefig(os.path.join(args.dir_out, 'output/prior_samples_s2_0=%d_rate_density=%d_prior_w_sig2=%d.png' % (i,j,k)))

            plt.close('all')



fig, ax = plt.subplots(2,2, figsize=(8,6), sharex='col', sharey='row')
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

idx_show_j = np.array([1,-2])
idx_show_k = np.array([1,-2])

# y: upcrossing
# x: rate_density
# color: prior_w_sig2
# fixed: s2_0
ax[0,0].set_prop_cycle('color',palette[0:2])
ax[0,0].plot(rate_density_list, np.squeeze(upcross[0][:,idx_show_k]),'-o')

ax[0,0].set_ylabel('Upcrossings')
#ax[0,0].set_xlabel('rate_density')

box = ax[0,0].get_position()
ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax[0,0].legend(prior_w_sig2_list, title='prior_w_sig2',loc='center left', bbox_to_anchor=(1, 0.5))
ax[0,0].legend(prior_w_sig2_list[idx_show_k], title=r'$\sigma^2_w$',loc='center left', bbox_to_anchor=(1, 0.5))



# y: upcrossing
# x: prior_w_sig2
# color: rate_density
# fixed: s2_0
ax[0,1].set_prop_cycle('color',palette[2:4])
ax[0,1].plot(prior_w_sig2_list, np.squeeze(upcross[0][idx_show_j,:]).T,'-o')

#ax[0,1].set_ylabel('Upcrossings')
ax[0,1].set_xlabel(r'$\sigma^2_w$')

box = ax[0,1].get_position()
ax[0,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend(rate_density_list[idx_show_j], title=r'$\lambda$',loc='center left', bbox_to_anchor=(1, 0.5))

# y: var
# x: rate_density
# color: prior_w_sig2
# fixed: s2_0
ax[1,0].set_prop_cycle('color',palette[0:2])
ax[1,0].plot(rate_density_list, np.squeeze(variance[0][:,idx_show_k]),'-o')

ax[1,0].set_ylabel('Variance')
ax[1,0].set_xlabel(r'$\lambda$')

box = ax[1,0].get_position()
ax[1,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,0].legend(prior_w_sig2_list[idx_show_k], title=r'$\sigma^2_w$',loc='center left', bbox_to_anchor=(1, 0.5))

# y: var
# x: prior_w_sig2
# color: rate_density
# fixed: s2_0
ax[0,1].set_prop_cycle('color',palette[0:2])
ax[1,1].plot(prior_w_sig2_list, np.squeeze(variance[0][idx_show_j,:]).T,'-o')

#ax[1,1].set_ylabel('Variance')
ax[1,1].set_xlabel(r'$\sigma^2_w$')

box = ax[1,1].get_position()
ax[1,1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[1,1].legend(rate_density_list[idx_show_j], title=r'$\lambda$',loc='center left', bbox_to_anchor=(1, 0.5))


fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_rbfn.png'))
fig.savefig(os.path.join(args.dir_out, 'upcross_vs_var_rbfn.pdf'))

