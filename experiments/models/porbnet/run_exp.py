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

parser = argparse.ArgumentParser()
parser.add_argument('--dir_base', type=str, default='../../../', help='directory of source code')
parser.add_argument('--dataset', type=str, default='motorcycle', help='dataset name')
parser.add_argument('--dataset_option', type=str, default=None, help='argument for dataset')
parser.add_argument('--seed', type=int, default=1, help='seed for dataset')
parser.add_argument('--dir_out', type=str, default='./', help='directory of output')

parser.add_argument('--validation_mode', action='store_true', help='flag for whether or not to split off validation set from training set (and use as test set)')

# prior
parser.add_argument('--dim_hidden_initial', type=int, default=30, help='initial number of hidden units')
parser.add_argument('--dim_hidden_max', type=int, default=75, help='maximum number of hidden units')
parser.add_argument('--s2_0', type=float, default=.175, help='inverse lengthscale when intensity = 1')
parser.add_argument('--T_lb', type=float, default=-1.0, help='lower bound of PP region')
parser.add_argument('--T_ub', type=float, default=2.0, help='upper bound of PP region')

parser.add_argument('--intensity_1', type=float, default=1.0, help='TEMP')
parser.add_argument('--intensity_2', type=float, default=None, help='TEMP')

parser.add_argument('--prior_w_sig2', type=float, default=.75, help='weights prior variance')
parser.add_argument('--prior_b_sig2', type=float, default=.75, help='bias prior variance')
parser.add_argument('--sig2', type=float, default=0.0356634620928351, help='observation noise variance')

# posterior samples
parser.add_argument('--burnin_n_samp', type=int, default=2500, help='number of samples (burn-in)')
parser.add_argument('--burnin_eps', type=float, default=0.00001, help='initial hmc step size (burn-in)')
parser.add_argument('--burnin_n_adapt_eps', type=int, default=200, help='number of times to adjust hmc step size (burn-in)')
parser.add_argument('--burnin_n_rep_hmc', type=int, default=1, help='number of hmc replications per loop (burn-in)')
parser.add_argument('--burnin_n_bigwrite', type=int, default=5, help='number of times to plot in tensorboard (burn-in)')

parser.add_argument('--burnin_resize_n_rep_resize', type=int, default=1, help='number of birth/death replications per loop (burn-in resize)')

parser.add_argument('--record_n_samp', type=int, default=5000, help='number of samples (record)')
parser.add_argument('--record_n_rep_hmc', type=int, default=1, help='number of hmc replications per loop (record)')
parser.add_argument('--record_n_bigwrite', type=int, default=500, help='number of times to plot in tensorboard (record)')
parser.add_argument('--record_n_rep_resize', type=int, default=1, help='number of birth/death replications per loop (record)')

parser.add_argument('--prior_intensity_alpha', type=float, default=None, help='')
parser.add_argument('--prior_intensity_beta', type=float, default=None, help='')

parser.add_argument('--prior_matching_file', type=str, default=None)

args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/porbnet/')
dir_data = os.path.join(args.dir_base, 'data/')

sys.path.append(dir_src)
sys.path.append(dir_data)
sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import networks_porbnet as networks
import util_porbnet as util
import data
import utils_load_dataset

if args.prior_matching_file is not None:
    util.reset_args_matched_prior(args)

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))
    if args.prior_matching_file is not None:
        f.write('Prior matching file %s' % args.prior_matching_file)

start_time = time.time()

torch.set_default_dtype(torch.float64)

## define model
torch.manual_seed(4)
np.random.seed(4) # includes one numpy part

if args.intensity_2 is not None:
    intensity = util.Piecewise(np.array([args.T_lb,0.5,args.T_ub]),np.array([args.intensity_1,args.intensity_2])) # HARDCODED FOR NOW
else:
    intensity = util.Piecewise(np.array([args.T_lb,args.T_ub]),np.array([args.intensity_1])) # HARDCODED FOR NOW

if args.dim_hidden_max is None:
    dim_hidden_max = 3*args.dim_hidden
else:
    dim_hidden_max = args.dim_hidden_max


net = networks.RBFN(dim_in=1, dim_hidden_initial=args.dim_hidden_initial, dim_hidden_max=args.dim_hidden_max, \
                    dim_out=1, intensity=intensity, s2_0=args.s2_0, \
                    prior_w_sig2 = args.prior_w_sig2, prior_b_sig2 = args.prior_b_sig2, \
                    sig2 = args.sig2, prior_sig2_alpha = None, prior_sig2_beta = None,
                    prior_intensity_alpha = args.prior_intensity_alpha, prior_intensity_beta = args.prior_intensity_beta)

## dataset
if args.dataset=='motorcycle':
    x, y, x_test, y_test = util.format_data_torch(*data.load_motorcycle(dir_data, seed=args.seed))
elif args.dataset=='finance':
    x, y, x_test, y_test = util.format_data_torch(*data.load_finance2(dir_data, seed=args.seed))
elif args.dataset=='mimic_gap':
    x, y, x_test, y_test = util.format_data_torch(*utils_load_dataset.load_mimic_patient_gap(dir_data, subject_id=args.dataset_option))
elif args.dataset=='GPdata':
    x, y, x_val, y_val, x_test, y_test, x_plot, f_plot = data.load_GPdata(dir_data, lengthscale=args.dataset_option)
    x, y, x_test, y_test = util.format_data_torch(x, y, x_test, y_test)
elif args.dataset=='finance_nogap':
    x, y, x_test, y_test = util.format_data_torch(*data.load_finance2(dir_data, gaps=None, seed=args.seed))
elif args.dataset=='mimic':
    x, y = utils_load_dataset.load_mimic_patient(patient_id=int(args.dataset_option), csv_filename=os.path.join(dir_data, 'hr.csv'))
    x, y, x_test, y_test = util.format_data_torch(*data.train_val_split(x, y, frac_train=0.75, seed=args.seed))
else:
    print('dataset not found')

if args.validation_mode:
    # split training set into smaller training set and validation set, use as training set and test set going forward
    x, y, x_test, y_test = data.train_val_split(x, y, frac_train=0.8, seed=args.seed+50)

x, y, x_test, y_test = data.standardize(x, y, x_test, y_test)

## check prior
x_plot = torch.linspace(0, 1, 100).reshape(-1,1)

y_samp_prior = net.sample_functions_prior(x_plot, n_samp=1000)
fig, ax = util.plot_functions(x_plot, y_samp_prior, x=x, y=y, x_test=x_test, y_test=y_test)
fig.savefig(os.path.join(args.dir_out, 'output/prior_sample_fuctions.png'))

fig, ax = util.plot_functions(x_plot, y_samp_prior,  x=x, y=y, x_test=x_test, y_test=y_test, plot_all=True)
fig.savefig(os.path.join(args.dir_out, 'output/prior_sample_fuctions_all.png'))

fig, ax = util.plot_prior_predictive(x_plot.numpy().reshape(-1), y_samp_prior.detach().numpy(), plot_all_functions=True)
fig.savefig(os.path.join(args.dir_out, 'output/prior_predictive.png'))


## initialize network
torch.manual_seed(args.seed+100)
np.random.seed(args.seed+200) # includes one numpy part
net.init_parameters() 

## plot of initial function before sampling
fig, ax = plt.subplots()
ax.plot(x_plot.detach().numpy(), net.forward(x_plot).detach().numpy())
ax.scatter(x.numpy(), y.numpy())
ax.scatter(x_test.numpy(), y_test.numpy())
ax.set_title('log_likelihood = %.5f' % net.log_likelihood(x,y))
fig.savefig(os.path.join(args.dir_out, 'output/initial_function.png'))

plt.close('all')

## fit

# Burn-in
writer = SummaryWriter(os.path.join(args.dir_out, 'log/burnin'))
_, _, eps_adjust = net.sample_posterior(x=x, y=y, n_samp=args.burnin_n_samp, x_plot=x_plot, eps=args.burnin_eps, n_adapt_eps=args.burnin_n_adapt_eps, \
                                        n_rep_hmc=args.burnin_n_rep_hmc, n_rep_resize=0, n_bigwrite=args.burnin_n_bigwrite, writer=writer, record=True, n_print=5)
writer.close()

# Burn-in
writer = SummaryWriter(os.path.join(args.dir_out, 'log/burnin_resize'))
_, _, eps_adjust = net.sample_posterior(x=x, y=y, n_samp=args.burnin_n_samp, x_plot=x_plot, eps=eps_adjust, n_adapt_eps=args.burnin_n_adapt_eps, \
                                        n_rep_hmc=args.burnin_n_rep_hmc, n_rep_resize=args.burnin_resize_n_rep_resize, n_bigwrite=args.burnin_n_bigwrite, writer=writer, record=True, n_print=5)
writer.close()


# Samples
writer = SummaryWriter(os.path.join(args.dir_out, 'log/samples'))
accept, samples, _ = net.sample_posterior(x=x, y=y, n_samp=args.record_n_samp, x_plot=x_plot, x_test=x_test, y_test=y_test, \
                                          eps=eps_adjust, n_print=20, n_rep_hmc=args.record_n_rep_hmc, n_rep_resize=args.record_n_rep_resize, \
                                          n_bigwrite=args.record_n_bigwrite, writer=writer, record=True)
writer.close()

## save
np.save(os.path.join(args.dir_out, 'output/samples.npy'), samples)
np.save(os.path.join(args.dir_out, 'output/accept.npy'), accept)

print("--- %s seconds elapsed ---" % (time.time() - start_time))
with open(os.path.join(args.dir_out, 'program_info.txt'), 'a') as f:
    f.write("seconds elapsed: %s\n" % (time.time() - start_time))


