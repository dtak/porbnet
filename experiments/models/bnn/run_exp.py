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
parser.add_argument('--dir_out', type=str, default='./output', help='directory of output')

parser.add_argument('--validation_mode', action='store_true', help='flag for whether or not to split off validation set from training set (and use as test set)')

# prior
parser.add_argument('--dim_hidden', type=int, default=30, help='number of hidden units')
parser.add_argument('--prior_w1_sig2', type=float, default=150., help='layer 1 weights prior variance')
parser.add_argument('--prior_b1_sig2', type=float, default=75., help='layer 1 bias prior variance')
parser.add_argument('--prior_w2_sig2', type=float, default=.75, help='layer 2 weights prior variance')
parser.add_argument('--prior_b2_sig2', type=float, default=.75, help='layer 2 bias prior variance')
parser.add_argument('--sig2', type=float, default=0.0356634620928351, help='observation noise variance')
parser.add_argument('--s2_0', type=float, default=1.0)

# posterior samples
parser.add_argument('--burnin_n_samp', type=int, default=5000, help='number of samples (burn-in)')
parser.add_argument('--burnin_eps', type=float, default=0.00001, help='initial hmc step size (burn-in)')
parser.add_argument('--burnin_n_adapt_eps', type=int, default=200, help='number of times to adjust hmc step size (burn-in)')
parser.add_argument('--burnin_n_rep_hmc', type=int, default=1, help='number of hmc replications per loop (burn-in)')
parser.add_argument('--burnin_n_bigwrite', type=int, default=5, help='number of times to plot in tensorboard (burn-in)')

parser.add_argument('--record_n_samp', type=int, default=5000, help='number of samples (record)')
parser.add_argument('--record_n_rep_hmc', type=int, default=1, help='number of hmc replications per loop (record)')
parser.add_argument('--record_n_bigwrite', type=int, default=500, help='number of times to plot in tensorboard (record)')

parser.add_argument('--prior_matching_file', type=str, default=None)

args = parser.parse_args()

dir_src = os.path.join(args.dir_base, 'src/bnn/')
dir_data = os.path.join(args.dir_base, 'data/')

sys.path.append(dir_src)
sys.path.append(dir_data)
sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import networks_bnn as networks
import util_bnn as util
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
net = networks.BNN(dim_in=1, dim_hidden=args.dim_hidden, dim_out=1, \
                    prior_w1_sig2 = args.prior_w1_sig2, prior_b1_sig2 = args.prior_b1_sig2, \
                    prior_w2_sig2 = args.prior_w2_sig2, prior_b2_sig2 = args.prior_b2_sig2, \
                    sig2=args.sig2, prior_sig2_alpha = None, prior_sig2_beta = None, s2_0=args.s2_0)

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

# center around zero
x = x - 0.5
x_test = x_test - 0.5

## check prior
x_plot = torch.linspace(-0.5, 0.5, 100).reshape(-1,1)

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

## fit

# Burn-in
writer = SummaryWriter(os.path.join(args.dir_out, 'log/burnin'))
_, _, eps_adjust = net.sample_posterior(x=x, y=y, n_samp=args.burnin_n_samp, x_plot=x_plot, eps=args.burnin_eps, n_adapt_eps=args.burnin_n_adapt_eps, \
                                        n_rep_hmc=args.burnin_n_rep_hmc, n_bigwrite=args.burnin_n_bigwrite, writer=writer, record=True, n_print=5)
writer.close()


# Samples
writer = SummaryWriter(os.path.join(args.dir_out, 'log/samples'))
accept, samples, _ = net.sample_posterior(x=x, y=y, n_samp=args.record_n_samp, x_plot=x_plot, x_test=x_test, y_test=y_test, \
                                          eps=eps_adjust, n_print=20, n_rep_hmc=args.record_n_rep_hmc, \
                                          n_bigwrite=args.record_n_bigwrite, writer=writer, record=True)
writer.close()


## save
np.save(os.path.join(args.dir_out, 'output/samples.npy'), samples)
np.save(os.path.join(args.dir_out, 'output/accept.npy'), accept)

print("--- %s seconds elapsed ---" % (time.time() - start_time))
with open(os.path.join(args.dir_out, 'program_info.txt'), 'a') as f:
    f.write("seconds elapsed: %s\n" % (time.time() - start_time))


