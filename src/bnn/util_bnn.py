import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

def log_normal(x, sig2):
    return -0.5*x.nelement()*(math.log(2*math.pi)) - 0.5*x.nelement()*torch.log(sig2) - torch.sum(x**2)/(2*sig2)

## data loading
def format_data_torch(x, y, x_test=None, y_test=None):
    x = torch.from_numpy(x).reshape(-1,1).to(torch.get_default_dtype())
    y = torch.from_numpy(y).reshape(-1,1).to(torch.get_default_dtype())

    if x_test is not None and y_test is not None:
        x_test = torch.from_numpy(x_test).reshape(-1,1).to(torch.get_default_dtype())
        y_test = torch.from_numpy(y_test).reshape(-1,1).to(torch.get_default_dtype())

    return x, y, x_test, y_test

def load_two_rbfs(n_obs=50, c_1=-1, c_2=1, s2_1=4, s2_2=4, lb=-2, ub=2, sig2=1e-5, seed=0):
    np.random.seed(seed)

    x = np.linspace(lb,ub,n_obs)
    f = lambda x: 0.5*np.exp(-0.5*s2_1*(x-c_1)**2) + 0.5*np.exp(-0.5*s2_2*(x-c_2)**2)
    y = f(x) + np.random.normal(0,np.sqrt(sig2),x.shape)

    return torch.from_numpy(x).reshape(-1,1).float(), torch.from_numpy(y).reshape(-1,1).float()

def test_log_likelihood(y, y_pred, sig2):
    'assumes normal likelihood function'
    norm = torch.distributions.normal.Normal(loc = y.reshape(1,-1), scale = sig2.reshape(-1,1).sqrt())
    prob = norm.log_prob(y_pred).exp()
    return torch.log(torch.prod(torch.mean(prob, dim=0)))

## plotting
def plot_functions(x_pred, y_pred, x=None, y=None, x_test=None, y_test=None, sig2=None, plot_all=False):
    x_pred_np = x_pred.numpy()
    y_pred_np = y_pred.detach().numpy()

    ci = np.percentile(y_pred_np, [5, 95], axis=0)

    fig, ax = plt.subplots()
    if x is not None and y is not None:
        ax.scatter(x, y)
        pass

    ax.plot(x_pred_np, np.mean(y_pred_np,0))
    ax.fill_between(x_pred_np.ravel(), ci[0,:], ci[1,:],alpha=.5)

    if x_test is not None and y_test is not None:
        ax.scatter(x_test, y_test)

    if sig2 is not None:
        ax.fill_between(x_pred_np.ravel(), ci[0,:]-np.sqrt(sig2), ci[1,:]+np.sqrt(sig2), alpha=.25)

    if plot_all:
        ax.plot(x_pred_np, y_pred_np.T,alpha=.05, color='red')

    return fig, ax

def plot_prior_predictive(x_pred, f_pred, upcross_level=0, bins=20, plot_all_functions=False):
    '''
    x_pred: (n_gridpoints,) numpy array
    f_pred: (n_samples, n_gridpoints) numpy array of function samples
    '''
    f_pred_mean = np.mean(f_pred,0)
    f_pred_std = np.std(f_pred,0)

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10,4))

    # amp. variance
    ax[0].plot(x_pred, f_pred_mean)
    ax[0].fill_between(x_pred.ravel(), f_pred_mean-f_pred_std, f_pred_mean+f_pred_std, alpha=.5)
    ax[0].set_title('amplitude variance')

    if plot_all_functions:
        ax[0].plot(x_pred, f_pred.T, alpha=.1, color='red')

    # upcrossings
    u = upcross_level*np.ones(x_pred.shape[0])
    up = np.logical_and(f_pred[:,:-1]<u[:-1], f_pred[:,1:]>u[1:])
    idx_up = [np.where(row)[0] for row in up]
    x_up = x_pred.ravel()[np.concatenate(idx_up)]
    ax[1].hist(x_up, bins=bins)
    ax[1].set_title('upcrossings --- total = %.3f' % np.sum(up))
    
    return fig, ax

## other stuff
def reset_args_matched_prior(args):

    df = pd.read_csv(args.prior_matching_file)
    if args.dataset_option=="None":
        row = df.loc[(df['dataset']==args.dataset) & (df['model']=='bnn'),:]
    else:
        row = df.loc[(df['dataset']==args.dataset) & (df['dataset_option']==args.dataset_option) & (df['model']=='bnn'),:]

    setattr(args, 'prior_b1_sig2', row['param1'].item())
    setattr(args, 'prior_w1_sig2', row['param1'].item())
    setattr(args, 'prior_b2_sig2', row['param2'].item())
    setattr(args, 'prior_w2_sig2', row['param2'].item())
    setattr(args, 'sig2', row['sig2'].item())

def rescale_eps(eps, accept, accept_target=.65, eps_min=1e-6):
    'Rescaling of step size'
    scale = 1 + 10*np.sign(accept - accept_target)*(accept - accept_target)**2
    scale = np.log(1+np.exp(scale)) + (1-np.log(1+np.exp(1)))
    return max(scale*eps, eps_min)