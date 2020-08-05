import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
#from scipy.io import loadmat
from util import (
        plot_posterior_predictive,
        compute_rmse,
        compute_loglik,
        compute_loglik2,
        generate_latex_table,
        sigmoid)
from distutils.dir_util import mkpath

import ipdb
from scipy import io

from util import plot_posterior_predictive, plot_intensity

# remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def standardize(x, y, x_test=None, y_test=None):
    # rescale
    xbias = x.min()
    xscale = x.max()-x.min()
    yscale = (np.abs(y-y.mean())).max()
    ybias = y.mean()
    x = (x - xbias) / xscale
    y = (y - ybias) / yscale

    if x_test is not None and y_test is not None:
        x_test = (x_test - xbias) / xscale
        y_test = (y_test - ybias) / yscale
        return x, y, x_test, y_test
    else:
        return x, y

"""
def plot_posterior_predictive(
        x_pred, y_pred, x=None, y=None, ax=None, alpha=0.05,
        x_test=None, y_test=None, sig2=None, plot_all=False,
        plot_uncertainty=True, diff_colors=False, s=36):
    ''' input: numpy arrays 
        x_pred: (n_pts,)
        y_pred: (n_smaples,n_pts,1)
    '''
    ci = np.percentile(y_pred, [5, 95], axis=0)

    if ax is None: fig, ax = plt.subplots()
    if x is not None and y is not None: # train data
        ax.scatter(x.ravel(), y.ravel(), label='train',s=s)
        pass

    if x_test is not None and y_test is not None:
        ax.scatter(x_test.ravel(), y_test.ravel(), label='test',s=s)

    if plot_uncertainty:
        ax.plot(x_pred, np.mean(y_pred,0), label='post. mean')
        ax.fill_between(x_pred.ravel(), ci[0,:,0], ci[1,:,0],alpha=.5,label='90% quantile')
        if sig2 is not None:
            ax.fill_between(x_pred.ravel(), ci[0,:,0]-np.sqrt(sig2),
                    ci[1,:,0]+np.sqrt(sig2), alpha=.25, label=r'$\sigma^2$')



    if plot_all: # plot samples
        if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred)
        if diff_colors:
            ax.plot(x_pred, y_pred.T,alpha=alpha)
        else:
            ax.plot(x_pred, y_pred.T,alpha=alpha, color='red')

    return ax

def plot_functions(x_pred, y_pred, x=None, y=None, x_test=None, y_test=None, sig2=None, plot_all=False, ax=None):
    x_pred_np = x_pred.numpy()
    y_pred_np = y_pred.detach().numpy()

    ci = np.percentile(y_pred_np, [5, 95], axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    
    if x is not None and y is not None:
        ax.scatter(x, y, label='training')
        pass

    ax.plot(x_pred_np, np.mean(y_pred_np,0))
    ax.fill_between(x_pred_np.ravel(), ci[0,:], ci[1,:],alpha=.5)

    if x_test is not None and y_test is not None:
        ax.scatter(x_test, y_test, label='test')

    if sig2 is not None:
        ax.fill_between(x_pred_np.ravel(), ci[0,:]-np.sqrt(sig2), ci[1,:]+np.sqrt(sig2), alpha=.25)

    if plot_all:
        ax.plot(x_pred_np, y_pred_np.T,alpha=.05, color='red')

    return ax
"""


if __name__ == "__main__":

    plt.rcParams.update({'font.size': 5})
    plt.rcParams.update({'axes.labelsize': 7})
    plt.rcParams.update({'axes.titlesize': 7})

    path = '../models/porbnet-sgcp/s2_0=2/GPdata/sin/60484367/task=5/output/samples.npy'

    samples = np.load(path, allow_pickle=True).item()

    DIR_OUT = 'output/'
    mkpath(DIR_OUT)

    i_max_avail = np.nonzero(samples['sig2'].numpy()!=0)[0][-1]
    i_max = samples['sig2'].shape[0]
    if i_max_avail < i_max:
        print('WARNING: only %d samples of %d are found' % (i_max_avail, i_max)) 
        i_max = i_max_avail
    x_plot = samples['x_plot'][:i_max_avail, :].numpy()
    y_plot = samples['y_plot_pred'].numpy()[:i_max_avail,:,np.newaxis]
    x = samples['x'].numpy()
    y = np.squeeze(samples['y'].numpy())
    x_test = samples['x_test'].numpy()
    y_test = np.squeeze(samples['y_test'].numpy())
    sig2 = np.mean(samples['sig2'][:i_max_avail].numpy())
    y_test_pred = samples['y_test_pred'][:i_max_avail,:].detach().numpy()


    rate_density_ub = samples['rate_density_ub'][:i_max_avail].numpy()
    gp_plot_samp = samples['gp_plot_pred'][:i_max_avail,:].numpy()


    rmse = compute_rmse(y_test, y_test_pred)
    loglik = compute_loglik2(y_test, y_test_pred, sig2)

    fig, ax = plt.subplots(1,2)

    ax[0].set_title('Function')
    plot_posterior_predictive(x_plot, y_plot, ax=ax[0], x=x,y=y, x_test=x_test, y_test=y_test, sig2=sig2, s=1)
    ax[0].set_xlabel(r'$x$')

    
    ax[0].legend()
    ax[0].set_xlabel(r'$x$')
    ax[0].set_yticks(np.array([-.5, 0, .5, 1]))

    plot_intensity(x_plot, gp_plot_samp, rate_density_ub.reshape(-1,1),ax[1], true_ls_name='sin')
    ax[1].set_title('Intensity')
    ax[1].set_xlabel(r'$x$')

    '''
    dataset_option = 'sin'
    if dataset_option == 'sin':
        #length_scale = lambda z: np.sin(z) + 1.1

        xbias = -4.9189
        xscale = 9.6491
        length_scale_orig = lambda z: np.sin(z) + 1.1
        length_scale = lambda z: length_scale_orig(z*xscale + xbias) / xscale

    elif dataset_option in ['inc','inc_gap']:
        def length_scale(x, ls_min=.25, ls_max=2, x_min=-5, x_max=2):
            slope = (ls_max - ls_min) / (x_max - x_min) 
            ls = (x - x_min)*slope + ls_min
            return ls**2


    #true_intensity_x = np.linspace(-5,5,100)
    true_intensity_x = x_plot
    true_intensity_y = 1/(length_scale(true_intensity_x))
    true_intensity_x, _ = standardize(true_intensity_x, true_intensity_x)

    
    
    ax2 = ax[1].twinx()
    color = 'tab:orange'
    ax2.set_ylabel('true inverse lengthscale', color=color, labelpad=1)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(true_intensity_x, true_intensity_y, '--', color=color, label=r'$1/l(x)$', linewidth=.5)
    #ax2.legend()
    ax[1].set_xlabel(r'$x$')

    color = 'tab:purple'
    ax[1].set_ylabel('inferred intensity', color=color, labelpad=1)
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].set_title('Intensity')
    '''


    plt.rcParams.update({'font.size': 3})
    plt.rcParams.update({'legend.fontsize': 5})
    plt.rcParams.update({'axes.labelsize': 4})
    plt.rcParams.update({'axes.titlesize': 8})
    fig.set_size_inches(3.28, 1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_OUT,'fig4_GPdata.pdf'), bbox_inches='tight', pad_inches=.025)
    fig.savefig(os.path.join(DIR_OUT,'fig4_GPdata.png'), bbox_inches='tight', pad_inches=.025)

