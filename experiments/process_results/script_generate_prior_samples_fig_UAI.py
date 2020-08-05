
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from distutils.dir_util import mkpath
import torch
torch.set_default_dtype(torch.float64)
import GPy

import ipdb

import sys

repo_path = '../..'
sys.path.append(repo_path+'/src/doucet/')
print(repo_path+'/src/doucet/')
from rbfn import RBFN

sys.path.append(repo_path+'/src/porbnet/')
from networks_porbnet import RBFN as PPRBFN
from util_porbnet import Piecewise

sys.path.append(repo_path+'/src/bnn/')
from networks_bnn import BNN


dir_process_results = os.path.join(repo_path,'experiments/process_results/')
sys.path.append(dir_process_results)
from util import plot_posterior_predictive

# remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# fonts
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'legend.fontsize': 6})
plt.rcParams.update({'axes.labelsize': 6})
plt.rcParams.update({'axes.titlesize': 8})

OUTPUT_PATH = os.path.join('output/figure_prior_samples/')

n_methods = 4
fig, ax = plt.subplots(3,n_methods, figsize=((3*n_methods,5.8)), sharex=True, sharey=True)
x_grid = np.linspace(-1.0,1.0,250)[:,np.newaxis]
n_samples = 50
n_samples_all = 10000
alpha=0.1
diff_colors=False

np.random.seed(20)
torch.manual_seed(20)

def compute_var_lines(ax, x_pred, f_pred):
    var_value = np.var(np.squeeze(f_pred),axis=0) # (n_samples,)
    ax.plot(x_pred,var_value,'k:')
    ax.plot(x_pred,-1.0*var_value,'k:')
    return np.mean(var_value)

###### PRIOR SAMPLES FROM BNN
i=3
bnn = BNN(1,10,1,sig2=0.01,prior_w1_sig2=5.0,prior_b2_sig2=0.001,\
        prior_w2_sig2=0.2,prior_b1_sig2=1.0)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = bnn.sample_functions_prior(x_tensor, n_samples)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[0,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
ax[0,i].set_title('BNN')
f_pred = bnn.sample_functions_prior(x_tensor, n_samples_all)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[0,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('bnn',var_val))

bnn = BNN(1,15,1,sig2=0.01,prior_w1_sig2=100.0,prior_b2_sig2=0.001,\
        prior_w2_sig2=0.2,prior_b1_sig2=1.0)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = bnn.sample_functions_prior(x_tensor, n_samples)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[1,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = bnn.sample_functions_prior(x_tensor, n_samples_all)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[1,i], x_grid, f_pred)
print('model=%s, lambda=high, var=%.3f' % ('bnn',var_val))

bnn = BNN(1,10,1,sig2=0.01,prior_w1_sig2=5.0,prior_b2_sig2=0.001,\
        prior_w2_sig2=0.4,prior_b1_sig2=1.0)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = bnn.sample_functions_prior(x_tensor, n_samples)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[2,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = bnn.sample_functions_prior(x_tensor, n_samples_all)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[2,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('bnn',var_val))

###### Prior samples from Doucet
x = np.random.randn(200,1)
y = np.random.randn(200,1)
print(x.shape)
print(y.shape)
k = 20

i=2
rbfn = RBFN(x,y,k,eps1=5.0,eps2=0.5,l=1.0,nu_0=1.5, gamma_0=1)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples, sample_K=True)
f_pred = f_pred[:,:,np.newaxis]
plot_posterior_predictive(x_grid, f_pred, ax=ax[0,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis]
var_val = compute_var_lines(ax[0,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('doucet',var_val))
ax[0,i].set_title('B-RBFN')

rbfn = RBFN(x,y,k,eps1=30.0,eps2=0.5,l=15.0,nu_0=1.5, gamma_0=1)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples, sample_K=True)[:,:,np.newaxis]
plot_posterior_predictive(x_grid, f_pred, ax=ax[1,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis]
var_val = compute_var_lines(ax[1,i], x_grid, f_pred)
print('model=%s, lambda=high, var=%.3f' % ('doucet',var_val))

#rbfn = RBFN(x,y,k,eps1=p_low[0],eps2=1.0,l=p_low[1],nu_0=1.5, gamma_0=1,\
#        alpha_del2=p_high_var, beta_del2=beta_del2)
rbfn = RBFN(x,y,k,eps1=5.0,eps2=0.5,l=200.0,nu_0=1.5, gamma_0=1)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples, sample_K=True)
f_pred = f_pred[:,:,np.newaxis]
plot_posterior_predictive(x_grid, f_pred, ax=ax[2,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = rbfn.sample_functions_prior(x_grid, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis]
var_val = compute_var_lines(ax[2,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('doucet',var_val))

###### Prior samples from PP-rbBNN
i=0
low_intensity = Piecewise(np.array([-5,5]),np.array([4.0]))
pprbfn = PPRBFN(sig2=0.01,intensity=low_intensity,s2_0=1.0, prior_b_sig2=0.01,\
        prior_w_sig2=0.7)
#pprbfn = PPRBFN(1,30,400,1,low_intensity, 1.0, sig2=0.01, prior_b_sig2=0.01,\
#        prior_w_sig2=0.7)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[0,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
ax[0,i].set_title('PoRB-Net') #PP-rbBNN')
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[0,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('pprbfn',var_val))

high_intensity = Piecewise(np.array([-5,5]),np.array([20.0]))
pprbfn = PPRBFN(sig2=0.01, intensity=high_intensity,s2_0=2.0, prior_b_sig2=0.01,\
        prior_w_sig2=0.7)
#pprbfn = PPRBFN(1,30,400,1,high_intensity, 2.0, prior_b_sig2=0.01,\
#        prior_w_sig2=0.7)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[1,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[1,i], x_grid, f_pred)
print('model=%s, lambda=high, var=%.3f' % ('pprbfn',var_val))

low_intensity = Piecewise(np.array([-5,5]),np.array([4.0]))
pprbfn = PPRBFN(sig2=0.01,intensity=low_intensity,s2_0=1.0, prior_b_sig2=0.01,\
        prior_w_sig2=1.4)
#pprbfn = PPRBFN(1,30,400,1,low_intensity, 1.0, sig2=0.01, prior_b_sig2=0.01,\
#        prior_w_sig2=0.7)
x_tensor = torch.from_numpy(x_grid).type('torch.DoubleTensor')
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
plot_posterior_predictive(x_grid, f_pred, ax=ax[2,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = pprbfn.sample_functions_prior(x_tensor, n_samples_all, sample_K=True)
f_pred = f_pred[:,:,np.newaxis].detach().numpy()
var_val = compute_var_lines(ax[2,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('pprbfn',var_val))

#for j in range(2):
    #for j in range(2):
    #i=3
    #ax[j,i].set_xlabel('x')

#ipdb.set_trace()

#for i in range(n_methods):
#    j=1
#    ax[i,j].set_ylabel('y')

###### PRIOR SAMPLES FROM GP
i=1

def get_function_samples(kernel, n_samples=100,xlims=[-5,5], x_grid=None,\
        hacknr=1.0):
    x=np.random.randn(1,1)
    if x_grid is None:
        x_grid = np.linspace(xlims[0],xlims[1],200)[:,np.newaxis]
    f_pred = np.zeros((n_samples,x_grid.shape[0],1))
    for r in range(n_samples):
        #y=2.3*np.random.rand(1,1)
        y=hacknr*np.random.randn(1,1)
        m = GPy.models.GPRegression(x,y,kernel)
        sample = m.posterior_samples_f(x_grid,size=1)
        f_pred[r,:,0] = np.squeeze(sample)
    return f_pred

kernel = GPy.kern.RBF(input_dim=1, variance=0.7, lengthscale=0.4)
f_pred = get_function_samples(kernel, n_samples=n_samples,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.15)
#f_pred = np.moveaxis(f_pred, [0, 1, 2], [1,2,0])
plot_posterior_predictive(x_grid, f_pred, ax=ax[0,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
ax[0,i].set_title('GP')
f_pred = get_function_samples(kernel, n_samples=n_samples_all,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.15)
#f_pred = np.moveaxis(f_pred, [0, 1, 2], [1,2,0])
var_val = compute_var_lines(ax[0,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('gp',var_val))

kernel = GPy.kern.RBF(input_dim=1, variance=0.7, lengthscale=0.05)
f_pred = get_function_samples(kernel, n_samples=n_samples,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.15)
plot_posterior_predictive(x_grid, f_pred, ax=ax[1,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = get_function_samples(kernel, n_samples=n_samples_all,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.15)
var_val = compute_var_lines(ax[1,i], x_grid, f_pred)
print('model=%s, lambda=high, var=%.3f' % ('gp',var_val))

kernel = GPy.kern.RBF(input_dim=1, variance=1.4, lengthscale=0.4)
f_pred = get_function_samples(kernel, n_samples=n_samples,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.6)
#f_pred = np.moveaxis(f_pred, [0, 1, 2], [1,2,0])
plot_posterior_predictive(x_grid, f_pred, ax=ax[2,i],alpha=alpha,\
        plot_all=True,plot_uncertainty=False,diff_colors=diff_colors)
f_pred = get_function_samples(kernel, n_samples=n_samples_all,xlims=[0,1.0],\
        x_grid=x_grid, hacknr=1.6)
#f_pred = np.moveaxis(f_pred, [0, 1, 2], [1,2,0])
var_val = compute_var_lines(ax[2,i], x_grid, f_pred)
print('model=%s, lambda=low, var=%.3f' % ('gp',var_val))

for i in range(4):
    ax[2,i].set_xlabel(r'$x$')
    for j in range(3):
        #ax[j,i].grid()
        ax[j,i].grid(linewidth='0.25',alpha=.5)
        ax[j,i].set_ylim([-3.0,3.0])
plt.show()

# legend for top left plot only
from matplotlib.lines import Line2D
colors = ['black','tab:red']
linestyle = [':','-']
lines = [Line2D([0], [0], color=c, linewidth=1, linestyle=l) for c,l in zip(colors,linestyle)]
labels = [r'$\mathbb{V}\ [f(x)]$',r'$f(x)$']
ax[0,0].legend(lines, labels, loc='upper left')

fig.set_size_inches(6.62, 3.4)
fig.tight_layout()

mkpath(OUTPUT_PATH)
filename = os.path.join(OUTPUT_PATH,'samples_prior.png')
plt.savefig(filename, bbox_inches='tight')
filename = os.path.join(OUTPUT_PATH,'samples_prior.pdf')
plt.savefig(filename, bbox_inches='tight',pad_inches=.025)
print('SAVED FIG: %s' % filename)
#ipdb.set_trace()
plt.close()
