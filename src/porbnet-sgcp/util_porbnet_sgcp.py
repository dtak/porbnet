import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

def sigmoid(z):
    return (1+np.exp(-z))**(-1)

def sigmoid_torch(z):
    return (1+torch.exp(-z))**(-1)

## data loading
def format_data_torch(x, y, x_test=None, y_test=None):
    x = torch.from_numpy(x).reshape(-1,1).to(torch.get_default_dtype())
    y = torch.from_numpy(y).reshape(-1,1).to(torch.get_default_dtype())

    if x_test is not None and y_test is not None:
        x_test = torch.from_numpy(x_test).reshape(-1,1).to(torch.get_default_dtype())
        y_test = torch.from_numpy(y_test).reshape(-1,1).to(torch.get_default_dtype())

    return x, y, x_test, y_test

def load_mimic(dir_data):
    x = np.genfromtxt(dir_data + '/X_small.csv', delimiter=',', skip_header=False).reshape(-1,1)
    y = np.genfromtxt(dir_data + '/y_small.csv', delimiter=',', skip_header=False).reshape(-1,1)
    y = y - np.mean(y)

    return torch.from_numpy(x), torch.from_numpy(y)

def load_from_bnn(dir_data, alpha, c_grid):
    dir_data = os.path.join(dir_data,'from_bnn/alpha=%s-c_grid=%d/' % (alpha, c_grid))

    x = torch.from_numpy(np.genfromtxt(os.path.join(dir_data, 'x.csv'), delimiter=',', skip_header=False).reshape(-1,1)).float()
    y = torch.from_numpy(np.genfromtxt(os.path.join(dir_data, 'y.csv'), delimiter=',', skip_header=False).reshape(-1,1)).float()
    x_val = torch.from_numpy(np.genfromtxt(os.path.join(dir_data, 'x_val.csv'), delimiter=',', skip_header=False).reshape(-1,1)).float()
    y_val = torch.from_numpy(np.genfromtxt(os.path.join(dir_data, 'y_val.csv'), delimiter=',', skip_header=False).reshape(-1,1)).float()

    return x, y, x_val, y_val

def load_two_rbfs(n_obs=50, c_1=-1, c_2=1, s2_1=4, s2_2=4, lb=-2, ub=2, sig2=1e-5, seed=0):
    np.random.seed(seed)

    x = np.linspace(lb,ub,n_obs)
    f = lambda x: 0.5*np.exp(-0.5*s2_1*(x-c_1)**2) + 0.5*np.exp(-0.5*s2_2*(x-c_2)**2)
    y = f(x) + np.random.normal(0,np.sqrt(sig2),x.shape)

    return torch.from_numpy(x).reshape(-1,1).to(torch.get_default_dtype()), torch.from_numpy(y).reshape(-1,1).to(torch.get_default_dtype())



## Some numpy code I haven't replaced with torch code
def integrate_piecewise(breaks,values):
    return sum([(b-a)*v for a,b,v in zip(breaks[:-1], breaks[1:], values)])

def piecewise_invcdf(z, breaks, values, normalize=False):
    if normalize:
        Z = integrate_piecewise(breaks,values)
        values = [v/Z for v in values]

    x = np.array(breaks)
    y = (x[1:]-x[:-1])*np.array(values)
    y = np.concatenate([np.array([0]),np.cumsum(y)])
    flist = [lambda s,x=x,y=y,m=m: 1/m*(s-y) + x for x,y,m in zip(x,y,values)]
    condlist = [np.logical_and(z>=a,z<b) for a,b in zip(y[:-1], y[1:])]
    return(np.piecewise(z, condlist, flist))  


## distributions
def log_gamma(x, alpha, beta):
    return x.nelement()*(alpha*torch.log(beta) - torch.lgamma(alpha)) + (alpha-1)*torch.sum(torch.log(x)) - beta*torch.sum(x)

def log_normal(x, sig2):
    return -0.5*x.nelement()*(math.log(2*math.pi)) - 0.5*x.nelement()*torch.log(sig2) - torch.sum(x**2)/(2*sig2)

def log_poisson_process(x, intensity):
    return -intensity.integral + torch.sum(intensity.log(x))

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

def plot_intensity(x_plot, gp_plot_pred, rate_density_ub, x):
    '''
    x_plot: points where intensity was evaluated (n_plot,1)
    gp_plot_pred: samples of gp (n_samp, n_plot)
    rate_density_ub:
    x: observed x data (n_obs, 1)
    '''
    post_rate_density  = rate_density_ub*sigmoid(gp_plot_pred)
    post_ci = np.percentile(post_rate_density, [5,95],axis=0)
    post_mean = np.mean(post_rate_density, axis=0)
    fig,ax = plt.subplots()
    ax.plot(x,np.zeros(x.shape[0]),'+k')
    ax.plot(x_plot, post_mean,'b')
    ax.fill_between(x_plot.reshape(-1), post_ci[0,:], post_ci[1,:], alpha=.3, color='b')
    ax.legend(['data','posterior mean','posterior 90% quantile'])
    ax.set_title('Posterior intensity')
    return fig, ax

def plot_centers_trace(samples_centers, samples_mask):
    '''
    samples_centers: (n_samp, n_centers)
    samles_mask: (n_samp, n_centers)
    '''

    samples_centers = np.squeeze(samples_centers)
    fig,ax = plt.subplots(figsize=(7,10))
    idxChange = np.any(samples_mask,axis=0)

    y= samples_centers[:,idxChange]
    ym = np.ma.masked_where(np.logical_not(samples_mask[:,idxChange]),y)

    plt.gca().set_prop_cycle(None)
    ax.plot(y, alpha=.3)

    plt.gca().set_prop_cycle(None)
    ax.plot(ym)
    ax.set_xlabel('Sample index')
    
    return fig, ax

## other stuff
def reset_args_matched_prior(args):

    df = pd.read_csv(args.prior_matching_file)
    if args.dataset_option=="None":
        row = df.loc[(df['dataset']==args.dataset) & (df['model']=='porbnet_sgcp'),:]
    else:
        row = df.loc[(df['dataset']==args.dataset) & (df['dataset_option']==args.dataset_option) & (df['model']=='porbnet_sgcp'),:]

    # param1
    if row['param1_name'].item() == 'rate_density_ub':
        rate_density_ub = row['param1'].item()
        setattr(args, 'rate_density_ub', rate_density_ub)
        
    elif row['param1_name'].item() == 's2_0':
        setattr(args, 's2_0', row['param1'].item())
        
    # param2
    setattr(args, 'prior_b_sig2', row['param2'].item())
    setattr(args, 'prior_w_sig2', row['param2'].item()*np.sqrt(np.pi/args.s2_0))

    setattr(args, 'sig2', row['sig2'].item())

    T=[-.25,1.25]
    dim_hidden_initial=int((T[1]-T[0])*(0.5*rate_density_ub))
    dim_hidden_max = np.maximum(2*dim_hidden_initial, 75)
    setattr(args, 'dim_hidden_initial', dim_hidden_initial)
    setattr(args, 'dim_hidden_max', dim_hidden_max)

def rescale_eps(eps, accept, accept_target=.65, eps_min=1e-6):
    'Rescaling of step size'
    scale = 1 + 10*np.sign(accept - accept_target)*(accept - accept_target)**2
    scale = np.log(1+np.exp(scale)) + (1-np.log(1+np.exp(1)))
    return (eps + max(scale*eps, eps_min))/2.0

class PiecewisePost(object):
    def __init__(self, x, samples):
        self.fit(samples)
        self.xmin = x.min()
        self.dx = x[1] - x[0] # assumes uniformely spaced samples
        
    def fit(self, samples):
        self.y = torch.mean(samples, dim=0)
        
    def __call__(self, xnew):
        return self.y[torch.floor((xnew-self.xmin) / self.dx).long()]

class Piecewise(object):
    def __init__(self, breaks, values, dx):
        '''
        Piecewise constant function class. 
        Records function as map x -> y, with dx distance between each x.
        
        breaks: list of function breaks (length B)
        values: list of function values between breaks (length B-1)
        dx: distance between each x value in map. Should divide break points. 
        '''
        self.breaks = breaks
        self.values = values
        self.dx = dx
    
        self.x = torch.arange(breaks[0], breaks[-1] + dx, dx)
        self.y = torch.empty(self.x.shape).float()
        
        for i in range(len(values)):
            self.y[(self.x>=breaks[i]) & (self.x<=breaks[i+1])] = values[i]
            
        self.log_y = torch.log(self.y)
        
        self.integral = torch.sum(torch.tensor([(breaks[i+1] - breaks[i])*values[i] for i in range(len(values))]))
   
        self.values_pdf = [v/self.integral.item() for v in values]
        

    def __call__(self, x, check_bounds=True):
        return self.y[torch.floor((x-self.breaks[0]) / self.dx).long()]
    
    def log(self, x, check_bounds=True):
        return self.log_y[torch.floor((x-self.breaks[0]) / self.dx).long()]
    
    def check_bounds(self,x):
        return False if torch.any((x < self.breaks[0]) | (x > self.breaks[-1])) else True
    
    def sample_conditional(self, n_samp):
        '''
        Samples from dist. with normalized piecewise constant function as pdf
        Uses numpy code but returns tensor
        '''
        return torch.as_tensor(piecewise_invcdf(np.random.uniform(size=n_samp),self.breaks,self.values_pdf)).float()
        
    def sample_pp(self):
        '''
        Samples from a poisson process using piecewise constant function as intensity
        Uses numpy code but returns tensor
        '''
        K = np.random.poisson(self.integral, size=1)
        return self.sample_conditional(K)

def sample_sgcp(T, gp, rate_density_ub):
    """
    T: window size
    gp: function that can sample from gp
    rate_density_ub: max intensity value

    returns sK, sM, gK, gM
    sK = unthinned events
    sM = thinned events
    gK = unthinned gp values
    gM = thinned gp values
    """
    V = T[1]-T[0]        
    poisson = torch.distributions.poisson.Poisson(torch.tensor([V*rate_density_ub])) # sample total number of events
    J = int(poisson.sample().item())
    
    # Repeat until you get nonzero total number of events. Is this ok?
    retry = 0
    max_retry = 100
    while J==0 and retry<max_retry:
        print('J=0 events sampled in sample_pp, retrying [%d/%d]' % (retry,max_retry))
        J = int(poisson.sample().item())
        retry+=1

    shat = torch.empty(J).uniform_(T[0],T[1])   # sample all locations
    r = torch.empty(J).uniform_(0,1)            # sample vertical aux. vars

    # Evaluate intensity function
    gphat = gp(shat)
    rate_density_eval = torch.sigmoid(gphat)

    # Accept subset of locations
    idx_K = r < rate_density_eval
    
    # return thinned and unthinned events and gp values
    return shat[idx_K], shat[~idx_K], gphat[idx_K], gphat[~idx_K]

def dataset_from_rbfn(rbfn, n_train=100, n_test=75, gap=False, seed=0):
    '''
    Randomly draws a sample function from rbfn as the true function, 
    except center parameters drawn from intensity (needs to be of Piecewise class)
    '''
    with torch.no_grad():

        np.random.seed(seed)
        torch.manual_seed(seed)

        ## sample x data
        x = torch.linspace(-5,5,n_train).reshape(-1,1)
        if gap:
            x = x[((x<-3.25) | (x>-1.75)) & ((x<1.75) | (x>3.25))].reshape(-1,1)
            x_test = torch.cat((torch.linspace(-3.25,-1.75,10),torch.linspace(1.75,3.25,10))).reshape(-1,1)
        else:
            x_test = torch.empty(n_test).uniform_(-5,5).reshape(-1,1)

        ## sample function
        rbfn.sample_parameters()
        rbfn.sample_parameters_sgcp()

        ## sample y data
        y = rbfn.forward(x) + np.sqrt(rbfn.sig2)*torch.randn(x.shape)
        y_test = rbfn.forward(x_test) + np.sqrt(rbfn.sig2)*torch.randn(x_test.shape)
        
        return x, y, x_test, y_test

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

