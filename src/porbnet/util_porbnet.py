import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import os
from functools import reduce

from scipy.stats import norm
from scipy.special import hyp2f1, gamma

import matplotlib.pyplot as plt

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

    return torch.from_numpy(x).reshape(-1,1).float(), torch.from_numpy(y).reshape(-1,1).float()



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
        row = df.loc[(df['dataset']==args.dataset) & (df['model']=='porbnet'),:]
    else:
        row = df.loc[(df['dataset']==args.dataset) & (df['dataset_option']==args.dataset_option) & (df['model']=='porbnet'),:]

    # param1
    if row['param1_name'].item() == 'intensity':
        intensity = row['param1'].item()
        setattr(args, 'intensity_1', intensity)

        setattr(args, 'prior_intensity_beta', 1.0)
        setattr(args, 'prior_intensity_alpha', alpha_for_sqrt_gamma(1.0, intensity))

    elif row['param1_name'].item() == 's2_0':
        setattr(args, 's2_0', row['param1'].item())

    # param2
    setattr(args, 'prior_b_sig2', row['param2'].item())
    setattr(args, 'prior_w_sig2', row['param2'].item()*np.sqrt(np.pi/args.s2_0))

    setattr(args, 'sig2', row['sig2'].item())

    T=[-.25,1.25]
    dim_hidden_initial=int((T[1]-T[0])*intensity)
    dim_hidden_max = np.maximum(2*dim_hidden_initial, 75)
    setattr(args, 'dim_hidden_initial', dim_hidden_initial)
    setattr(args, 'dim_hidden_max', dim_hidden_max)

def rescale_eps(eps, accept, accept_target=.65, eps_min=1e-6):
    'Rescaling of step size'
    scale = 1 + 10*np.sign(accept - accept_target)*(accept - accept_target)**2
    scale = np.log(1+np.exp(scale)) + (1-np.log(1+np.exp(1)))
    return (eps + max(scale*eps, eps_min))/2.0

def gcd(a, b, tol=1e-5) : 
    if (a < b) : 
        return gcd(b, a) 
    
    if (np.abs(b) < 1e-5) : 
        return a 
    else : 
        return (gcd(b, a - np.floor(a / b) * b)) 

class Piecewise(object):
    def __init__(self, breaks, values):
        '''
        Piecewise constant function class. 
        Records function as map x -> y, with dx distance between each x.
        
        breaks: list of function breaks (length B)
        values: list of function values between breaks (length B-1)
        dx: distance between each x value in map. Should divide break points. 
        '''
        self.breaks = breaks
        self.dx = reduce(gcd, np.array(breaks[1:]) - np.array(breaks[:-1])).astype('float')
    
        self.x = torch.arange(breaks[0], breaks[-1] + self.dx, self.dx)

        self.reset_values(values)
            
    def reset_values(self, values):

        self.values = values
        self.y = torch.empty(self.x.shape)

        for i in range(len(values)):
            self.y[(self.x>=self.breaks[i]) & (self.x<=self.breaks[i+1])] = values[i]

        self.log_y = torch.log(self.y)
        self.integral = torch.sum(torch.tensor([(self.breaks[i+1] - self.breaks[i])*values[i] for i in range(len(values))]))
        self.values_pdf = [v/self.integral.item() for v in values]

    def __call__(self, x):
        # if x < breaks[0], return values[0]. if x > breaks[-1] return values[-1]
        return self.y[torch.floor(torch.clamp((x-self.breaks[0]) / self.dx, 0.0, float(self.y.size()[0]-1))).long()]

    def log(self, x):
        return self.log_y[torch.floor(torch.clamp((x-self.breaks[0]) / self.dx, 0.0, float(self.y.size()[0]-1))).long()]
    
    def check_bounds(self, x):
        return False if torch.any((x < self.breaks[0]) | (x > self.breaks[-1])) else True
    
    def sample_conditional(self, n_samp):
        '''
        Samples from dist. with normalized piecewise constant function as pdf
        Uses numpy code but returns tensor
        '''
        return torch.as_tensor(piecewise_invcdf(np.random.uniform(size=n_samp),self.breaks,self.values_pdf)).to(torch.get_default_dtype())
        
    def sample_pp(self):
        '''
        Samples from a poisson process using piecewise constant function as intensity
        Uses numpy code but returns tensor
        '''
        K = np.random.poisson(self.integral, size=1)
        return self.sample_conditional(K)

class ConstantIntensity(object):
    def __init__(self, bounds, y):
        '''
        bounds: (2,D) array (LB, UB)
        value: scalar intensity value
        '''
        self.y = torch.tensor(y)
        self.bounds = bounds

        self.D = bounds.shape[1]

        self.log_y = torch.log(self.y)

        self.integral = torch.prod(self.bounds[:,1] - self.bounds[:,0])*self.y
   
        self.y_pdf = self.y/self.integral
        

    #def __call__(self, x, check_bounds=True):
    def __call__(self, x):
        return self.y*torch.ones((x.shape[0],1))
    
    #def log(self, x, check_bounds=True):
    def log(self, x):
        return self.log_y*torch.ones((x.shape[0],1))
    
    def check_bounds(self, x):
        # x: N x D
        return torch.all(x > self.bounds[0,:]) and torch.all(x < self.bounds[1,:])

        
    def sample_conditional(self, n_samp):
        '''
        Samples from uniform distribution: n_samp x D
        '''
        return self.bounds[0,:] + (self.bounds[1,:]-self.bounds[0,:])*torch.rand(n_samp, self.D)

    def sample_pp(self):
        '''
        Samples from a poisson process using piecewise constant function as intensity
        Uses numpy code but returns tensor
        '''
        K = np.random.poisson(self.integral, size=1)
        return self.sample_conditional(K)


def dataset_from_rbfn(rbfn, intensity, n_train=100, n_test=75, gap=False, seed=0):
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

        # resample center parameters (so intensity can be used)
        if "c" in dir(rbfn):
            rbfn.c.data = intensity.sample_conditional(rbfn.c.shape)
        elif "cK" in dir(rbfn):
            rbfn.cK.data = intensity.sample_conditional(rbfn.cK.shape)

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

## covariance
def cov_porbnet_fixed_intensity(x1, x2, C0, C1, b_sig2, w_sig2, s2_0, intensity):
    '''
    Covariance for porbnet with 1d inputs and uniform intensity (fixed)
    '''
    s2 = s2_0*intensity**2
    xm = (x1+x2)/2
    
    C0_tilde = (C0-xm)*np.sqrt(2*s2)
    C1_tilde = (C1-xm)*np.sqrt(2*s2)

    return b_sig2 + w_sig2*np.sqrt(np.pi/s2_0)*np.exp(-s2*((x1-x2)/2)**2)*(norm.cdf(C1_tilde)-norm.cdf(C0_tilde))

def cov_porbnet_gamma_intensity(x1, x2, C0, C1, b_sig2, w_sig2, s2_0, intensity_alpha, intensity_beta):
    '''
    Covariance for porbnet with 1d inputs and uniform intensity with gamma prior
    '''
    dx2 = (1/4)*s2_0*(x1-x2)**2 + intensity_beta
    xm = (x1+x2)/2
    Ct0 = xm - C0
    Ct1 = C1 - xm
    V = dx2**(-intensity_alpha) * \
            (Ct0*hyp2f1(1/2,intensity_alpha,3/2, -s2_0*Ct0**2/dx2) + \
             Ct1*hyp2f1(1/2,intensity_alpha,3/2, -s2_0*Ct1**2/dx2))

    E_lambda = mean_sqrt_gamma(intensity_alpha, intensity_beta)

    return b_sig2 + w_sig2*E_lambda*intensity_beta**(intensity_alpha)*V

def mean_sqrt_gamma(alpha, beta):
    try:
        return np.exp(loggamma(alpha+0.5)-loggamma(alpha)-0.5*np.log(beta))
    except:
        # in case of numerical issue, use approximation
        return alpha**(0.5)*beta**(-0.5)

def alpha_for_sqrt_gamma(beta, K):
    '''
    Suppose X ~ Gamma(alpha, beta)
    return alpha such that E[sqrt(X)] = K

    uses approximation to ratio of Gamma functions
    '''
    return beta*K**2





