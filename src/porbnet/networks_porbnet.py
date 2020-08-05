import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt

import util_porbnet as util
from util_porbnet import Piecewise

import ipdb

torch.set_default_dtype(torch.float64)

DEFAULT_INTENSITY = Piecewise(np.array([-10,10]),np.array([10.0]))

class RBFN(nn.Module):
    '''
    PoRB-NET SGCP model

    dim_in:                             input dimension <-- only dim_in == 1 will work
    dim_hidden_initial:                 hidden dimension
    dim_hidden_max:                     maximum hidden dimension
    dim_out:                            output dimension <-- only dim_out == 1 will work
    intensity:
    s2_0:                               s2_0 parameter
    prior_w_sig2:                       variance parameter for normal prior on weights
    prior_b_sig2:                       variance parameter for normal prior on bias
    sig2:                               variance of observational noise
    prior_sig2_alpha:                   alpha parameter for gamma prior on sig2
    prior_sig2_beta:                    beta parameter for gamma prior on sig2
    prior_intensity_alpha:              alpha parameter for gamma prior on uniform intensity
    prior_intensity_beta:               beta parameter for gamma prior on uniform intensity
    intensity_proposal_std:             standard deviation of normal proposal distribution for uniform intensity
    infer_wb:                           whether to infer weights and bias
    infer_c:                            whether to infer centers
    infer_intensity:                    whether to infer intensity (only works for uniform intensity)
    '''
    def __init__(self, dim_in=1, dim_hidden_initial=30, dim_hidden_max=400, \
        dim_out=1, intensity=DEFAULT_INTENSITY, s2_0=1.0, \
        prior_w_sig2 = 1., prior_b_sig2 = 1., \
        sig2 = None, prior_sig2_alpha = None, prior_sig2_beta = None,
        prior_intensity_alpha = None, prior_intensity_beta = 1,
        intensity_proposal_std=1.,
        infer_wb=True, infer_c=True, infer_intensity=True):
        super(RBFN, self).__init__()

        # inference overrides
        self.infer_wb = infer_wb
        self.infer_c = infer_c
        self.infer_intensity = infer_intensity # only works for uniform intensity

        # architecture
        self.dim_in = dim_in
        self.dim_hidden_initial = dim_hidden_initial
        self.dim_hidden = dim_hidden_initial # this will change
        self.dim_hidden_max = dim_hidden_max
        self.dim_out = dim_out

        # parameters
        if self.infer_c:
            self.c = nn.Parameter(torch.Tensor(dim_in, dim_hidden_max))
        else:
            self.c = torch.empty(dim_in, dim_hidden_max)

        if self.infer_wb:
            self.w = nn.Parameter(torch.Tensor(dim_out, dim_hidden_max))
            self.b = nn.Parameter(torch.Tensor(dim_out))
        else:
            self.w = torch.empty(dim_out, dim_hidden_max)
            self.b = torch.empty(dim_out)

        self.c_intensity = intensity # can be updated

        self.mask = torch.zeros(dim_hidden_max, dtype=torch.bool)
        self.mask[:dim_hidden_initial] = 1

        # momentum parameters
        if self.infer_c:
            self.p_c = torch.empty(self.c.shape)

        if self.infer_wb:
            self.p_w = torch.empty(self.w.shape)
            self.p_b = torch.empty(self.b.shape)

        # priors
        self.prior_c_intensity = intensity # never updated
        self.prior_w_sig2 = torch.tensor(np.sqrt(s2_0/np.pi) * prior_w_sig2)
        self.prior_b_sig2 = torch.tensor(prior_b_sig2)

        if prior_sig2_alpha is None or prior_sig2_beta is None:
            self.infer_sig2 = False
            self.sig2 = torch.tensor(sig2)
        else:
            self.infer_sig2 = True
            self.prior_sig2_alpha = prior_sig2_alpha
            self.prior_sig2_beta = prior_sig2_beta

        if self.infer_intensity and prior_intensity_alpha is not None and prior_intensity_beta is not None \
            and len(intensity.values)==1:
            # only implemented for uniform intensity (len(intensity.values)==1)
            self.prior_intensity_alpha = prior_intensity_alpha
            self.prior_intensity_beta = prior_intensity_beta
            self.intensity_proposal_std = intensity_proposal_std
        else:
            self.infer_intensity = False


        self.h = lambda x: s2_0*x**2
        self.s2_0 = s2_0

	# initialize
        self.init_parameters()
        self._sample_momentum()

        self.V = intensity.breaks[-1]-intensity.breaks[0] # Volume of region

        self.have_ground_truth = False
        
        self.trained = False
        self.eps_adapted = 1e-5

    def init_parameters(self):
        '''
        Initialize parameters (not necessarily from the prior)
        '''

        if self.infer_c:
            self.c.data = self.prior_c_intensity.sample_conditional(self.c.shape)

        if self.infer_wb:
            self.w.data.normal_(0,self.prior_w_sig2.sqrt() * 1e-2)
            self.b.data.normal_(0,self.prior_w_sig2.sqrt() * 1e-2)

        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

    def sample_parameters(self, sample_K=False, sample_intensity=False):
        '''
        Sample network parameters (w, b, c) from the prior

        sample_K: whether or not to sample width
        '''   
        if sample_K:
            centers = self.prior_c_intensity.sample_pp()
            self.dim_hidden = centers.shape[0]
            #centers = centers.unsqueeze(0)
            self.c.data[0,:self.dim_hidden] = centers
            self.mask = torch.zeros(self.dim_hidden_max, dtype=torch.bool)
            self.mask[:self.dim_hidden] = 1
        else:
            self.c.data = self.prior_c_intensity.sample_conditional(self.c.shape)

        if sample_intensity:
            if self.infer_intensity:
                gamma = torch.distributions.gamma.Gamma(self.prior_intensity_alpha, self.prior_intensity_beta)
                intensity2_samp = gamma.sample()
                self.c_intensity.reset_values([intensity2_samp.sqrt().item()])
            else:
                print('Intensity not sampled, can only sample when inferring intensity')
        self.w.data.normal_(0,self.prior_w_sig2.sqrt())
        self.b.data.normal_(0,self.prior_b_sig2.sqrt())

        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

    def sample_functions_prior(self, x, n_samp, sample_K=False, sample_intensity=False):
        '''
        Sample functions from the prior

        x: points to sample function (n,dim_in)
        n_samp: number of samples
        sample_K: whether or not to sample width

        returns (n_samp, n) tensor
        '''
        y_samp_prior = torch.zeros((n_samp, x.shape[0]))
        for i in range(n_samp):
            self.sample_parameters(sample_K=sample_K,sample_intensity=sample_intensity)
            y_samp_prior[i,:] = self.forward(x).reshape(-1)
        return y_samp_prior

    def _sample_momentum(self):
        '''
        samples momentum parameters 
        '''
        if self.infer_c:
            self.p_c.data.normal_(0,1)

        if self.infer_wb:
            self.p_w.data.normal_(0,1)
            self.p_b.data.normal_(0,1)

    def forward(self, x):
        '''
        Forward pass of network

        x: (n, dim_in)

        returns (n, dim_out) tensor
        '''
        s2 = self.h(self.c_intensity(self.c[:, self.mask].reshape(-1))).reshape(-1)
        h = torch.exp(-0.5 * (x - self.c[:, self.mask])**2 * s2)
        return F.linear(h, self.w[:, self.mask], self.b)

    def log_prior(self):
        '''
        log of prior distribution
        '''
        return util.log_poisson_process(self.c[:, self.mask], self.prior_c_intensity) \
                   + util.log_normal(self.w[:, self.mask], self.prior_w_sig2) \
                   + util.log_normal(self.b, self.prior_b_sig2)

    def _log_kinetic_energy(self):
        '''
        Log of kinetic energy
        '''
        ke = 0.

        if self.infer_wb:
            ke += torch.sum(self.p_w[:, self.mask]**2) + torch.sum(self.p_b**2)
            
        if self.infer_c:
            ke += torch.sum(self.p_c[:, self.mask]**2)

        ke *= .5
        return ke

    def _update_p(self, eps, x, y, rollback_mu):
        '''
        Leapfrog step for momentum
        '''

        # backprop
        self._zero_gradients()
        U = self._log_potential_energy(x, y, rollback_mu)
        if torch.isinf(U):
            return 1
        else:
            U.backward()

            if self.infer_c:
                self.p_c[:, self.mask] -= eps*self.c.grad[:, self.mask]

            if self.infer_wb:
                self.p_w[:, self.mask] -= eps*self.w.grad[:, self.mask]
                self.p_b -= eps*self.b.grad

            return 0

    def _update_q(self, eps):
        '''
        Leapfrog step for parameters
        '''
        if self.infer_c:
            self.c[:, self.mask] += eps*self.p_c[:, self.mask]

        if self.infer_wb:
            self.w[:, self.mask] += eps*self.p_w[:, self.mask]
            self.b += eps*self.p_b

    def _negate_momentum(self):
        '''
        Flip sign of momentum
        '''
        if self.infer_c:
            self.p_c[:, self.mask] *= -1.

        if self.infer_wb:
            self.p_w[:, self.mask] *= -1.
            self.p_b *= -1.

    def _copy_parameters(self):
        '''
        Store a copy of parameters
        '''
        self.c_copy = self.c.data.clone()
        self.w_copy = self.w.data.clone()
        self.b_copy = self.b.data.clone()

    def _reset_parameters(self):
        '''
        Reset parameters to stored copy
        '''
        self.c.data = self.c_copy.clone()
        self.w.data = self.w_copy.clone()
        self.b.data = self.b_copy.clone()
            
    def log_likelihood(self, x, y):
        '''
        Log likelihood

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        return -0.5*x.shape[0]*(math.log(2*math.pi) + torch.log(self.sig2)) - torch.sum((y - self.forward(x))**2)/(2*self.sig2)

    def log_posterior(self, x, y):
        '''
        Log posterior

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        return self.log_likelihood(x, y) + self.log_prior()

    def _zero_gradients(self):
        '''
        Set gradients to zero
        '''
        if self.c.grad is not None:
            self.c.grad.zero_()
        if self.w.grad is not None:
            self.w.grad.zero_()
        if self.b.grad is not None:
            self.b.grad.zero_()
    
    def _log_potential_energy(self, x, y, rollback_mu):
        '''
        Potential energy of network

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        g0 = self.c[:, self.mask] - self.prior_c_intensity.breaks[0] 
        g1 = self.prior_c_intensity.breaks[-1] - self.c[:, self.mask]
        
        U1 = torch.sum(torch.log(1 + torch.exp(-rollback_mu * g0)))
        U2 = torch.sum(torch.log(1 + torch.exp(-rollback_mu * g1)))

        return -self.log_posterior(x, y) + U1 + U2

    def update_intensity(self, x, y):
        '''
        Update intensity (gamma prior) with MH
        Only works for uniform intensity (not piecewise)

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''

        # record current state
        intensity_current = torch.tensor(self.c_intensity.values[0])

        # propose new state
        intensity_proposed = intensity_current + torch.randn(1)*self.intensity_proposal_std
        self.c_intensity.reset_values([intensity_proposed.item()])
        log_lik_proposed = self.log_likelihood(x, y).clone()

        a = torch.exp(
            log_lik_proposed - self.log_lik_current \
            + (self.dim_hidden + 2*(self.prior_intensity_alpha-1))*(torch.log(intensity_proposed) - torch.log(intensity_current)) \
            - self.V*(intensity_proposed - intensity_current) \
            - self.prior_intensity_beta*(intensity_proposed**2 - intensity_current**2)
            )

        if torch.rand(1) < a:
            # accept
            self.log_lik_current = log_lik_proposed.clone()
            return True

        else:
            # reject (change back)
            self.c_intensity.reset_values([intensity_current.item()])
            return False


    def update_sig2(self, x, y):
        '''
        Gibbs update for observational noise

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        SSR = torch.sum((y - self.forward(x))**2)
        gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha + 0.5*x.shape[0], self.prior_sig2_beta + 0.5*SSR)
        self.sig2 = 1/gamma.sample()
    
    def birth(self, x, y):
        '''
        Birth step for hidden unit

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        if self.dim_hidden == self.dim_hidden_max:
            return False
        else:            
            k_proposed = torch.min(torch.nonzero(self.mask==0)).item()
            
            # Record current parameters in case you need to go back
            c_current = self.c[:, k_proposed].clone()
            w_current = self.w[:, k_proposed].clone()

            # Propose new parameters
            c_proposed = self.prior_c_intensity.sample_conditional(self.dim_in)
            w_proposed = self.prior_w_sig2.sqrt()*torch.randn(self.dim_out)
            
            # Temporarily update and compute acceptance prob
            self.mask[k_proposed] = 1
            self.c[:, k_proposed] = c_proposed
            self.w[:, k_proposed] = w_proposed

            log_lik_proposed = self.log_likelihood(x,y)

            a = np.exp(log_lik_proposed - self.log_lik_current + np.log(self.prior_c_intensity.integral) - np.log(self.dim_hidden+1))

            if torch.rand(1) < a:
                # Accept
                self.dim_hidden += 1
                self.log_lik_current = log_lik_proposed.clone()
                return True
            else:
                # Reject (change back)
                self.mask[k_proposed] = 0
                self.c[:, k_proposed] = c_current.clone()
                self.w[:, k_proposed] = w_current.clone()
                return False
        
    def death(self, x, y):
        '''
        Death step for hidden unit

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        if self.dim_hidden == 1:
            return False
        else:
            # Choose node to delete
            k_proposed = torch.nonzero(self.mask)[torch.randint(self.dim_hidden, (1,1))].item()

            # Temporarily update and compute acceptance prob
            self.mask[k_proposed] = 0 

            log_lik_proposed = self.log_likelihood(x,y)

            a = torch.exp(log_lik_proposed - self.log_lik_current + np.log(self.dim_hidden) - np.log(self.prior_c_intensity.integral))

            if torch.rand(1) < a:
                # Accept
                self.dim_hidden -= 1
                self.log_lik_current = log_lik_proposed.clone()
                return True
            else:
                # Reject (change back)
                self.mask[k_proposed] = 1
                return False
            
    def hmc(self, x, y, eps, L):
        '''
        Single step of Hamiltonian Monte Carlo (HMC)

        Adapted from Neal (2012)

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        eps: leapfrog stepsize (float)
        L: number of leapfrog steps (int)
        '''
        rollback_mu = .75*1/eps

        self._copy_parameters()
        
        # sample momentum
        self._sample_momentum()

        # record current potential and kenetic energy
        current_K = self._log_kinetic_energy()
        current_U = self._log_potential_energy(x, y, rollback_mu)

        # half step for momentum: p -= eps*grad_U(q)/2
        error_code = self._update_p(eps/2, x, y, rollback_mu)
        
        for i in range(L):

            if error_code == 1:
                print('Skipping hmc step, potential energy is infinite')
                self._reset_parameters()
                return False

            with torch.no_grad():
                # full step for positions: q += eps*p
                self._update_q(eps)
            
            # full step for momentum: p -= eps*grad_U(q), except at end
            if i != L-1:
                error_code = self._update_p(eps, x, y, rollback_mu)
            
        # half step for momentum: p -= eps*grad_U(q)/2
        error_code = self._update_p(eps/2, x, y, rollback_mu)
        if error_code == 1:
            print('Skipping hmc step, potential energy is infinite')
            self._reset_parameters()
            return False

        # Negate the momentum at the end of the trajectory
        self._negate_momentum()

        proposed_K = self._log_kinetic_energy()
        
        # Evaluate potential and kinetic energies at end of trajectory
        proposed_U = self._log_potential_energy(x, y, rollback_mu)
        
        with torch.no_grad():
            if torch.rand(1) < torch.exp(current_U-proposed_U+current_K-proposed_K):
                # accept
                return True
            else:
                # reject, go back to old parameters
                self._reset_parameters()
                return False

    def _init_record(self, x, y, x_plot, x_test, y_test, n_samp, n_rep_hmc, n_rep_resize, \
        predict_y, predict_y_plot, predict_y_test):
        '''
        Intialize dictionary for recoding posterior samples
        '''
        samples = {
            'x': x,
            'y': y,
            'dim_hidden': torch.zeros(n_samp),
            'sig2': torch.zeros(n_samp),
            'c': torch.zeros((n_samp,)+self.c.shape),
            'w': torch.zeros((n_samp,)+self.w.shape),
            'mask': torch.zeros((n_samp, self.dim_hidden_max)),
            'predict_y': predict_y,
            'predict_y_plot': predict_y_plot,
            'predict_y_test': predict_y_test,
            'x_test': x_test,
            'y_test': y_test
        }
        if predict_y:
            samples.update({
                'y_pred': torch.zeros((n_samp, x.shape[0]))
            })
        if predict_y_plot:
            samples.update({
                'x_plot': x_plot,
                'y_plot_pred': torch.zeros((n_samp, x_plot.shape[0]))
            })
        if predict_y_test:
            samples.update({
                'y_test_pred': torch.zeros((n_samp, x_test.shape[0]))
            })
        return samples

    def _record_samp(self, i, samples):
        '''
        Record single posterior sample
        '''
        samples['dim_hidden'][i] = self.dim_hidden
        samples['sig2'][i] = self.sig2
        samples['c'][i,:,:] = self.c.detach()
        samples['w'][i,:,:] = self.w.detach()
        samples['mask'][i,:] = self.mask

        if samples['predict_y']:
            samples['y_pred'][i,:] = self.forward(samples['x']).reshape(-1)

        if samples['predict_y_plot']:
            samples['y_plot_pred'][i,:] = self.forward(samples['x_plot']).reshape(-1)

        if samples['predict_y_test']:
            samples['y_test_pred'][i,:] = self.forward(samples['x_test']).reshape(-1)

    def _smallwrite(self, i, eps, samples, accept, accept_cum, writer):
        writer.add_scalar('eps', eps, i)
        writer.add_scalar('log_prob/log_likelihood', self.log_likelihood(samples['x'],samples['y']), i)
        writer.add_scalar('log_prob/log_prior', self.log_prior(), i)
        
        if self.infer_sig2:
            writer.add_scalar('sig2', self.sig2, i)
        
        if self.have_ground_truth:
            writer.add_scalars('dist_param_truth', self.dist_param_truth(), i)
            writer.add_scalars('dim_hidden', {'true': self.dim_hidden_true, 'inferred': self.dim_hidden}, i)
            if self.infer_intensity:
                writer.add_scalars('intensity', {'true': self.intensity_true, 'inferred': self.c_intensity.values[0]}, i)
        else:
            writer.add_scalar('dim_hidden', self.dim_hidden, i)
            if self.infer_intensity:
                writer.add_scalar('intensity', self.c_intensity.values[0], i)

        writer.add_scalars('acceptance', accept_cum, i)
        writer.add_scalar('log_prob/log_poisson_process', util.log_poisson_process(self.c[:, self.mask], self.prior_c_intensity), i) # TEMPORARY

    def _bigwrite(self, i, eps, samples, accept, accept_cum, writer):
        fig, ax = util.plot_functions(samples['x_plot'], samples['y_plot_pred'][:i,:], samples['x'], samples['y'], samples['x_test'], samples['y_test'])
        writer.add_figure('functions',fig,i)

        fig, ax = util.plot_centers_trace(samples['c'][:i,:].detach().numpy(), samples['mask'][:i,:].numpy())
        writer.add_figure('parameters/c_trace',fig,i)

        fig, ax = util.plot_centers_trace(samples['w'][:i,:].detach().numpy(), samples['mask'][:i,:].numpy())
        writer.add_figure('parameters/w_trace',fig,i)

        if samples['predict_y_test']:
            test_ll = util.test_log_likelihood(samples['y_test'].reshape(1,-1), samples['y_test_pred'][:i,:], samples['sig2'][:i].reshape(-1,1))
            writer.add_scalar('test_log_likelihood', test_ll, i)
            ax.set_title('test log lik: %.3f' % test_ll.item())

    def train(self, x, y, n_samp, frac_samp_resize=.5, L=100, eps=None, n_adapt_eps=None, numpy=True, n_print=0):
        '''
        train model

        x:  training features (n_train,d_in) 
        y:  training outcomes (n_train,d_out)
        n_samp:  number of samples
        frac_samp_resize:  fraction of samples for which network is allowed to resize
        L:  number of leapfrog steps for hmc
        eps:  step size for hmc. default uses self.eps_adapted.
        n_eps_adapt:  number of times to adapt step size. default uses n_samp/10 if untrained and 1 if trained.
        numpy:  whether to assume data inputs are numpy arrays
        n_print:  number of times to print status
        '''

        n_samp_resize = int(frac_samp_resize*n_samp)
        n_samp_fixed = n_samp - n_samp_resize

        if self.trained:
            n_adapt_eps_use = 1 if n_adapt_eps is None else n_adapt_eps
        else:
            n_adapt_eps_use = int(n_samp/10) if n_adapt_eps is None else n_adapt_eps
        eps_use = self.eps_adapted if eps is None else eps

        if numpy:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        # fixed network size
        if n_samp_fixed > 0:
            _, eps_use = self.sample_posterior(x=x, y=y, n_samp=n_samp_fixed, x_plot=None, eps=eps_use, L=100, n_adapt_eps=n_adapt_eps_use, \
                                                 n_rep_hmc=1, n_rep_resize=0, record=False, n_print=n_print)

        # with resizing
        if n_samp_resize > 0:
            _, eps_use = self.sample_posterior(x=x, y=y, n_samp=n_samp_resize, x_plot=None, eps=eps_use, L=100, n_adapt_eps=n_adapt_eps_use, \
                                                 n_rep_hmc=1, n_rep_resize=1, record=False, n_print=n_print)

        self.eps_adapted = eps_use
        self.trained = True


    def f_gradient(self, x_test, weights):
        x_test_ = np.asarray(x_test)

        with torch.no_grad():
            self.set_network_params(weights)

        x = torch.autograd.Variable(
                torch.from_numpy(x_test_[None, :]), requires_grad=True)

        m = self.forward(x)[0] # scalar

        m.backward()

        g = x.grad.data.numpy()[0, :]
        return g

    def set_network_params(self, weights):
        self.c.data = weights['c'].clone()
        self.w.data = weights['w'].clone()
        self.b.data = weights['b'].clone()
        self.mask.data = weights['mask'].clone()
        self.sig2 = weights['sig2'].clone()

    def get_network_params(self):
        weights = dict()
        weights['c'] = self.c.data.clone()
        weights['w'] = self.w.data.clone()
        weights['b'] = self.b.data.clone()
        weights['mask'] = self.mask.data.clone()
        weights['sig2'] = self.sig2.clone()
        return weights

    def predict(self, x_predict, x, y, n_samp, numpy=True, n_print=0, \
            return_individual_predictions=False):
        '''
        computes mean and standard deviation of posterior predictive at x_predict.
        network parameters reset to initial values once samples have been collected.

        x_predict:  locations to predict (n_predict,d) 
        x:  training features (n_train,d_in) 
        y:  training outcomes (n_train,d_out)
        n_samp:  number of samples
        numpy:  whether to assume data inputs are numpy arrays and also return numpy arrays
        n_print:  number of times to print status
        '''
        if numpy:
            x_predict = torch.from_numpy(x_predict)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        # copy parameters
        c_init = self.c.data.clone()
        w_init = self.w.data.clone()
        b_init = self.b.data.clone()
        mask_init = self.mask.clone()
        sig2_init = self.sig2.clone()

        accept, samples, _ = self.sample_posterior(x=x, y=y, n_samp=n_samp, x_plot=x_predict, eps=self.eps_adapted, n_adapt_eps=0, \
                                                  n_rep_hmc=1, n_rep_resize=1, record=True, n_print=n_print)

        # reset parameters
        self.c.data = c_init.clone()
        self.w.data = w_init.clone()
        self.b.data = b_init.clone()
        self.mask = mask_init.clone()
        self.dim_hidden = int(torch.sum(self.mask))
        self.sig2 = sig2_init.clone()

        mean = torch.mean(samples['y_plot_pred'],0)
        std = torch.std(samples['y_plot_pred'],0)

        if numpy:
            mean = mean.numpy()
            std = std.numpy()

        if return_individual_predictions:
            funcs = samples['y_plot_pred']
            if numpy: funcs = funcs.numpy()
            return (mean, std, funcs)
        else:
            return (mean, std)


    def sample_posterior(self, x, y, n_samp, 
        eps=.001, L=100, n_adapt_eps=0, \
        n_rep_hmc=1, n_rep_resize=1, \
        predict_y = False, x_plot = None, x_test = None, y_test = None, \
        n_print=0, n_bigwrite=0, \
        writer=None, record=True):
        '''
        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        n_samp: number of samples
        eps: leapfrog stepsize in hmc for network parameters
        L: number of leapfrog steps in hmc for network parameters
        n_adapt_eps: number of times to update eps
        n_rep_hmc: number of times per sample to repeat network hmc
        n_rep_resize: number of times per sample to repeat birth/death steps
        predict_y: whether to predict y on training data
        x_plot: locations for recording function samples (n, dim_in)
        x_test: test data (n, dim_in)
        y_test: test outcomes (n, dim_out)
        n_print: number of times to print status
        n_bigwrite: number of times to do big write (plots, etc)
        writer: tensorflow writer
        record: whether to record results (including data)
        '''

        predict_y_plot = False if x_plot is None else True
        predict_y_test = False if x_test is None or y_test is None else True
        
        # samples per event
        samples_per = lambda n: int(n_samp / n) if n>0 and n<n_samp else n_samp
        n_samp_print = samples_per(n_print)
        n_samp_bigwrite = samples_per(n_bigwrite)
        n_samp_adapt_eps = samples_per(n_adapt_eps)
        i_adapt_eps = 0
        
        ## allocate space for results
        accept = {
            'hmc': np.full((n_samp, n_rep_hmc), np.nan),
            'birth': np.full((n_samp, n_rep_resize), np.nan),
            'death': np.full((n_samp, n_rep_resize), np.nan),
            'intensity': np.full((n_samp, 1), np.nan)
        }
        accept_cum = dict.fromkeys(accept, 0)

        if record:
            samples = self._init_record(x, y, x_plot, x_test, y_test, n_samp, n_rep_hmc, n_rep_resize, \
                                        predict_y, predict_y_plot, predict_y_test)
        
        # main loop
        for i in range(n_samp):
            self.i = i
            self.log_lik_current = self.log_likelihood(x,y)
            
            if self.infer_wb or self.infer_c:
                for j in range(n_rep_hmc):
                    ## hmc
                    accept['hmc'][i, j] = self.hmc(x, y, eps, L)

            # adapt eps
            if i >0 and i % n_samp_adapt_eps == 0:
                eps_old = eps
                accept_prev_iter = np.mean(accept['hmc'][n_samp_adapt_eps*i_adapt_eps:i,:])
                eps = util.rescale_eps(eps, accept_prev_iter)
                i_adapt_eps+=1
                print('Epsilon adapted from %.6f to %.6f based on acceptance of %.3f' % (eps_old,eps,accept_prev_iter))

            with torch.no_grad():
                self.log_lik_current = self.log_likelihood(x,y) # assumes log likelihod computed
                
                ## birth/death
                for j in range(n_rep_resize):
                    if torch.rand(1) < 0.5:
                        accept['birth'][i, j] = self.birth(x,y)
                    else:
                        accept['death'][i, j] = self.death(x,y)

                ## noise
                if self.infer_sig2:
                    self.update_sig2(x,y)

                ## intensity
                if self.infer_intensity:
                    accept['intensity'][i] = self.update_intensity(x,y)

                ## saving
                for accept_type in accept_cum.keys():
                    accept_cum[accept_type] = np.nanmean(accept[accept_type][:i,:])

                if record:
                    self._record_samp(i, samples)

                    if writer is not None:
                        self._smallwrite(i, eps, samples, accept, accept_cum, writer)

                        if i>0 and i % n_samp_bigwrite == 0 and writer is not None:
                            self._bigwrite(i, eps, samples, accept, accept_cum, writer)

                # print
                if i >0 and i % n_samp_print == 0:
                    print('sample [%d/%d], acceptance_hmc: %.3f, acceptance_birth: %.3f, acceptance_death: %.3f' % \
                        (i, n_samp, accept_cum['hmc'], accept_cum['birth'], accept_cum['death']))

        if record:
            return accept, samples, eps
        else:
            return accept, eps

    def store_true_parameters(self):
        '''
        Stores a copy of all parameters.
        For use if network is initialized to the ground truth parameters. 
        '''
        self.have_ground_truth = True

        self.b_true = self.b.clone()
        self.w_true = self.w.clone()
        self.c_true = self.c.clone()
        self.dim_hidden_true = self.dim_hidden
        self.intensity_true = self.c_intensity.values[0]

    def dist_param_truth(self):
        '''
        Compute distance to stored ground truth parameters
        Not useful if network resizes
        '''
        return { \
        'b': torch.sum((self.b - self.b_true)**2), \
        'w': torch.sum((self.w - self.w_true)**2), \
        'c': torch.sum((self.c - self.c_true)**2), \
        'dim_hidden': np.abs(self.dim_hidden - self.dim_hidden_true), \
        'intensity': np.abs(self.c_intensity.values[0] - self.intensity_true)
        }

        
            
