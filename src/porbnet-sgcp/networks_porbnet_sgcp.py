import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt

import util_porbnet_sgcp as util
import sys
import os

import GP as GP

class RBFN(nn.Module):
    '''
    PoRB-NET SGCP model

    dim_in:                             input dimension <-- only dim_in == 1 will work
    dim_hidden_initial:                 hidden dimension
    dim_hidden_max:                     maximum hidden dimension
    dim_out:                            output dimension <-- only dim_out == 1 will work
    s2_0:                               s2_0 parameter
    T:                                  poisson process region as list of boundaries (e.g., [0,1])
    length_scale_sgcp:                  length scale of intensity gp
    variance_sgcp:                      amplitudge variance of intensity gp
    proposal_std_cM:                    standard deviation of proposal distribution for thinned centers
    rate_density_ub:                    value for maximum intensity (i.e., lambda^*)
    prior_w_sig2:                       variance parameter for normal prior on weights
    prior_b_sig2:                       variance parameter for normal prior on bias
    sig2:                               variance of observational noise
    prior_sig2_alpha:                   alpha parameter for gamma prior on sig2
    prior_sig2_beta:                    beta parameter for gamma prior on sig2
    infer_wb:                           whether to infer weights and bias
    infer_cK:                           whether to infer unthinned locations
    infer_cM:                           whether to infer thinned locations
    infer_gM:                           whether to infer gp at thinned locations
    infer_gK:                           whether to infer gp at unthinned locations
    infer_M:                            whether to infer number of thinned units
    infer_K:                            whether to infer number of unthinned units (i.e. dim_hidden)
    infer_rate_density_ub:              whether to infer rate_density_ub
    rate_density_ub_prior_alpha:        alpha parameter for gamma prior on rate_density_ub
    rate_density_ub_prior_beta:         beta parameter for gamma prior on rate_density_ub
    rate_density_ub_proposal_std:       standard deviation of proposal distribution for thinned centers
    '''
    def __init__(self, dim_in, dim_hidden_initial, dim_hidden_max, dim_out, \
        s2_0, \
        T, length_scale_sgcp, variance_sgcp, proposal_std_cM, \
        rate_density_ub, \
        prior_w_sig2 = 1., prior_b_sig2 = 1., \
        sig2 = None, prior_sig2_alpha = None, prior_sig2_beta = None,
        infer_wb =True, infer_cK = True, infer_cM=True, infer_gM=True, infer_gK=True, infer_M=True, infer_K=True, infer_rate_density_ub=True,
        rate_density_ub_prior_alpha = None, rate_density_ub_prior_beta = None, rate_density_ub_proposal_std = 2.,
        use_gp_term = True, set_gp_to_mean = False):
        super(RBFN, self).__init__()

        # inference overrides
        self.infer_wb = infer_wb
        self.infer_cK = infer_cK
        self.infer_cM = infer_cM
        self.infer_gM = infer_gM
        self.infer_gK = infer_gK
        self.infer_M = infer_M
        self.infer_K = infer_K
        self.infer_rate_density_ub = infer_rate_density_ub

        self.use_gp_term = use_gp_term
        self.set_gp_to_mean = set_gp_to_mean
        
        # architecture
        self.dim_in = dim_in
        self.dim_hidden_initial = dim_hidden_initial
        self.dim_hidden = dim_hidden_initial # this will change
        self.dim_hidden_max = dim_hidden_max
        self.dim_out = dim_out
        
        # parameters
        if self.infer_cK:
            self.cK = nn.Parameter(torch.Tensor(dim_in, dim_hidden_max))
        else:
            self.cK = torch.empty(dim_in, dim_hidden_max) 

        if self.infer_gM or self.infer_gK:
            self.cK.requires_grad = True
        
        if self.infer_wb:
            self.w = nn.Parameter(torch.Tensor(dim_out, dim_hidden_max))
            self.b = nn.Parameter(torch.Tensor(dim_out))
        else:
            self.w = torch.empty(dim_out, dim_hidden_max)
            self.b = torch.empty(dim_out)
        
        self.mask = torch.zeros(dim_hidden_max, dtype=torch.bool)
        self.mask[:dim_hidden_initial] = 1
        
        # momentum parameters
        if self.infer_cK:
            self.p_c = torch.empty(self.cK.shape)
        
        if self.infer_wb:
            self.p_w = torch.empty(self.w.shape)
            self.p_b = torch.empty(self.b.shape)
        
        # priors        
        self.prior_w_sig2 = torch.tensor(np.sqrt(s2_0/np.pi) * prior_w_sig2)
        self.prior_b_sig2 = torch.tensor(prior_b_sig2)
        
        if prior_sig2_alpha is None or prior_sig2_beta is None:
            self.infer_sig2 = False
            self.sig2 = torch.tensor(sig2)
        else:
            self.infer_sig2 = True
            self.prior_sig2_alpha = prior_sig2_alpha
            self.prior_sig2_beta = prior_sig2_beta


        self.rate_density_ub_prior_alpha = rate_density_ub_prior_alpha
        self.rate_density_ub_prior_beta = rate_density_ub_prior_beta
        self.rate_density_ub_proposal_std = rate_density_ub_proposal_std

        ########### SGCP
        self.T = T
        self.rate_density_ub = torch.tensor(rate_density_ub)
        self.prob_birth_thinned = 0.5
        self.proposal_std_cM = proposal_std_cM
        self.fudge = 8e-4 #self.fudge = 8e-3
        self.V = T[1]-T[0] # Volume of region

        self.gK = nn.Parameter(torch.Tensor(dim_hidden_max))

        # GP:
        self.kernel = GP.RBFkernel(length_scale_sgcp, variance_sgcp)
        self.gp = GP.GP(self.kernel, self.fudge)

        self.h = lambda x: s2_0*x**2

        self.p_gK = torch.empty(self.dim_hidden_max)
        ###########
        
        # initialize
        self.sample_parameters() # added for fixing parameters version
        self.init_parameters()
        self._sample_momentum()
        self.sample_parameters_sgcp()

        self.have_ground_truth = False

    def init_parameters(self):
        '''
        Initialize parameters (not necessarily from the prior)
        '''

        T_length = self.T[1] - self.T[0]
        #self.cK.data.uniform_(self.T[0] + T_length/3., self.T[1] - T_length/3.,)
        if self.infer_cK:
            self.cK.data.uniform_(0, 1) # FIXED TO (0,1)!!!!!
        
        if self.infer_wb:
            self.w.data.normal_(0, self.prior_w_sig2.sqrt() * 1e-2)
            self.b.data.normal_(0, self.prior_b_sig2.sqrt() * 1e-2)

        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

    def sample_parameters(self, force_width=False):
        '''
        Sample network parameters (w, b, cK, sig2) from the prior

        Notes: 
            - assumes gp has zero mean
            - width does not change
        '''        
        self.cK.data.uniform_(self.T[0],self.T[1])
        self.w.data.normal_(0, self.prior_w_sig2.sqrt())
        self.b.data.normal_(0, self.prior_b_sig2.sqrt())
        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

        self.sample_parameters_sgcp()

    def sample_parameters_proper(self):
        
        # rate_density_ub
        if self.infer_rate_density_ub:
            gamma = torch.distributions.gamma.Gamma(self.rate_density_ub_prior_alpha, self.rate_density_ub_prior_beta)
            self.rate_density_ub = gamma.sample()

        # sample from poisson process
        cK, cM, gK, gM = util.sample_sgcp(T=self.T, \
                                          gp=lambda c: self.gp.sample_y(c.reshape(-1,1), prior=True).reshape(-1), \
                                          rate_density_ub=self.rate_density_ub)
        #cK, cM, gK, gM = util.sample_sgcp(T=self.T, \
        #                                  gp=lambda c: torch.zeros(c.shape).reshape(-1), \
        #                                  rate_density_ub=self.rate_density_ub)

        # width
        self.dim_hidden = cK.shape[0]
        self.M = cM.shape[0]
        self.mask.zero_()
        self.mask[:self.dim_hidden] = 1
        
        # centers
        self.cK.data.zero_()
        self.cK.data[:, self.mask] = cK
        self.cM = cM

        # gp
        self.gK.data.zero_()
        self.gK.data[self.mask] = gK
        self.gM = gM

        # Re-fit gp whenever gMK is updated
        self.fit_gp()

        # output weights
        self.w.data.normal_(0, self.prior_w_sig2.sqrt())
        self.b.data.normal_(0, self.prior_b_sig2.sqrt())

        # observation noise variance
        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

    def sample_functions_prior(self, x, n_samp, proper=False):
        '''
        Sample functions from the prior

        x: points to sample function (n,dim_in)
        n_samp: number of samples

        returns (n_samp, n) tensor
        '''
        y_samp_prior = torch.zeros((n_samp, x.shape[0]))
        for i in range(n_samp):
            if proper:
                self.sample_parameters_proper()
            else:
                self.sample_parameters()
            y_samp_prior[i,:] = self.forward(x).reshape(-1)
        return y_samp_prior

    def _sample_momentum(self):
        '''
        samples network momentum parameters 
        '''
        if self.infer_cK:
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
        g = self.gp.predict(self.cK[:, self.mask].reshape(-1,1)).reshape(-1)
        s2 = self.h(self.rate_density_ub * util.sigmoid_torch(g)).reshape(-1)
        h = torch.exp(-0.5 * (x - self.cK[:, self.mask])**2 * s2)
        return F.linear(h, self.w[:, self.mask], self.b)

    def log_prior(self):
        '''
        log of prior distribution
        '''

        if self.use_gp_term:
            gMK = self.gMK()
            K = self.kernel(self.cMK().reshape(-1,1)) + self.fudge*torch.eye(self.M+self.dim_hidden)
            K[K < 1e-10] = 0.
            chol_L = torch.cholesky(K)
            alpha = torch.cholesky_solve(gMK.reshape(-1,1), chol_L)
            gp_part = -0.5*torch.logdet(K) - 0.5 * gMK.reshape(1,-1) @ alpha
        else:
            gp_part = 0.
      
        log_prior_no_c = util.log_normal(self.w[:, self.mask], self.prior_w_sig2) \
                        + util.log_normal(self.b, self.prior_b_sig2)

        
        return log_prior_no_c + gp_part

    def _log_kinetic_energy(self):
        '''
        Kinetic energy of network
        '''
        ke = 0.

        if self.infer_wb:
            ke += torch.sum(self.p_w[:, self.mask]**2) + torch.sum(self.p_b**2)
            
        if self.infer_cK:
            ke += torch.sum(self.p_c[:, self.mask]**2)

        ke *= .5
        return ke

    def _update_p(self, eps, x, y, rollback_mu):
        '''
        Leapfrog step for network momentum
        '''

        # backprop
        self._zero_gradients_network()
        U = self._log_potential_energy(x, y, rollback_mu)
        if torch.isinf(U):
            return 1
        else:
            U.backward()

            if self.infer_cK:
                self.p_c[:, self.mask] -= eps*self.cK.grad[:, self.mask]
            
            if self.infer_wb:
                self.p_w[:, self.mask] -= eps*self.w.grad[:, self.mask]
                self.p_b -= eps*self.b.grad

            return 0
        

    def _update_q(self, eps):
        '''
        Leapfrog step for network parameters
        '''
        if self.infer_cK:
            self.cK[:, self.mask] += eps*self.p_c[:, self.mask]
        
        if self.infer_wb:
            self.w[:, self.mask] += eps*self.p_w[:, self.mask]
            self.b += eps*self.p_b

    def _negate_momentum(self):
        '''
        Flip sign of network momentum
        '''
        if self.infer_cK:
            self.p_c[:, self.mask] *= -1.

        if self.infer_wb:
            self.p_w[:, self.mask] *= -1.
            self.p_b *= -1.

    def _copy_parameters(self):
        '''
        Store a copy of network parameters
        '''
        self.cK_copy = self.cK.data.clone()
        self.w_copy = self.w.data.clone()
        self.b_copy = self.b.data.clone()

    def _reset_parameters(self):
        '''
        Reset network parameters to stored copy
        '''
        self.cK.data = self.cK_copy.clone()
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

    def _zero_gradients_network(self):
        '''
        Set network gradients cK.grad, w.grad, b.grad to zero
        '''
        if self.cK.grad is not None:
            self.cK.grad.zero_()
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
        g0 = self.cK[:, self.mask] - self.T[0] 
        g1 = self.T[1] - self.cK[:, self.mask]
        
        U1 = torch.sum(torch.log(1 + torch.exp(-rollback_mu * g0)))
        U2 = torch.sum(torch.log(1 + torch.exp(-rollback_mu * g1)))

        return -self.log_posterior(x, y) + U1 + U2

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
        Birth step for unthinned unit

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''
        if self.dim_hidden == self.dim_hidden_max:
            return False
        else:            
            k_proposed = torch.min(torch.nonzero(self.mask==0)).item()
            
            # Record current parameters in case you need to go back
            #c_current = self.cK[:, k_proposed].clone()
            #w_current = self.w[:, k_proposed].clone()

            # Propose new parameters
            c_proposed = torch.empty(1).uniform_(self.T[0], self.T[1]) 
            w_proposed = self.prior_w_sig2.sqrt()*torch.randn(self.dim_out)
            g_proposed = self.gp.sample_y(c_proposed.reshape(-1,1)) 

            # Temporarily update and compute acceptance prob
            self.mask[k_proposed] = 1
            self.cK[:, k_proposed] = c_proposed.clone()
            self.w[:, k_proposed] = w_proposed.clone()
            self.gK[k_proposed] = g_proposed.clone()
            self.fit_gp()

            log_lik_proposed = self.log_likelihood(x,y)

            a = torch.exp(
                    log_lik_proposed \
                    + torch.log(self.rate_density_ub) \
                    + math.log(self.V) \
                    - self.log_lik_current \
                    - math.log(self.dim_hidden+1) \
                    - torch.log(1+torch.exp(-g_proposed)) 
                )

            if torch.rand(1) < a:
                # Accept
                self.dim_hidden += 1
                self.log_lik_current = log_lik_proposed.clone()
                return True
            
            else:
                # Reject (no need to change parameters back, just mask unit)
                self.mask[k_proposed] = 0
                self.fit_gp()
                return False
        
    def death(self, x, y):
        '''
        Death step for unthinned unit

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
            self.fit_gp()

            log_lik_proposed = self.log_likelihood(x,y)

            a = torch.exp(
                    log_lik_proposed \
                    + math.log(self.dim_hidden) \
                    + torch.log(1+torch.exp(-self.gK[k_proposed])) \
                    - self.log_lik_current \
                    - torch.log(self.rate_density_ub) \
                    - math.log(self.V)
                )

            if torch.rand(1) < a:
                # Accept
                self.dim_hidden -= 1
                self.log_lik_current = log_lik_proposed.clone()
                return True
            
            else:
                # Reject (change back)
                self.mask[k_proposed] = 1
                self.fit_gp()
                return False

    def update_rate_density_ub(self, x, y):
        '''
        MH step for rate_density_ub

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        '''

        # record current state
        rate_density_ub_current = self.rate_density_ub.clone()

        # propose new state
        rate_density_ub_proposed = self.rate_density_ub + torch.randn(1)*self.rate_density_ub_proposal_std
        self.rate_density_ub = rate_density_ub_proposed.clone()
        log_lik_proposed = self.log_likelihood(x, y).clone()

        a = torch.exp(
            log_lik_proposed - self.log_lik_current \
            + (self.rate_density_ub_prior_alpha+self.M+self.dim_hidden-1)
            * (torch.log(rate_density_ub_proposed) - torch.log(rate_density_ub_current)) \
            - (rate_density_ub_proposed - rate_density_ub_current)*(self.V+self.rate_density_ub_prior_beta)
            )

        if torch.rand(1) < a:
            # accept
            self.log_lik_current = log_lik_proposed.clone()
            return True

        else:
            # reject (change back)
            self.rate_density_ub = rate_density_ub_current.clone()
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
                'c': torch.zeros((n_samp,)+self.cK.shape),
                'mask': torch.zeros((n_samp, self.dim_hidden_max), dtype=torch.bool),
                'rate_density_ub': torch.zeros(n_samp),
                'gK': torch.zeros(n_samp, self.dim_hidden_max),
                'w': torch.zeros(n_samp, self.dim_hidden_max),
                'predict_y': predict_y,
                'predict_y_plot': predict_y_plot,
                'predict_y_test': predict_y_test,
                'x_test': x_test,
                'y_test': y_test,
            }

        if predict_y:
            samples.update({
                'y_pred': torch.zeros((n_samp, x.shape[0]))
            })

        if predict_y_plot:
            samples.update({
                'x_plot': x_plot,
                'y_plot_pred': torch.zeros((n_samp, x_plot.shape[0])),
                'gp_plot_pred': torch.zeros((n_samp, x_plot.shape[0]))
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
        samples['c'][i,:,:] = self.cK.detach()
        samples['mask'][i,:] = self.mask
        samples['rate_density_ub'][i] = self.rate_density_ub
        samples['gK'][i,:] = self.gK.detach()
        samples['w'][i,:] = self.w.detach().reshape(-1)

        if samples['predict_y']:
            samples['y_pred'][i,:] = self.forward(samples['x']).reshape(-1)

        if samples['predict_y_plot']:
            samples['y_plot_pred'][i,:] = self.forward(samples['x_plot']).reshape(-1)
            samples['gp_plot_pred'][i,:] = self.gp.predict(samples['x_plot']).reshape(-1) 
            #samples['gp_plot_samp'][i,:] = self.gp.sample_y(samples['x_plot']).reshape(-1)

        if samples['predict_y_test']:
            samples['y_test_pred'][i,:] = self.forward(samples['x_test']).reshape(-1)



    def _smallwrite(self, i, eps, eps_sgcp, samples, accept, accept_cum, writer):
        '''
        Add to tensorflow writer (called frequently)
        '''

        writer.add_scalar('eps', eps, i)
        writer.add_scalar('eps_sgcp', eps_sgcp, i)

        writer.add_scalar('dim_hidden', self.dim_hidden, i)
        writer.add_scalar('log_prob/log_likelihood', self.log_likelihood(samples['x'],samples['y']), i)
        writer.add_scalar('log_prob/log_prior', self.log_prior(), i)

        writer.add_scalar('log_prob/log_full_cond_gMK', -self.neg_log_full_cond_gMK(samples['x'],samples['y']), i)

        if self.infer_sig2:
            writer.add_scalar('sig2', self.sig2, i)

        writer.add_scalars('acceptance', accept_cum, i)

        if self.have_ground_truth:
            writer.add_scalars('dist_ground_truth', self.dist_param_truth(), i)

        writer.add_scalars('average_gp_values', {'gK': torch.mean(self.gK[self.mask]), 'gM': torch.mean(self.gM)}, i)
        writer.add_scalars('n_thinned_and_n_unthinned', {'K': self.dim_hidden, 'M': self.M}, i)

        if self.infer_rate_density_ub:
            if self.have_ground_truth:
                writer.add_scalars('rate_density_ub', {'true': self.rate_density_ub_true, 
                                                       'inferred': self.rate_density_ub}, i)
            else:
                writer.add_scalar('rate_density_ub', self.rate_density_ub, i)

        ### intensity plot
        #if i % 5 == 0:
        #    fig, ax = plt.subplots()
        #    c_plot = torch.linspace(-.25,1.25,100).reshape(-1,1)
        #    intensity = self.gp.predict(c_plot)
        #    ax.plot(c_plot.numpy(), intensity.numpy())
        #    ax.scatter(self.cM.numpy(), self.gM.numpy(), label='M')
        #    ax.scatter(self.cK[:, self.mask].detach().numpy(), self.gK[self.mask].detach().numpy(), label='K')
        #    ax.legend()
        #    writer.add_figure('gp',fig,i)
        #    plt.close('all')
        ###

    def _bigwrite(self, i, eps, eps_sgcp, samples, accept, accept_cum, writer):
        '''
        Add to tensorflow write (called infrequently)
        '''
        fig, ax = util.plot_functions(samples['x_plot'], samples['y_plot_pred'][:i,:], samples['x'], samples['y'], samples['x_test'], samples['y_test'])
        writer.add_figure('functions',fig,i)

        fig, ax = util.plot_intensity(samples['x_plot'].numpy(), samples['gp_plot_pred'][:i,:].numpy(), samples['rate_density_ub'][:i].reshape(-1,1).numpy(), samples['x'].numpy())
        if self.have_ground_truth:
            cMK_true = torch.cat((self.cM_true.reshape(-1), self.cK_true[:, self.mask_true].reshape(-1)))
            gMK_true = torch.cat((self.gM_true, self.gK_true[self.mask_true]))
            intensity_true = self.rate_density_ub_true * util.sigmoid_torch(gMK_true)
            ax.scatter(cMK_true.numpy(), intensity_true.numpy(), color='orange', label='true')
        writer.add_figure('intensity/function',fig,i)

        #fig, ax = util.plot_intensity(samples['x_plot'].numpy(), samples['gp_plot_samp'][:i,:].numpy(), samples['rate_density_ub'][:i].reshape(-1,1).numpy(), samples['x'].numpy())
        #if self.have_ground_truth:
        #    cMK_true = torch.cat((self.cM_true.reshape(-1), self.cK_true[:, self.mask_true].reshape(-1)))
        #    gMK_true = torch.cat((self.gM_true, self.gK_true[self.mask_true]))
        #    intensity_true = self.rate_density_ub_true * util.sigmoid_torch(gMK_true)
        #    ax.scatter(cMK_true.numpy(), intensity_true.numpy(), color='orange', label='true')
        #writer.add_figure('intensity/function_samples',fig,i)

        fig, ax = util.plot_centers_trace(samples['gK'][:i,:].detach().numpy(), samples['mask'][:i,:].numpy())
        writer.add_figure('parameters/gK_trace',fig,i)

        fig, ax = util.plot_centers_trace(samples['c'][:i,:].detach().numpy(), samples['mask'][:i,:].numpy())
        writer.add_figure('parameters/c_trace',fig,i)

        fig, ax = util.plot_centers_trace(samples['w'][:i,:].detach().numpy(), samples['mask'][:i,:].numpy())
        writer.add_figure('parameters/w_trace',fig,i)

        fig, ax = plt.subplots()
        ax.hist(torch.squeeze(samples['c'][:i,:])[samples['mask'][:i,:]].detach())
        writer.add_figure('parameters/c_hist',fig,i)

        plt.close('all')

        if samples['predict_y_test']:
            test_ll = util.test_log_likelihood(samples['y_test'].reshape(1,-1), samples['y_test_pred'][:i,:], samples['sig2'][:i].reshape(-1,1))
            writer.add_scalar('test_log_likelihood', test_ll, i)
            ax.set_title('test log lik: %.3f' % test_ll.item())
            
    def sample_posterior(self, x, y, n_samp, 
        eps=.001, L=100, n_adapt_eps=0, \
        eps_sgcp=.001, L_sgcp=100, \
        n_rep_hmc=1, n_rep_resize=1, \
        predict_y = False, x_plot = None, x_test = None, y_test = None, \
        n_print=0, n_bigwrite=0, \
        writer=None, record=True, \
        n_checkpoint=0, dir_checkpoint = './', \
        step1=True, step2=True, step3=True, step4=True):
        '''
        Sample from posterior

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        n_samp: number of samples
        eps: leapfrog stepsize in hmc for network parameters
        L: number of leapfrog steps in hmc for network parameters
        n_adapt_eps: number of times to update eps
        eps_sgcp: leapfrog stepsize in hmc for network parameters
        L_sgcp: number of leapfrog steps in hmc for network parameters
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
        n_checkpoint: number of times to save results
        dir_checkpoint: directory to save results
        step1: whether to perform step 1 (network parameters)
        step2: whether to perform step 2 (network size)
        step3: whether to perform step 3 (intensity)
        step4: whether to perform step 4 (observational noise)
        '''

        if n_adapt_eps > 0:
            writer.add_scalar('eps', eps, 0)

        predict_y_plot = False if x_plot is None else True
        predict_y_test = False if x_test is None or y_test is None else True
        
        # samples per event
        samples_per = lambda n: int(n_samp / n) if n>0 and n<n_samp else n_samp
        n_samp_print = samples_per(n_print)
        n_samp_bigwrite = samples_per(n_bigwrite)
        n_samp_adapt_eps = samples_per(n_adapt_eps)
        n_samp_checkpoint = samples_per(n_checkpoint)
        i_adapt_eps = 0
        
        ## allocate space for results
        accept = {
            'hmc': np.full((n_samp, n_rep_hmc), np.nan),
            'birth': np.full((n_samp, n_rep_resize), np.nan),
            'death': np.full((n_samp, n_rep_resize), np.nan),
            'n_thinned': np.full((n_samp,1), np.nan),
            'thinned_locations': np.full((n_samp,1), np.nan),
            'hmc_gp': np.full((n_samp,1), np.nan),
            'rate_density_ub': np.full((n_samp,1), np.nan)
        }
        accept_cum = dict.fromkeys(accept, 0)

        if record:
            samples = self._init_record(x, y, x_plot, x_test, y_test, n_samp, n_rep_hmc, n_rep_resize, \
                                        predict_y, predict_y_plot, predict_y_test)
        
        # main loop
        for i in range(n_samp):
            self.i = i
            self.log_lik_current = self.log_likelihood(x,y)

            ### Step 1: Update network parameters (w, b, c) with HMC
            if step1:
                for j in range(n_rep_hmc):
                    accept['hmc'][i, j] = self.hmc(x, y, eps, L)

                    if self.set_gp_to_mean:
                        with torch.no_grad():
                            self.gK[self.mask] = self.gp.predict(self.cK[:, self.mask].reshape(-1,1)).reshape(-1)
                            self.gM = self.gp.predict(self.cM.reshape(-1,1)).reshape(-1)

                with torch.no_grad():
                    # adapt leapfrog stepsize for network hmc
                    if i>0 and i % n_samp_adapt_eps == 0:
                        eps_old = eps
                        accept_prev_iter = np.mean(accept['hmc'][n_samp_adapt_eps*i_adapt_eps:i,:])
                        eps = util.rescale_eps(eps, accept_prev_iter)
                        i_adapt_eps+=1
                        print('Epsilon adapted from %.6f to %.6f based on acceptance of %.3f' % (eps_old,eps,accept_prev_iter))
                    
                    # update sgcp after hmc
                    self.fit_gp()

            ### Step 2: birth/death of hidden units
            if step2 and self.infer_K:
                with torch.no_grad():
                    self.log_lik_current = self.log_likelihood(x,y) # assumes log likelihod computed

                    for j in range(n_rep_resize):
                        if torch.rand(1) < 0.5:
                            accept['birth'][i, j] = self.birth(x,y)
                        else:
                            accept['death'][i, j] = self.death(x,y)

            ### Step 3: update intensity
            if step3:
                accept_sgcp = self.update_sgcp(x, y, eps_sgcp, L_sgcp)
                accept['n_thinned'][i] = accept_sgcp[0]
                accept['thinned_locations'][i] = accept_sgcp[1]
                accept['hmc_gp'][i] = accept_sgcp[2]
                accept['rate_density_ub'][i] = accept_sgcp[3]

                # adapt leapfrog stepsize for intensity hmc
                if i>0 and i % n_samp_adapt_eps == 0:
                    eps_sgcp_old = eps_sgcp
                    accept_prev_iter = np.mean(accept['hmc_gp'][n_samp_adapt_eps*(i_adapt_eps-1):i])
                    eps_sgcp = util.rescale_eps(eps_sgcp, accept_prev_iter)
                    print('Epsilon SGCP adapted from %.6f to %.6f based on acceptance of %.3f' % (eps_sgcp_old,eps_sgcp,accept_prev_iter))
                

            with torch.no_grad():

                ### Step 4: Update noise
                if step4 and self.infer_sig2:
                    self.update_sig2(x,y)

                ### Save samples
                for accept_type in accept_cum.keys():
                    accept_cum[accept_type] = np.nanmean(accept[accept_type][:i,:])

                if record:
                    self._record_samp(i, samples)

                self._smallwrite(i, eps, eps_sgcp, samples, accept, accept_cum, writer)
                if i>0 and i % n_samp_bigwrite == 0:
                    self._bigwrite(i, eps, eps_sgcp, samples, accept, accept_cum, writer)

                # print
                if i >0 and i % n_samp_print == 0:
                    print('sample [%d/%d], acceptance_hmc: %.3f, acceptance_birth: %.3f, acceptance_death: %.3f' % \
                        (i, n_samp, accept_cum['hmc'], accept_cum['birth'], accept_cum['death']))

                if i>0 and i % n_samp_checkpoint == 0:
                    print('saving checkpoint at iteration %d' % i)
                    np.save(os.path.join(dir_checkpoint, 'samples.npy'), samples)
                    np.save(os.path.join(dir_checkpoint, 'accept.npy'), accept)

        if record:
            return accept, samples, eps, eps_sgcp
        else:
            return accept, eps, eps_sgcp

    ####### 
    # SGCP
    #######

    def init_parameters_sgcp(self):
        '''
        initialize intensity parameters M, cM, gMK
        '''

        with torch.no_grad():
            if self.infer_M:
                self.M = int(np.maximum(1,int(self.V*self.rate_density_ub - self.dim_hidden))) # total expected number of events - K

            if self.infer_cM:
                self.cM = torch.empty(self.M).uniform_(self.T[0],self.T[1]) # location of thinned events 

            gMK = self.gp.sample_y(self.cMK().reshape(-1,1), prior=False).reshape(-1)
            if self.infer_gM:
                self.gM = torch.randn(self.M)*1e-2

            if self.infer_gK:
                self.gK.data = torch.randn(self.gK.shape)*1e-2

            # Re-fit gp whenever gMK is updated
            self.fit_gp()


    def sample_parameters_sgcp(self):
        '''
        Sample intensity parameters M, cM, gMK from the prior
        '''
        with torch.no_grad():
            self.M = int(np.maximum(1,int(self.V*self.rate_density_ub - self.dim_hidden))) # total expected number of events - K

            self.cM = torch.empty(self.M).uniform_(self.T[0],self.T[1]) # location of thinned events 

            # sample gp
            self.gK.data = torch.zeros(self.gK.shape) 
            gMK = self.gp.sample_y(self.cMK().reshape(-1,1), prior=True).reshape(-1)
            self.gM = gMK[:self.M]
            self.gK.data[self.mask] = gMK[self.M:]

            # Re-fit gp whenever gMK is updated
            self.fit_gp()

    def gMK(self): 
        '''
        return all function values at unmasked locations
        '''
        return torch.cat((self.gM, self.gK[self.mask]))

    def cMK(self):
        '''
        return all unmasked locations
        '''
        return torch.cat((self.cM, self.cK[:, self.mask].reshape(-1)))

    def fit_gp(self):
        '''
        Fit GP posterior gMK ~ GP(cMK)
        '''
        cMK = self.cMK()
        gMK = self.gMK()
        return self.gp.fit(cMK.detach().reshape(-1,1), gMK.detach())

    def update_sgcp(self, x, y, eps, L):
        '''
        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        eps: leapfrog stepsize (float)
        L: number of leapfrog steps (int)
        '''
        with torch.no_grad():
            # number of thinned events M 
            if self.infer_M:
                accept1 = self.update_n_thinned(x, y, n_steps=10) 
            else:
                accept1 = np.nan

            # location of thinned events cM
            if self.infer_cM:
                accept2 = self.update_thinned_locations(x,y) 
            else:
                accept2 = np.nan

            # GP gMK
            if self.infer_gM or self.infer_gK:
                accept3 = self.update_gp_function(x, y, n_steps=1, eps=eps, L=L)
            else:
                accept3 = np.nan

            # rate_density_ub
            if self.infer_rate_density_ub:
                accept4 = self.update_rate_density_ub(x,y)
            else:
                accept4 = np.nan
            
            return np.mean(accept1), np.mean(accept2), np.mean(accept3), accept4

    def update_n_thinned(self, x, y, n_steps=10):
        '''
        Updating number of thinned centers
        '''
        accept = np.zeros(n_steps,dtype=np.bool)
        for i in range(n_steps):
            if torch.rand(1) < .5:
                accept[i] = self.birth_sgcp(x, y)
            else:
                accept[i] = self.death_sgcp(x, y) 
        return accept

    def birth_sgcp(self, x, y):
        '''
        Birth step for thinned center
        '''

        # Propose new location \tilde{s'}
        cM_prime = torch.empty(1).uniform_(self.T[0], self.T[1])

        # Sample g(\tilde{s'}), assuming gp had already been fit to g_{M+K}
        gM_prime = self.gp.sample_y(cM_prime.reshape(-1,1))

        # Temporarily update intensity
        self.cM = torch.cat((self.cM, cM_prime))
        self.gM = torch.cat((self.gM, gM_prime.reshape(-1)))
        self.fit_gp()
        log_lik_proposed = self.log_likelihood(x,y)
        
        # Acceptance prob.
        a_ins = torch.exp(log_lik_proposed - self.log_lik_current)*\
                ((1-self.prob_birth_thinned)*self.V*self.rate_density_ub) / ((self.M+1)*self.prob_birth_thinned*(1+torch.exp(gM_prime)))

        if torch.rand(1) < a_ins:
            # Accept
            self.M += 1
            self.log_lik_current = log_lik_proposed.clone()
            return True
        
        else:
            # Reject
            self.cM = self.cM[:-1]
            self.gM = self.gM[:-1]
            self.fit_gp()
            return False

    def death_sgcp(self, x, y):
        '''
        Death step for thinned center
        '''

        # Uniformly propose thinned event to delete
        if self.M > 1:
            m = np.random.choice(self.M)
        else:
            return False # just skip this one

        # Record current state in case you need to go back
        cM_current = self.cM[m].clone()
        gM_current = self.gM[m].clone()

        # Temporarily update intensity
        self.cM = torch.cat([self.cM[:m], self.cM[m+1:]])
        self.gM = torch.cat([self.gM[:m], self.gM[m+1:]])
        self.fit_gp()
        log_lik_proposed = self.log_likelihood(x,y)

        # Acceptance prob.
        a_del = torch.exp(log_lik_proposed - self.log_lik_current)*\
                (self.M*self.prob_birth_thinned*(1+torch.exp(gM_current))) / ((1-self.prob_birth_thinned)*self.V*self.rate_density_ub)

        if torch.rand(1) < a_del:
            # Accept
            self.M -= 1
            self.log_lik_current = log_lik_proposed.clone()
            return True
        
        else:
            # Reject (go back)
            self.cM = torch.cat([self.cM[:m], cM_current.reshape(1), self.cM[m:]])
            self.gM = torch.cat([self.gM[:m], gM_current.reshape(1), self.gM[m:]])
            self.fit_gp()
            return False

    def update_thinned_locations(self, x, y):
        '''
        MH updates for thinned centers
        '''
        accept = np.zeros(self.M, dtype=np.bool)
        perm_locations = np.random.permutation(self.M) # shuffling ordering to improve mixing
        for m in perm_locations:

            cM_current = self.cM[m].clone()
            gM_current = self.gM[m].clone()

            # Propose new location \tilde{s'}
            cM_prime = self.cM[m] + self.proposal_std_cM*torch.randn(1)

            if cM_prime < self.T[0] or cM_prime > self.T[1]:
                continue

            # Sample g(\tilde{s'}), assuming gp had already been fit to g_{M+K}
            gM_prime = self.gp.sample_y(cM_prime.reshape(-1,1))

            # Temporarily update intensity
            self.cM[m] = cM_prime.clone()
            self.gM[m] = gM_prime.clone()
            self.fit_gp()

            log_lik_proposed = self.log_likelihood(x,y)

            # Acceptance prob.
            a_loc = torch.exp(log_lik_proposed - self.log_lik_current + torch.log(1+torch.exp(self.gM[m])) - torch.log(1+torch.exp(gM_prime)))

            if torch.rand(1) < a_loc:
                # Accept
                accept[m] = True
            else:
                self.cM[m] = cM_current.clone()
                self.gM[m] = gM_current.clone()
                self.fit_gp()

        return accept

    def update_gp_function(self, x, y, n_steps, eps, L):
        '''
        Update GP function in intensity with hmc
        '''
        self.fit_gp()

        accept = np.zeros(n_steps,dtype=np.bool)
        self._copy_parameters_sgcp()
        for i in range(n_steps):
            accept = self.hmc_sgcp(x, y, eps, L)
        return accept

    def neg_log_full_cond_gMK(self, x, y):
        '''
        Objective function for gMK HMC
        '''
        gMK = self.gMK()

        ### cholesky
        alpha = torch.cholesky_solve(gMK.reshape(-1,1), self.gp.L) # can we assume this is already updated?
        sgcp_part = 0.5 * gMK.reshape(1,-1) @ alpha \
               + torch.sum(torch.log(1+torch.exp(-self.gK[self.mask]))) \
               + torch.sum(torch.log(1+torch.exp(self.gM)))

        net_part = -self.log_likelihood(x, y)

        return net_part + sgcp_part      

    def _log_potential_energy_sgcp(self, x, y):
        '''
        negative log posterior
        '''
        return self.neg_log_full_cond_gMK(x,y)

    def _log_kinetic_energy_sgcp(self):
        '''
        Kinetic energy for hmc update of GP
        '''
        ke = 0.0

        if self.infer_gM:
            ke += torch.sum(self.p_gM**2)
        if self.infer_gK:
            ke += torch.sum(self.p_gK[self.mask]**2)

        return .5*ke

    def _sample_momentum_sgcp(self):
        '''
        Sample momemtum for hmc update of GP
        '''
        if self.infer_gM:
            self.p_gM = torch.randn(self.M)
        if self.infer_gK:
            self.p_gK.data.normal_(0,1)

    def _update_p_sgcp(self, eps, x, y):
        '''
        Update momentum for hmc update of GP
        '''
        if self.infer_gM or self.infer_gK:

            self.gM.requires_grad = True # kind of a hack
            if self.gM.grad is not None:
                self.gM.grad.zero_()
            if self.gK.grad is not None:
                self.gK.grad.zero_()

            with torch.enable_grad():
                U = self._log_potential_energy_sgcp(x, y)
                U.backward()
            self.gM.requires_grad = False # kind of a hack

            self.p_gM -= eps*self.gM.grad
            self.p_gK[self.mask] -= eps*self.gK.grad[self.mask]

    def _update_q_sgcp(self, eps):
        '''
        Update positions for hmc update of GP
        '''
        if self.infer_gM:
            self.gM += eps*self.p_gM
        
        if self.infer_gK:
            self.gK[self.mask] += eps*self.p_gK[self.mask]

        # UPDATE GP 
        self.gp.alpha = torch.cholesky_solve(self.gMK().reshape(-1,1), self.gp.L)

    def _copy_parameters_sgcp(self):
        '''
        Make copy of GP function 
        '''
        self.gM_copy = self.gM.clone()
        self.gK_copy = self.gK.data.clone()
        self.alpha_copy = self.gp.alpha.clone()

    def _reset_parameters_sgcp(self):
        '''
        Reset GP function to stored copy
        '''
        self.gM = self.gM_copy.clone()
        self.gK.data = self.gK_copy.clone()
        self.gp.alpha = self.alpha_copy.clone()

    def _negate_momentum_sgcp(self):
        '''
        Multipy momentum variables by -1
        '''
        if self.infer_gM:
            self.p_gM *= -1.
        if self.infer_gK:
            self.p_gK[self.mask] *= -1.

    def hmc_sgcp(self, x, y, eps, L):
        '''
        Single step of Hamiltonian Monte Carlo (HMC)

        Adapted from Neal (2012)

        x: features (n, dim_in)
        y: outcomes (n, dim_out)
        eps: leapfrog stepsize (float)
        L: number of leapfrog steps (int)
        '''

        # sample momentum
        self._sample_momentum_sgcp()

        # record current potential and kenetic energy
        current_K = self._log_kinetic_energy_sgcp()
        current_U = self._log_potential_energy_sgcp(x, y)

        # half step for momentum: p -= eps*grad_U(q)/2
        self._update_p_sgcp(eps/2, x, y)
        
        for i in range(L):
            # full step for positions: q += eps*p
            self._update_q_sgcp(eps)
            
            # full step for momentum: p -= eps*grad_U(q), except at end
            if i != L-1:
                self._update_p_sgcp(eps, x, y)

        # half step for momentum: p -= eps*grad_U(q)/2
        self._update_p_sgcp(eps/2, x, y)

        # Negate the momentum at the end of the trajectory
        self._negate_momentum_sgcp()
        
        # Evaluate potential and kinetic energies at end of trajectory
        proposed_K = self._log_kinetic_energy_sgcp()
        proposed_U = self._log_potential_energy_sgcp(x, y)

        if torch.rand(1) < torch.exp(current_U-proposed_U+current_K-proposed_K):
            # accept (gMK and gp.alpha are already updated)
            self._copy_parameters_sgcp()
            return True
        else:
            # reject, go back to old parameters
            self._reset_parameters_sgcp()
            return False


    def store_true_parameters(self):
        '''
        Stores a copy of all parameters.
        For use if network is initialized to the ground truth parameters. 
        '''
        with torch.no_grad():
            self.have_ground_truth = True

            self.b_true = self.b.clone()
            self.w_true = self.w.clone()
            self.cM_true = self.cM.clone()
            self.cK_true = self.cK.clone()
            self.gM_true = self.gM.clone()
            self.gK_true = self.gK.clone()
            self.M_true = self.M
            self.K_true = self.dim_hidden
            self.mask_true = self.mask.clone()
            self.rate_density_ub_true = self.rate_density_ub.clone()

    def dist_param_truth(self):
        '''
        Compute distance to stored ground truth parameters
        Not useful if network resizes
        '''
        with torch.no_grad():
            d = { \
            'b': torch.sum((self.b - self.b_true)**2).item(), \
            'w': torch.sum((self.w - self.w_true)**2).item(), \
            'cK': torch.sum((self.cK - self.cK_true)**2).item(), \
            'gK': torch.sum((self.gK - self.gK_true)**2).item(), \
            'M': np.abs(self.M - self.M_true), \
            'K': np.abs(self.dim_hidden - self.K_true), \
            'rate_density_ub': torch.abs(self.rate_density_ub - self.rate_density_ub_true).item() \
            }

            if not self.infer_M:
                d['cM'] = torch.sum((self.cM - self.cM_true)**2).item()
                d['gM'] = torch.sum((self.gM - self.gM_true)**2).item()

            return d








