import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt

import util_bnn as util

class BNN(nn.Module):
    '''
    Regular BNN with one hidden layer
    '''
    def __init__(self, dim_in=1, dim_hidden=30, dim_out=1, 
        prior_w1_sig2 = 1., prior_b1_sig2 = 1., \
        prior_w2_sig2 = 1., prior_b2_sig2 = 1., \
        sig2 = None, prior_sig2_alpha = None, prior_sig2_beta = None, s2_0=1.0):
        super(BNN, self).__init__()
        
        # architecture
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        # parameters
        self.w1 = nn.Parameter(torch.Tensor(dim_hidden, dim_in))
        self.b1 = nn.Parameter(torch.Tensor(dim_hidden))
        
        self.w2 = nn.Parameter(torch.Tensor(dim_out, dim_hidden))
        self.b2 = nn.Parameter(torch.Tensor(dim_out))
        
        self.s2_0 = s2_0
        
        # momentum parameters
        self.p_w1 = torch.empty(self.w1.shape)
        self.p_b1 = torch.empty(self.b1.shape)
        self.p_w2 = torch.empty(self.w2.shape)
        self.p_b2 = torch.empty(self.b2.shape)
        
        # priors
        self.prior_w1_sig2 = torch.tensor(prior_w1_sig2)
        self.prior_b1_sig2 = torch.tensor(prior_b1_sig2)
        self.prior_w2_sig2 = torch.tensor(prior_w2_sig2)
        self.prior_b2_sig2 = torch.tensor(prior_b2_sig2)
        
        if prior_sig2_alpha is None or prior_sig2_beta is None:
            self.infer_sig2 = False
            self.sig2 = torch.tensor(sig2)
        else:
            self.infer_sig2 = True
            self.prior_sig2_alpha = prior_sig2_alpha
            self.prior_sig2_beta = prior_sig2_beta
        
        # initialize
        self.init_parameters()
        self.sample_momentum()

        self.trained = False
        self.eps_adapted = 1e-5

    def init_parameters(self):
        # initialize parameters (might be prior, might not)
        self.w1.data.normal_(0, self.prior_w1_sig2.sqrt() * 1e-2)
        self.b1.data.normal_(0, self.prior_b1_sig2.sqrt() * 1e-2)
        self.w2.data.normal_(0, self.prior_w2_sig2.sqrt() * 1e-2)
        self.b2.data.normal_(0, self.prior_b2_sig2.sqrt() * 1e-2)

        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()
        
    def sample_parameters(self):
        # sample from prior. 
        self.w1.data.normal_(0,self.prior_w1_sig2.sqrt())
        self.b1.data.normal_(0,self.prior_b1_sig2.sqrt())
        self.w2.data.normal_(0,self.prior_w2_sig2.sqrt())
        self.b2.data.normal_(0,self.prior_b2_sig2.sqrt())

        if self.infer_sig2:
            gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha, self.prior_sig2_beta)
            self.sig2 = 1/gamma.sample()

    def sample_functions_prior(self, x, n_samp):
        y_samp_prior = torch.zeros((n_samp, x.shape[0]))
        for i in range(n_samp):
            self.sample_parameters()
            y_samp_prior[i,:] = self.forward(x).reshape(-1)
        return y_samp_prior

    def sample_momentum(self):
        self.p_w1.data.normal_(0,1)
        self.p_b1.data.normal_(0,1)
        self.p_w2.data.normal_(0,1)
        self.p_b2.data.normal_(0,1)

    def forward(self, x):
        z = F.linear(x, self.w1, self.b1)
        h = torch.exp(-0.5 * self.s2_0 * z**2)
        return F.linear(h, self.w2, self.b2)

    def log_prior(self):
        return util.log_normal(self.w1, self.prior_w1_sig2) \
             + util.log_normal(self.b1, self.prior_b1_sig2) \
             + util.log_normal(self.w2, self.prior_w2_sig2) \
             + util.log_normal(self.b2, self.prior_b2_sig2) \

    def log_kinetic_energy(self):
        return 0.5*(torch.sum(self.p_w1**2) + \
                    torch.sum(self.p_b1**2) + \
                    torch.sum(self.p_w2**2) + \
                    torch.sum(self.p_b2**2))

    def update_p(self, eps):
        self.p_w1 -= eps*self.w1.grad
        self.p_b1 -= eps*self.b1.grad
        self.p_w2 -= eps*self.w2.grad
        self.p_b2 -= eps*self.b2.grad

    def update_q(self, eps):
        self.w1 += eps*self.p_w1
        self.b1 += eps*self.p_b1
        self.w2 += eps*self.p_w2
        self.b2 += eps*self.p_b2

    def negate_momentum(self):
        self.p_w1 *= -1.
        self.p_b1 *= -1.
        self.p_w2 *= -1.
        self.p_b2 *= -1.

    def copy_parameters(self):
        self.w1_copy = self.w1.data.clone()
        self.b1_copy = self.b1.data.clone()
        self.w2_copy = self.w2.data.clone()
        self.b2_copy = self.b2.data.clone()

    def reset_parameters(self):
        self.w1.data = self.w1_copy.clone()
        self.b1.data = self.b1_copy.clone()
        self.w2.data = self.w2_copy.clone()
        self.b2.data = self.b2_copy.clone()
            
    def log_likelihood(self, x, y):
        return -0.5*x.shape[0]*(math.log(2*math.pi) + torch.log(self.sig2)) - torch.sum((y - self.forward(x))**2)/(2*self.sig2)

    def log_posterior(self, x, y):
        return self.log_likelihood(x, y) + self.log_prior()
    
    def log_potential_energy(self, x, y, zero_gradients=True):
        if zero_gradients:
            [p.grad.zero_() for p in self.parameters()]
        return -self.log_posterior(x, y)

    def update_sig2(self, x, y):
        SSR = torch.sum((y - self.forward(x))**2)
        gamma = torch.distributions.gamma.Gamma(self.prior_sig2_alpha + 0.5*x.shape[0], self.prior_sig2_beta + 0.5*SSR)
        self.sig2 = 1/gamma.sample()
            
    def hmc(self, x, y, eps, L):
        'Single step of Hamiltonian Monte Carlo (HMC)'
        'Adapted from Neal (2012)'
        # assumes current_U already computed (and backproped)
        
        # sample momentum
        self.sample_momentum()

        # record current potential and kenetic energy
        current_K = self.log_kinetic_energy()

        # half step for momentum: p -= eps*grad_U(q)/2
        self.update_p(eps/2)
        
        for i in range(L):

            with torch.no_grad():
                # full step for positions: q += eps*p
                self.update_q(eps)
            
            # backprop
            U = self.log_potential_energy(x, y, zero_gradients=True)
            U.backward()
            
            # full step for momentum: p -= eps*grad_U(q), except at end
            with torch.no_grad():
                if i != L-1:
                    self.update_p(eps)
            
        # half step for momentum: p -= eps*grad_U(q)/2
        self.update_p(eps/2)

        # Negate the momentum at the end of the trajectory
        self.negate_momentum()

        proposed_K = self.log_kinetic_energy()
        
        # Evaluate potential and kinetic energies at end of trajectory
        proposed_U = self.log_potential_energy(x,y, zero_gradients=True)
        proposed_U.backward() # For next iteration
        
        with torch.no_grad():
            if torch.rand(1) < torch.exp(self.current_U-proposed_U+current_K-proposed_K):
                # accept
                self.current_U = proposed_U.clone()
                self.copy_parameters()
                return True
            else:
                # reject, go back to old parameters
                self.reset_parameters()
                return False

    def train(self, x, y, n_samp, L=100, eps=None, n_adapt_eps=None, numpy=True, n_print=0):
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

        if self.trained:
            n_adapt_eps_use = 1 if n_adapt_eps is None else n_adapt_eps
            eps_use = self.eps_adapted if eps is None else eps
        else:
            eps_use = 1e-05 if eps is None else eps
            n_adapt_eps_use = int(n_samp/10) if n_adapt_eps is None else n_adapt_eps

        if numpy:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        _, eps_use = self.sample_posterior(x=x, y=y, n_samp=n_samp, x_plot=None, eps=eps_use, L=100, n_adapt_eps=n_adapt_eps_use, \
                                           n_rep_hmc=1, record=False, n_print=n_print)

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
        self.w1.data = weights['w1'].clone()
        self.b1.data = weights['b1'].clone()
        self.w2.data = weights['w2'].clone()
        self.b2.data = weights['b2'].clone()
        self.sig2 = weights['sig2'].clone()

    def get_network_params(self):
        weights = dict()
        weights['w1'] = self.w1.data.clone()
        weights['b1'] = self.b1.data.clone()
        weights['w2'] = self.w2.data.clone()
        weights['b2'] = self.b2.data.clone()
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
        w1_init = self.w1.data.clone()
        b1_init = self.b1.data.clone()
        w2_init = self.w2.data.clone()
        b2_init = self.b2.data.clone()
        sig2_init = self.sig2.clone()

        accept, samples, _ = self.sample_posterior(x=x, y=y, n_samp=n_samp, x_plot=x_predict, eps=self.eps_adapted, n_adapt_eps=0, \
                                                  n_rep_hmc=1, record=True, n_print=n_print)

        # reset parameters
        self.w1.data = w1_init.clone()
        self.b1.data = b1_init.clone()
        self.w2.data = w2_init.clone()
        self.b2.data = b2_init.clone()
        self.sig2 = sig2_init.clone()

        mean = torch.mean(samples['y_plot_pred'],0)
        std = torch.std(samples['y_plot_pred'],0)
        #import pdb; pdb.set_trace()

        #if std[0] == 0.0:
        #    import pdb; pdb.set_trace()

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
        n_rep_hmc=1, \
        predict_y = False, x_plot = None, x_test = None, y_test = None, \
        n_print=0, n_bigwrite=0, \
        writer=None, record=True):
        '''
        if record==True, copies of the data are stored (in addition to posterior samples)
        '''

        if n_adapt_eps > 0 and writer is not None:
            writer.add_scalar('eps', eps, 0)

        predict_y_plot = False if x_plot is None else True
        predict_y_test = False if x_test is None or y_test is None else True
        
        # samples per event
        n_samp_print = int(n_samp / n_print) if n_print>0 else n_samp
        n_samp_bigwrite = int(n_samp / n_bigwrite) if n_bigwrite>0 else n_samp
        n_samp_adapt_eps = int(n_samp / n_adapt_eps) if n_adapt_eps>0 else n_samp
        i_adapt_eps = 0

        # just need to initialize energy or will break, kind of a hack for now
        self.current_U = self.log_potential_energy(x, y, zero_gradients=False) 
        self.current_U.backward()

        ## allocate space for results
        accept = {
            'hmc': np.full((n_samp, n_rep_hmc), np.nan)
        }
        accept_cum = {
            'hmc': 0
        }

        if record:
            samples = {
                'x': x,
                'y': y,
                'sig2': torch.zeros(n_samp),
                'w1': torch.zeros((n_samp,)+self.w1.shape),
                'w2': torch.zeros((n_samp,)+self.w2.shape),
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
                    'x_test': x_test,
                    'y_test': y_test,
                    'y_test_pred': torch.zeros((n_samp, x_test.shape[0]))
                })

        # main loop
        for i in range(n_samp):

            for j in range(n_rep_hmc):
                ## hmc (assumes current_U computed and backproped)
                self.current_U = self.log_potential_energy(x, y, zero_gradients=True)
                self.current_U.backward()
                self.copy_parameters()

                accept['hmc'][i, j] = self.hmc(x, y, eps, L)

            # adapt eps
            if i >0 and i % n_samp_adapt_eps == 0:
                eps_old = eps
                accept_prev_iter = np.mean(accept['hmc'][n_samp_adapt_eps*i_adapt_eps:i,:])
                eps_new = util.rescale_eps(eps, accept_prev_iter)
                eps = (eps_old + eps_new)/2.0
                i_adapt_eps+=1
                if writer is not None:
                    writer.add_scalar('eps', eps, i)
                print('Epsilon adapted from %.6f to %.6f based on acceptance of %.3f' % (eps_old,eps,accept_prev_iter))

            with torch.no_grad():
                self.log_lik_current = self.log_likelihood(x,y) # assumes log likelihod computed

                ## noise
                if self.infer_sig2:
                    self.update_sig2(x,y)

                ## saving

                # writer
                for accept_type in accept_cum.keys():
                    accept_cum[accept_type] = np.nanmean(accept[accept_type][:i,:])

                if writer is not None:
                    writer.add_scalar('log_prob/log_likelihood', self.log_likelihood(x,y), i)
                    writer.add_scalar('log_prob/log_prior', self.log_prior(), i)

                    if self.infer_sig2:
                        writer.add_scalar('sig2', self.sig2, i)

                    writer.add_scalars('acceptance', accept_cum, i)

                # record
                if record:
                    samples['sig2'][i] = self.sig2
                    samples['w1'][i,:,:] = self.w1
                    samples['w2'][i,:,:] = self.w2

                    if predict_y:
                        samples['y_pred'][i,:] = self.forward(x).reshape(-1)

                    if predict_y_plot:
                        samples['y_plot_pred'][i,:] = self.forward(x_plot).reshape(-1)

                    if predict_y_test:
                        samples['y_test_pred'][i,:] = self.forward(x_test).reshape(-1)


                    if i>0 and i % n_samp_bigwrite == 0 and writer is not None:
                        fig, ax = util.plot_functions(x_plot, samples['y_plot_pred'][:i,:], x, y, x_test, y_test)
                        writer.add_figure('functions',fig,i)

                        if predict_y_test:
                            test_ll = util.test_log_likelihood(y_test.reshape(1,-1), samples['y_test_pred'][:i,:], samples['sig2'][:i].reshape(-1,1))
                            writer.add_scalar('test_log_likelihood', test_ll, i)
                            ax.set_title('test log lik: %.3f' % test_ll.item())
                            # ADD: training and test coverage

                # print
                if i >0 and i % n_samp_print == 0:
                    print('sample [%d/%d], acceptance_hmc: %.3f' % \
                        (i, n_samp, accept_cum['hmc']))

        if record:
            return accept, samples, eps
        else:
            return accept, eps
        
            
