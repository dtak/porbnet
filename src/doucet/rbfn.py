import numpy as np
import numpy.random as npr
import util_doucet as util

from scipy.special import factorial

class RBFN(object):
	def __init__(self, x, y, k, k_fixed=False, l=1, iota=0, \
		nu_0=1, gamma_0=1, sig2 = None,\
		alpha_del2=2, beta_del2=1, del2 = None,\
		eps1=.1, eps2=.1):
		'''
		x: N x d features
		y: N x c outcomes
		k: initial hidden dimension
		k_fixed: whether number of units is fixed (boolean)
		l: activation scale hyperparameter
		iota: expansion of data region (>0)
		'''

		self.x = x
		self.y = y

		self.N = x.shape[0]
		self.d = x.shape[1]

		self.c = y.shape[1]
		self.k = k # width
		self.k_max = k if k_fixed else self.N - (self.d + 1)

		self.k_fixed = k_fixed

		self.l = l
		self.iota = iota

		self.phi = lambda z: np.exp(-0.5*l*z**2) # activation function

		# mask for expanding network. 1=included, 0=exclude. 
		# First 1+d components should never change (always 1) but mask is easier to use this way
		# Use self.mask[-self.k_max:] to get mask for all hidden units only
		# Use np.nonzero(self.mask[-self.k_max:])[0] to get indices of included hidden units
		self.mask = np.zeros(1+self.d+self.k_max, dtype=np.bool)
		self.mask[:1+self.d+self.k] = 1

		## priors

		# sig2 ~ InverseGamma(nu_0/2, gamma_0/2) -- represents observation noise
		self.gamma_0 = gamma_0
		self.nu_0 = nu_0

		# del2 ~ InverseGamma(alpha_del2, beta_del2) -- represents signal-to-noise ratio
		self.alpha_del2 = alpha_del2
		self.beta_del2 = beta_del2

		# Lam ~ Gamma(0.5 + eps1, eps2) -- represents expected number of units
		self.eps1 = eps1
		self.eps2 = eps2

		## Data region
		self.Xi = np.max(x, 0) - np.min(x, 0)
		self.Omega = np.vstack((np.min(x, 0) - self.iota * self.Xi, np.max(x, 0) + self.iota * self.Xi)) # top row is LB, bottom row is UB
		self.H = np.prod((1+2*iota) * self.Xi) # hypervolume of center space

		self.vsigma_star = .05 # for splits and merges

		if sig2 is not None and del2 is not None:
			self.scale_fixed = True
			self.sig2 = sig2 * np.ones(self.c)
			self.del2 = del2 * np.ones(self.c)
			print('using fixed scale')
		else:
			self.scale_fixed = False

		## Intialize parameters to prior
		self.init_parameters()


	def init_parameters(self,sample_K=False):
		'''
		Samples parameters from prior
		'''

		## Hyperparameters

		# del2
		if not self.scale_fixed:
			self.del2 = 1/npr.gamma(self.alpha_del2, 1/self.beta_del2, self.c)

		# Lam
		self.Lam = npr.gamma(0.5 + self.eps1, 1/self.eps2)
		if sample_K:
			self.k = np.random.poisson(self.Lam)
			self.mask = np.zeros(1+self.d+self.k_max, dtype=np.bool)
			self.mask[:1+self.d+self.k] = 1

		## Parameters

		# mu
		self.mu = np.vstack([np.random.uniform(self.Omega[0,:], self.Omega[1,:], self.d) for _ in range(self.k_max)]) # k x d
		self.D = self.compute_design_matrix(self.x) # intialize design matrix: N x (1+d+k_max)

		# sig2
		if not self.scale_fixed:
			self.sig2 = 1/npr.gamma(self.nu_0/2, 1/(self.gamma_0/2), self.c)

		# alpha
		self.alpha = np.zeros((1+self.d+self.k_max, self.c)) 
		DD = self.D[:,self.mask].T@self.D[:,self.mask]
		for i in range(self.c):
			Sigma = util.stable_inv(1/self.del2[i] * DD)
			self.alpha[self.mask,i] = npr.multivariate_normal(np.zeros(1+self.d+self.k), self.sig2[i]*Sigma) # Could sample more efficiently directly from precision
	
	def sample_functions_prior(self, x, n_samp, sample_K=False):
		y_samp_prior = np.zeros((n_samp, x.size))
		for i in range(n_samp):
			self.init_parameters(sample_K=sample_K)
			y_samp_prior[i,:] = self.forward(x.reshape(-1,1)).ravel()
		return y_samp_prior

	def unpack_alpha(self):
		b = self.alpha[0,:]
		beta = self.alpha[1:1+self.d,:] 
		a = self.alpha[self.mask, :][1+self.d:,:]

		return b, beta, a

	def compute_design_matrix(self, x):
		'''
		x: N x D
		Assemble the (N, M=1+D+K) design matrix D(mu, x) from Eq 2.2
		'''
		return np.hstack((np.ones((x.shape[0],1)), x, self.compute_basis_expansion(x)))

	def compute_basis_expansion(self, x):
		'''
		phi(|| x - mu ||)
		'''
		u = x[:,np.newaxis,:] - self.mu[np.newaxis,:,:] # N x K x D
		return self.phi(np.linalg.norm(u, ord=2, axis=2))

	def forward(self, x=None):
		'''
		x: N x D input
		'''
		D = self.D if x is None else self.compute_design_matrix(x)

		'''
		# Equivalent method:
		b, beta, a = self.unpack_alpha()

		basis = self.compute_basis_expansion(x)

		return basis@a + b + x@beta
		'''

		return D[:,self.mask] @ self.alpha[self.mask,:] # N x C

	def log_likelihood(self):
		term1 = -self.c*self.N/2*(np.log(2*np.pi)) - self.N/2*np.sum(np.log(self.sig2))
		term2 = 0
		y_plot_pred = self.forward() # N x c
		for i in range(self.c):
			dy = (self.y[:,i].reshape(-1,1) - y_plot_pred.reshape(-1,1))
			term2 += (-1/(2*self.sig2[i]) * dy.T @ dy).item()
		return term1 + term2

	def log_prior(self):
		term1a = -0.5*self.c*np.log(2*np.pi) - 0.5*np.sum(np.log(self.sig2))
		
		term1b = 0
		DD = self.D[:,self.mask].T@self.D[:,self.mask]
		for i in range(self.c):
			Sigma_inv = 1/self.del2[i] * DD
			term1b += -1/(2*self.sig2[i]) * self.alpha[self.mask,i].reshape(1,-1) @ Sigma_inv @ self.alpha[self.mask,i].reshape(-1,1)

		term2 = -self.k * np.log(self.H) # Assumes mu parameters are in bounds

		k_max = self.N - (self.d + 1) # Don't use self.k_max. Different when k_fixed == True
		term3 = self.k*np.log(self.Lam) - np.log(factorial(self.k)) - np.log(np.sum(self.Lam**np.arange(k_max) / factorial(np.arange(k_max))))

		return term1a + term1b + term2 + term3 


	def train(self, n_samp, x_plot = None, x_test = None, y_test = None, n_print = 0, record=False):
		'''
		MCMC sampling
		'''
		n_samp_print = int(n_samp / n_print) if n_print>0 else np.inf

		# allocate space for acceptance rates
		accept = {
				'Lam': np.zeros(n_samp),
				'mu_unif': np.full(n_samp, np.nan),
				'mu_norm': np.full(n_samp, np.nan)
			}

		if self.k_fixed == False:
			accept.update({
				'birth': np.full(n_samp, np.nan),
				'death': np.full(n_samp, np.nan),
				'split': np.full(n_samp, np.nan),
				'merge': np.full(n_samp, np.nan)
				})

		# Allocate space for samples if recording
		if record:
			#if x_plot == 'grid':
			#	n_pred = 100
			#	XX = np.meshgrid(*[np.linspace(lb,ub,n_pred) for lb,ub in self.Omega.T])
			#	x_plot = np.vstack([xx.ravel() for xx in XX]).T

			samples = {
				'del2': np.zeros((n_samp,) + self.del2.shape),
				'Lam': np.zeros((n_samp,1)),
				'mu': np.zeros((n_samp,) + self.mu.shape),
				'sig2': np.zeros((n_samp,) + self.sig2.shape),
				'alpha': np.zeros((n_samp,) + self.alpha.shape),
				'k': np.zeros((n_samp,1)),
				'x_plot': x_plot,
				'y_plot_pred': np.zeros((n_samp,) + (x_plot.shape[0], self.c)),
				'log_likelihood': np.zeros((n_samp,1)),
				'log_prior': np.zeros((n_samp,1)),
				'log_posterior': np.zeros((n_samp,1)),
				'mask': np.zeros((n_samp, 1+self.d+self.k_max), dtype=np.bool),
				'x': self.x,
				'y': self.y
			}

			predict_y_plot = x_plot is not None
			predict_y_test = x_test is not None and y_test is not None

			if predict_y_test:
				samples.update({
					'x_test': x_test,
					'y_test': y_test,
					'y_test_pred': np.zeros((n_samp,) + (x_test.shape[0], self.c)),
				})

		for samp in range(n_samp):

			#if record: # TEMP
			#	samples['log_likelihood'][samp] = self.log_likelihood() #TEMP
			#	samples['log_prior'][samp,:] = self.log_prior() #TEMP
			#	samples['sig2'][samp,:] = self.sig2 #TEMP
			if record:
				samples['mu'][samp,:,:] = self.mu

				samples['sig2'][samp,:] = self.sig2
				samples['alpha'][samp,:,:] = self.alpha

				samples['del2'][samp, :] = self.del2
				samples['Lam'][samp] = self.Lam

				samples['k'][samp] = self.k

				samples['log_likelihood'][samp] = self.log_likelihood()
				samples['log_prior'][samp,:] = self.log_prior() 
				samples['log_posterior'][samp,:] = samples['log_likelihood'][samp,:] + samples['log_prior'][samp,:]

				samples['mask'][samp,self.mask] = True

				if predict_y_plot:
					samples['y_plot_pred'][samp,:,:] = self.forward(x_plot)

				if predict_y_test:
					samples['y_test_pred'][samp,:,:] = self.forward(x_test)

			if self.k_fixed:
				accept['mu_unif'][samp], accept['mu_norm'][samp] = self.update_mu_values()

			else:
				b_k, d_k, s_k, m_k = self.mu_step_probs()
				u = npr.uniform()
				if u <= b_k:
					accept['birth'][samp] = self.update_mu_birth()
				elif u <= b_k + d_k:
					accept['death'][samp] = self.update_mu_death()
				elif u <= b_k + d_k + s_k:
					accept['split'][samp] = self.update_mu_split()
				elif u <= b_k + d_k + s_k + m_k:
					accept['merge'][samp] = self.update_mu_merge()
				else:
					accept['mu_unif'][samp], accept['mu_norm'][samp] = self.update_mu_values() 

			self.update_nuisance()

			accept['Lam'][samp] = self.update_hyperparameters()

			if samp>0 and samp % n_samp_print == 0:
				if record:
					print('sample: [%d/%d], accept Lam: %.3f, accept mu (unif): %.3f, accept mu (norm): %.3f' % \
						(samp, n_samp, np.mean(accept['Lam'][:samp]), \
						np.nanmean(accept['mu_unif'][:samp]), np.nanmean(accept['mu_norm'][:samp])))
				else:
					print('sample: [%d/%d]' % (samp, n_samp))

		if record:
			return accept, samples
		else:
			return accept

	def mu_step_probs(self, c_star=0.25):

		# birth prob
		if self.k==0 or self.k==self.k_max:
			b_k = 0
		else:
			b_k = c_star * np.minimum(1, self.Lam / self.k)

		# death prob
		if self.k==0:
			d_k = 0
		else:
			d_k = c_star * np.minimum(1, (self.k-1) / self.Lam )

		# split porb
		if self.k==0 or self.k==self.k_max:
			s_k = 0
		else:
			s_k = b_k

		# merge prob
		if self.k==0 or self.k==1:
			m_k = 0
		else:
			m_k = d_k

		return b_k, d_k, s_k, m_k


	def update_mu_values(self, omega_bar = 0.5, sig2_RW = 1):

		accept_unif = np.full(self.k_max, np.nan)
		accept_norm = np.full(self.k_max, np.nan)

		DD = self.D[:,self.mask].T@self.D[:,self.mask]

		for j in np.nonzero(self.mask[-self.k_max:])[0]:

			## Sample from one of two proposal distributions
			if npr.uniform() < omega_bar:
				# Global uniform
				proposal_type = 'unif'
				mu_star = np.random.uniform(self.Omega[0,:], self.Omega[1,:], self.d) # Make this a method?
			else:
				# Local normal
				proposal_type = 'norm'
				mu_star = npr.normal(self.mu[j,:], sig2_RW*np.ones(self.d))

				# Check if mu_star is outside Omega. If so skip update (since acceptance prob will be zero)
				if np.any(np.concatenate((mu_star < self.Omega[0,:], mu_star > self.Omega[1,:]))):
					continue

			D_j_current = self.D[:, 1+self.d+j].copy() # For changing back if rejected

			## Accept/reject

			# posterior before update
			log_r_num = self.log_gamma_yPy_(DD)

			# posterior after update
			self.D[:, 1+self.d+j] = self.phi(np.linalg.norm(self.x - mu_star.reshape(1,-1), ord=2, axis=1)) # Change single column of D based on mu_star
			DD_star = self.D[:,self.mask].T@self.D[:,self.mask]

			log_r_den = self.log_gamma_yPy_(DD_star)

			# acceptance ratio
			r = np.exp((self.N + self.nu_0)/2 * (log_r_num - log_r_den))

			accept = npr.uniform() < r
			if accept:
				self.mu[j,:] = mu_star # Update mu (D already updated)
				DD = DD_star
			else:
				self.D[:, 1+self.d+j] = D_j_current # Change D back (mu stays the same)

			if proposal_type == 'unif':
				accept_unif[j] = accept
			elif proposal_type == 'norm':
				accept_norm[j] = accept

		return np.nanmean(accept_unif), np.nanmean(accept_norm)


	def update_nuisance(self):
		'''
		Updates sig2 and alpha
		'''
		#self.D = self.compute_design_matrix(self.x)
		DD = self.D[:,self.mask].T@self.D[:,self.mask]

		# Update each output dimension separately
		for i in range(self.c):
			M, P = self.compute_M_P_(i, DD)
			h = M @ self.D[:,self.mask].T @ self.y[:,i].reshape(-1,1)

			# Sample from full conditionals 
			beta = (self.gamma_0 + self.y[:,i].reshape(1,-1) @ P @ self.y[:,i].reshape(-1,1)) / 2
			if beta < 0:
				print('P not PSD, skipping sig2 step')
			else:
				if not self.scale_fixed:
					self.sig2[i] = 1/npr.gamma(shape = (self.nu_0 + self.N)/2, \
									   	   	   scale =  1/beta) # Equation 4.1

			self.alpha[self.mask,i] = npr.multivariate_normal(h.reshape(-1), self.sig2[i]*M) # Equation 4.2


	def update_hyperparameters(self):
		'''
		updates del2 and Lam
		'''
		DD = self.D[:,self.mask].T@self.D[:,self.mask]

		if not self.scale_fixed:
			for i in range(self.c):

				# del2
				alpha_del2_post = self.alpha_del2 + (1+self.d+self.k)/2
				beta_del2_post = self.beta_del2 + 1/(2*self.sig2[i]) * self.alpha[self.mask,i].reshape(1,-1) @ DD @ self.alpha[self.mask,i].reshape(-1,1)
				self.del2[i] = 1/npr.gamma(alpha_del2_post, 1/beta_del2_post)

		# Lambda
		Lam_star = npr.gamma(0.5+self.eps1+self.k+1, 1/(1+self.eps2))
		r = np.exp(self.log_p_over_q_lam_(Lam_star) - self.log_p_over_q_lam_(self.Lam))

		accept_Lam = npr.uniform() < r
		if accept_Lam:
			self.Lam = Lam_star

		return accept_Lam

	def update_mu_birth(self):

		# propose new mu
		mu_star = np.random.uniform(self.Omega[0,:], self.Omega[1,:], self.d) 

		# posterior before birth
		DD = self.D[:,self.mask].T@self.D[:,self.mask]
		log_r_num = self.log_gamma_yPy_(DD)

		# posterior after birth
		j_star = self.mask[-self.k_max:].argmin() # always place new unit at first unmasked unit
		self.mask[1+self.d+j_star] = 1 # unmask the proposed unit
		self.D[:, 1+self.d+j_star] = self.phi(np.linalg.norm(self.x - mu_star.reshape(1,-1), ord=2, axis=1)) # basis expansion
		DD_star = self.D[:,self.mask].T@self.D[:,self.mask] # recompute DD
		log_r_den = self.log_gamma_yPy_(DD_star)

		# MH step
		r_birth = np.exp(-0.5*np.sum(np.log(1+self.del2)) + (self.N + self.gamma_0)/2 * (log_r_num - log_r_den) - np.log(self.k+1)) # Eq. 4.7

		accept = npr.uniform() < r_birth
		if accept:
			# D, mask already updated
			self.k += 1
			self.mu[j_star,:] = mu_star
		else:
			self.mask[1+self.d+j_star] = 0
			# no need to change D back since proposed unit is now masked

		return accept

	def update_mu_death(self):
		
		# propose unit to delete
		j_star = npr.choice(np.nonzero(self.mask[-self.k_max:])[0])

		# posterior before death
		DD = self.D[:,self.mask].T@self.D[:,self.mask]
		log_r_num = self.log_gamma_yPy_(DD)

		# posterior after death
		self.mask[1+self.d+j_star] = 0 # mask the proposed unit
		DD_star = self.D[:,self.mask].T@self.D[:,self.mask] # recompute DD
		log_r_den = self.log_gamma_yPy_(DD_star)

		# MH step
		r_birth = np.exp(0.5*np.sum(np.log(1+self.del2)) + (self.N + self.gamma_0)/2 * (log_r_num - log_r_den) + np.log(self.k)) # Eq. 4.7

		accept = npr.uniform() < r_birth
		if accept:
			# mask already updated, D doesn't need to be updated
			self.k -= 1
		else:
			self.mask[1+self.d+j_star] = 1 # change mask back

		return accept


	def update_mu_split(self):
		
		# propose unit to split
		j_split = npr.choice(np.nonzero(self.mask[-self.k_max:])[0])
		mu_split = self.mu[j_split,:]

		# propose two new centers
		u_ms = npr.uniform()
		mu_1 = mu_split - u_ms*self.vsigma_star
		mu_2 = mu_split + u_ms*self.vsigma_star

		# check if distance between mu_1 and mu_2 is less than mu_1 to any other mu
		if self.k>1:
			idx_other = np.concatenate((self.mask[-self.k_max:][:j_split], np.array([False]), self.mask[-self.k_max:][j_split+1:]))
			dist = np.linalg.norm(mu_split.reshape(1,-1) - self.mu[idx_other], 2, 1)
			distance_small = np.linalg.norm(mu_1 - mu_2, 2, 0) < dist.min()
		else:
			distance_small = True

		# proceed if mu_1 and mu_2 are in bounds and closer together than any other center to mu_split
		if np.all(np.concatenate((mu_1 >= self.Omega[0,:], mu_1 <= self.Omega[1,:]))) and \
		   np.all(np.concatenate((mu_2 >= self.Omega[0,:], mu_2 <= self.Omega[1,:]))) and \
		   distance_small:

			# posterior before split
			DD = self.D[:,self.mask].T@self.D[:,self.mask]
			log_r_num = self.log_gamma_yPy_(DD)

			# posterior after split
			j_1 = j_split
			j_2 = self.mask[-self.k_max:].argmin()
			
			self.mask[1+self.d+j_2] = 1 # unmask second unit (first unit already unmasked)
			
			D_j_1_current = self.D[:, 1+self.d+j_1].copy() # in case merge is rejected
			D_j_2_current = self.D[:, 1+self.d+j_2].copy() # in case merge is rejected

			self.D[:, 1+self.d+j_1] = self.phi(np.linalg.norm(self.x - mu_1.reshape(1,-1), ord=2, axis=1)) 
			self.D[:, 1+self.d+j_2] = self.phi(np.linalg.norm(self.x - mu_2.reshape(1,-1), ord=2, axis=1)) 
			
			DD_star = self.D[:,self.mask].T@self.D[:,self.mask] # recompute DD
			log_r_den = self.log_gamma_yPy_(DD_star)

			# MH step
			r_split = np.exp(-0.5*np.sum(np.log(1+self.del2)) + (self.N + self.gamma_0)/2 * (log_r_num - log_r_den) + \
						np.log(self.k * self.vsigma_star / (self.H*(self.k+1)))) # Eq. 4.10

			accept = npr.uniform() < r_split
			if accept:
				# mask, D already changed
				self.k += 1
				self.mu[1+self.d+j_1,:] = mu_1
				self.mu[1+self.d+j_2,:] = mu_2
			else:
				# change back mask at j_2 (j_1 stays unmasked), D at j_1 and j_2
				self.mask[1+self.d+j_2] = 0

				self.D[:, 1+self.d+j_1] = D_j_1_current
				self.D[:, 1+self.d+j_2] = D_j_2_current

			return accept

		else:
			return False


	def update_mu_merge(self):
		
		# propose unit at random 
		j_star = npr.choice(np.nonzero(self.mask[-self.k_max:])[0])
		mu_star = self.mu[1+self.d+j_star,:]

		# find closest center
		dist = np.linalg.norm(mu_star.reshape(1,-1) - self.mu[self.mask[-self.k_max:]], 2, 1)
		j_star_nearest = np.nonzero(self.mask[-self.k_max:])[0][dist.argmin()]
		mu_star_nearest = self.mu[1+self.d+j_star_nearest,:]

		# proceeed if two centers are close enough
		if np.linalg.norm(mu_star - mu_star_nearest, 2, 0) < 2*self.vsigma_star:

			# posterior before merge
			DD = self.D[:,self.mask].T@self.D[:,self.mask]
			log_r_num = self.log_gamma_yPy_(DD)

			# posterior after merge
			mu_merge = (mu_star + mu_star_nearest)/2
			self.mask[1+self.d+j_star_nearest] = 0 # mask nearest unit
			D_j_star_current = self.D[:, 1+self.d+j_star].copy() # in case merge is rejected
			self.D[:, 1+self.d+j_star] = self.phi(np.linalg.norm(self.x - mu_merge.reshape(1,-1), ord=2, axis=1)) # assume merge unit placed at j_star
			DD_star = self.D[:,self.mask].T@self.D[:,self.mask] # recompute DD
			log_r_den = self.log_gamma_yPy_(DD_star)

			# MH step
			r_merge = np.exp(0.5*np.sum(np.log(1+self.del2)) + (self.N + self.gamma_0)/2 * (log_r_num - log_r_den) + \
						np.log(self.k * self.H / (self.vsigma_star*(self.k-1)))) # Eq. 4.10

			accept = npr.uniform() < r_merge
			if accept:
				# j_star_nearest already masked, D already updated
				self.k -= 1
				self.mu[1+self.d+j_star,:] = mu_merge
			else:
				self.mask[1+self.d+j_star_nearest] = 1 # unmask nearest unit
				self.D[:, 1+self.d+j_star]

			return accept

		else:
			return False

	### Helper functions

	def log_p_over_q_lam_(self, Lam):
		k_max = self.N - (self.d + 1) # Don't use self.k_max. Different when k_fixed == True
		return Lam - np.log(Lam) - np.log(np.sum(Lam**np.arange(k_max) / factorial(np.arange(k_max))))

	def compute_M_P_(self, i, DD, fudge_std=1e-8):
		'''
		BC: Results are VERY sensitive to matrix inversion
		'''
		M = util.stable_inv((1 + 1/self.del2[i])*DD)
		P = np.identity(self.N) - self.D[:,self.mask] @ M @ self.D[:,self.mask].T
		return M, P

	def log_gamma_yPy_(self, DD):
		'''
		Computes log prod_i gamma_u + y.T P y, as in various acceptance ratios
		'''
		res = np.zeros(self.c)
		for i in range(self.c):
			_, P = self.compute_M_P_(i, DD)
			res[i] = self.gamma_0 + self.y[:,i].reshape(1,-1) @ P @ self.y[:,i].reshape(-1,1)

		return np.sum(np.log(res))






