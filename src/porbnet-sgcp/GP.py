import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def cdist(x1, x2):
    # squared euclidean distance, same as scipy.spatial.distance.cdist(x1,x2,'sqeuclidean')
    xx1 = x1.unsqueeze(1).expand(x1.shape[0],x2.shape[0],x1.shape[1])
    xx2 = x2.unsqueeze(0).expand(x1.shape[0],x2.shape[0],x2.shape[1])
    return torch.pow(xx1 - xx2,2).sum(2)

class RBFkernel:
    def __init__(self, length_scale, variance):
        self.length_scale = torch.tensor(length_scale)
        self.variance = torch.tensor(variance)

    def __call__(self, x1, x2=None):
        if x2 is None:
            dist = cdist(x1, x1)
        else:
            dist = cdist(x1, x2)
        return self.variance*torch.exp(-0.5*dist/self.length_scale**2)

class GP:
    '''
    Designed to mimic functionality of GaussianProcessRegressor
    from sklearn. Only difference should be fudge factor added to covariance. 

    length_scale: length scale of RBF kernel
    var_y: observation variance
    fudge: added to diagonal of covariance matrix when sampling
    '''
    def __init__(self, kernel, var_y=None, fudge=1e-8):
        self.kernel = kernel
        self.fudge = torch.tensor(fudge)
        self.trained = False

        if var_y is None:
            self.var_y = self.fudge
        else:
            self.var_y = torch.tensor(var_y)

    def predict(self, Xstar):
        '''
        Xstar: (n, 1)
        returns (n, 1) matrix (same as sklearn function)
        '''
        if self.trained:
            Kstar = self.kernel(Xstar, self.X)
            return Kstar @ self.alpha
        else:
            return torch.zeros((Xstar.shape[0],1))

    def sample_y(self, Xstar, prior=False):
        '''
        Xstar: (n, 1)
        returns (n, 1) matrix (same as sklearn function) # Is this right?
        '''
        if not prior and self.trained:
            Kstar = self.kernel(Xstar, self.X)
            mu = (Kstar @ self.alpha).reshape(-1)
            v, _ = torch.solve(Kstar.transpose(0,1), self.L)
            Sigma = self.kernel(Xstar) - v.transpose(0,1) @ v

        else:
            mu = torch.zeros(Xstar.shape[0])
            Sigma = self.kernel(Xstar)
        
        Sigma = (Sigma + Sigma.transpose(0,1))/2
        Sigma = Sigma + self.fudge*torch.eye(Sigma.shape[0]) ## SHOULD FUDGE BE ADDED AGAIN?

        try:
            mvn = MultivariateNormal(mu, Sigma)
            return mvn.sample()
        except:
            print('error with mvn, trying numpy version')

            Sigma = Sigma + 10*self.fudge*torch.eye(Sigma.shape[0]) # add fudge again
            sample = np.random.multivariate_normal(mu.detach().numpy(), Sigma.detach().numpy())
            return torch.from_numpy(sample).to(torch.get_default_dtype())

    def fit(self, X, y, cutoff=1e-10):
        '''
        X:  (n, 1)
        y:  (n,)
        '''
        K = self.kernel(X)
        self.X = X
        self.y = y
        self.trained = True        
        self.L = torch.cholesky(K + self.var_y*torch.eye(X.shape[0]))
        self.alpha = torch.cholesky_solve(y.reshape(-1,1), self.L)

    def grad_mean(self, Xstar):
        '''
        Xstar: (nstar, 1)

        gradient of posterior mean

        ASSUMES RBF KERNEL. To do: fix this
        '''
        if self.trained:
            Xbar = Xstar.reshape(-1,1) - self.X.reshape(1,-1) # (nstar, n)
            Kstar = self.kernel(Xstar.reshape(-1,1), self.X) # (nstar, n)
            return -1/self.kernel.length_scale**2 * \
                   torch.einsum('ij,ij->i', Xbar, Kstar * self.alpha.transpose(0,1))
        else:
            print('GP needs to be fitted first')


