import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stable_inv(A, fudge_mean=1e-8, cutoff=1e-10):
    ''''
    Returns psuedo inverse of matrix A, trying to be as numerically stable as possible
    '''
    fudge =  np.abs(np.random.normal(fudge_mean, fudge_mean*1e-3, 1))
    A[A < cutoff] = 0.0
    A = (A + A.T)/2.0
    return np.linalg.pinv(A + fudge*np.eye(A.shape[0])) + fudge*np.eye(A.shape[0])

def test_log_likelihood(y, y_pred, sig2):
    '''
    y: (N, 1) or (1, N) or (N,)
    y_pred: (Nsamp, N)
    sig2: (Nsamp, 1) or (1, Nsamp) or (Nsamp,)
    '''
    n = norm(loc = y.reshape(1,-1), scale = np.sqrt(sig2).reshape(-1,1))
    prob = n.pdf(y_pred)
    return np.log(np.prod(np.mean(prob, axis=0)))


def plot_functions(x_plot, y_plot_pred, x, y, sig2=None, plot_all = False):
    ci = np.percentile(y_plot_pred, [5, 95], axis=0)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x_plot, np.mean(y_plot_pred,0))
    ax.fill_between(x_plot.ravel(), ci[0,:], ci[1,:],alpha=.5)

    if sig2 is not None:
        ax.fill_between(x_plot.ravel(), ci[0,:]-np.sqrt(sig2), ci[1,:]+np.sqrt(sig2),alpha=.25)

    if plot_all:
        ax.plot(x_plot, y_plot_pred.T,alpha=.05, color='red')
    return fig, ax
