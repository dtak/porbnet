import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
import pandas as pd
import GPy
from scipy.stats import norm

def mean_upcrossings(y, upcross_level=0):
    '''
    returns mean number of upcrossings of upcross_level

    y: (n_samples, n_gridpoints) array of function values
    '''
    u = upcross_level*np.ones(y.shape)
    return np.sum(np.logical_and(y[:,:-1]<u[:,:-1], y[:,1:]>u[:,1:])) / y.shape[0]

def get_gp_function_samples(kernel, n_samples=100,xlims=[-5,5], x_grid=None, aux=1.0):
    ''' returns (n_samples,n_points) function values '''

    x=np.random.randn(1,1)
    if x_grid is None:
        x_grid = np.linspace(xlims[0],xlims[1],200)[:,np.newaxis]
    f_pred = np.zeros((n_samples,x_grid.shape[0],1))
    for r in range(n_samples):
        #y=2.3*np.random.rand(1,1)
        y = aux*np.random.randn(1,1)
        m = GPy.models.GPRegression(x,y,kernel)
        sample = m.posterior_samples_f(x_grid,size=1)
        f_pred[r,:,0] = np.squeeze(sample)
    return f_pred

### DATA RELATED

def normalize_data(x,y,xbias=None,ybias=None,xscale=None,yscale=None,\
        return_only_data=False):
    # TODO: verify dims, adapt to higher dims
    x_original = x.copy()
    y_original = y.copy()
    if xbias is None: xbias = np.min(x_original)
    if ybias is None: ybias = np.mean(y_original)
    if xscale is None: xscale = np.max(x_original) - np.min(x_original);
    if yscale is None: yscale = np.max(np.abs(y_original - ybias))
    x = (x - xbias)*1./xscale;
    y = (y - ybias)*1./yscale;
    if return_only_data:
        return (x,y,x_original,y_original)
    else:
        return (x,y,x_original,y_original,xbias,xscale,ybias,yscale)

def renormalize_data(x_norm,y_norm,xbias,xscale,ybias,yscale):
    x = x_norm*xscale + xbias;
    y = y_norm*yscale + ybias;
    return x,y

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

### postprocessing


'''
# Old version
def compute_loglik(y_test, Yt_hat, s2y):
    T = Yt_hat.shape[0]
    ll = (logsumexp(-(0.5 / s2y) * (y_test[None] - Yt_hat)**2., 0) - np.log(T)
        - 0.5*np.log(2*np.pi) + 0.5*np.log(s2y))
    test_ll = np.mean(ll)
    return test_ll
'''

def compute_loglik(y, y_pred, sig2):
    '''
    y:          (n_obs,) or (n_obs,1) or (1,n_obs)
    y_pred:     (n_samp, n_obs)
    sig2:       scalar or (n_samp,) or (n_samp,1) or (1,n_samp)
    '''
    if not isinstance(sig2, np.ndarray):
        sig2 = np.array([sig2])
    return norm.logpdf(y.reshape(1,-1), loc = y_pred, scale = np.sqrt(sig2).reshape(-1,1)).mean()

def compute_loglik2(y, y_pred, sig2):
    '''
    y:          (n_obs,) or (n_obs,1) or (1,n_obs)
    y_pred:     (n_samp, n_obs)
    sig2:       scalar or (n_samp,) or (n_samp,1) or (1,n_samp)
    '''
    if not isinstance(sig2, np.ndarray):
        sig2 = np.array([sig2])
    logp = norm.logpdf(y.reshape(1,-1), loc = y_pred, scale = np.sqrt(sig2).reshape(-1,1))
    
    return -np.log(y_pred.shape[0])+np.mean(np.log(np.sum(np.exp(logp),axis=0)))

def compute_loglik_alt(y, y_pred, y_pred_var):
    '''
    y:          (n_obs,) or (n_obs,1) or (1,n_obs)
    y_pred:     (n_obs, n_samp)
    y_pred_var: (n_obs, n_samp)]
    '''
    return np.mean(-0.5*np.log(2*np.pi) \
                   -0.5*np.log(y_pred_var) \
                   -0.5*np.square(y.reshape(-1,1) - y_pred)/y_pred_var)

def compute_rmse(y_test, Yt_hat):
    if Yt_hat.ndim == 1:
        MC_pred = Yt_hat;
    else:
        MC_pred = np.mean(Yt_hat, 0)
    rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5
    return rmse


## plotting

def plot_posterior_predictive(
        x_pred, y_pred, x=None, y=None, ax=None, alpha=0.05,
        x_test=None, y_test=None, sig2=None, plot_all=False,
        plot_uncertainty=True, diff_colors=False,s=36):
    ''' input: numpy arrays 
        x_pred: (n_pts,)
        y_pred: (n_smaples,n_pts,1)
    '''
    ci = np.percentile(y_pred, [5, 95], axis=0)

    if ax is None: fig, ax = plt.subplots()
    if x is not None and y is not None: # train data
        ax.scatter(x.ravel(), y.ravel(),s=s,label='train',color='tab:gray')
        pass

    if plot_uncertainty:
        for q in [2.5,5,10]:
            ci = np.percentile(y_pred, [q, 100-q], axis=0)
            ax.fill_between(x_pred.ravel(), ci[0,:,0], ci[1,:,0],alpha=.2, color='tab:blue',lw=0)
        if sig2 is not None:
            #ax.fill_between(x_pred.ravel(), ci[0,:,0]-np.sqrt(sig2), ci[1,:,0]+np.sqrt(sig2), alpha=.25, label=r'$\sigma^2$')
            pass

        ax.plot(x_pred, np.mean(y_pred,0), linewidth=.5, color='tab:blue',alpha=1)

    if x_test is not None and y_test is not None:
        ax.scatter(x_test.ravel(), y_test.ravel(),s=s,label='test',color='tab:red')


    if plot_all: # plot samples
        if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred)
        if diff_colors:
            ax.plot(x_pred, y_pred.T,alpha=alpha)
        else:
            ax.plot(x_pred, y_pred.T,alpha=alpha, color='red')

    ax.grid(linewidth='0.25',alpha=.5)

    return ax

def plot_prior_predictive(x_pred, f_pred, upcross_level=0, bins=20, plot_all_functions=False):
    '''
    x_pred: (n_gridpoints,) numpy array
    f_pred: (n_samples, n_gridpoints) numpy array of function samples
    upcross_level: each sample that crosses this value from below is counted as an upcrossing
    plot_all_functions: whether to show the function samples in the first plot
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

def sigmoid(z):
    return (1+np.exp(-z))**(-1)

def plot_intensity(x_plot, gp_plot_pred, rate_density_ub, ax=None, true_ls_name=None):
    '''
    x_plot: points where intensity was evaluated (n_plot,1)
    gp_plot_pred: samples of gp (n_samp, n_plot)
    rate_density_ub:
    x: observed x data (n_obs, 1)
    '''
    post_rate_density  = rate_density_ub*sigmoid(gp_plot_pred)
    post_ci = np.percentile(post_rate_density, [5,95],axis=0)
    post_mean = np.mean(post_rate_density, axis=0)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_plot, post_mean,color='tab:purple',linewidth=.5,label=r'inferred $\lambda(c)$')

    for q in [2.5,5,10]:
        post_ci = np.percentile(post_rate_density, [q, 100-q],axis=0)
        ax.fill_between(x_plot.reshape(-1), post_ci[0,:], post_ci[1,:], alpha=.2, color='tab:purple',lw=0)

    ax.grid(linewidth='0.25',alpha=.5)
    #ax.legend(['posterior mean'])
    #ax.set_title('Posterior intensity')

    if true_ls_name is not None:

        length_scale, label = get_true_ls(true_ls_name, inverse=True)
        ax.plot(x_plot, length_scale(x_plot), '--', linewidth=.5, color='tab:orange', label=label)
        ax.legend()

    return ax

def get_true_ls(true_ls_name, inverse=False):

    # scaling is harded coded from data/GPdata/create_datatsets.ipynb
    if true_ls_name in ['sin']:
        xbias = -4.9189
        xscale = 9.6491
        length_scale_orig = lambda z: np.sin(z) + 1.1
        label = r'true $l_{sin}^{-1}(x)$'

    elif true_ls_name in ['stat_gap']:
        xbias = -4.9876
        xscale = 9.7886
        length_scale_orig = lambda z: np.ones(z.shape)
        label = r'true $l_{const}^{-1}(x)$'


    elif true_ls_name in ['inc','inc_gap']:
        xbias = -4.9876
        xscale = 9.7886

        if true_ls_name == 'inc_gap':
            label = r'true $l_{inc}^{-1}(x)$'
        elif true_ls_name == 'inc':
            label = r'true $l_{inc2}^{-1}(x)$'

        def length_scale_orig(x, ls_min=.25, ls_max=2, x_min=-5, x_max=5):
            slope = (ls_max - ls_min) / (x_max - x_min) 
            ls = (x - x_min)*slope + ls_min
            return ls**2

    if inverse:
        return lambda z: 1/(length_scale_orig(z*xscale + xbias) / xscale), label
    else:
        return lambda z: length_scale_orig(z*xscale + xbias) / xscale, label



def plot_lengthscale(x_plot, ls_plot, inverse=False, ax=None, true_ls_name=None):
    '''
    x: observed x data (n_obs, 1)
    '''
    if inverse:
        ls_plot = 1/ls_plot

    post_mean = np.mean(ls_plot, axis=0)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_plot, post_mean, color='tab:orange',linewidth=.5, label=r'inferred $l^{-1}(x)$')

    for q in [2.5,5,10]:
        post_ci = np.percentile(ls_plot, [q, 100-q],axis=0)
        ax.fill_between(x_plot.reshape(-1), post_ci[0,:], post_ci[1,:], alpha=.2, color='tab:orange',lw=0)

    ax.grid(linewidth='0.25',alpha=.5)

    if true_ls_name is not None:
        length_scale, label = get_true_ls(true_ls_name, inverse=inverse)

        ax.plot(x_plot, length_scale(x_plot), '--', linewidth=.5, color='tab:orange', label=label)
        ax.legend()

    return ax

##
# LATEX TABLE

def generate_latex_table(data, column_labels, row_labels=None, \
        highlight_best=0, float_format='%.3f', data_highlight=None):
    '''
    For generating latex tables
    Parameters:
      data:          two dimensional array of data
      column_labels: column header
      row_labels:    row labels
      highlight_best: highlight maximum (=1) or minimum (=2) number in each column
    Return:
      latex_table:   latex table as a string
    '''

    if row_labels is not None:
        table = pd.DataFrame(index=row_labels, columns=column_labels, data=data)
    else:
        table = pd.DataFrame(columns=column_labels, data=data)

    #latex_table = '\\begin{table}{h}\n' + table.to_latex() + '\\end{table}'
    if float_format is not None:
        latex_table = table.to_latex(float_format=float_format)
    else:
        latex_table = table.to_latex()
    latex_table = latex_table.replace('textbackslash ','').replace('\\$','$')

    header = latex_table.split('\n')[:2]
    col_headers = latex_table.split('\n')[2]
    col_rule = latex_table.split('\n')[3]
    table_rows = latex_table.split('\n')[4:-3]
    footers = latex_table.split('\n')[-3:]

    col_headers = col_headers.split('&')
    for label, col in zip(column_labels, range(1, table.shape[1] + 1)):
        col_headers[col] = col_headers[col].replace(label, '\\textbf{' + label + '}')
    col_headers = '&'.join(col_headers)

    if row_labels is not None:
        for label, row in zip(row_labels, range(table.shape[0])):
            table_rows[row] = table_rows[row].replace(label, '\\textbf{' + label + '}')
        table_rows = '\n'.join(table_rows)

    latex_table = '\n'.join(header + [col_headers] + [table_rows] + footers)

    if highlight_best > 0:
        if data_highlight is None:
            data_highlight = data

        if highlight_best == 1:
            row_index = np.argmax(data_highlight, axis=0)
        if highlight_best == 2:
            row_index = np.argmin(data_highlight, axis=0)

        table_rows = table_rows.split('\n')
        data_rows = [row.split('&') for row in table_rows]

        for col, row in zip(np.array(range(table.shape[1])), row_index):
            if float_format is not None:
                val = float_format % table.iloc[row,col]
            else:
                val = table.iloc[row,col]

            if row_labels is None:
                data_rows[row][col] = data_rows[row][col].replace(
                        val, '\\textbf{' + val + '}')
            else:
                data_rows[row][col + 1] = data_rows[row][col + 1].replace(
                        val, '\\textbf{' + val + '}')

        table_rows = '\n'.join(['&'.join(row) for row in data_rows])

        latex_table = '\n'.join(header + [col_headers] + [table_rows] + footers)

    return latex_table
