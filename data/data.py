import numpy as np
import os
import scipy.io
import pandas as pd
import ipdb
import sys

# various functions for loading or simulating data. 
# data always returned as numpy arrays with dimension (n_observations, n_features)

def standardize(x, y, x_test=None, y_test=None):
    # rescale
    xbias = x.min()
    xscale = x.max()-x.min()
    yscale = (np.abs(y-y.mean())).max()
    ybias = y.mean()
    x = (x - xbias) / xscale
    y = (y - ybias) / yscale

    if x_test is not None and y_test is not None:
        x_test = (x_test - xbias) / xscale
        y_test = (y_test - ybias) / yscale
        return x, y, x_test, y_test
    else:
        return x, y

def train_val_split(x, y, frac_train=0.8, seed=0):
    np.random.seed(seed)
    n_train = int(frac_train*x.shape[0])
    idx_shuffle = np.arange(x.shape[0])
    np.random.shuffle(idx_shuffle)

    x_train = x[idx_shuffle[:n_train]]
    y_train = y[idx_shuffle[:n_train]]

    x_val = x[idx_shuffle[n_train:]]
    y_val = y[idx_shuffle[n_train:]]

    return x_train, y_train, x_val, y_val

def rescale(y, t0, t1):
    return (y - y.min()) / (y.max() - y.min()) * (t1-t0) + t0

def load_two_rbfs_gap(n_obs=50, c_1=-1, c_2=1, s2_1=2, s2_2=2, lb=-2, ub=2, sig2=1e-5, seed=0):
    np.random.seed(seed)

    x = np.concatenate((np.linspace(lb,-0.5,int(n_obs)/2), np.linspace(0.5,ub,int(n_obs)/2)))
    f = lambda x: -0.25*np.exp(-0.5*s2_1*(x-c_1)**2) + 0.25*np.exp(-0.5*s2_2*(x-c_2)**2)
    y = f(x) + np.random.normal(0,np.sqrt(sig2),x.shape)

    n_test = int(n_obs/5)
    x_test = np.linspace(lb,ub,n_obs)
    y_test = f(x_test) + np.random.normal(0,np.sqrt(sig2),x_test.shape)

    return x.reshape(-1,1), y.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_two_rbfs(n_obs=50, c_1=-1, c_2=1, s2_1=4, s2_2=4, lb=-2, ub=2, sig2=1e-5, seed=0):
    np.random.seed(seed)

    x = np.linspace(lb,ub,n_obs)
    f = lambda x: 0.5*np.exp(-0.5*s2_1*(x-c_1)**2) + 0.5*np.exp(-0.5*s2_2*(x-c_2)**2)
    y = f(x) + np.random.normal(0,np.sqrt(sig2),x.shape)

    n_test = int(n_obs/5)
    x_test = np.linspace(lb,ub,n_obs)
    y_test = f(x_test) + np.random.normal(0,np.sqrt(sig2),x_test.shape)

    return x.reshape(-1,1), y.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_adaptivegp(dir_data, key='Dl', frac_train=0.5, seed=0):
    np.random.seed(seed)

    # load data
    mat = scipy.io.loadmat(os.path.join(dir_data,'adaptivegp/datasets.mat'))
    x = mat[key][0][0][0]

    if key == 'M':
        y = mat[key][0][0][1]
    else:
        y = mat[key][0][0][2]

    # train/test split
    n = x.shape[0]
    n_train = int(frac_train*n)
    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n), idx_train)

    x_train = x[idx_train,:]
    y_train = y[idx_train,:]
    x_test = x[idx_test,:]
    y_test = y[idx_test,:]

    return x_train.reshape(-1,1), y_train.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_motorcycle(dir_data, frac_train=.75, seed=0):
    np.random.seed(seed)

    # read data
    data=pd.read_csv(os.path.join(dir_data,'motorcycle/motorcycle.csv'), sep=',',header=0)
    n = data.shape[0]

    # train/test split
    n_train = int(frac_train*n)
    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n),idx_train)

    x = data['times'].values[idx_train]
    x_test = data['times'].values[idx_test]

    y = data['accel'].values[idx_train]
    y_test = data['accel'].values[idx_test]

    return x.reshape(-1,1), y.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_mimic(dir_data, frac_train=.75, seed=0):
    np.random.seed(seed)

    # load data
    x = np.genfromtxt(os.path.join(dir_data,'hr_pid=11/X.csv'), delimiter=',', skip_header=True)
    y = np.genfromtxt(os.path.join(dir_data,'hr_pid=11/Y.csv'), delimiter=',', skip_header=True)
    y = rescale(y, -1, 1)

    # train/test split
    n = x.shape[0]
    n_train = int(frac_train*n)
    
    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n), idx_train)

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]

    # downsample training data in non-spiky regions to reduce dataset size
    np.random.seed(1)

    breaks = [0,35,45,80,100,140]
    keep_prob = [0.4,1,0.4,1,0.4]

    idx_subsample = []

    for i in range(len(breaks)-1):
        idx = np.nonzero(np.logical_and(x_train >= breaks[i], x_train < breaks[i+1]))[0]
        
        keep = np.random.binomial(1, keep_prob[i], idx.size).astype(np.bool)
        
        idx_subsample.append(idx[keep])
        
    idx_subsample = np.concatenate(idx_subsample)

    x_train_subsample = x_train[idx_subsample]
    y_train_subsample = y_train[idx_subsample]

    return x_train_subsample.reshape(-1,1), y_train_subsample.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_finance(dir_data, frac_train=.75, seed=0):
    np.random.seed(seed)

    # read data
    data = pd.read_csv(os.path.join(dir_data,'finance/finance.csv'))
    price = data['Close'].to_numpy()
    logret = np.log(price[2:]) - np.log(price[1:-1])

    n = logret.size
    time = np.arange(n)

    # train/test split
    n_train = int(frac_train*n)

    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n), idx_train)

    x = time[idx_train]
    y = logret[idx_train]
    x_test = time[idx_test]
    y_test = logret[idx_test]

    x = np.array([float(r) for r in x])
    y = np.array([float(r) for r in y])
    x_test = np.array([float(r) for r in x_test])
    y_test = np.array([float(r) for r in y_test])

    return x.reshape(-1,1), y.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_finance2(dir_data, frac_train=.75, frac_train_downsample=.75, gaps=[(10,40),(220, 250)], seed=0):
    '''
    frac_train_downsample:  fraction of training observations kept in downsampling (by uniformly removing observations)
    '''

    np.random.seed(seed)

    # read data
    data = pd.read_csv(os.path.join(dir_data,'finance2/CBOE_volatility.csv'))
    time = pd.to_datetime(data['DATE']).values.astype(np.int64)
    volatility = data['INDEX'].values
    n = volatility.size
    time = np.arange(n)

    mask = (volatility != '.').nonzero()[0]
    time = time[mask]
    volatility = volatility[mask]

    volatility = volatility.astype(float)

    n = volatility.size
    assert volatility.size == time.size

    x = time
    y = volatility

    print('max x:', x.max())
    # train/test split
    if gaps is None:
        n_train = int(frac_train*n)

        idx_train = np.random.choice(n, n_train, replace=False)
        idx_test = np.setdiff1d(np.arange(n), idx_train)

    else:
        idx_test = np.zeros(x.shape[0], dtype=np.bool)
        for (lb, ub) in gaps:
            idx_test = np.logical_or(idx_test, np.logical_and(x >= lb, x <= ub))
        idx_test= np.nonzero(idx_test)

        idx_train = np.setdiff1d(np.arange(n), idx_test)
        n_train = idx_train.shape[0]

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]

    # downsampling of training observations
    if frac_train_downsample < 1:
        n_train_downsample = int(frac_train_downsample*n_train)
        idx_downsample = np.random.choice(np.arange(n_train), n_train_downsample, replace=False)
        x_train = x_train[idx_downsample]
        y_train = y_train[idx_downsample]



    x_train = np.array([float(r) for r in x_train])
    y_train = np.array([float(r) for r in y_train])
    x_test = np.array([float(r) for r in x_test])
    y_test = np.array([float(r) for r in y_test])

    print('n_train', x_train.shape)
    return x_train.reshape(-1,1), y_train.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_GPdata(dir_data, lengthscale='sin'):
 
    x_train = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'x_train.csv'), delimiter=',', skip_header=False)
    y_train = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'y_train.csv'), delimiter=',', skip_header=False)
    
    x_val = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'x_val.csv'), delimiter=',', skip_header=False)
    y_val = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'y_val.csv'), delimiter=',', skip_header=False)
    
    x_test = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'x_test.csv'), delimiter=',', skip_header=False)
    y_test = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'y_test.csv'), delimiter=',', skip_header=False)
     
    x_plot = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'x_plot.csv'), delimiter=',', skip_header=False)
    f_plot = np.genfromtxt(os.path.join(dir_data, 'GPdata/', lengthscale, 'f_plot.csv'), delimiter=',', skip_header=False)

    return x_train, y_train, x_val, y_val, x_test, y_test, x_plot, f_plot
    
def load_from_m2_sgcp(dir_data, frac_train=0.75, seed=0):
    np.random.seed(seed)

    # load data
    x = np.genfromtxt(os.path.join(dir_data,'from_m2_sgcp/x.csv'), delimiter=',', skip_header=False)
    y = np.genfromtxt(os.path.join(dir_data,'from_m2_sgcp/y.csv'), delimiter=',', skip_header=False)

    # train/test split
    n = x.shape[0]
    n_train = int(frac_train*n)

    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n), idx_train)

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]

    return x_train.reshape(-1,1), y_train.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def load_from_m2(dir_data, intensity='low', variance='low', frac_train=0.75, seed=0):
    '''
    intensity and variance should be either 'low' or 'high'
    '''
    folder = 'from_m2/dataset_%sintensity_%svariance/' % (intensity, variance)

    # load data
    x = np.genfromtxt(os.path.join(dir_data,folder,'x.csv'), delimiter=',', skip_header=False)
    y = np.genfromtxt(os.path.join(dir_data,folder,'y.csv'), delimiter=',', skip_header=False)

    # train/test split
    n = x.shape[0]
    n_train = int(frac_train*n)

    idx_train = np.random.choice(n, n_train, replace=False)
    idx_test = np.setdiff1d(np.arange(n), idx_train)

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]

    return x_train.reshape(-1,1), y_train.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)








