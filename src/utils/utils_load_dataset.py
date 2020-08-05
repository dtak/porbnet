import numpy as np
#import torch
from collections import OrderedDict

#from sklearn.model_selection import train_test_split

import os
import pickle
from datetime import datetime
import pdb

class Split():

    def __init__(self,x,y, **kwargs):
        assert type(x) == type(y)
        self.x = x
        self.y = y
        if type(x) == torch.Tensor: self.tensorized == True
        else: self.tensorized = False

    def to_torch(self):
        if not(self.tensorized):
            torch_kwargs = dict()
            for attr, value in self.__dict__.items():
                if type(value) == np.ndarray:
                    torch_kwargs[attr] = torch.from_numpy(value.astype(np.float32))
            self.__dict__.update(torch_kwargs)
            self.tensorized = True

    def to_numpy(self):
        if self.tensorized:
            numpy_kwargs = dict()
            for attr, value in self.__dict__.items():
                if type(value) == torch.Tensor:
                    numpy_kwargs[attr] = value.data.numpy()
            self.__dict__.update(numpy_kwargs)
            self.tensorized = False

    def copy(self):
        kwargs = {}
        state = self.tensorized
        if state: self.to_numpy()
        new_split = Split(self.x.copy(), self.y.copy(), **kwargs)
        if state: self.to_torch()
        return new_split

class Dataset():

    def __init__(self,train=None, val=None, test=None,
            X_colnames=None, Y_colnames=None, **kwargs):
        if train is not None: self.train = train # Split object
        if val is not None: self.val = val
        if test is not None: self.test = test
        if X_colnames is not None: self.X_colnames = X_colnames
        if Y_colnames is not None: self.Y_colnames = Y_colnames
        self.to_numpy()
        self.tensorized = False

    def to_torch(self):
        for attr, split in self.__dict__.items():
            if type(split) == Split: split.to_torch()
        self.tensorized = True

    def to_numpy(self):
        for attr, split in self.__dict__.items():
            if type(split) == Split: split.to_numpy()
        self.tensorized = False

    def copy(self):
        kwargs = {}
        for attr, value in self.__dict__.items():
            if type(value) == Split or type(value) == np.ndarray:
                kwargs[attr] = value.copy()
            else:
                kwargs[attr] = value
        new_db = Dataset(**kwargs)
        return new_db

#MIMIC_PATH = '/n/dtak/mimic-iii-v1-4/hypotension_management_dataset/query-data/'
CSV_FILE_PATH = os.path.join(os.environ['HOME'],'datasets/mimic/hr.csv')
ATTR_LIST = ['hr','map','dbp','sbp']
DEFAULT_FIX_ATTR = ['subject_id','icustay_id']

def load_unique_fields(csv_filename,req_attr_timeserie=DEFAULT_FIX_ATTR,
        timeserie_ids_pkl_fn='./timeserie_ids.pkl'):
    if os.path.exists(timeserie_ids_pkl_fn):
        with open(timeserie_ids_pkl_fn,'rb') as f:
            timeserie_ids = pickle.load(f)
    else:
        # do computation
        f = open(csv_filename,'r')
        timeserie_ids = OrderedDict()
        header_fields = [x.strip() for x in f.readline().split(',')]
        idxs_timeserie_attr = np.in1d(
                header_fields,req_attr_timeserie).nonzero()[-1]
        for ll, line in enumerate(f.readlines()):
            fields = np.array([x.strip() for x in line.split(',')])
            key = ','.join(fields[idxs_timeserie_attr])
            if key in timeserie_ids.keys():
                timeserie_ids[key].append(ll)
            else:
                timeserie_ids[key] = []
        # save pickle file
        with open(timeserie_ids_pkl_fn,'wb') as f:
            pickle.dump(timeserie_ids,f)
        f.close()
    return timeserie_ids

def load_mimic_patient(patient_id,
        csv_filename=CSV_FILE_PATH,
        req_attr_timeserie=DEFAULT_FIX_ATTR):

    # if first time, create list subject_ids
    # key = timeserie id, either a combination of fields or single fields
    # value = idxs in the file
    timeserie_ids = load_unique_fields(csv_filename,DEFAULT_FIX_ATTR) # ordered dict
    V = list(timeserie_ids.items())
    line_idxs = V[patient_id][1]

    x = []
    y = []
    f = open(csv_filename,'r')
    for ll, line in enumerate(f.readlines()):
        if (ll == 0):
            header = [x.strip() for x in line.split(',')]
            idx_x = header.index('charttime')
            idx_y = header.index('value')
            continue
        if ll in line_idxs:
            fields = [x.strip() for x in line.split(',')]
            date = datetime.strptime(fields[idx_x],'%Y-%m-%d %H:%M:%S')
            x.append( (date - datetime(1970,1,1)).total_seconds() )
            y.append(float(fields[idx_y]))
        if ll > max(line_idxs):
            break
    f.close()
    x = np.array(x)
    x = (x - min(x))/float(3600) # x-axis in hours
    y = np.array(y)
    return (x,y)

def load_mimic_patient_gap(dir_data, subject_id):
    subject_id = int(subject_id)
    #gaps = {11: [(1,10), (58,72)], 1228: [(13,25),(60,72)], 1472: [(22,37),(65,75)], 1535: [(6,12)]}
    gaps = {11: [(54,76)], 1228: [(13,25),(60,72)], 1472: [(22,37),(65,75)], 1535: [(6,12)]}

    if subject_id not in gaps.keys():
        print('mimic subject_id not found')
        return None

    x, y = load_mimic_patient(subject_id, csv_filename=os.path.join(dir_data, 'hr.csv'))
    
    idx_test = np.full(x.shape, False)
    for (lb, ub) in gaps[subject_id]:
        idx_test = np.logical_or(idx_test, np.logical_and(x>=lb, x<=ub))
    idx_train = np.logical_not(idx_test)
 
    return x[idx_train], y[idx_train], x[idx_test], y[idx_test]

def load_2D_sin(binarize=False):
    ''' generate labels given sinusoidal function '''
    x_aux = np.arange(0,10,0.1)
    #X = np.array([(x,y) for x in x_aux for y in x_aux])
    X = np.meshgrid(x_aux,x_aux)

    u = 1.0; v = 1.0; phi = 0.0
    Y = np.sin(2*np.pi*(u*X[0])+phi) + \
        np.sin(2*np.pi*(v*X[1])+phi) + \
            np.random.randn(X[0].shape[0],X[0].shape[1])*np.sqrt(0.1)
    if binarize:
        Y[Y<0.5] = 0.0
        Y[Y>=0.5] = 1.0
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1,1)
    ax.imshow(Y)
    plt.show()
    import ipdb; ipdb.set_trace()
    return Y

def generate_label_sin(x,y,sig2=0.1,B=5.0):
    u = 1/10.0; v = 1/10.0; phi = 2.5 #- np.pi/8
    z = np.cos(2*np.pi*(u*x+v*y)) + \
        np.sin(2*np.pi*(u*x+v*y)+phi)    
    noise = np.random.randn(z.shape[0])*np.sqrt(sig2)
    z = z + noise   
    return z

def generate_label_xor(x,y,sig2=0.1,B=5.0):
    z = np.not_equal(x > B, y > B).astype('float64')  
    noise = np.random.randn(z.shape[0])*np.sqrt(sig2)
    z = z + noise   
    return z


def generate_2D_data(N,sig2=0.01,sig2_sampling=0.1,B=5.0,\
                    binarize=True, label='sin'):
    ''' 
	function to generate 2D synthetic data, either for regression or classification
	sig2: label noise 
        sig2_sampling: spread of clusters
        N: nr. datapoints
    '''
    # sample points from mixture of Gaussians
    X = []; Y = []
    mus = np.array([[0.0,0.0],[0.0,B],[B,0.0],[B,B]])
    K = mus.shape[0]
    n_per_cluster = np.random.multinomial(N,np.ones(K)*1.0/K)
    for k in range(K):
        x_tmp = np.random.multivariate_normal(
            mus[k],sig2_sampling*np.eye(2),size=n_per_cluster[k])
        if label=='sin':
            y_tmp = generate_label_sin(x_tmp[:,0],x_tmp[:,1],\
                                   sig2=sig2,B=B)
        elif label=='xor':
            y_tmp = generate_label_xor(x_tmp[:,0],x_tmp[:,1],\
                                   sig2=sig2,B=B/2.)
        X.append(x_tmp)
        Y.append(y_tmp)
    X = np.vstack(X)
    Y = np.hstack(Y)
    idxs = np.random.permutation(N)
    X = X[idxs,:]
    Y = Y[idxs]
    if binarize:
        Y[Y>=0.5] = 1.0
        Y[Y<0.5] = 0.0
    return (X,Y)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f,ax_vec = plt.subplots(4,4,figsize=(16,16))
    for k,ax in enumerate(ax_vec.ravel()):
        x,y = load_mimic_patient(k+11)
        ax.plot(x,y,'-o')
        ax.set_xlabel('time (hours)')
        ax.set_ylabel('HR')
    plt.show()

