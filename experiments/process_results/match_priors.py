import GPy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path
import torch
import util
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dir_base', type=str, default='../../', help='directory of source code')
parser.add_argument('--dir_out', type=str, default='output/match_priors', help='directory of output')

parser.add_argument('--n_samp', type=int, default=1000)
parser.add_argument('--n_grid', type=int, default=2)

parser.add_argument('--skip_bnn', action='store_true')
parser.add_argument('--skip_porbnet', action='store_true')
parser.add_argument('--skip_porbnet_sgcp', action='store_true')

parser.add_argument('--prior_b2_sig2_add', default=.1, type=float, help='assume this is used for actual inference models')

parser.add_argument('--plot', action='store_true')

# bnn
parser.add_argument('--bnn_param1_min', type=float, default=50)
parser.add_argument('--bnn_param1_max', type=float, default=100)

parser.add_argument('--bnn_param2_min', type=float, default=.05)
parser.add_argument('--bnn_param2_max', type=float, default=.1)

parser.add_argument('--bnn_prior_b1_sig2', type=float, default=10.0)
parser.add_argument('--bnn_dim_hidden', type=int, default=50)
parser.add_argument('--bnn_s2_0', type=float, default=1.0)

# porbnet
parser.add_argument('--porbnet_param1_name', type=str, default='intensity', help='set to "intensity" or "s2_0"')
parser.add_argument('--porbnet_param1_min', type=float, default=10)
parser.add_argument('--porbnet_param1_max', type=float, default=20)

parser.add_argument('--porbnet_param2_min', type=float, default=.05)
parser.add_argument('--porbnet_param2_max', type=float, default=.1)

parser.add_argument('--intensity', type=float, default=10.0)
parser.add_argument('--porbnet_s2_0', type=float, default=1.0)

parser.add_argument('--porbnet_run_cross', action='store_true', help='run single slice of each parameter instead of gird')

args = parser.parse_args()


sys.path.append(os.path.join(args.dir_base,'src/bnn'))
import networks_bnn
import util_bnn

sys.path.append(os.path.join(args.dir_base,'src/porbnet'))
import networks_porbnet
import util_porbnet

sys.path.append(os.path.join(args.dir_base,'src/porbnet-sgcp'))
import networks_porbnet_sgcp
import util_porbnet_sgcp

def avg_upcrossings(f_pred, level=0):
    u = level*np.ones(f_pred.shape[1])
    upcross = np.logical_and(f_pred[:,:-1]<u[:-1], f_pred[:,1:]>u[1:])
    return np.mean(np.sum(upcross,1))

def avg_variance(f_samp):
    return np.mean(np.var(f_samp,0))

def find_all_files(folder, file):
    return [str(path) for path in Path(folder).rglob(file)]

def run_cross(get_f_samp, ls_param_list, var_param_list, variance_expression=None):
    grid_shape = (len(ls_param_list), len(var_param_list))
    var_grid = np.zeros(grid_shape)
    upcross_grid = np.zeros(grid_shape)

    for i, ls_param in enumerate(ls_param_list):
        f_samp, _ = get_f_samp(ls_param, var_param_list[0])
        upcross_grid[i,:] = avg_upcrossings(f_samp)

    for j, var_param in enumerate(var_param_list):
        f_samp, net = get_f_samp(ls_param_list[0], var_param)
        if variance_expression is None:
            var_grid[:,j] = avg_variance(f_samp)
        else:
            var_grid[:,j] = variance_expression(net)

    return var_grid, upcross_grid

def run_grid(get_f_samp, param1_list, param2_list, variance_expression=None):
    grid_shape = (len(param1_list), len(param1_list))
    var_grid = np.zeros(grid_shape)
    upcross_grid = np.zeros(grid_shape)

    for i, param1 in enumerate(param1_list):
        for j, param2 in enumerate(param2_list):

            f_samp, net = get_f_samp(param1, param2)
            
            if variance_expression is None:
                var_grid[i,j] = avg_variance(f_samp)
            else:
                var_grid[i,j] = variance_expression(net)

            upcross_grid[i,j] = avg_upcrossings(f_samp)

    return var_grid, upcross_grid

def match_grid(var_grid, upcross_grid, var_tgt, upcross_tgt):
    grid_shape = var_grid.shape
    loss = np.abs((var_grid-var_tgt)/var_tgt) + np.abs((upcross_grid-upcross_tgt)/upcross_tgt)
    i_star, j_star = np.unravel_index(loss.argmin(),loss.shape)
    return i_star, j_star


def get_f_samp_bnn(args, wb1_sig2, w2_sig2, x_plot, n_samp=1000):
    net = networks_bnn.BNN(dim_in=1, dim_hidden=args.bnn_dim_hidden, dim_out=1, \
                   prior_w1_sig2 = wb1_sig2, prior_b1_sig2 = wb1_sig2, \
                   prior_w2_sig2 = w2_sig2, prior_b2_sig2 = 1e-8, \
                   sig2=.01, s2_0=args.bnn_s2_0)
    return net.sample_functions_prior(x_plot, n_samp=n_samp).detach().numpy(), net

def get_f_samp_porbnet(args, intensity, s2_0, w_sig2, x_plot, n_samp=1000):
    beta=1.0
    alpha = util_porbnet.alpha_for_sqrt_gamma(beta, intensity)
    T=[-.25,1.25]
    dim_hidden_initial=int((T[1]-T[0])*intensity)
    dim_hidden_max = np.maximum(2*dim_hidden_initial, 100)
    intensity_func = util_porbnet.Piecewise(np.array(T),np.array([intensity]))
    net = networks_porbnet.RBFN(dim_in=1, dim_hidden_initial=dim_hidden_initial, dim_hidden_max=dim_hidden_max, \
                    dim_out=1, intensity=intensity_func, s2_0=s2_0, \
                    prior_w_sig2 = w_sig2*np.sqrt(np.pi/s2_0), prior_b_sig2 = 1e-8, \
                    sig2 = .01,
                    prior_intensity_alpha=alpha, prior_intensity_beta=1.0)
    return net.sample_functions_prior(x_plot, n_samp=n_samp, sample_K=True, sample_intensity=True).detach().numpy(), net

def get_f_samp_porbnet_sgcp(args, rate_density_ub, s2_0, w_sig2, x_plot, n_samp=1000):
    #beta=1.0
    #alpha = rate_density_ub*beta
    T=[-.25,1.25]
    dim_hidden_initial=int((T[1]-T[0])*(0.5*rate_density_ub))
    dim_hidden_max = np.maximum(2*dim_hidden_initial, 200)
    net = networks_porbnet_sgcp.RBFN(dim_in=1, dim_hidden_initial=dim_hidden_initial, dim_hidden_max=dim_hidden_max, dim_out=1, \
                    s2_0 = s2_0, \
                    T=T, length_scale_sgcp=.25, variance_sgcp=5.0, proposal_std_cM=.5, \
                    rate_density_ub=rate_density_ub,\
                    prior_w_sig2 = w_sig2, prior_b_sig2 = 1e-8, \
                    sig2=.01, use_gp_term=True, set_gp_to_mean=False, infer_rate_density_ub=False)
    return net.sample_functions_prior(x_plot, n_samp=n_samp, proper=True).detach().numpy(), net

def avg_variance_porbnet(x_plot, net):
    '''
    Computes the average variance over x_plot for porbnet using closed form expression
    Assumes intensity is sampled
    '''
    var = util_porbnet.cov_porbnet_gamma_intensity(x_plot, x_plot, \
            C0=net.prior_c_intensity.breaks[0], \
            C1=net.prior_c_intensity.breaks[-1], \
            b_sig2=1e-8, \
            w_sig2=net.prior_w_sig2, \
            s2_0=net.s2_0, \
            intensity_alpha=net.prior_intensity_alpha, \
            intensity_beta=net.prior_intensity_beta)

    return torch.mean(var).item()

if __name__ == "__main__":
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    files = find_all_files(os.path.join(args.dir_base, 'experiments/models/gp/'), 'samples.npy')

    #mapping = pd.DataFrame(columns=['dataset_idx','gp_path','kernel_variance','kernel_lengthscale','kernel_upcross',\
    #    'model', 'param1_name', 'param2_name', 'param1', 'param2', 'var', 'upcross'])
    mapping = pd.DataFrame()

    dataset_options = {'mimic_gap': ['11', '1228', '1472', '1535'], 
                       'GPdata': ['inc','inc_gap','sin','stat_gap'],
                       'motorcycle': None,
                       'finance': None,
                       'finance_nogap': None,
                       'mimic': ['11', '20', '29', '58', '69', '302', '391', '416', '475', '518', '575', '675', '977', '1241', '1245', '1250', '1256', '1259']}

    x_plot = torch.linspace(0,1,100).reshape(-1,1)
    x_plot_bnn = x_plot - 0.5


    ### Gridsearches

    ## bnn
    if not args.skip_bnn:
        print('running bnn grid')

        get_f_samp_bnn_eval = lambda p1, p2: get_f_samp_bnn(args, p1, p2, x_plot_bnn, args.n_samp)
        param1_list_bnn = np.linspace(args.bnn_param1_min, args.bnn_param1_max, args.n_grid)
        param2_list_bnn = np.linspace(args.bnn_param2_min, args.bnn_param2_max, args.n_grid)

        var_grid_bnn, upcross_grid_bnn = run_grid(\
                get_f_samp = get_f_samp_bnn_eval, \
                param1_list = param1_list_bnn, \
                param2_list = param2_list_bnn)

        var_grid_bnn += param2_list_bnn.reshape(1,-1)

        np.savetxt(os.path.join(args.dir_out,'var_grid_bnn.csv'), var_grid_bnn)
        np.savetxt(os.path.join(args.dir_out,'upcross_grid_bnn.csv'), upcross_grid_bnn)

    ## porbnet
    if not args.skip_porbnet:
        print('running porbnet grid')

        if args.porbnet_param1_name=='intensity':
            get_f_samp_porbnet_eval = lambda p1, p2: get_f_samp_porbnet(args, intensity=p1, s2_0=args.porbnet_s2_0, w_sig2=p2, x_plot=x_plot, n_samp=args.n_samp)
        elif args.porbnet_param1_name=='s2_0':
            get_f_samp_porbnet_eval = lambda p1, p2: get_f_samp_porbnet(args, intensity=args.porbnet_intensity, s2_0=p1, w_sig2=p2, x_plot=x_plot, n_samp=args.n_samp)

        param1_list_porbnet = np.linspace(args.porbnet_param1_min, args.porbnet_param1_max, args.n_grid)
        param2_list_porbnet = np.linspace(args.porbnet_param2_min, args.porbnet_param2_max, args.n_grid)

        if args.porbnet_run_cross:
            var_grid_porbnet, upcross_grid_porbnet  = run_cross(\
                get_f_samp = get_f_samp_porbnet_eval, \
                ls_param_list = param1_list_porbnet, \
                var_param_list = param2_list_porbnet,
                variance_expression = lambda net: avg_variance_porbnet(x_plot, net))

        else:
            var_grid_porbnet, upcross_grid_porbnet  = run_grid(\
                get_f_samp = get_f_samp_porbnet_eval, \
                param1_list = param1_list_porbnet, \
                param2_list = param2_list_porbnet,
                variance_expression = lambda net: avg_variance_porbnet(x_plot, net))

        var_grid_porbnet += param2_list_porbnet.reshape(1,-1)

        np.savetxt(os.path.join(args.dir_out,'var_grid_porbnet.csv'), var_grid_porbnet)
        np.savetxt(os.path.join(args.dir_out,'upcross_grid_porbnet.csv'), upcross_grid_porbnet)

    ## porbnet-sgcp
    if not args.skip_porbnet_sgcp:
        print('running porbnet_sgcp grid')

        if args.porbnet_param1_name=='intensity':
            porbnet_sgcp_param1_name='rate_density_ub'
            porbnet_sgcp_param1_min=2*args.porbnet_param1_min
            porbnet_sgcp_param1_max=2*args.porbnet_param1_max
            porbnet_sgcp_s2_0=args.porbnet_s2_0
            get_f_samp_porbnet_sgcp_eval = lambda p1, p2: get_f_samp_porbnet_sgcp(args, rate_density_ub=p1, s2_0=porbnet_sgcp_s2_0, w_sig2=p2, x_plot=x_plot, n_samp=args.n_samp)
        
        elif args.porbnet_param1_name=='s2_0':
            porbnet_sgcp_param1_name='s2_0'
            porbnet_sgcp_param1_min=args.porbnet_param1_min
            porbnet_sgcp_param1_max=args.porbnet_param1_max
            porbnet_sgcp_rate_density_ub=2*args.porbnet_intensity
            get_f_samp_porbnet_sgcp_eval = lambda p1, p2: get_f_samp_porbnet_sgcp(args, rate_density_ub=porbnet_sgcp_rate_density_ub, s2_0=p1, w_sig2=p2, x_plot=x_plot, n_samp=args.n_samp)

        param1_list_porbnet_sgcp = np.linspace(porbnet_sgcp_param1_min, porbnet_sgcp_param1_max, args.n_grid)
        param2_list_porbnet_sgcp = np.linspace(args.porbnet_param2_min, args.porbnet_param2_max, args.n_grid)

        if args.porbnet_run_cross:
            var_grid_porbnet_sgcp, upcross_grid_porbnet_sgcp = run_cross(\
                get_f_samp = get_f_samp_porbnet_sgcp_eval, \
                ls_param_list = param1_list_porbnet_sgcp, \
                var_param_list = param2_list_porbnet_sgcp)

        else:
            var_grid_porbnet_sgcp, upcross_grid_porbnet_sgcp = run_grid(\
                get_f_samp = get_f_samp_porbnet_sgcp_eval, \
                param1_list = param1_list_porbnet_sgcp, \
                param2_list = param2_list_porbnet_sgcp)

        var_grid_porbnet_sgcp += param2_list_porbnet_sgcp.reshape(1,-1)

        np.savetxt(os.path.join(args.dir_out,'var_grid_porbnet_sgcp.csv'), var_grid_porbnet_sgcp)
        np.savetxt(os.path.join(args.dir_out,'upcross_grid_porbnet_sgcp.csv'), upcross_grid_porbnet_sgcp)

    for i, file in enumerate(files):
        print('matching file %s' % file)

        dir_plots = os.path.join(args.dir_out, 'dataset_idx=%d/' % i)
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)

        ### try to match dataset name
        try:
            file_split = file.split('/')
            dataset = list(filter(lambda x: x in dataset_options.keys(), file_split))[0]
            if dataset_options[dataset] is not None:
                dataset_option = list(filter(lambda x: x in dataset_options[dataset], file_split))[0]
            else:
                dataset_option = None
        except:
            print('warning: could not match dataset from file %s' % file)
            dataset = None
            dataset_option = None

            
        ### load gp hyperparameters
        samples = np.load(file, allow_pickle=True).item()

        kernel_lengthscale = samples['kernel_lengthscale']
        kernel_variance = samples['kernel_variance']
        kernel_upcross = 1/(2*np.pi*kernel_lengthscale)
        sig2 = samples['sig2']

        # plot
        kernel = GPy.kern.RBF(input_dim=1, variance=kernel_variance, lengthscale=kernel_lengthscale)
        f_samp_gp = np.random.multivariate_normal(np.zeros(x_plot.numpy().shape[0]),kernel.K(x_plot.numpy(), x_plot.numpy()), args.n_samp)
        fig, ax = util.plot_prior_predictive(x_plot.numpy(), f_samp_gp, upcross_level=0, bins=20, plot_all_functions=False)
        fig.savefig(os.path.join(dir_plots, 'prior_predictive_gp.png'))

        # save (same for each row)
        row_data = {'file_idx':i,\
                    'dataset':dataset,\
                    'dataset_option':dataset_option,\
                    'gp_path':file,\
                    'kernel_variance':kernel_variance,\
                    'kernel_lengthscale':kernel_lengthscale,\
                    'kernel_upcross':kernel_upcross,\
                    'sig2':sig2}

        ### bnn
        if not args.skip_bnn:

            i_star, j_star = match_grid(var_grid_bnn, \
                                        upcross_grid_bnn, \
                                        var_tgt=kernel_variance, \
                                        upcross_tgt=kernel_upcross)

            # plot
            if args.plot:
                f_samp_bnn, _ = get_f_samp_bnn_eval(param1_list_bnn[i_star], param2_list_bnn[j_star])
                fig, ax = util.plot_prior_predictive(x_plot_bnn.numpy(), f_samp_bnn, upcross_level=0, bins=20, plot_all_functions=False)
                fig.savefig(os.path.join(dir_plots, 'prior_predictive_bnn.png'))

            # save
            row_bnn = {'model':'bnn',\
                       'param1_name':'wb1_sig2',\
                       'param2_name':'w2_sig2',\
                       'param1':param1_list_bnn[i_star],\
                       'param2':param2_list_bnn[j_star],\
                       'var':var_grid_bnn[i_star, j_star],\
                       'upcross':upcross_grid_bnn[i_star, j_star],
                       'param1_on_boundary':True if i_star==0 or i_star==len(param1_list_bnn)-1 else False,
                       'param2_on_boundary':True if j_star==0 or j_star==len(param2_list_bnn)-1 else False,
                       'bnn_width':args.bnn_dim_hidden}

            mapping = mapping.append({**row_data, **row_bnn}, ignore_index=True)

        ### porbnet
        if not args.skip_porbnet:

            i_star, j_star = match_grid(var_grid_porbnet, \
                            upcross_grid_porbnet, \
                            var_tgt=kernel_variance, \
                            upcross_tgt=kernel_upcross)

            # plot
            var_porbnet_est = None
            if args.plot:
                f_samp_porbnet, _ = get_f_samp_porbnet_eval(param1_list_porbnet[i_star], param2_list_porbnet[j_star])
                fig, ax = util.plot_prior_predictive(x_plot.numpy(), f_samp_porbnet, upcross_level=0, bins=20, plot_all_functions=False)
                fig.savefig(os.path.join(dir_plots, 'prior_predictive_porbnet.png'))

                var_porbnet_est = avg_variance(f_samp_porbnet) + param2_list_porbnet[j_star]

            # save
            row_porbnet = {\
                'model':'porbnet',\
                'param1_name':args.porbnet_param1_name,\
                'param2_name':'w_sig2',\
                'param1':param1_list_porbnet[i_star],\
                'param2':param2_list_porbnet[j_star],\
                'var':var_porbnet_est,\
                'var_true':var_grid_porbnet[i_star, j_star],\
                'upcross':upcross_grid_porbnet[i_star, j_star],
                'param1_on_boundary':True if i_star==0 or i_star==len(param1_list_porbnet)-1 else False,
                'param2_on_boundary':True if j_star==0 or j_star==len(param2_list_porbnet)-1 else False}

            mapping = mapping.append({**row_data, **row_porbnet}, ignore_index=True)


        ### porbnet-sgcp
        if not args.skip_porbnet_sgcp:

            i_star, j_star = match_grid(var_grid_porbnet_sgcp, \
                upcross_grid_porbnet_sgcp, \
                var_tgt=kernel_variance, \
                upcross_tgt=kernel_upcross)

            # plot
            if args.plot:
                f_samp_porbnet_sgcp, _ = get_f_samp_porbnet_sgcp_eval(param1_list_porbnet_sgcp[i_star], param2_list_porbnet_sgcp[j_star])
                fig, ax = util.plot_prior_predictive(x_plot.numpy(), f_samp_porbnet_sgcp, upcross_level=0, bins=20, plot_all_functions=False)
                fig.savefig(os.path.join(dir_plots, 'prior_predictive_porbnet_sgcp.png'))

            # save
            row_porbnet_sgcp = {\
                'model':'porbnet_sgcp',\
                'param1_name':porbnet_sgcp_param1_name,\
                'param2_name':'w_sig2',\
                'param1':param1_list_porbnet_sgcp[i_star],\
                'param2':param2_list_porbnet_sgcp[j_star],\
                'var':var_grid_porbnet_sgcp[i_star, j_star],\
                'upcross':upcross_grid_porbnet_sgcp[i_star, j_star],
                'param1_on_boundary':True if i_star==0 or i_star==len(param1_list_porbnet_sgcp)-1 else False,
                'param2_on_boundary':True if j_star==0 or j_star==len(param2_list_porbnet_sgcp)-1 else False}

            mapping = mapping.append({**row_data, **row_porbnet_sgcp}, ignore_index=True)

        plt.close('all')

    mapping.to_csv(os.path.join(args.dir_out,'prior_matching.csv'))









