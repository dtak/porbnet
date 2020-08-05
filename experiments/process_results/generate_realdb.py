import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse
#from scipy.io import loadmat
from util import (
        plot_posterior_predictive,
        compute_rmse,
        compute_loglik,
        compute_loglik2,
        generate_latex_table,
        plot_lengthscale)
from distutils.dir_util import mkpath

import ipdb
from scipy import io

from util import plot_posterior_predictive, plot_intensity

plt.rcParams.update({'font.size': 8})

# remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT, DIR_IN_PREFIX='./', figsize_inches=(8, 8), transpose=False, figname='_'):

    MODEL_TITLES = {'gp': 'GP', 'bnn': 'BNN', 'porbnet': 'PoRB-Net*', 'porbnet-sgcp': 'PoRB-Net', 'adaptivegp': 'LGP'}
    DB_TITLES = {'sin': 'sin', 'inc_gap': 'inc', 'inc': 'inc2', 'inc (synthetic)': 'inc (synthetic)', 'stat_gap': 'const', 'mimic1': 'mimic1', 'mimic1 (real)': 'mimic1 (real)', 'mimic2': 'mimic2', 'mimic3': 'mimic3', 'mimic4': 'mimic4', 'finance': 'finance', 'motorcycle': 'motorcycle'}

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT) 

    n_model = len(MODEL_LIST)
    n_db = len(DB_LIST)
    n_model_int = len(list(filter(lambda x: x in ['adaptivegp','porbnet-sgcp'], MODEL_LIST)))

    if transpose:
        fig, ax = plt.subplots(n_db, n_model, figsize=(n_model, n_db), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(n_model, n_db, figsize=(n_db,n_model), sharex=True, sharey=True)
    
    RMSE_TABLE = np.full((n_model, n_db), np.nan)
    LOGLIK_TABLE = np.full((n_model, n_db), np.nan)

    f_out = open(os.path.join(DIR_OUT,'plot_order.txt'), 'w')
    f_out.write('row,column,path\n')

    if 'porbnet-sgcp' in MODEL_LIST or 'adaptivegp' in MODEL_LIST:
        have_intensity = True
        fig_int, ax_int = plt.subplots(n_model_int, n_db, figsize=(n_db,n_model_int), sharex=True, sharey='row')
        f_out_int = open(os.path.join(DIR_OUT,'plot_order_intensity.txt'), 'w')
        f_out_int.write('row,column,path\n')
    else:
        have_intensity = False

    i_int = 0
    for i, model in enumerate(MODEL_LIST):

        # row labels

        if transpose:
            if model == 'porbnet':
                ax[0,i].set_title(r'PoRB-Net $\dag$')
            else:
                ax[0,i].set_title(MODEL_TITLES[model])
        else:
            if model == 'porbnet':
                ax[i,0].set_ylabel(r'PoRB-Net $\dag$')
            else:
                ax[i,0].set_ylabel(MODEL_TITLES[model])

        for j, db in enumerate(DB_LIST):

            # column labels
            if transpose:
                ax[j,0].set_ylabel(DB_TITLES[db])
            else:
                ax[0,j].set_title(DB_TITLES[db])

            try:
                if model == 'adaptivegp':
                    path = os.path.join(DIR_IN_PREFIX, DIR_IN[i][j])
                    samples = io.loadmat(path)
                else:
                    path = os.path.join(DIR_IN_PREFIX, DIR_IN[i][j])
                    samples = np.load(path, allow_pickle=True).item()
            except:
                print('Unable to load entry (%d, %d) from %s ' % (i,j,path))
                continue

            f_out.write('%d,%d,%s\n' % (i,j,path))

            if have_intensity:
                f_out_int.write('%d,%d,%s\n' % (i,j,path))


            # dataset specific
            if model in ['bnn', 'porbnet-sgcp', 'porbnet']:
                # check if all samples avalilable by checking if sig2 is ever zero
                i_max_avail = np.nonzero(samples['sig2'].numpy()!=0)[0][-1]
                i_max = samples['sig2'].shape[0]
                if i_max_avail < i_max-1:
                    print('WARNING: only %d samples of %d are found' % (i_max_avail, i_max)) 
                    i_max = i_max_avail

                x_plot = samples['x_plot'][:i_max_avail, :].numpy()
                y_plot = samples['y_plot_pred'].numpy()[:i_max_avail,:,np.newaxis]
                x = samples['x'].numpy()
                y = np.squeeze(samples['y'].numpy())
                x_test = samples['x_test'].numpy()
                y_test = np.squeeze(samples['y_test'].numpy())
                sig2 = np.mean(samples['sig2'][:i_max_avail].numpy())
                y_test_pred = samples['y_test_pred'][:i_max_avail,:].detach().numpy()

                if model == 'porbnet-sgcp':
                    rate_density_ub = samples['rate_density_ub'][:i_max_avail].numpy()
                    gp_plot_samp = samples['gp_plot_pred'][:i_max_avail,:].numpy()


            elif model in ['gp']:
                x_plot = samples['x_plot']
                y_plot = samples['y_plot_samp']

                y_test_pred = np.squeeze(samples['y_test_samp'])

                x = samples['x']
                y = samples['y']
                x_test = samples['x_test']
                y_test = samples['y_test']
                sig2 = samples['sig2']

            elif model == 'adaptivegp':

                x_plot = samples['x_plot']

                #y_plot = samples['y_plot_pred'][:,:,np.newaxis]
                y_plot = samples['ysamp_plot_pred']
                idx_nan = np.any(np.isnan(y_plot),1)
                print('number nan in ysamp_plot_pred:', sum(idx_nan))
                y_plot = y_plot[~idx_nan,:][:,:,np.newaxis]

                y_test_pred = samples['ysamp_test_pred']
                idx_nan = np.any(np.isnan(y_test_pred),1)
                print('number nan in ysamp_test_pred:', sum(idx_nan))
                y_test_pred = y_test_pred[~idx_nan,:]

                x = samples['x']
                y = samples['y']
                x_test = samples['x_test']
                y_test = samples['y_test']
                sig2 = samples['sig2']

                l_test_pred = samples['l_plot_pred'][50:,:]

                


                ##
                

                '''
                save(fullfile(path_out, 'output.mat'),...
        'samples', 'sig2', ...
        'y_plot_pred','ystd_plot_pred','l_plot_pred','s_plot_pred','o_plot_pred',...
        'y_train_pred','ystd_train_pred','l_train_pred','s_train_pred','o_train_pred',...
        'y_test_pred','ystd_test_pred','l_test_pred','s_test_pred','o_test_pred',...
        'mse', 'nlpd', ...
        'x_plot', 'x', 'y', 'x_test', 'y_test')
        '''

                    

            ## HACK
            if model == 'bnn':
                x = x + 0.5
                x_test = x_test + 0.5
                x_plot = x_plot + 0.5
            ##

            if transpose:
                plot_posterior_predictive(x_plot, y_plot, ax=ax[j,i] ,x=x,y=y, x_test=x_test, y_test=y_test, sig2=sig2, s=2)
                #ax[j,i].grid(linewidth='0.5')
            else:
                plot_posterior_predictive(x_plot, y_plot, ax=ax[i,j] ,x=x,y=y, x_test=x_test, y_test=y_test, sig2=sig2, s=2)
                #ax[i,j].grid(linewidth='0.5')

            print('db=%s, sig2=%.2f' % (db,sig2))
            

            if model == 'adaptivegp':
                RMSE_TABLE[i,j] = np.sqrt(samples['mse'])
                LOGLIK_TABLE[i,j] = samples['nlpd'] # analytical (uses variance not covariance)
                LOGLIK_TABLE[i,j] = compute_loglik2(y_test, y_test_pred, sig2) # empirical
            elif model == 'gp':
                RMSE_TABLE[i,j] = samples['rmse_test'] # analytical (uses variance not covariance)
                #LOGLIK_TABLE[i,j] = samples['ll_test'] # empirical
                LOGLIK_TABLE[i,j] = compute_loglik2(y_test, y_test_pred, sig2)
            else:

                RMSE_TABLE[i,j] = compute_rmse(y_test, y_test_pred)
                LOGLIK_TABLE[i,j] = compute_loglik2(y_test, \
                                                         y_test_pred, \
                                                         sig2)
            
            if model in ['porbnet-sgcp','adaptivegp']:
                if n_model_int>1:
                    if transpose:
                        ax_int_0 = ax_int[j, i_int]
                    else:
                        ax_int_0 = ax_int[i_int,j]

                        if i_int==0:
                            ax_int_0.set_title(DB_TITLES[db])
                else:
                    ax_int_0 = ax_int[j]
                jj=j

                # get dataset name for plotting true intensity
                try:
                    true_ls_name = list(filter(lambda x: x in ['sin','stat_gap','inc','inc_gap'], path.split('/')))[0]
                except:
                    true_ls_name = None


                if model == 'porbnet-sgcp':
                    plot_intensity(x_plot, \
                    gp_plot_samp, \
                    rate_density_ub.reshape(-1,1),\
                    ax_int_0,
                    true_ls_name=true_ls_name)

                    if j==0:
                        ax_int_0.set_ylabel(MODEL_TITLES[model]+'\nintensity')
                elif model == 'adaptivegp':
                    plot_lengthscale(x_plot, l_test_pred, inverse=True, ax=ax_int_0, true_ls_name=true_ls_name)
                    if j==0:
                        ax_int_0.set_ylabel(MODEL_TITLES[model]+'\ninverse lengthcale')
                
                if j==len(DB_LIST)-1:
                    i_int+=1



    df_rmse = pd.DataFrame(RMSE_TABLE, index=MODEL_LIST, columns=DB_LIST)
    df_rmse.to_csv(os.path.join(DIR_OUT,'rmse.csv'), index=True, header=True, sep=',')

    df_rmse = pd.DataFrame(LOGLIK_TABLE, index=MODEL_LIST, columns=DB_LIST)
    df_rmse.to_csv(os.path.join(DIR_OUT,'loglik.csv'), index=True, header=True, sep=',')

    ### Make latex table
    #RMSE_TABLE = np.core.defchararray.add(RMSE_TABLE.astype('str'))
    #LOGLIK_TABLE = np.core.defchararray.add(LOGLIK_TABLE.astype('str'))

    RMSE_TABLE = np.round(RMSE_TABLE,2).astype('str')
    LOGLIK_TABLE = np.round(LOGLIK_TABLE,2).astype('str')


    #latex_table = generate_latex_table(LATEX_TABLE, db_list+db_list, \
    #        row_labels=model_list)
    rmse_table = generate_latex_table(RMSE_TABLE, DB_LIST, \
            row_labels=MODEL_LIST)
    loglik_table = generate_latex_table(LOGLIK_TABLE, DB_LIST, \
            row_labels=MODEL_LIST)

    filename = os.path.join(DIR_OUT,'rmse.tex')
    with open(filename,'w') as f: f.write(rmse_table)
    filename = os.path.join(DIR_OUT,'loglik.tex')
    with open(filename,'w') as f: f.write(loglik_table)

    print('LATEX TABLE SAVED AT: %s' % filename)
    ###

    for j in range(ax.shape[1]):
        ax[-1,j].set_xlabel(r'$x$')


    mkpath(DIR_OUT)
    filename = os.path.join(DIR_OUT,'realdb_'+figname+'.pdf')
    fig.set_size_inches(figsize_inches[0], figsize_inches[1])
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight',pad_inches=.025)
    print('SAVED FIG: %s' % filename)
    f_out.close()

    if have_intensity:
        filename_int = os.path.join(DIR_OUT,'intensity_realdb_'+figname+'.pdf')
        fig_int.set_size_inches(figsize_inches[0],figsize_inches[1]/n_model*n_model_int)
        fig_int.tight_layout()
        fig_int.savefig(filename_int, bbox_inches='tight', pad_inches=.025)
        f_out_int.close()

    plt.close('all')
        

def find_all_files(folder, file):
    return [str(path) for path in Path(folder).rglob(file)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--figname', type=str, default='main')
    args = parser.parse_args()


    if args.figname == 'main':

        DIR_IN = [[
        '../models/porbnet-sgcp/s2_0=2/GPdata/inc_gap/60484367/task=4/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/475/60508105/task=8/output/samples.npy',
        ],[
        '../models/adaptivegp/results_1k/GPdata/inc_gap/output.mat',
        '../models/adaptivegp/results_1k/mimic/475/output.mat',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/inc_gap/60539083/task=4/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/475/60539083/task=19/output/samples.npy',
        ]]
        DIR_IN = np.array(DIR_IN)

        MODEL_LIST = ['porbnet-sgcp', 'adaptivegp', 'bnn']
        DB_LIST = ['inc (synthetic)', 'mimic1 (real)']

        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update({'legend.fontsize': 6})
        plt.rcParams.update({'axes.labelsize': 6})
        plt.rcParams.update({'axes.titlesize': 8})

        plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT='output/realdb/'+args.figname, figsize_inches=(6.62, 3.75), transpose=True, figname=args.figname)


    elif args.figname == 'appendix1':

        DIR_IN = [[
        '../models/porbnet/s2_0=2/GPdata/sin/60508230/task=5/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/inc_gap/60508230/task=4/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/inc/60508230/task=3/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/stat_gap/60508230/task=6/output/samples.npy',
        ],[
        '../models/porbnet-sgcp/s2_0=2/GPdata/sin/60484367/task=5/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/inc_gap/60484367/task=4/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/inc/60484367/task=3/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/stat_gap/60484367/task=6/output/samples.npy',
        ],[
        '../models/gp/GPdata/sin/samples.npy',
        '../models/gp/GPdata/inc_gap/samples.npy',
        '../models/gp/GPdata/inc/samples.npy',
        '../models/gp/GPdata/stat_gap/samples.npy',
        ],[
        '../models/adaptivegp/results_1k/GPdata/sin/output.mat',
        '../models/adaptivegp/results_1k/GPdata/inc_gap/output.mat',
        '../models/adaptivegp/results_1k/GPdata/inc/output.mat',
        '../models/adaptivegp/results_1k/GPdata/stat_gap/output.mat',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/sin/60539104/task=5/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/inc_gap/60539083/task=4/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/inc/60539104/task=3/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/stat_gap/60539083/task=6/output/samples.npy',
        ]]
        DIR_IN = np.array(DIR_IN)

        MODEL_LIST = ['porbnet', 'porbnet-sgcp', 'gp', 'adaptivegp', 'bnn']
        DB_LIST = ['sin', 'inc_gap', 'inc', 'stat_gap']
        # inc is inc_gap in files, inc2 is inc in files, const is stat_gap in files


        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update({'legend.fontsize': 6})
        plt.rcParams.update({'axes.labelsize': 6})
        plt.rcParams.update({'axes.titlesize': 8})
        plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT='output/realdb/'+args.figname, figsize_inches=(6.62, 8.5), figname=args.figname)

    elif args.figname == 'appendix2':

        DIR_IN = [[
        '../models/porbnet/s2_0=2/mimic/475/60508230/task=19/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/1241/60508230/task=24/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/518/60508230/task=20/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/1259/60508230/task=28/output/samples.npy',
        ],[
        '../models/porbnet-sgcp/s2_0=2/mimic/475/60508105/task=8/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/1241/60508105/task=13/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/518/60508105/task=9/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/1259/60508105/task=17/output/samples.npy',
        ],[
        '../models/gp/mimic/475/samples.npy',
        '../models/gp/mimic/1241/samples.npy',
        '../models/gp/mimic/518/samples.npy',
        '../models/gp/mimic/1259/samples.npy',
        ],[
        '../models/adaptivegp/results_1k/mimic/475/output.mat',
        '../models/adaptivegp/results_1k/mimic/1241/output.mat',
        '../models/adaptivegp/results_1k/mimic/518/output.mat',
        '../models/adaptivegp/results_1k/mimic/1259/output.mat',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/475/60539083/task=19/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/1241/60539104/task=24/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/518/60539104/task=20/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/mimic/1259/60510326/task=17/output/samples.npy',
        ]]
        DIR_IN = np.array(DIR_IN)

        MODEL_LIST = ['porbnet', 'porbnet-sgcp', 'gp', 'adaptivegp', 'bnn']
        DB_LIST = ['mimic1', 'mimic2', 'mimic3', 'mimic4']
        # inc is inc_gap in files, inc2 is inc in files, const is stat_gap in files


        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update({'legend.fontsize': 6})
        plt.rcParams.update({'axes.labelsize': 6})
        plt.rcParams.update({'axes.titlesize': 8})
        plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT='output/realdb/'+args.figname, figsize_inches=(6.62, 8.5), figname=args.figname)

    elif args.figname == 'appendix3':

        DIR_IN = [[
        '../models/porbnet/s2_0=2/finance/None/60508230/task=1/output/samples.npy',
        '../models/porbnet/s2_0=2/motorcycle/None/60508230/task=2/output/samples.npy',
        ],[
        '../models/porbnet-sgcp/s2_0=2/finance/None/60484367/task=1/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/motorcycle/None/60484367/task=2/output/samples.npy',
        ],[
        '../models/gp/finance/samples.npy',
        '../models/gp/motorcycle/samples.npy',
        ],[
        '../models/adaptivegp/results_1k/finance/output.mat',
        '../models/adaptivegp/results_1k/motorcycle/output.mat',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=100/finance/None/60539104/task=1/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/motorcycle/None/60539083/task=2/output/samples.npy',
        ]]
        DIR_IN = np.array(DIR_IN)

        MODEL_LIST = ['porbnet', 'porbnet-sgcp', 'gp', 'adaptivegp', 'bnn']
        DB_LIST = ['finance', 'motorcycle']
        # inc is inc_gap in files, inc2 is inc in files, const is stat_gap in files


        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update({'legend.fontsize': 6})
        plt.rcParams.update({'axes.labelsize': 6})
        plt.rcParams.update({'axes.titlesize': 8})
        plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT='output/realdb/'+args.figname, figsize_inches=(3.3, 8.5), figname=args.figname)


    elif args.figname == 'all':

        DIR_IN = [[
        '../models/porbnet/s2_0=2/GPdata/sin/60508230/task=5/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/inc_gap/60508230/task=4/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/inc/60508230/task=3/output/samples.npy',
        '../models/porbnet/s2_0=2/GPdata/stat_gap/60508230/task=6/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/475/60508230/task=19/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/1241/60508230/task=24/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/518/60508230/task=20/output/samples.npy',
        '../models/porbnet/s2_0=2/mimic/1259/60508230/task=28/output/samples.npy',
        '../models/porbnet/s2_0=2/finance/None/60508230/task=1/output/samples.npy',
        '../models/porbnet/s2_0=2/motorcycle/None/60508230/task=2/output/samples.npy',
        ],[
        '../models/porbnet-sgcp/s2_0=2/GPdata/sin/60484367/task=5/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/inc_gap/60484367/task=4/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/inc/60484367/task=3/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/GPdata/stat_gap/60484367/task=6/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/475/60508105/task=8/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/1241/60508105/task=13/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/518/60508105/task=9/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/mimic/1259/60508105/task=17/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/finance/None/60484367/task=1/output/samples.npy',
        '../models/porbnet-sgcp/s2_0=2/motorcycle/None/60484367/task=2/output/samples.npy',
        ],[
        '../models/gp/GPdata/sin/samples.npy',
        '../models/gp/GPdata/inc_gap/samples.npy',
        '../models/gp/GPdata/inc/samples.npy',
        '../models/gp/GPdata/stat_gap/samples.npy',
        '../models/gp/mimic/475/samples.npy',
        '../models/gp/mimic/1241/samples.npy',
        '../models/gp/mimic/518/samples.npy',
        '../models/gp/mimic/1259/samples.npy',
        '../models/gp/finance/samples.npy',
        '../models/gp/motorcycle/samples.npy',
        ],[
        '../models/adaptivegp/results_1k/GPdata/sin/output.mat',
        '../models/adaptivegp/results_1k/GPdata/inc_gap/output.mat',
        '../models/adaptivegp/results_1k/GPdata/inc_gap/output.mat',
        '../models/adaptivegp/results_1k/GPdata/stat_gap/output.mat',
        '../models/adaptivegp/results_1k/mimic/475/output.mat',
        '../models/adaptivegp/results_1k/mimic/1241/output.mat',
        '../models/adaptivegp/results_1k/mimic/518/output.mat',
        '../models/adaptivegp/results_1k/mimic/1259/output.mat',
        '../models/adaptivegp/results_1k/finance/output.mat',
        '../models/adaptivegp/results_1k/motorcycle/output.mat',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/sin/60539083/task=5/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/inc_gap/60539083/task=4/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/inc/60539083/task=3/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/GPdata/stat_gap/60539083/task=6/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/475/60539083/task=19/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/1241/60539083/task=24/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/518/60539083/task=20/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/mimic/1259/60539083/task=28/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/finance/None/60539083/task=1/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=25/motorcycle/None/60539083/task=2/output/samples.npy',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=50/GPdata/sin/60480895/task=5/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/GPdata/inc_gap/60480895/task=4/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/GPdata/inc/60480895/task=3/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/GPdata/stat_gap/60480895/task=6/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/mimic/475/60510326/task=8/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/mimic/1241/60510326/task=13/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/mimic/518/60510326/task=9/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/mimic/1259/60510326/task=17/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/finance/None/60480895/task=1/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=50/motorcycle/None/60480895/task=2/output/samples.npy',
        ],[
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/sin/60539104/task=5/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/inc_gap/60539104/task=4/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/inc/60539104/task=3/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/GPdata/stat_gap/60539104/task=6/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/475/60539104/task=19/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/1241/60539104/task=24/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/518/60539104/task=20/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/mimic/1259/60539104/task=28/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/finance/None/60539104/task=1/output/samples.npy',
        '../models/bnn/s2_0=2_dim_hidden=100/motorcycle/None/60539104/task=2/output/samples.npy',
        ]]
        DIR_IN = np.array(DIR_IN)

        MODEL_LIST = ['porbnet', 'porbnet-sgcp', 'gp', 'adaptivegp', 'bnn', 'bnn', 'bnn']
        DB_LIST = ['sin', 'inc_gap', 'inc', 'stat_gap', '475', '1241', '518', '1259', 'finance', 'motorcycle']

        plt.rcParams.update({'font.size': 4})
        plt.rcParams.update({'axes.labelsize': 8})
        plt.rcParams.update({'axes.titlesize': 8})
        plot_results(MODEL_LIST, DB_LIST, DIR_IN, DIR_OUT='output/realdb/'+args.figname, figsize_inches=(6,12), figname=args.figname)



        
