import GPy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


dir_base = '../../../'
dir_data = os.path.join(dir_base, 'data/')
sys.path.append(dir_data)
sys.path.append(os.path.join(dir_base, 'src/utils'))
import data
import utils_load_dataset

def reshape_dataset(x, y, x_test, y_test):
    return x.reshape(-1,1), y.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)


def run_gp(dataset, dataset_option, seed, x_plot, sig2, dir_data, dir_out):

    ## dataset
    if dataset=='motorcycle':
        x, y, x_test, y_test = reshape_dataset(*data.load_motorcycle(dir_data, seed=seed))
    elif dataset=='finance':
        x, y, x_test, y_test = reshape_dataset(*data.load_finance2(dir_data, seed=seed))
    elif dataset=='mimic_gap':
        x, y, x_test, y_test = reshape_dataset(*utils_load_dataset.load_mimic_patient_gap(dir_data, subject_id=dataset_option))
    elif dataset=='GPdata':
        x, y, x_val, y_val, x_test, y_test, _, _ = data.load_GPdata(dir_data, lengthscale=dataset_option)
        x, y, x_test, y_test = reshape_dataset(x, y, x_test, y_test)
    elif dataset=='finance_nogap':
        x, y, x_test, y_test = reshape_dataset(*data.load_finance2(dir_data, gaps=None, seed=seed))
    elif dataset=='mimic':
        x, y = utils_load_dataset.load_mimic_patient(patient_id=int(dataset_option), csv_filename=os.path.join(dir_data, 'hr.csv'))
        x, y, x_test, y_test = reshape_dataset(*data.train_val_split(x, y, frac_train=0.75, seed=seed))


    x, y, x_test, y_test = data.standardize(x, y, x_test, y_test)

    ## fit gp
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    m = GPy.models.GPRegression(x,y,kernel)
    m.Gaussian_noise.variance.fix()
    m.Gaussian_noise.variance = sig2
    m.kern.lengthscale.constrain_bounded(1/(5*2*np.pi),10)
    m.optimize_restarts(num_restarts = 10, verbose=False)


    # ploterior metrics
    y_plot_samp = m.posterior_samples_f(x_plot, size=1000)
    y_plot_samp = np.moveaxis(y_plot_samp, [0, 1, 2], [1,2,0])

    y_plot_pred, y_plot_pred_var = m.predict(x_plot, full_cov=False, include_likelihood=False)
    y_test_pred, y_test_pred_var = m.predict(x_test, full_cov=False, include_likelihood=False)

    y_test_samp = m.posterior_samples_f(x_test, size=1000)
    y_test_samp = np.moveaxis(y_test_samp, [0, 1, 2], [1,2,0])

    ll_test = np.mean(m.log_predictive_density(x_test, y_test))

    rmse_test = np.sqrt(np.mean((y_test_pred - y_test)**2))

    # save
    samples = {
        'x_plot': x_plot,
        'y_plot_pred': np.squeeze(y_plot_pred),
        'y_plot_pred_var': y_test_pred_var,
        'y_plot_samp': y_plot_samp,
        'y_test_pred': np.squeeze(y_test_pred),
        'y_test_pred_var': y_test_pred_var,
        'y_test_samp': y_test_samp,
        'x': x,
        'y': y,
        'x_test': x_test,
        'y_test': y_test,
        'sig2': m.Gaussian_noise.variance.item(),
        'll_test': ll_test,
        'rmse_test': rmse_test,
        'kernel_lengthscale': m.kern.lengthscale.item(),
        'kernel_variance': m.kern.variance.item()
    }

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    np.save(os.path.join(dir_out, 'samples.npy'), samples)

if __name__ == "__main__":


    x_plot = np.linspace(0,1,100).reshape(-1,1)
    dataset_list = ['GPdata', 'mimic_gap', 'motorcycle', 'finance', 'finance_nogap', 'mimic']
    dataset_option_dict = {'GPdata': ['inc','inc_gap','sin','stat_gap'], 
                           'mimic_gap': ['11','1228','1472','1535'],
                           'motorcycle': [''],
                           'finance': [''],
                           'finance_nogap': [''],
                           'mimic': ['11', '20', '29', '58', '69', '302', '391', '416', '475', '518', '575', '675', '977', '1241', '1245', '1250', '1256', '1259']}
    sig2_dict = {'GPdata': .02, 'mimic_gap': .01, 'motorcycle': .04, 'finance': .0035, 'finance_nogap': .0035, 'mimic': .01}
    dir_out_base = './'

    for i, dataset in enumerate(dataset_list):
        for j, dataset_option in enumerate(dataset_option_dict[dataset]):
            print('running dataset=%s, dataset_option=%s' % (dataset, dataset_option))

            run_gp(dataset=dataset, \
                dataset_option=dataset_option, \
                seed=1, \
                x_plot=x_plot, \
                sig2=sig2_dict[dataset], \
                dir_data=dir_data, \
                dir_out=os.path.join(dir_out_base, dataset, dataset_option))




