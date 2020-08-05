import numpy as np
import data
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir_base', type=str, default='../', help='directory of source code')
parser.add_argument('--seed', type=int, default=1, help='seed for dataset')
args = parser.parse_args()

sys.path.append(os.path.join(args.dir_base, 'src/utils'))
import utils_load_dataset

dir_data = os.path.join(args.dir_base, 'data/')
if not os.path.exists('matlab/'):
    os.makedirs('matlab/') 

def save_dataset(dir_out, x, y, x_test, y_test, standardize=True):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out) 
    if standardize:
        x, y, x_test, y_test = data.standardize(x, y, x_test, y_test)
    np.savetxt(os.path.join(dir_out,'x.csv'), x)
    np.savetxt(os.path.join(dir_out,'y.csv'), y)
    np.savetxt(os.path.join(dir_out,'x_test.csv'), x_test)
    np.savetxt(os.path.join(dir_out,'y_test.csv'), y_test)

# motorcycle
x, y, x_test, y_test = data.load_motorcycle(dir_data, seed=args.seed)
save_dataset('matlab/motorcycle/', x, y, x_test, y_test)

# finance
x, y, x_test, y_test = data.load_finance2(dir_data, seed=args.seed)
save_dataset('matlab/finance/', x, y, x_test, y_test)

# finance_nogap
x, y, x_test, y_test = data.load_finance2(dir_data, gaps=None, seed=args.seed)
save_dataset('matlab/finance_nogap/', x, y, x_test, y_test)

# mimic_gap
for subject_id in [11, 1228, 1472, 1535]:
    x, y, x_test, y_test = utils_load_dataset.load_mimic_patient_gap(dir_data, subject_id=subject_id)
    save_dataset('matlab/mimic_gap/%s' % subject_id, x, y, x_test, y_test)

# GPdata
for lengthscale in ['inc', 'inc_gap', 'sin', 'stat_gap']:
    x, y, x_val, y_val, x_test, y_test, x_plot, f_plot = data.load_GPdata(dir_data, lengthscale=lengthscale)
    save_dataset('matlab/GPdata/%s' % lengthscale, x, y, x_test, y_test)

# mimic
for subject_id in [11, 20, 29, 58, 69, 302, 391, 416, 475, 518, 575, 675, 977, 1241, 1245, 1250, 1256, 1259]:
    x, y = utils_load_dataset.load_mimic_patient(patient_id=int(subject_id), csv_filename=os.path.join(dir_data, 'hr.csv'))
    x, y, x_test, y_test = data.train_val_split(x, y, frac_train=0.75, seed=args.seed)
    save_dataset('matlab/mimic/%s' % subject_id, x, y, x_test, y_test)

