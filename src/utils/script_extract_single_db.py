from utils_load_dataset import load_mimic_patient
import pickle
import pandas as pd
import os
from distutils.dir_util import mkpath

def save_xy_files(output_path,X_df,Y_df,ext='csv'):
    if (ext=='csv'):
        X_df.to_csv(os.path.join(output_path,'X.csv'),index=False)
        Y_df.to_csv(os.path.join(output_path,'Y.csv'),index=False)
    else:
        raise ValueError('Unknown or not implemented file extension')

pid=11
x,y = load_mimic_patient(pid)

X_df = pd.DataFrame(data=x)
Y_df = pd.DataFrame(data=y)

output_path='hr_pid=%d' % pid
mkpath(output_path)
save_xy_files(output_path,X_df,Y_df)
