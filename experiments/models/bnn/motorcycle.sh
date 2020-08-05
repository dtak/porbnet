#!/bin/bash
#SBATCH -J motorcycle 						   # A single job name for the array
#SBATCH -c 4                               # Request four cores
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 0-02:00                         # Runtime in D-HH:MM format
#SBATCH -p shared
#SBATCH --mem=8G                           # Memory total in MB (for all cores)
#SBATCH -o %A_%a.out                 # File to which STDOUT will be written. %A and %a represent the job ID and the job array index, respectively
#SBATCH -e %A_%a.err                 # File to which STDERR will be written. %A and %a represent the job ID and the job array index, respectively
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=beaucoker@g.harvard.edu   # Email to which notifications will be sent

prior_wb1_sig2_arr=(100 100 100 500 500 500 1000 1000 1000)
prior_wb2_sig2_arr=(.025 .05 .1 .025 .05 .1 .025 .05 .1)

module load Anaconda3/5.0.1-fasrc02

source activate venv282

python run_exp.py --dataset motorcycle --seed 1 \
 --dir_out motorcycle_$SLURM_ARRAY_JOB_ID/task=$SLURM_ARRAY_TASK_ID \
 --prior_w1_sig2 ${prior_wb1_sig2_arr[$SLURM_ARRAY_TASK_ID]} --prior_b1_sig2 ${prior_wb1_sig2_arr[$SLURM_ARRAY_TASK_ID]} \
 --prior_w2_sig2 ${prior_wb2_sig2_arr[$SLURM_ARRAY_TASK_ID]} --prior_b2_sig2 ${prior_wb2_sig2_arr[$SLURM_ARRAY_TASK_ID]} \
 --dim_hidden 50 \
 --sig2 0.04

source deactivate