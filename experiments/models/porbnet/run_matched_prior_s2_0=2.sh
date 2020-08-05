#!/bin/bash
#SBATCH -J s2_0=2 						   # A single job name for the array
#SBATCH -c 4                               # Request four cores
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 0-06:00                         # Runtime in D-HH:MM format
#SBATCH -p shared
#SBATCH --mem=12G                           # Memory total in MB (for all cores)
#SBATCH -o %A_%a.out                 # File to which STDOUT will be written. %A and %a represent the job ID and the job array index, respectively
#SBATCH -e %A_%a.err                 # File to which STDERR will be written. %A and %a represent the job ID and the job array index, respectively
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=beaucoker@g.harvard.edu   # Email to which notifications will be sent

dataset_arr=(finance motorcycle GPdata GPdata GPdata GPdata mimic_gap mimic_gap mimic_gap mimic_gap)
dataset_option_arr=(None None inc inc_gap sin stat_gap 11 1228 1472 1535)

module load Anaconda3/5.0.1-fasrc02

source activate venv282

python run_exp.py \
 --dataset ${dataset_arr[$SLURM_ARRAY_TASK_ID]} \
 --dataset_option ${dataset_option_arr[$SLURM_ARRAY_TASK_ID]}\
 --seed 1 \
 --dir_out s2_0=2/${dataset_arr[$SLURM_ARRAY_TASK_ID]}/${dataset_option_arr[$SLURM_ARRAY_TASK_ID]}/$SLURM_ARRAY_JOB_ID/task=$SLURM_ARRAY_TASK_ID \
 --s2_0 2.0 \
 --prior_matching_file ../../process_results/output/match_priors/porbnet/s2_0=2/prior_matching.csv \
 --T_lb -.25 --T_ub 1.25 \
 --burnin_n_bigwrite 2 \
 --record_n_bigwrite 2

source deactivate