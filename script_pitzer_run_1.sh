#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --job-name=tmp_main_ssl
#SBATCH --account=PAS2622

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o //users/PAS2301/alialavi/projects/AAD-CMAA/op_script_pitzer_run_1.out

module load miniconda3

module load cuda/11.8.0
source activate /users/PAS2301/alialavi/miniconda3/envs/tfp
/users/PAS2301/alialavi/miniconda3/envs/tfp/bin/python /users/PAS2301/alialavi/projects/AAD-CMAA/main.py
