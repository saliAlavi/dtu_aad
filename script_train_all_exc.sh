#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=tmp_main_ssl
#SBATCH --account=PAS2622

#SBATCH --mem=100gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o //users/PAS2301/alialavi/projects/AAD-CMAA/op_script_train_all_exc.out

module load miniconda3
module load cuda/11.2.2
source activate /users/PAS2301/alialavi/miniconda3/envs/tf1
/users/PAS2301/alialavi/miniconda3/envs/tf1/bin/python /users/PAS2301/alialavi/projects/AAD-CMAA/train_except.py
