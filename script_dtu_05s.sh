#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=train_all_05s
#SBATCH --account=PAS2301

#SBATCH --mem=64gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o //users/PAS2301/alialavi/projects/AAD-CMAA/op_script_dtu_05s.out

module load miniconda3

module load cuda/11.2.2
source activate /users/PAS2301/alialavi/miniconda3/envs/tf1
/users/PAS2301/alialavi/miniconda3/envs/tf1/bin/python /users/PAS2301/alialavi/projects/AAD-CMAA/train_all_3.py
