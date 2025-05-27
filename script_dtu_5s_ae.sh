#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=train_dtu_5s_ae
#SBATCH --account=PAS2301

#SBATCH --mem=64gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o //users/PAS2301/alialavi/projects/AAD-CMAA/logs/outputs/op_script_dtu_5s_ae.out

module load miniconda3

module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8
source activate /users/PAS2301/alialavi/miniconda3/envs/tf1
/users/PAS2301/alialavi/miniconda3/envs/tf1/bin/python /users/PAS2301/alialavi/projects/AAD-CMAA/train_ae_dtu_cont_5s.py
