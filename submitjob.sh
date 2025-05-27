#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --job-name=train_dtu_cont
#SBATCH --account=PAS2301
#SBATCH --mem=64gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /users/PAS2301/alialavi/projects/AAD-CMAA/sbatch_output.out

#module load miniconda3
module load cuda/11.8.0
#module load cudnn/8.6.0.163-11.8
#source activate /users/PAS2301/alialavi/miniconda3/envs/tf
source activate tf
cd /users/PAS2301/alialavi/projects/AAD-CMAA
python train_all.py
