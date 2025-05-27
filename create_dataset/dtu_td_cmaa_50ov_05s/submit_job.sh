#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --job-name=ds_dtu_05
#SBATCH --account=PAS2301
#SBATCH --mem=64gb

#SBATCH --cpus-per-task=8
#SBATCH -o /users/PAS2301/alialavi/projects/AAD-CMAA/create_dataset/dtu_td_cmaa_50ov_05s/sbatch_output.out

current_dir_name=$(basename "$PWD")
folder_name="/fs/scratch/PAS2622/tensorflow_datasets"
DEST_DIR="/fs/scratch/PAS2622/tensorflow_datasets/$current_dir_name"
DATA_DIR="/fs/scratch/PAS2622/tensorflow_datasets/"
module load miniconda3
source activate /users/PAS2301/alialavi/miniconda3/envs/tf1
cd /users/PAS2301/alialavi/projects/AAD-CMAA/create_dataset/{$current_dir_name}
ln -s "$DEST_DIR" "$SOURCE_DIR"
/users/PAS2301/alialavi/miniconda3/envs/tf1/bin/tfds build --data_dir "$DATA_DIR"


