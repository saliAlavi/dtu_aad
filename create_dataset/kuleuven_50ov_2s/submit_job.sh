#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --job-name=create_ku_2s
#SBATCH --account=PAS2301
#SBATCH --mem=64gb

#SBATCH --cpus-per-task=8
#SBATCH -o /users/PAS2301/alialavi/projects/AAD-CMAA/create_dataset/kuleuven_50ov_2s/sbatch_output.out

current_dir_name=$(basename "$PWD")
folder_name="/fs/scratch/PAS2622/tensorflow_datasets"
DEST_DIR="/fs/scratch/PAS2622/tensorflow_datasets/$current_dir_name"
module load miniconda3
source activate /users/PAS2301/alialavi/miniconda3/envs/tf1
cd /users/PAS2301/alialavi/projects/AAD-CMAA/create_dataset/{$current_dir_name}
/users/PAS2301/alialavi/miniconda3/envs/tf1/bin/tfds build --data_dir "$DEST_DIR"


# Check if the folder exists
if [ ! -d "$folder_name" ]; then
  # If it doesn't exist, create it
  mkdir "$folder_name"
  echo "Folder '$folder_name' created."
else
  echo "Folder '$folder_name' already exists."
fi

SOURCE_DIR="/users/PAS2301/alialavi/tensorflow_datasets/$current_dir_name"

# rsync -a --info=progress2 "$SOURCE_DIR" "$DEST_DIR" && rm -rf "$SOURCE_DIR"
ln -s "$DEST_DIR" "$SOURCE_DIR"