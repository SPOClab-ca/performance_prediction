#!/bin/bash

#SBATCH -c 2
#SBATCH --mem=64GB
#SBATCH --partition=rtx6000,t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/preprocess_%j.out
#SBATCH --error=../slurm/preprocess_%j.err
#SBATCH --qos=normal
#SBATCH --dependency=afterburstbuffer:5176777

. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load transformers4

python -u preprocess_data.py
