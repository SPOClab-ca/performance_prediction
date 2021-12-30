#!/bin/bash

#SBATCH -c 2
#SBATCH --mem=32GB
#SBATCH --partition=cpu
#SBATCH --output=../slurm/probing_%j.out
#SBATCH --error=../slurm/probing_%j.err
#SBATCH --qos=nopreemption


. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load transformers4

python -u probing.py --corruption_step $1 --task bigram_shift
python -u probing.py --corruption_step $1 --task coordination_inversion
python -u probing.py --corruption_step $1 --task obj_number
python -u probing.py --corruption_step $1 --task odd_man_out
python -u probing.py --corruption_step $1 --task past_present
python -u probing.py --corruption_step $1 --task subj_number
python -u probing.py --corruption_step $1 --task tree_depth