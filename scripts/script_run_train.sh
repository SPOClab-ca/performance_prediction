#!/bin/bash

#SBATCH -c 2
#SBATCH --mem=32GB
#SBATCH --partition=cpu
#SBATCH --output=../slurm/%j.out
#SBATCH --error=../slurm/%j.err
#SBATCH --qos=nopreemption

. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load transformers4

python -u probe.py --task bigram_shift