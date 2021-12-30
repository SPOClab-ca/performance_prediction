#!/bin/bash

#SBATCH -c 2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2,t4v1,rtx6000
#SBATCH --output=../slurm/glue_classify_%j.out
#SBATCH --qos=normal

. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load transformers4

python -u glue_classify.py \
    --task $2 \
    --model roberta-base \
    --corruption_step 0 \
    --batch_size 4 \
    --init_lr $1 \
    --num_epochs 3 \
    --checkpoint_dir /checkpoint/$USER/$SLURM_JOB_ID \
    --slurm_id $SLURM_JOB_ID
