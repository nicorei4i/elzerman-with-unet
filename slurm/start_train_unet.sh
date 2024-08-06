#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name="Elzerman Training unet"
#SBATCH --time=0-01:00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=/home/md334167/elzerman-with-unet/logs/%j_%n_%x.txt
#SBATCH --error=/home/md334167/elzerman-with-unet/logs/%j_%n_%x.err

module load GCC/9.4.0
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/md334167/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/md334167/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/md334167/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/md334167/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate DL

# run the training
python -u /home/md334167/elzerman-with-unet/unet_train.py