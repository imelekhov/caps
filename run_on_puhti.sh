#!/bin/bash
#SBATCH --job-name=caps
#SBATCH --account=project_2003267
#SBATCH --partition=gputest
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1,nvme:3500

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/scratch/project_2003267/anaconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/scratch/project_2003267/anaconda2/etc/profile.d/conda.sh" ]; then
        . "/scratch/project_2003267/anaconda2/etc/profile.d/conda.sh"
    else
        export PATH="/scratch/project_2003267/anaconda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda -V
conda activate caps
rsync /scratch/project_2003267/datasets/CAPS-MegaDepth-release-light.tar $TMPDIR
tar -xvf $TMPDIR/CAPS-MegaDepth-release-light.tar -C $TMPDIR

export PROJECT_OUTPUT_PATH=/scratch/project_2003267/caps_output
srun python3 train.py --datadir $TMPDIR/CAPS-MegaDepth-release-light \
                      --n_iters 60 \
                      --exp_name puhti_test \
                      --logdir $PROJECT_OUTPUT_PATH/logs \
                      --outdir $PROJECT_OUTPUT_PATH/out
