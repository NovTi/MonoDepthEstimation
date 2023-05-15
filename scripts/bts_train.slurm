#!/bin/bash

#SBATCH --job-name=bts-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=25:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2

# job info
config_path=$1
exp_name=$2
exp_id=$3

# Singularity path
ext3_path=/scratch/$USER/my_env/py10/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/my_env/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/NYU/RobortVision/MonoDepth
python -m src.train --config ${config_path} --exp_name ${exp_name} --exp_id ${exp_id}
"