#!/bin/bash
#SBATCH --job-name=multi_gpu_exp
#SBATCH --account=amandamurray19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40:8
#SBATCH --mem=792G
#SBATCH --time=72:00:00
#SBATCH --qos=savio_debug
#SBATCH --partition=savio4_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda3
source activate tf_env

cd /path/to/your/code

python train_with_memmap.py
