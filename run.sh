#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/%u/East-African-Dataset/logs/%j.log
#SBATCH --job-name=east_african_translation
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HF_HOME="/tmp/$USER/.cache/huggingface"
export EVALUATE_CACHE_DIR="/tmp/$USER/.cache/evaluate"
export TRANSFORMERS_CACHE="/tmp/$USER/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE"

module purge
module  load cuda
module load anaconda
conda activate east_african_dataset

cd /projects/$USER/East-African-Dataset

python -u run.py
