#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/%u/East-African-Dataset/logs/%j.log
#SBATCH --job-name=east_african_translation
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

# export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="/projects/$USER/.cache/huggingface"
export EVALUATE_CACHE_DIR="/projects/$USER/.cache/evaluate"
export TRANSFORMERS_CACHE="/projects/$USER/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE"

module purge
module load anaconda
conda activate east_african_dataset

cd /projects/$USER/East-African-Dataset

python -u run.py
