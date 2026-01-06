#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --job-name=train_lora_qwen
#SBATCH --output=outputs/train_lora_qwen_%j.out

set -e

echo "Job started on $(hostname) at $(date)"

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv || conda activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=.
python -m src.training.lora --model Qwen3VL --debug

echo "--- THE END ---"
echo "Job ended at $(date)"