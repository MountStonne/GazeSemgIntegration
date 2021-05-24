#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem=32000M
#SBATCH --time=10-0:00

source ~/GazeSemgIntegration/bin/activate
python SSI_SKW_KURT_AAC.py
