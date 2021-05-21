#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-12:00

source ~/GazeSemgIntegration/bin/activate
python features_analysis.py
