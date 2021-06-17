#!/bin/bash
#SBATCH --time=0-14:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=4096M

source ~/GazeSemgIntegration/bin/activate
python WL2.py