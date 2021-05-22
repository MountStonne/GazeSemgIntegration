#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=10-0:00

source ~/GazeSemgIntegration/bin/activate
python features_analysis2.py
