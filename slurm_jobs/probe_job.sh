#!/bin/bash
#SBATCH --job-name=probe_variants
#SBATCH --account=cvcs2026
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=probe_out_%j.txt
#SBATCH --error=probe_err_%j.txt
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"

eval "$(conda shell.bash hook)"
conda activate openvla

cd /work/cvcs2026/VenusIA/openvla

python probe_variants.py