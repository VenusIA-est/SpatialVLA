#!/bin/bash
#SBATCH --job-name=eval_openvla_vanilla
#SBATCH --account=cvcs2026
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=vanilla_out_%j.txt
#SBATCH --error=vanilla_err_%j.txt
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"

eval "$(conda shell.bash hook)"
conda activate openvla

cd /work/cvcs2026/VenusIA/openvla
export MUJOCO_GL=egl

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b \
    --model_family openvla \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 15