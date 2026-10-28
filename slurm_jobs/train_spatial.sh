#!/bin/bash
#SBATCH --job-name=openvla-lora
#SBATCH --account=cvcs2026
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_%j.log

eval "$(conda shell.bash hook)"
conda activate openvla
cd /work/cvcs2026/VenusIA/openvla


export MUJOCO_GL=egl
export WANDB_MODE=offline

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /work/cvcs2026/VenusIA/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /work/cvcs2026/VenusIA/openvla/checkpoints/openvla_run2 \
  --batch_size 12 \
  --grad_accumulation_steps 1 \
  --learning\_rate 5e-4 \
  --lora_rank 32 \
  --save_steps 5000