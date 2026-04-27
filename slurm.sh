#!/bin/bash
# ==============================================================================
# SLURM Job Script — DrQ Habitat Image-Goal Navigation (HER)
# ==============================================================================
# Submit:    sbatch slurm.sh
# Monitor:   squeue -u $USER
# Cancel:    scancel <jobid>
# ==============================================================================

#SBATCH --job-name=habitat_her
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egyaase@maine.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:l40:1
#SBATCH --output=logs/habitat_her_%j.log

# ==============================================================================
# Environment Setup
# ==============================================================================
ENV_NAME="habitat"

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "============================================"

module load anaconda3
$INIT_CONDA
conda activate ${ENV_NAME}

# Create log directory
mkdir -p logs

# ==============================================================================
# Training — Recommended HPC Hyperparameters
# ==============================================================================
# Replay buffer  : 1M  (10x default — more diverse training data)
# Max steps      : 5M  (5x default — longer training on HPC)
# Batch size     : 256 (2x default — larger gradient batches, more stable)
# Start training : 10K (2x default — let buffer fill before updates)
# ==============================================================================

python train_habitat_her.py \
    --replay_buffer_size 1000000 \
    --max_steps 5000000 \
    --batch_size 256 \
    --start_training 10000 \
    --debug_render False \
    --video_interval 50000 \
    --video_length 500 \
    --checkpoint_interval 10000 \
    --randomize_scenes True \
    --log_interval 1000 \
    --tqdm True

echo "============================================"
echo "End time     : $(date)"
echo "============================================"