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
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/egyaase/robot-learning-tools/NVIDIA-Linux-x86_64-545.23.06"
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

# ── GPU diagnostics ──────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi failed — GPU may not be visible"

# ── Verify EGL ICD file exists ──────────────────────────────────────────────
if [ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]; then
    echo "WARNING: EGL ICD file not found at $__EGL_VENDOR_LIBRARY_FILENAMES"
    echo "Searching for alternative NVIDIA EGL ICD..."
    ALT=$(find /usr -name "10_nvidia.json" 2>/dev/null | head -1)
    if [ -n "$ALT" ]; then
        echo "Found: $ALT"
        export __EGL_VENDOR_LIBRARY_FILENAMES="$ALT"
    else
        echo "ERROR: No NVIDIA EGL ICD found. Habitat-Sim will likely segfault."
    fi
fi

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
echo "=== DISPLAY (will unset for headless EGL) ==="
echo $DISPLAY
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
echo "=== EGL CHECK ==="
python -c "
import ctypes
egl = ctypes.cdll.LoadLibrary('libEGL.so.1')
print('EGL loaded OK')
"
echo "=== VULKAN CHECK ==="
vulkaninfo --summary 2>/dev/null || echo "No vulkan"

python train_image_goal_hpc.py

echo "============================================"
echo "End time     : $(date)"
echo "============================================"