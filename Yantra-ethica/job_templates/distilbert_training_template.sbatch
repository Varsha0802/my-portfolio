#!/bin/bash
#SBATCH --account=lnun5155@cgu.edu_1759
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1               # Prefer A100, fallback to available
#SBATCH --time=04:00:00                 # 4 hours
#SBATCH --job-name=distilbert_training
#SBATCH --output=../results/model_ethos/training_%j.out
#SBATCH --error=../results/model_ethos/training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=balan.mahesh@cgu.edu

echo "=== DistilBERT Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules
module load gcc/13.3.0
module load python/3.11.9

# Try to load CUDA if available
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "CUDA module not found"

# Project setup
PROJECT_ROOT="/project/LNUN5155_1759/cgu_bias_detector"
cd $PROJECT_ROOT

# Activate environment
source bias_detection_env/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Show system info
echo "=== System Information ==="
nvidia-smi
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run training script (placeholder - will be updated when scripts are ready)
echo "=== Starting Training ==="
echo "Training script will go here.."
python scripts/train_model_ethos.py 
echo "=== Training completed at: $(date) ==="


    #  --config models/configs/distilbert_config.yaml \
    #  --output_dir models/checkpoints/run_${SLURM_JOB_ID} \
    #  --logging_dir results/logs/run_${SLURM_JOB_ID}