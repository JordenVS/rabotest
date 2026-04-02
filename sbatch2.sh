#!/bin/bash
#SBATCH --job-name=first-test
#SBATCH --output=slurm.test-%j.out
#SBATCH --error=slurm.test-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --account=ewi-insy-reit
#SBATCH --partition=general
#SBATCH --qos=short

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=================================================="
echo ""

# Print allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo ""

set -euo pipefail 

HF_Model_Path="sentence-transformers/all-MiniLM-L6-v2"

module use /opt/insy/modulefiles
module load python
module load devtoolset/11
cd $HOME/rabotest
source .venv/Scripts/activate
uv sync

# Set HF home & cache dir
export HF_HOME="hf_cache"
export HF_HUB_CACHE="$HF_HOME"
echo "HF_HUB_CACHE set to: $HF_HUB_CACHE"

# Download HF model
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download $HF_Model_Path \
    --local-dir "$HF_HUB_CACHE/$(basename $HF_Model_Path)" \
    --local-dir-use-symlinks False

# Environment variables for enabling containers:
#export APPTAINER_PATH=/tudelft.net/staff-umbrella/REITcourses/apptainer/pytorch2.2.1-cuda12.1.sif
export APPTAINER_PATH=/tudelft.net/staff-umbrella/REITcourses/apptainer/llm_on_pytorch2.3.1-cuda12.1-cudnn8-runtime.sif

# Check that container file exists
if [ ! -f $APPTAINER_PATH ]; then
    ls $APPTAINER_PATH
    exit 1
fi

echo "Running GPU ML example with PyTorch..."
echo "Container: $APPTAINER_PATH"
echo ""    

apptainer exec --nv $APPTAINER_PATH python3 gpu_ml_example.py
