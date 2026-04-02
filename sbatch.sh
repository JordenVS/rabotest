#!/bin/bash
#SBATCH --job-name=first-test
#SBATCH --output=slurm.test-%j.out
#SBATCH --error=slurm.test-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:05:00
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

export APPTAINER_PATH=/tudelft.net/staff-umbrella/REITcourses/apptainer/llm_on_pytorch2.3.1-cuda12.1-cudnn8-runtime.sif

# Check that container file exists
if [ ! -f ${APPTAINER_PATH} ]; then
    ls ${APPTAINER_PATH}
    exit 1
else
    echo "Using apptainer: ${APPTAINER_PATH}"
fi 

module use /opt/insy/modulefiles
module load python
module load devtoolset/11

APPTAINER_CWD=$HOME/rabotest/apptainer_workdir
mkdir -p $APPTAINER_CWD

# Set HF home & cache dir
export HF_HOME="hf_cache"
export HF_HUB_CACHE="$HF_HOME"
echo "HF_HUB_CACHE set to: $HF_HUB_CACHE"

srun apptainer run --nv \
  --env HF_HOME=FIXME \
  ${APPTAINER_PATH} \
  hf_excercise.py

cd $HOME/rabotest


python main.py    

rm -r /tmp/$USER/