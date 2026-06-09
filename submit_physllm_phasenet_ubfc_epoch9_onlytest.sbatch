#!/bin/bash
#SBATCH -J physllm_phase_test
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_bundle/slurm_logs/physllm_phase_test-%j.out
#SBATCH --error=/public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_bundle/slurm_logs/physllm_phase_test-%j.err

set -euo pipefail

source /public_hw/home/cit_yingxinlai/anaconda3/etc/profile.d/conda.sh
conda activate rllm

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd /public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_bundle
mkdir -p slurm_logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
which python
python --version
nvidia-smi

python main.py --config_file configs/PhaseNet_UBFC_UBFC_072_100_onlytest_epoch9.yaml
