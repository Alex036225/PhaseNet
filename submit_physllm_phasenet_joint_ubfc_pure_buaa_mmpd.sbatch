#!/bin/bash
#SBATCH -J physllm_phase_joint4
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_bundle/slurm_logs/physllm_phase_joint4-%j.out
#SBATCH --error=/public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_bundle/slurm_logs/physllm_phase_joint4-%j.err

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

TRAIN_CFG="configs/PhaseNet_JOINT_UBFC_PURE_BUAA_MMPD_72x72_160_train_test.yaml"
ONLYTEST_UBFC="configs/PhaseNet_JOINT_UBFC_PURE_BUAA_MMPD_72x72_160_onlytest_UBFC-rPPG.yaml"
ONLYTEST_PURE="configs/PhaseNet_JOINT_UBFC_PURE_BUAA_MMPD_72x72_160_onlytest_PURE.yaml"
ONLYTEST_BUAA="configs/PhaseNet_JOINT_UBFC_PURE_BUAA_MMPD_72x72_160_onlytest_BUAA.yaml"
ONLYTEST_MMPD="configs/PhaseNet_JOINT_UBFC_PURE_BUAA_MMPD_72x72_160_onlytest_MMPD.yaml"

echo "[1/5] Training with $TRAIN_CFG"
python main.py --config_file "$TRAIN_CFG"

echo "[2/5] Testing UBFC"
python main.py --config_file "$ONLYTEST_UBFC"

echo "[3/5] Testing PURE"
python main.py --config_file "$ONLYTEST_PURE"

echo "[4/5] Testing BUAA"
python main.py --config_file "$ONLYTEST_BUAA"

echo "[5/5] Testing MMPD"
python main.py --config_file "$ONLYTEST_MMPD"
