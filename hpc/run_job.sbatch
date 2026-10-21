#!/bin/bash
# Template. Submit from the repo root on Hypatia:
#   sbatch hpc/run_job.sbatch scripts/launcher.py train path/to/config.yaml
# Queues (docs): short(2d, default) medium(15d) long(30d) bigmem(15d) gpu(15d, max 2 GPU)
#SBATCH --job-name=kirad
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=sf.valencia@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o %x-log.o%j
#SBATCH -e %x-err.o%j

set -euo pipefail
export DATA_DIR="${DATA_DIR:-$HOME/data}"     # datasets live outside the repo
export RUNS_DIR="${RUNS_DIR:-$HOME/runs}"     # outputs/checkpoints outside the repo
mkdir -p "$RUNS_DIR"

module purge
module load anaconda
source activate "${ENV_NAME:-kintsugi}"

cd "$HOME/kintsugi-hpc-uniandes"
echo "Running commit: $(git rev-parse --short HEAD)"   # reproducibility: log what ran
python "$@"
