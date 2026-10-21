#!/bin/bash
# Submit from the repo root on Hypatia (~/Kintusgi/kintsugi-hpc-uniandes), e.g. stage 1:
#   sbatch hpc/run_job.sbatch scripts/launcher.py train hpc/configs/stage1.yaml \
#       --experiment-dir research/stable/ordinal_regression_mtl_mar_2024
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
REPO_DIR="$HOME/Kintusgi/kintsugi-hpc-uniandes"
export KINTSUGI_RAD_PATH="$REPO_DIR"   # where kirad.utils.get_package_path finds kintsugi-rad
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"   # so `import kirad` works without pip install

module purge
module load anaconda
# Activate the env automatically (batch shells don't read ~/.bashrc; conda scripts break under set -u).
set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-kintsugi}"
set -u

cd "$REPO_DIR"
echo "Running commit: $(git rev-parse --short HEAD)"   # reproducibility: log what ran
python "$@"
