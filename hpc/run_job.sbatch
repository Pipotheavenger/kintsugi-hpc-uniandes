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
# Some nodes (gpu_economia: nodea-gpu-3) have no git executable: read the commit from .git directly and let
# GitPython import without it (kirad.utils.get_package_info then logs KIRAD_GIT_COMMIT).
if command -v git >/dev/null 2>&1; then
    KIRAD_GIT_COMMIT=$(git rev-parse HEAD)
else
    export GIT_PYTHON_REFRESH=quiet
    HEAD_REF=$(sed 's/^ref: //' .git/HEAD)
    KIRAD_GIT_COMMIT=$(cat ".git/$HEAD_REF" 2>/dev/null || grep " $HEAD_REF\$" .git/packed-refs | cut -d' ' -f1)
fi
export KIRAD_GIT_COMMIT
echo "Running commit: ${KIRAD_GIT_COMMIT:0:7}"   # reproducibility: log what ran
python "$@"
