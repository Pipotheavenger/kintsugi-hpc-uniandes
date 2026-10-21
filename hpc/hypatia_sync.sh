#!/usr/bin/env bash
# Run ON Hypatia. Mirrors GitHub `main` exactly (fast-forward only) and, on first
# use, clones the repo and creates the conda env.
#   bash hpc/hypatia_sync.sh            # sync only
#   bash hpc/hypatia_sync.sh --env      # sync + create conda env "$ENV_NAME" if missing (never updates it)
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Pipotheavenger/kintsugi-hpc-uniandes.git}"
REPO_DIR="${REPO_DIR:-$HOME/Kintusgi/kintsugi-hpc-uniandes}"
ENV_NAME="${ENV_NAME:-kintsugi}"
ENV_FILE="envs/kipy-gpu-full.yaml"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --single-branch -b main "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# Track only main and never create merge commits on the cluster.
git config pull.ff only
git checkout main
git branch --set-upstream-to=origin/main main >/dev/null
git pull --ff-only

# Warn if someone edited files on the cluster (they would block future pulls).
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: local changes on Hypatia:"; git status --short
fi
echo "Hypatia now at: $(git log --oneline -1)"

if [ "${1:-}" = "--env" ]; then
  module load anaconda
  # Never modify an existing env (no `conda env update`): only create it if missing.
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists: left untouched."
  else
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi
fi
