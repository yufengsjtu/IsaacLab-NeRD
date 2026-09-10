#!/bin/bash
# In-pod environment setup for an image that already contains simulator/runtime
# dependencies. Reinstall the uploaded Isaac Lab tree so jobs import that copy
# instead of the Isaac Lab distribution bundled in the container image.

set -ex

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/code/IsaacLab-NeRD}"
PIP="python3 -m pip install --no-cache-dir"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[FATAL] IsaacLab-NeRD source tree not found at $PROJECT_ROOT"
    exit 1
fi

python3 "$PROJECT_ROOT/osmo_scripts/lib/local_isaaclab.py" bootstrap --project-root "$PROJECT_ROOT"
python3 -c "import lazy_loader"

# TensorBoard is used for live training monitoring and reading SummaryWriter logs.
if ! python3 -c "import tensorboard" 2>/dev/null; then
    $PIP tensorboard
fi
python3 -m tensorboard.main --help >/dev/null

if [[ "${ENABLE_WANDB:-false}" == "true" || "${ENABLE_WANDB:-0}" == "1" ]]; then
    if ! python3 -c "import wandb" 2>/dev/null; then
        $PIP wandb
    fi
    python3 -c "import wandb"
fi

echo "=== init.sh DONE ==="
