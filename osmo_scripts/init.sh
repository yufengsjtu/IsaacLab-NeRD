#!/bin/bash
# In-pod environment setup for an image that already contains standard Isaac Lab
# and its simulator/runtime dependencies. Only install IsaacLab-NeRD on top.

set -ex

PIP="python3 -m pip install --no-cache-dir"
PROJECT_ROOT="$HOME/code/IsaacLab-NeRD"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[FATAL] IsaacLab-NeRD source tree not found at $PROJECT_ROOT"
    exit 1
fi

# NeRD project as editable. Dependencies are expected to be provided by the image.
$PIP -e "$PROJECT_ROOT/source/isaaclab_neural" --no-deps

# TensorBoard is used for live training monitoring and reading SummaryWriter logs.
if ! python3 -c "import tensorboard" 2>/dev/null; then
    $PIP tensorboard
fi
python3 -m tensorboard.main --help >/dev/null

echo "=== init.sh DONE ==="
