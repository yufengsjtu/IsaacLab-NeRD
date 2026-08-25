#!/bin/bash
# Standalone OSMO entrypoint for paired-tail contact analysis.

set -euo pipefail
set -x

CODE_INPUT_DIR="${TAIL_CODE_DIR:-/osmo/data/input/0}"
DATASET_INPUT_DIR="${TAIL_DATASET_DIR:-/osmo/data/input/1}"
A_METRICS_INPUT_DIR="${TAIL_A_METRICS_DIR:-/osmo/data/input/2}"
C_METRICS_INPUT_DIR="${TAIL_C_METRICS_DIR:-/osmo/data/input/3}"
OUTPUT_PARENT_URL="${TAIL_OUTPUT_PARENT_URL:?TAIL_OUTPUT_PARENT_URL is required}"
ARTIFACT_NAME="${TAIL_ARTIFACT_NAME:?TAIL_ARTIFACT_NAME is required}"
PROJECT_ROOT=/root/code/IsaacLab-NeRD
OUTPUT_DIR=/tmp/paired-tail-contact-output

install_python_shims() {
    if [[ -x /isaac-sim/python.sh ]]; then
        mkdir -p /tmp/shims
        printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /tmp/shims/python3
        chmod +x /tmp/shims/python3
        export PATH="/tmp/shims:$PATH"
    fi
}

find_exactly_one() {
    local root="$1"
    local pattern="$2"
    local count

    count="$(find "$root" -type f -name "$pattern" -print | wc -l)"
    if [[ "$count" -ne 1 ]]; then
        echo "[FATAL] Expected exactly one $pattern below $root, found $count." >&2
        exit 1
    fi
    find "$root" -type f -name "$pattern" -print | head -n 1
}

install_python_shims
archive="$(find_exactly_one "$CODE_INPUT_DIR" '*.tar.gz')"
dataset="$(find_exactly_one "$DATASET_INPUT_DIR" '*.hdf5')"
a_metrics="$(find_exactly_one "$A_METRICS_INPUT_DIR" 'per_window_metrics.npz')"
c_metrics="$(find_exactly_one "$C_METRICS_INPUT_DIR" 'per_window_metrics.npz')"
mkdir -p "$(dirname "$PROJECT_ROOT")" "$OUTPUT_DIR"
tar -xzf "$archive" -C "$(dirname "$PROJECT_ROOT")"
python3 -m pip install --no-cache-dir -e "$PROJECT_ROOT/source/isaaclab_neural" --no-deps
python3 -m isaaclab_neural.eval.analyze_paired_tail_contacts \
    --dataset "$dataset" \
    --a-metrics "$a_metrics" \
    --c-metrics "$c_metrics" \
    --output "$OUTPUT_DIR/tail_self_collision_analysis.json" \
    --tail-fraction 0.001

stage_root="$(mktemp -d)"
mkdir -p "$stage_root/$ARTIFACT_NAME"
cp -a "$OUTPUT_DIR"/. "$stage_root/$ARTIFACT_NAME"/
osmo data upload "$OUTPUT_PARENT_URL" "$stage_root/$ARTIFACT_NAME"
osmo data check "${OUTPUT_PARENT_URL%/}/$ARTIFACT_NAME/"
osmo data list "${OUTPUT_PARENT_URL%/}/$ARTIFACT_NAME/" --recursive --no-pager
