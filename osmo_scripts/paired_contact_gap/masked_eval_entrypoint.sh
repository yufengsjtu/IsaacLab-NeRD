#!/bin/bash
# Standalone OSMO entrypoint for C-active self-collision masking evaluation.

set -euo pipefail
set -x

CODE_INPUT_DIR="${MASKED_EVAL_CODE_DIR:-/osmo/data/input/0}"
DATASET_INPUT_DIR="${MASKED_EVAL_DATASET_DIR:-/osmo/data/input/1}"
MANIFEST_INPUT_DIR="${MASKED_EVAL_MANIFEST_DIR:-/osmo/data/input/2}"
CHECKPOINT_INPUT_DIR="${MASKED_EVAL_CHECKPOINT_DIR:-/osmo/data/input/3}"
OUTPUT_PARENT_URL="${MASKED_EVAL_OUTPUT_PARENT_URL:?MASKED_EVAL_OUTPUT_PARENT_URL is required}"
ARTIFACT_NAME="${MASKED_EVAL_ARTIFACT_NAME:?MASKED_EVAL_ARTIFACT_NAME is required}"
PROJECT_ROOT=/root/code/IsaacLab-NeRD
OUTPUT_DIR=/tmp/masked-self-eval-output

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
paired_manifest="$(find_exactly_one "$MANIFEST_INPUT_DIR" 'paired_contact_policy_eval_manifest.json')"
checkpoint_manifest="$(find_exactly_one "$CHECKPOINT_INPUT_DIR" 'checkpoint_manifest.json')"
mkdir -p "$(dirname "$PROJECT_ROOT")" "$OUTPUT_DIR"
tar -xzf "$archive" -C "$(dirname "$PROJECT_ROOT")"
python3 -m pip install --no-cache-dir -e "$PROJECT_ROOT/source/isaaclab_neural" --no-deps
python3 -m isaaclab_neural.eval.run_masked_self_collision_eval \
    --dataset "$dataset" \
    --paired-dataset-manifest "$paired_manifest" \
    --checkpoint-manifest "$checkpoint_manifest" \
    --design c_active \
    --expected-seeds 0,1,2 \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 512 \
    --num-workers 8 \
    --num-envs 1 \
    --seed 0 \
    --device cuda:0 \
    --headless

stage_root="$(mktemp -d)"
mkdir -p "$stage_root/$ARTIFACT_NAME"
cp -a "$OUTPUT_DIR"/. "$stage_root/$ARTIFACT_NAME"/
osmo data upload "$OUTPUT_PARENT_URL" "$stage_root/$ARTIFACT_NAME"
osmo data check "${OUTPUT_PARENT_URL%/}/$ARTIFACT_NAME/"
osmo data list "${OUTPUT_PARENT_URL%/}/$ARTIFACT_NAME/" --recursive --no-pager
