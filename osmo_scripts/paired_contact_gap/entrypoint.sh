#!/bin/bash
# Standalone OSMO entrypoint for paired contact-gap generation and evaluation.

set -euo pipefail
set -x

MODE="${PAIRED_EVAL_MODE:?PAIRED_EVAL_MODE is required}"
CODE_INPUT_DIR="${PAIRED_EVAL_CODE_DIR:-/osmo/data/input/0}"
OUTPUT_PARENT_URL="${PAIRED_EVAL_OUTPUT_PARENT_URL:?PAIRED_EVAL_OUTPUT_PARENT_URL is required}"
ARTIFACT_NAME="${PAIRED_EVAL_ARTIFACT_NAME:?PAIRED_EVAL_ARTIFACT_NAME is required}"
PROJECT_ROOT=/root/code/IsaacLab-NeRD
OUTPUT_DIR=/tmp/paired-contact-gap-output

install_python_shims() {
    if [[ -x /isaac-sim/python.sh ]]; then
        mkdir -p /tmp/shims
        printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /tmp/shims/python3
        chmod +x /tmp/shims/python3
        export PATH="/tmp/shims:$PATH"
    fi
}

extract_code() {
    local archive archive_count

    archive_count="$(find "$CODE_INPUT_DIR" -type f \( -name '*.tar.gz' -o -name '*.tgz' \) -print | wc -l)"
    if [[ "$archive_count" -ne 1 ]]; then
        echo "[FATAL] Expected exactly one immutable code archive, found $archive_count." >&2
        exit 1
    fi
    archive="$(find "$CODE_INPUT_DIR" -type f \( -name '*.tar.gz' -o -name '*.tgz' \) -print | head -n 1)"
    mkdir -p "$(dirname "$PROJECT_ROOT")"
    tar -xzf "$archive" -C "$(dirname "$PROJECT_ROOT")"
    if [[ ! -f "$PROJECT_ROOT/isaaclab.sh" ]]; then
        echo "[FATAL] Extracted code archive does not contain IsaacLab-NeRD." >&2
        exit 1
    fi
}

install_project() {
    python3 -m pip install --no-cache-dir -e "$PROJECT_ROOT/source/isaaclab_neural" --no-deps
}

find_exactly_one() {
    local root="$1"
    local pattern="$2"
    local match match_count

    match_count="$(find "$root" -type f -name "$pattern" -print | wc -l)"
    if [[ "$match_count" -ne 1 ]]; then
        echo "[FATAL] Expected exactly one $pattern below $root, found $match_count." >&2
        exit 1
    fi
    match="$(find "$root" -type f -name "$pattern" -print | head -n 1)"
    printf '%s\n' "$match"
}

upload_output() {
    local stage_root child_url

    stage_root="$(mktemp -d)"
    mkdir -p "$stage_root/$ARTIFACT_NAME"
    cp -a "$OUTPUT_DIR"/. "$stage_root/$ARTIFACT_NAME"/
    osmo data upload "$OUTPUT_PARENT_URL" "$stage_root/$ARTIFACT_NAME"
    child_url="${OUTPUT_PARENT_URL%/}/$ARTIFACT_NAME/"
    osmo data check "$child_url"
    osmo data list "$child_url" --recursive --no-pager
}

generate_dataset() {
    local policy_checkpoint

    policy_checkpoint="$PROJECT_ROOT/pretrained/control_policy/anymal_c_rough_terrain/model_1499.pt"
    if [[ ! -f "$policy_checkpoint" ]]; then
        echo "[FATAL] Rough-terrain policy checkpoint is missing from the code archive." >&2
        exit 1
    fi
    python3 -m isaaclab_neural.eval.generate_paired_contact_dataset \
        --task Isaac-Velocity-Rough-Anymal-C-v0 \
        --policy-checkpoint "$policy_checkpoint" \
        --dataset-dir "$OUTPUT_DIR" \
        --sample-mode policy \
        --initial-states-source env \
        --contact-mode newton_native \
        --contact-packing-policy body_round_robin_pair_atomic \
        --contact-representation raw15_tokens \
        --max-contact-tokens 64 \
        --num-contacts-per-env 64 \
        --num-envs 1024 \
        --num-transitions 1000000 \
        --trajectory-length 400 \
        --write-chunk-transitions 409600 \
        --seed 40 \
        --states-frame body \
        --anchor-frame-step every \
        --states-embedding-type identical \
        --prediction-type relative \
        --orientation-prediction-parameterization quaternion \
        --data-device cpu \
        --device cuda:0 \
        --headless \
        --force-overwrite
}

evaluate_checkpoints() {
    local checkpoint_manifest dataset dataset_count dataset_manifest dataset_pattern design dataset_key

    design="${PAIRED_EVAL_DESIGN:?PAIRED_EVAL_DESIGN is required}"
    dataset_key="${PAIRED_EVAL_DATASET_KEY:?PAIRED_EVAL_DATASET_KEY is required}"
    dataset_manifest="$(
        find_exactly_one "${PAIRED_EVAL_DATASET_DIR:?PAIRED_EVAL_DATASET_DIR is required}" \
            paired_contact_policy_eval_manifest.json
    )"
    checkpoint_manifest="$(
        find_exactly_one "${PAIRED_EVAL_CHECKPOINT_DIR:?PAIRED_EVAL_CHECKPOINT_DIR is required}" \
            checkpoint_manifest.json
    )"
    case "$dataset_key" in
        raw15)
            dataset_pattern='*Raw15-Paired-Eval/dataset_lstm_actuator_policy_valid.hdf5'
            ;;
        contact_tokens)
            dataset_pattern='*ContactTokens-Paired-Eval/dataset_lstm_actuator_policy_valid.hdf5'
            ;;
        *)
            echo "[FATAL] Unsupported paired dataset key: $dataset_key" >&2
            exit 1
            ;;
    esac
    dataset_count="$(find "$PAIRED_EVAL_DATASET_DIR" -type f -path "$dataset_pattern" -print | wc -l)"
    if [[ "$dataset_count" -ne 1 ]]; then
        echo "[FATAL] Expected one paired $dataset_key HDF5, found $dataset_count." >&2
        exit 1
    fi
    dataset="$(find "$PAIRED_EVAL_DATASET_DIR" -type f -path "$dataset_pattern" -print | head -n 1)"

    python3 -m isaaclab_neural.eval.run_paired_checkpoint_eval \
        --dataset "$dataset" \
        --paired-dataset-manifest "$dataset_manifest" \
        --checkpoint-manifest "$checkpoint_manifest" \
        --design "$design" \
        --expected-seeds 0,1,2 \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 512 \
        --num-workers 8 \
        --num-envs 1 \
        --seed 0 \
        --device cuda:0 \
        --headless
}

install_python_shims
extract_code
install_project
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
case "$MODE" in
    generate)
        generate_dataset
        ;;
    evaluate)
        evaluate_checkpoints
        ;;
    *)
        echo "[FATAL] Unsupported PAIRED_EVAL_MODE: $MODE" >&2
        exit 1
        ;;
esac
upload_output
