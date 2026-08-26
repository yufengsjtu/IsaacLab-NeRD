#!/bin/bash
# Standalone OSMO entrypoint for fixed-Epoch-199 sampling evaluation.

set -euo pipefail
set -x

ENCODER="${SAMPLING_EVAL_ENCODER:?SAMPLING_EVAL_ENCODER is required}"
OLD_DATASET_ROOT="${SAMPLING_EVAL_OLD_DATASET_ROOT:?SAMPLING_EVAL_OLD_DATASET_ROOT is required}"
NEW_DATASET_ROOT="${SAMPLING_EVAL_NEW_DATASET_ROOT:?SAMPLING_EVAL_NEW_DATASET_ROOT is required}"
CODE_INPUT_DIR="${SAMPLING_EVAL_CODE_DIR:-/osmo/run/workspace/code}"
CHECKPOINT_ROOT="${SAMPLING_EVAL_CHECKPOINT_ROOT:-/osmo/run/workspace/checkpoints}"
RESULT_ROOT="${SAMPLING_EVAL_RESULT_ROOT:-/osmo/run/workspace/results}"
PROJECT_ROOT=/root/code/IsaacLab-NeRD

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
        echo "[FATAL] Extracted archive does not contain IsaacLab-NeRD." >&2
        exit 1
    fi
}

wait_for_code() {
    local waited=0

    while [[ ! -f "$CODE_INPUT_DIR/READY" ]]; do
        if (( waited >= 21600 )); then
            echo "[FATAL] Timed out waiting six hours for code rsync." >&2
            exit 1
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

wait_for_checkpoints() {
    local waited=0

    while [[ ! -f "$CHECKPOINT_ROOT/READY" ]]; do
        if (( waited >= 21600 )); then
            echo "[FATAL] Timed out waiting six hours for checkpoint rsync." >&2
            exit 1
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

dataset_filename() {
    case "$1" in
        exp_trajectory) printf '%s\n' dataset_valid.hdf5 ;;
        zero_action_trajectory) printf '%s\n' dataset_zero_action_valid.hdf5 ;;
        lstm_actuator_zero_action_trajectory) printf '%s\n' dataset_lstm_actuator_zero_action_valid.hdf5 ;;
        lstm_actuator_policy_trajectory) printf '%s\n' dataset_lstm_actuator_policy_valid.hdf5 ;;
        *) echo "[FATAL] Unsupported regime: $1" >&2; exit 1 ;;
    esac
}

evaluate_suite() {
    local sampling="$1"
    local regime="$2"
    local dataset_root suite_id dataset output_dir rollout_count

    if [[ "$sampling" == old ]]; then
        dataset_root="$OLD_DATASET_ROOT"
    else
        dataset_root="$NEW_DATASET_ROOT"
    fi
    suite_id="${sampling}_${regime}"
    dataset="$dataset_root/$(dataset_filename "$regime")"
    output_dir="$RESULT_ROOT/$suite_id"
    rollout_count=0
    if [[ "$regime" == lstm_actuator_policy_trajectory ]]; then
        rollout_count=1024
    fi
    if [[ ! -f "$dataset" ]]; then
        echo "[FATAL] Validation suite does not exist: $dataset" >&2
        exit 1
    fi

    python3 -m osmo_scripts.sampling_strategy_eval.run_eval \
        --encoder "$ENCODER" \
        --checkpoint-manifest "$CHECKPOINT_ROOT/checkpoint_manifest_${ENCODER}.json" \
        --dataset-manifest "$PROJECT_ROOT/osmo_scripts/sampling_strategy_eval/dataset_manifest_${ENCODER}.json" \
        --suite-id "$suite_id" \
        --dataset "$dataset" \
        --output-dir "$output_dir" \
        --batch-size 512 \
        --num-workers 8 \
        --num-envs 64 \
        --max-windows 51200 \
        --window-seed 20260826 \
        --rollout-count "$rollout_count" \
        --rollout-horizon 10 \
        --rollout-seed 20260826 \
        --seed 0 \
        --device cuda:0 \
        --headless
}

analyze_results() {
    local args=()
    local sampling regime

    for sampling in old new; do
        for regime in \
            exp_trajectory \
            zero_action_trajectory \
            lstm_actuator_zero_action_trajectory \
            lstm_actuator_policy_trajectory; do
            args+=(--result "$RESULT_ROOT/${sampling}_${regime}")
        done
    done
    python3 -m osmo_scripts.sampling_strategy_eval.analyze \
        "${args[@]}" \
        --output-json "$RESULT_ROOT/analysis.json" \
        --output-markdown "$RESULT_ROOT/analysis.md" \
        --bootstrap-samples 10000 \
        --bootstrap-seed 20260826
}

wait_for_result_download() {
    touch "$RESULT_ROOT/DONE"
    while [[ ! -f "$RESULT_ROOT/RESULTS_DOWNLOADED" ]]; do
        sleep 15
    done
}

install_python_shims
wait_for_code
extract_code
python3 -m pip install --no-cache-dir -e "$PROJECT_ROOT/source/isaaclab_neural" --no-deps
wait_for_checkpoints
mkdir -p "$RESULT_ROOT"
cd "$PROJECT_ROOT"
for sampling in old new; do
    for regime in \
        exp_trajectory \
        zero_action_trajectory \
        lstm_actuator_zero_action_trajectory \
        lstm_actuator_policy_trajectory; do
        evaluate_suite "$sampling" "$regime"
    done
done
analyze_results
wait_for_result_download
