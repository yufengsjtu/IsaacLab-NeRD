#!/bin/bash
# OSMO task entrypoint. Downloads code/data, runs the selected preset, and uploads output.

set -euo pipefail
set -x

WORKFLOW_ID="${WORKFLOW_ID:?WORKFLOW_ID is required}"
OUTPUT_LOCAL_PATH="${OUTPUT_LOCAL_PATH:-/tmp/runs/output}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/code/IsaacLab-NeRD}"
STORAGE_BACKEND="${STORAGE_BACKEND:-nvdataset}"
SWIFT_CODE_DIR="${SWIFT_CODE_DIR:-/tmp/swift/code}"
SWIFT_DATA_DIR="${SWIFT_DATA_DIR:-/tmp/swift/data}"
SWIFT_OUTPUT_PREFIX="${SWIFT_OUTPUT_PREFIX:-$WORKFLOW_ID}"
NVDATASET_CODE_DIR="${NVDATASET_CODE_DIR:-/tmp/nvdatasets/code}"
NVDATASET_DATA_DIR="${NVDATASET_DATA_DIR:-/tmp/nvdatasets/data}"
NVDATASET_OUTPUT_DIR="${NVDATASET_OUTPUT_DIR:-$WORKFLOW_ID}"
DATASET_CACHE_MODE="${DATASET_CACHE_MODE:-auto}"
NVDATASET_INDEX_URL="${NVDATASET_INDEX_URL:-https://artifactory.pdx.nvidia.com/artifactory/api/pypi/sw-ngc-data-platform-pypi-local/simple}"

export OUTPUT_LOCAL_PATH
export NGC_API_KEY="${NGC_API_KEY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

write_shell_exports() {
    {
        printf 'export WORKFLOW_ID=%q\n' "$WORKFLOW_ID"
        printf 'export OUTPUT_LOCAL_PATH=%q\n' "$OUTPUT_LOCAL_PATH"
        printf 'export STORAGE_BACKEND=%q\n' "$STORAGE_BACKEND"
        printf 'export SWIFT_CODE_DIR=%q\n' "$SWIFT_CODE_DIR"
        printf 'export SWIFT_OUTPUT_CONTAINER=%q\n' "${SWIFT_OUTPUT_CONTAINER:-}"
        printf 'export SWIFT_OUTPUT_PREFIX=%q\n' "${SWIFT_OUTPUT_PREFIX:-}"
        printf 'export NVDATASET_CODE_DIR=%q\n' "$NVDATASET_CODE_DIR"
        printf 'export NVDATASET_OUTPUT_DATASET=%q\n' "${NVDATASET_OUTPUT_DATASET:-}"
        printf 'export NVDATASET_OUTPUT_DESCRIPTION=%q\n' "${NVDATASET_OUTPUT_DESCRIPTION:-}"
        printf 'export NVDATASET_OUTPUT_DIR=%q\n' "${NVDATASET_OUTPUT_DIR:-}"
    } >> ~/.bashrc
}

install_python_shims() {
    local isaacsim_python="/isaac-sim/python.sh"

    # Isaac Sim images often expose Python only through python.sh. Use wrapper
    # scripts instead of symlinks because python.sh resolves sidecar paths from
    # its own executable directory.
    if [[ -x "$isaacsim_python" ]]; then
        mkdir -p /tmp/shims
        printf '#!/bin/bash\nexec "%s" "$@"\n' "$isaacsim_python" > /tmp/shims/python3
        cp /tmp/shims/python3 /tmp/shims/python
        chmod +x /tmp/shims/python3 /tmp/shims/python
        export PATH="/tmp/shims:$PATH"
    fi
}

install_storage_deps() {
    case "$STORAGE_BACKEND" in
        swift)
            python3 -m pip install --quiet -U python-swiftclient
            python3 /tmp/swift_io.py check
            ;;
        nvdataset)
            python3 -m pip install --quiet --force-reinstall cffi
            python3 -m pip install --quiet -U --extra-index-url "$NVDATASET_INDEX_URL" nvdataset
            python3 /tmp/nvdataset_io.py check
            ;;
        *)
            echo "[FATAL] Unsupported storage backend: $STORAGE_BACKEND"
            exit 2
            ;;
    esac

    export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
    hash -r
}

upload_storage_output() {
    local status=$?
    local output_dir_name output_upload_root upload_log upload_status

    trap - EXIT
    set +e
    if [[ "$STORAGE_BACKEND" == "swift" && -n "${SWIFT_OUTPUT_CONTAINER:-}" ]]; then
        output_dir_name="${SWIFT_OUTPUT_PREFIX:-$WORKFLOW_ID}"
    elif [[ "$STORAGE_BACKEND" == "nvdataset" && -n "${NVDATASET_OUTPUT_DATASET:-}" ]]; then
        output_dir_name="${NVDATASET_OUTPUT_DIR:-$WORKFLOW_ID}"
    else
        exit "$status"
    fi
    if [[ -n "$output_dir_name" ]]; then
        output_dir_name="${output_dir_name//\//_}"
        output_upload_root="/tmp/storage/output_upload"
        rm -rf "$output_upload_root"
        mkdir -p "$output_upload_root/$output_dir_name"
        sync "$OUTPUT_LOCAL_PATH" || true
        cp -a "$OUTPUT_LOCAL_PATH"/. "$output_upload_root/$output_dir_name"/
        upload_log="$output_upload_root/$output_dir_name/nvdataset_upload.log"
        {
            echo "=== Uploading $output_upload_root through $STORAGE_BACKEND storage ==="
            date -u +"Upload started at %Y-%m-%dT%H:%M:%SZ"
        } | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"

        if [[ "$STORAGE_BACKEND" == "swift" ]]; then
            python3 /tmp/swift_io.py upload-directory \
                --container "$SWIFT_OUTPUT_CONTAINER" \
                --source-dir "$output_upload_root/$output_dir_name" \
                --prefix "$SWIFT_OUTPUT_PREFIX" \
                --resume 2>&1 | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"
        else
            python3 /tmp/nvdataset_io.py upload-directory \
                --dataset "$NVDATASET_OUTPUT_DATASET" \
                --source-dir "$output_upload_root" \
                --description "${NVDATASET_OUTPUT_DESCRIPTION:-}" 2>&1 | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"
        fi
        upload_status=${PIPESTATUS[0]}
        {
            echo "Upload finished with status $upload_status"
            date -u +"Upload finished at %Y-%m-%dT%H:%M:%SZ"
        } | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"
        if [[ "$status" -eq 0 ]]; then
            status=$upload_status
        fi
    fi
    exit "$status"
}

download_code() {
    if [[ "$STORAGE_BACKEND" == "swift" ]]; then
        if [[ -z "${SWIFT_CODE_CONTAINER:-}" || -z "${SWIFT_CODE_OBJECT:-}" ]]; then
            echo "[FATAL] Set SWIFT_CODE_CONTAINER and SWIFT_CODE_OBJECT for the Swift backend."
            exit 1
        fi
        mkdir -p "$SWIFT_CODE_DIR"
        python3 /tmp/swift_io.py download-object \
            --container "$SWIFT_CODE_CONTAINER" \
            --object-name "$SWIFT_CODE_OBJECT" \
            --output "$SWIFT_CODE_DIR/IsaacLab-NeRD.tar.gz"
        return
    fi
    if [[ -z "${NVDATASET_CODE_DATASET:-}" ]]; then
        echo "[FATAL] Set nvdataset_code_dataset to the NV-Datasets dataset containing IsaacLab-NeRD."
        exit 1
    fi

    python3 /tmp/nvdataset_io.py download \
        --dataset "$NVDATASET_CODE_DATASET" \
        --snapshot "${NVDATASET_CODE_SNAPSHOT:-}" \
        --output-dir "$NVDATASET_CODE_DIR"
}

download_data() {
    if [[ "$DATASET_CACHE_MODE" == "off" ]]; then
        echo "Dataset cache mode is off; skipping storage input download."
        DATASET_INPUT_PATH=""
        return
    fi
    if [[ "$STORAGE_BACKEND" == "swift" ]]; then
        if [[ -z "${SWIFT_DATA_CONTAINER:-}" ]]; then
            DATASET_INPUT_PATH=""
            return
        fi
        DATASET_INPUT_PATH="$SWIFT_DATA_DIR"
        if ! python3 /tmp/swift_io.py download-prefix \
            --container "$SWIFT_DATA_CONTAINER" \
            --prefix "${DATASET_SUBDIR:-}" \
            --output-dir "$DATASET_INPUT_PATH"; then
            echo "Swift data cache $SWIFT_DATA_CONTAINER/${DATASET_SUBDIR:-} unavailable."
            rm -rf "$DATASET_INPUT_PATH"
            mkdir -p "$DATASET_INPUT_PATH"
        fi
        return
    fi
    if [[ -z "${NVDATASET_DATA_DATASET:-}" ]]; then
        DATASET_INPUT_PATH=""
        return
    fi

    DATASET_INPUT_PATH="$NVDATASET_DATA_DIR"
    if ! python3 /tmp/nvdataset_io.py download \
        --dataset "$NVDATASET_DATA_DATASET" \
        --snapshot "${NVDATASET_DATA_SNAPSHOT:-}" \
        --prefix "${DATASET_SUBDIR:-}" \
        --output-dir "$DATASET_INPUT_PATH"; then
        echo "NV-Datasets data dataset $NVDATASET_DATA_DATASET unavailable; datasets will be generated locally and uploaded."
        rm -rf "$DATASET_INPUT_PATH"
        mkdir -p "$DATASET_INPUT_PATH"
    fi
}

extract_code() {
    local code_archive=""
    local code_input_dir
    local repo_input=""
    local candidate

    mkdir -p "$HOME/code"
    shopt -s nullglob globstar
    code_input_dir="$NVDATASET_CODE_DIR"
    if [[ "$STORAGE_BACKEND" == "swift" ]]; then
        code_input_dir="$SWIFT_CODE_DIR"
    fi
    for candidate in "$code_input_dir"/**/*.tar.gz "$code_input_dir"/**/*.tgz; do
        if [[ -f "$candidate" ]]; then
            code_archive="$candidate"
            break
        fi
    done

    if [[ -n "$code_archive" ]]; then
        tar -xzf "$code_archive" -C "$HOME/code"
    else
        for candidate in "$code_input_dir"/IsaacLab-NeRD "$code_input_dir"/*/IsaacLab-NeRD "$code_input_dir" "$code_input_dir"/*; do
            if [[ -f "$candidate"/isaaclab.sh && -d "$candidate"/source ]]; then
                repo_input="$candidate"
                break
            fi
        done
    fi

    if [[ -d "$PROJECT_ROOT" ]]; then
        :
    elif [[ -n "$repo_input" ]]; then
        mkdir -p "$PROJECT_ROOT"
        cp -a "$repo_input"/. "$PROJECT_ROOT"/
    else
        echo "[FATAL] Code storage must contain an IsaacLab-NeRD tarball, tree, or repository contents."
        exit 1
    fi
}

run_experiment() {
    local args=(
        --preset "${OSMO_EXPERIMENT_PRESET:?OSMO_EXPERIMENT_PRESET is required}"
        --workflow-base-name "${WORKFLOW_BASE_NAME:?WORKFLOW_BASE_NAME is required}"
        --dataset-subdir "${DATASET_SUBDIR:?DATASET_SUBDIR is required}"
        --dataset-cache-mode "$DATASET_CACHE_MODE"
        --storage-backend "$STORAGE_BACKEND"
    )

    if [[ -n "${DATASET_INPUT_PATH:-}" ]]; then
        args+=(--dataset-input-path "$DATASET_INPUT_PATH")
    fi
    if [[ "$STORAGE_BACKEND" == "swift" && -n "${SWIFT_DATA_CONTAINER:-}" ]]; then
        args+=(--swift-data-container "$SWIFT_DATA_CONTAINER")
    elif [[ -n "${NVDATASET_DATA_DATASET:-}" ]]; then
        args+=(
          --nvdataset-data-dataset "$NVDATASET_DATA_DATASET"
          --nvdataset-data-description "${NVDATASET_DATA_DESCRIPTION:-}"
        )
    fi
    if [[ "${ENABLE_WANDB:-false}" == "true" || "${ENABLE_WANDB:-0}" == "1" ]]; then
        args+=(--enable-wandb)
        if [[ -n "${WANDB_PROJECT_NAME:-}" ]]; then
            args+=(--wandb-project-name "$WANDB_PROJECT_NAME")
        fi
        if [[ -n "${WANDB_EXP_NAME:-}" ]]; then
            args+=(--wandb-exp-name "$WANDB_EXP_NAME")
        fi
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            args+=(--wandb-entity "$WANDB_ENTITY")
        fi
    fi

    cd "$PROJECT_ROOT"
    bash /tmp/init.sh
    python3 osmo_scripts/run_experiment.py "${args[@]}"
}

mkdir -p "$OUTPUT_LOCAL_PATH"
exec > >(tee -a "$OUTPUT_LOCAL_PATH/entry.log") 2>&1
write_shell_exports
install_python_shims
install_storage_deps
trap upload_storage_output EXIT

mkdir -p /tmp/nvdatasets /tmp/swift
download_code
download_data
extract_code
run_experiment
