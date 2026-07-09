#!/bin/bash
# OSMO task entrypoint. Downloads code/data, runs the selected preset, and uploads output.

set -euo pipefail
set -x

WORKFLOW_ID="${WORKFLOW_ID:?WORKFLOW_ID is required}"
OUTPUT_LOCAL_PATH="${OUTPUT_LOCAL_PATH:-/tmp/runs/output}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/code/IsaacLab-NeRD}"
NVDATASET_CODE_DIR="${NVDATASET_CODE_DIR:-/tmp/nvdatasets/code}"
NVDATASET_DATA_DIR="${NVDATASET_DATA_DIR:-/tmp/nvdatasets/data}"
NVDATASET_OUTPUT_DIR="${NVDATASET_OUTPUT_DIR:-$WORKFLOW_ID}"
DATASET_CACHE_MODE="${DATASET_CACHE_MODE:-auto}"

export OUTPUT_LOCAL_PATH
export NGC_API_KEY="${NGC_API_KEY:-}"

write_shell_exports() {
    {
        printf 'export WORKFLOW_ID=%q\n' "$WORKFLOW_ID"
        printf 'export OUTPUT_LOCAL_PATH=%q\n' "$OUTPUT_LOCAL_PATH"
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

install_nvdataset_deps() {
    python3 -m pip install --quiet --force-reinstall cffi
    python3 -m pip install --quiet \
        --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-ngc-data-platform-pypi/simple \
        "PyJWT[crypto]" python-dateutil nvdataset
    python3 /tmp/nvdataset_io.py check

    export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
    hash -r
}

upload_nvdataset_output() {
    local status=$?
    local output_dir_name output_upload_root upload_log upload_status

    trap - EXIT
    set +e
    if [[ -n "${NVDATASET_OUTPUT_DATASET:-}" ]]; then
        output_dir_name="${NVDATASET_OUTPUT_DIR:-$WORKFLOW_ID}"
        output_dir_name="${output_dir_name//\//_}"
        output_upload_root="/tmp/nvdatasets/output_upload"
        rm -rf "$output_upload_root"
        mkdir -p "$output_upload_root/$output_dir_name"
        sync "$OUTPUT_LOCAL_PATH" || true
        cp -a "$OUTPUT_LOCAL_PATH"/. "$output_upload_root/$output_dir_name"/
        upload_log="$output_upload_root/$output_dir_name/nvdataset_upload.log"
        {
            echo "=== Uploading $output_upload_root to NV-Datasets dataset $NVDATASET_OUTPUT_DATASET ==="
            date -u +"Upload started at %Y-%m-%dT%H:%M:%SZ"
        } | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"

        python3 /tmp/nvdataset_io.py upload-directory \
            --dataset "$NVDATASET_OUTPUT_DATASET" \
            --source-dir "$output_upload_root" \
            --description "${NVDATASET_OUTPUT_DESCRIPTION:-}" 2>&1 | tee -a "$OUTPUT_LOCAL_PATH/entry.log" "$upload_log"
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
    local repo_input=""
    local candidate

    mkdir -p "$HOME/code"
    shopt -s nullglob globstar
    for candidate in "$NVDATASET_CODE_DIR"/**/*.tar.gz "$NVDATASET_CODE_DIR"/**/*.tgz; do
        if [[ -f "$candidate" ]]; then
            code_archive="$candidate"
            break
        fi
    done

    if [[ -n "$code_archive" ]]; then
        tar -xzf "$code_archive" -C "$HOME/code"
    else
        for candidate in "$NVDATASET_CODE_DIR"/IsaacLab-NeRD "$NVDATASET_CODE_DIR"/*/IsaacLab-NeRD "$NVDATASET_CODE_DIR" "$NVDATASET_CODE_DIR"/*; do
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
        echo "[FATAL] nvdataset_code_dataset must contain an IsaacLab-NeRD tarball, tree, or repository contents."
        exit 1
    fi
}

run_experiment() {
    local args=(
        --preset "${OSMO_EXPERIMENT_PRESET:?OSMO_EXPERIMENT_PRESET is required}"
        --workflow-base-name "${WORKFLOW_BASE_NAME:?WORKFLOW_BASE_NAME is required}"
        --dataset-subdir "${DATASET_SUBDIR:?DATASET_SUBDIR is required}"
        --dataset-cache-mode "$DATASET_CACHE_MODE"
    )

    if [[ -n "${DATASET_INPUT_PATH:-}" ]]; then
        args+=(
            --dataset-input-path "$DATASET_INPUT_PATH"
            --nvdataset-data-dataset "${NVDATASET_DATA_DATASET:-}"
            --nvdataset-data-description "${NVDATASET_DATA_DESCRIPTION:-}"
        )
    fi

    cd "$PROJECT_ROOT"
    bash /tmp/init.sh
    python3 osmo_scripts/run_experiment.py "${args[@]}"
}

mkdir -p "$OUTPUT_LOCAL_PATH"
exec > >(tee -a "$OUTPUT_LOCAL_PATH/entry.log") 2>&1
write_shell_exports
install_python_shims
install_nvdataset_deps
trap upload_nvdataset_output EXIT

mkdir -p /tmp/nvdatasets
download_code
download_data
extract_code
run_experiment
