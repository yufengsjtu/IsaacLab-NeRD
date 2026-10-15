#!/bin/bash
# OSMO task entrypoint. Downloads code/data, overlays the uploaded Isaac Lab
# tree on the container image, runs the selected preset, and uploads output.

set -euo pipefail

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

set -x

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
    local venv_root="${VIRTUAL_ENV:-}"
    local sim_root lab_root env_site

    mkdir -p /tmp/shims

    # Isaac Lab 3 docker images install lockfile deps into /opt/isaaclab-venv.
    # Kit's python.sh does not see those packages (for example lazy_loader), so
    # prefer the venv whenever it exists.
    if [[ -z "$venv_root" && -x /opt/isaaclab-venv/bin/python ]]; then
        venv_root="/opt/isaaclab-venv"
    fi
    if [[ -n "$venv_root" && -x "$venv_root/bin/python" ]]; then
        export VIRTUAL_ENV="$venv_root"
        export PATH="$venv_root/bin:$PATH"
        hash -r
        lab_root="${ISAACLAB_PATH:-/workspace/isaaclab}"
        sim_root="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
        if [[ -d "$lab_root/_isaac_sim" ]]; then
            sim_root="$lab_root/_isaac_sim"
        fi
        set +u
        if [[ -f "$sim_root/setup_conda_env.sh" ]]; then
            # shellcheck disable=SC1091
            . "$sim_root/setup_conda_env.sh" >/dev/null 2>&1 || true
        elif [[ -f "$sim_root/setup_python_env.sh" ]]; then
            export ISAAC_PATH="$sim_root"
            export CARB_APP_PATH="$ISAAC_PATH/kit"
            export EXP_PATH="$ISAAC_PATH/apps"
            # shellcheck disable=SC1091
            . "$sim_root/setup_python_env.sh" >/dev/null 2>&1 || true
            env_site="$(
                "$venv_root/bin/python" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true
            )"
            if [[ -n "$env_site" ]]; then
                export PYTHONPATH="${env_site}${PYTHONPATH:+:$PYTHONPATH}"
            fi
        fi
        set -u
        echo "Using Isaac Lab image venv: $venv_root/bin/python"
        return
    fi

    # Older images often expose Python only through python.sh. Use wrapper
    # scripts instead of symlinks because python.sh resolves sidecar paths from
    # its own executable directory.
    if [[ -x "$isaacsim_python" ]]; then
        printf '#!/bin/bash\nexec "%s" "$@"\n' "$isaacsim_python" > /tmp/shims/python3
        cp /tmp/shims/python3 /tmp/shims/python
        chmod +x /tmp/shims/python3 /tmp/shims/python
        export PATH="/tmp/shims:$PATH"
        echo "Using Isaac Sim python.sh: $isaacsim_python"
    fi
}

install_storage_deps() {
    case "$STORAGE_BACKEND" in
        swift)
            python3 -m pip install --quiet -U python-swiftclient
            python3 /tmp/swift_io.py check
            ;;
        nvdataset)
            if [[ -z "${NVDATASET_PYTHON:-}" && -x /opt/nvdataset-venv/bin/python ]]; then
                export NVDATASET_PYTHON=/opt/nvdataset-venv/bin/python
            fi
            if ! python3 /tmp/nvdataset_io.py check; then
                echo "[FATAL] Use an image with a supported preinstalled NV-Datasets SDK; no runtime package-index access is attempted." >&2
                exit 1
            fi
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
    local local_pythonpath
    local_pythonpath="$(
        python3 osmo_scripts/lib/local_isaaclab.py export-pythonpath --project-root "$PROJECT_ROOT"
    )"
    export PYTHONPATH="${local_pythonpath}${PYTHONPATH:+:$PYTHONPATH}"
    python3 osmo_scripts/lib/local_isaaclab.py verify --project-root "$PROJECT_ROOT"
    local out="$OUTPUT_LOCAL_PATH/randomized_acceptance" ck=osmo_scripts/diagnostics/ckpt
    mkdir -p "$out"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
    python3 osmo_scripts/diagnostics/control_step_velocity.py all \
        --policy pretrained/control_policy/so101_keyboard/model_14999.pt \
        --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/SO101/transformer_keyboard_control_step.yaml \
        --train-file unused --valid-file unused --skip-onestep \
        --checkpoints $ck/rc_e299.pt@episode_start $ck/rn_e299.pt@episode_start $ck/rc_e199.pt@episode_start $ck/rn_e199.pt@episode_start \
        --keyboard-scene randomized \
        --seeds 42 43 44 45 46 47 --per-gpu 3 \
        --out "$out"
}

mkdir -p "$OUTPUT_LOCAL_PATH"
exec > >(tee -a "$OUTPUT_LOCAL_PATH/entry.log") 2>&1
write_shell_exports
install_python_shims
install_storage_deps
trap upload_storage_output EXIT

mkdir -p /tmp/nvdatasets /tmp/swift
download_code
data_download_started=$SECONDS
download_data
echo "[Data download] seconds=$((SECONDS - data_download_started)) cache_mode=$DATASET_CACHE_MODE"
extract_code
run_experiment
