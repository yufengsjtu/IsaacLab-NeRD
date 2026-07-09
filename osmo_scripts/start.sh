#!/usr/bin/env bash
# Package code, upload it to NV-Datasets, set OSMO credentials, and submit a workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_FILE="${WORKFLOW_FILE:-$SCRIPT_DIR/osmo_workflow.yaml}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
CODE_ONLY="${CODE_ONLY:-0}"
REFRESH_CREDENTIAL="${REFRESH_CREDENTIAL:-0}"

NVDATASET_CREDENTIAL="${NVDATASET_CREDENTIAL:-nvdataset_cred}"
NVDATASET_CODE_DATASET="${NVDATASET_CODE_DATASET:-IsaacLab-NeRD-Code}"
NVDATASET_DATA_DATASET="${NVDATASET_DATA_DATASET:-IsaacLab-NeRD-Datasets}"
NVDATASET_OUTPUT_DATASET="${NVDATASET_OUTPUT_DATASET:-IsaacLab-NeRD-Output}"
NVDATASET_OUTPUT_DESCRIPTION="${NVDATASET_OUTPUT_DESCRIPTION:-IsaacLab-NeRD OSMO training outputs for $RUN_ID.}"
DATASET_CACHE_MODE="${DATASET_CACHE_MODE:-auto}"
OSMO_EXPERIMENT_PRESET="${OSMO_EXPERIMENT_PRESET:-anymal_newton_native}"
PRESET_FILE="${PRESET_FILE:-}"

WORKFLOW_NAME_OVERRIDE=""
DATASET_SUBDIR_OVERRIDE=""
OSMO_NUM_GPU_OVERRIDE=""
OSMO_NUM_CPU_OVERRIDE=""
OSMO_MEMORY_OVERRIDE=""
OSMO_STORAGE_OVERRIDE=""
OSMO_PLATFORM_OVERRIDE=""
OSMO_POOL_OVERRIDE=""

NGC_API_KEY="${NGC_API_KEY:-}"
NVDATASET_TENANTID="${NVDATASET_TENANTID:-}"

usage() {
    cat <<'EOF'
Usage: ./osmo_scripts/start.sh [options] [-- osmo-submit-options]

Common options:
  --preset NAME              Preset under osmo_scripts/presets/ (default: anymal_newton_native)
  --preset-file PATH         Use a custom preset YAML file
  --workflow-name NAME       Override preset workflow base name; run id is appended
  --dataset-subdir NAME      Override generated dataset cache subdirectory
  --code-only                Upload/replace code dataset without submitting OSMO
  --code-dataset NAME        NV-Datasets code dataset (default: IsaacLab-NeRD-Code)
  --data-dataset NAME        NV-Datasets generated data dataset
  --output-dataset NAME      NV-Datasets output dataset
  --num-gpu N                Override preset OSMO GPU resource
  --num-cpu N                Override preset OSMO CPU resource
  --memory SIZE              Override preset OSMO memory resource
  --storage SIZE             Override preset OSMO storage resource
  --platform NAME            Override preset OSMO platform
  --pool NAME                OSMO pool passed to validate and submit
  --refresh-credential       Reset the OSMO generic credential before submit
  --workflow-file PATH       Override osmo_workflow.yaml path
  -- <osmo-options>          Forward extra OSMO options such as --pool to validate and submit

Credentials:
  NGC_API_KEY
  NVDATASET_TENANTID
EOF
}

OSMO_ARGS=()
while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --preset|--experiment-preset)
            OSMO_EXPERIMENT_PRESET="${2:?Missing value for --preset}"
            shift 2
            ;;
        --preset-file)
            PRESET_FILE="${2:?Missing value for --preset-file}"
            shift 2
            ;;
        --code-only|--upload-code-only)
            CODE_ONLY=1
            shift
            ;;
        --code-dataset)
            NVDATASET_CODE_DATASET="${2:?Missing value for --code-dataset}"
            shift 2
            ;;
        --data-dataset)
            NVDATASET_DATA_DATASET="${2:?Missing value for --data-dataset}"
            shift 2
            ;;
        --output-dataset)
            NVDATASET_OUTPUT_DATASET="${2:?Missing value for --output-dataset}"
            shift 2
            ;;
        --output-description)
            NVDATASET_OUTPUT_DESCRIPTION="${2:?Missing value for --output-description}"
            shift 2
            ;;
        --workflow-name)
            WORKFLOW_NAME_OVERRIDE="${2:?Missing value for --workflow-name}"
            shift 2
            ;;
        --dataset-subdir)
            DATASET_SUBDIR_OVERRIDE="${2:?Missing value for --dataset-subdir}"
            shift 2
            ;;
        --dataset-cache-mode)
            DATASET_CACHE_MODE="${2:?Missing value for --dataset-cache-mode}"
            shift 2
            ;;
        --num-gpu|--num-gpus)
            OSMO_NUM_GPU_OVERRIDE="${2:?Missing value for --num-gpu}"
            shift 2
            ;;
        --num-cpu|--num-cpus)
            OSMO_NUM_CPU_OVERRIDE="${2:?Missing value for --num-cpu}"
            shift 2
            ;;
        --memory)
            OSMO_MEMORY_OVERRIDE="${2:?Missing value for --memory}"
            shift 2
            ;;
        --storage)
            OSMO_STORAGE_OVERRIDE="${2:?Missing value for --storage}"
            shift 2
            ;;
        --platform)
            OSMO_PLATFORM_OVERRIDE="${2:?Missing value for --platform}"
            shift 2
            ;;
        --pool)
            OSMO_POOL_OVERRIDE="${2:?Missing value for --pool}"
            shift 2
            ;;
        --credential)
            NVDATASET_CREDENTIAL="${2:?Missing value for --credential}"
            shift 2
            ;;
        --refresh-credential)
            REFRESH_CREDENTIAL=1
            shift
            ;;
        --workflow-file)
            WORKFLOW_FILE="${2:?Missing value for --workflow-file}"
            shift 2
            ;;
        --main-scripts-localpath)
            echo "[WARN] --main-scripts-localpath is deprecated and ignored; use --preset/--preset-file instead." >&2
            shift 2
            ;;
        --)
            shift
            OSMO_ARGS+=("$@")
            break
            ;;
        *)
            OSMO_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$NGC_API_KEY" ]]; then
    echo "[FATAL] Set NGC_API_KEY before running this script." >&2
    exit 2
fi
if [[ -z "$NVDATASET_TENANTID" ]]; then
    echo "[FATAL] Set NVDATASET_TENANTID before running this script." >&2
    exit 2
fi

resolve_args=(
    --preset "$OSMO_EXPERIMENT_PRESET"
    --run-id "$RUN_ID"
)
if [[ -n "$PRESET_FILE" ]]; then
    resolve_args+=(--preset-file "$PRESET_FILE")
fi
if [[ -n "$WORKFLOW_NAME_OVERRIDE" ]]; then
    resolve_args+=(--workflow-name "$WORKFLOW_NAME_OVERRIDE")
fi
if [[ -n "$DATASET_SUBDIR_OVERRIDE" ]]; then
    resolve_args+=(--dataset-subdir "$DATASET_SUBDIR_OVERRIDE")
fi
if [[ -n "$OSMO_NUM_GPU_OVERRIDE" ]]; then
    resolve_args+=(--num-gpu "$OSMO_NUM_GPU_OVERRIDE")
fi
if [[ -n "$OSMO_NUM_CPU_OVERRIDE" ]]; then
    resolve_args+=(--num-cpu "$OSMO_NUM_CPU_OVERRIDE")
fi
if [[ -n "$OSMO_MEMORY_OVERRIDE" ]]; then
    resolve_args+=(--memory "$OSMO_MEMORY_OVERRIDE")
fi
if [[ -n "$OSMO_STORAGE_OVERRIDE" ]]; then
    resolve_args+=(--storage "$OSMO_STORAGE_OVERRIDE")
fi
if [[ -n "$OSMO_PLATFORM_OVERRIDE" ]]; then
    resolve_args+=(--platform "$OSMO_PLATFORM_OVERRIDE")
fi
eval "$(python3 "$SCRIPT_DIR/lib/preset.py" resolve-submit "${resolve_args[@]}")"

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

ARCHIVE_PATH="$TMP_DIR/IsaacLab-NeRD.tar.gz"

echo "=== Packaging code snapshot ==="
python3 "$SCRIPT_DIR/lib/package_code.py" \
    --project-root "$PROJECT_ROOT" \
    --archive-path "$ARCHIVE_PATH"

echo "=== Ensuring local nvdataset package ==="
export NVDATASET_TENANTID
export NGC_API_KEY
if ! python3 "$SCRIPT_DIR/lib/nvdataset_io.py" check >/dev/null 2>&1; then
    python3 -m pip install --quiet \
        --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-ngc-data-platform-pypi/simple \
        cffi "PyJWT[crypto]" python-dateutil nvdataset
    python3 "$SCRIPT_DIR/lib/nvdataset_io.py" check
fi

echo "=== Uploading code archive to NV-Datasets: $NVDATASET_CODE_DATASET ==="
python3 "$SCRIPT_DIR/lib/nvdataset_io.py" replace-files \
    --dataset "$NVDATASET_CODE_DATASET" \
    --description "IsaacLab-NeRD code snapshot archive for OSMO." \
    "$ARCHIVE_PATH"

if [[ "$CODE_ONLY" == "1" ]]; then
    echo "=== Code-only mode complete. Skipping OSMO credential setup and workflow submit. ==="
    exit 0
fi

credential_exists() {
    local credential_list

    credential_list="$(osmo credential list 2>/dev/null || true)"
    [[ "$credential_list" == *"$NVDATASET_CREDENTIAL"* ]]
}

if [[ "$REFRESH_CREDENTIAL" != "1" ]] && credential_exists; then
    echo "=== Reusing existing OSMO NV-Datasets credential: $NVDATASET_CREDENTIAL ==="
else
    echo "=== Setting OSMO NV-Datasets credential: $NVDATASET_CREDENTIAL ==="
    if ! credential_output="$(osmo credential set "$NVDATASET_CREDENTIAL" --type GENERIC \
        --payload nvapi_key="$NGC_API_KEY" tenant_id="$NVDATASET_TENANTID" 2>&1)"; then
        if [[ "$credential_output" == *"duplicate key value"* || "$credential_output" == *"already exists"* ]]; then
            echo "Credential $NVDATASET_CREDENTIAL already exists; reusing it."
        elif credential_exists; then
            echo "$credential_output" >&2
            echo "Credential $NVDATASET_CREDENTIAL is present after set failed; reusing it."
        else
            echo "$credential_output" >&2
            exit 1
        fi
    else
        echo "$credential_output"
    fi
fi

SUBMIT_ARGS=(
    --set
    "num_gpu=$OSMO_NUM_GPU"
    "num_cpu=$OSMO_NUM_CPU"
    --set-string
    "workflow_name=$WORKFLOW_NAME"
    "workflow_base_name=$WORKFLOW_BASE_NAME"
    "dataset_subdir=$DATASET_SUBDIR"
    "dataset_cache_mode=$DATASET_CACHE_MODE"
    "osmo_experiment_preset=$OSMO_EXPERIMENT_PRESET"
    "memory=$OSMO_MEMORY"
    "storage=$OSMO_STORAGE"
    "resource_platform=$OSMO_PLATFORM"
    "nvdataset_credential=$NVDATASET_CREDENTIAL"
    "nvdataset_code_dataset=$NVDATASET_CODE_DATASET"
    "nvdataset_code_snapshot="
    "nvdataset_data_dataset=$NVDATASET_DATA_DATASET"
    "nvdataset_output_dataset=$NVDATASET_OUTPUT_DATASET"
    "nvdataset_output_description=$NVDATASET_OUTPUT_DESCRIPTION"
)
OSMO_RESOURCE_ARGS=()
if [[ -n "$OSMO_POOL_OVERRIDE" ]]; then
    OSMO_RESOURCE_ARGS+=(--pool "$OSMO_POOL_OVERRIDE")
fi

echo "=== Resolved OSMO resources: gpu=$OSMO_NUM_GPU cpu=$OSMO_NUM_CPU memory=$OSMO_MEMORY storage=$OSMO_STORAGE platform=$OSMO_PLATFORM pool=${OSMO_POOL_OVERRIDE:-<default>} ==="

echo "=== Validating OSMO workflow ==="
osmo workflow validate "${SUBMIT_ARGS[@]}" "${OSMO_RESOURCE_ARGS[@]}" "${OSMO_ARGS[@]}" -- "$WORKFLOW_FILE"

echo "=== Submitting OSMO workflow ==="
osmo workflow submit "${SUBMIT_ARGS[@]}" "${OSMO_RESOURCE_ARGS[@]}" "${OSMO_ARGS[@]}" -- "$WORKFLOW_FILE"
