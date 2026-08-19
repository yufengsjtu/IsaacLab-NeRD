#!/usr/bin/env bash
# Package code, upload it to object storage, set OSMO credentials, and submit a workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_FILE="${WORKFLOW_FILE:-$SCRIPT_DIR/osmo_workflow.yaml}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
CODE_ONLY="${CODE_ONLY:-0}"
REFRESH_CREDENTIAL="${REFRESH_CREDENTIAL:-0}"

STORAGE_BACKEND="${STORAGE_BACKEND:-nvdataset}"
STORAGE_CREDENTIAL="${STORAGE_CREDENTIAL:-}"
STORAGE_AUTH_URL_ENV=""
STORAGE_AUTH_URL_KEY=""
STORAGE_AUTH_VERSION_ENV=""
STORAGE_AUTH_VERSION_KEY=""
STORAGE_USER_ENV=""
STORAGE_USER_KEY=""
STORAGE_KEY_ENV=""
STORAGE_KEY_KEY=""
STORAGE_TENANT_ENV=""
STORAGE_TENANT_KEY=""
OSMO_DATA_CREDENTIAL="${OSMO_DATA_CREDENTIAL:-css_team-nvr-srl}"
OSMO_SWIFT_BASE="${OSMO_SWIFT_BASE:-swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan}"
OSMO_CODE_URL=""
OSMO_DATASET_INPUT_URL=""
OSMO_DATASET_UPLOAD_URL=""
OSMO_OUTPUT_URL=""
SWIFT_CREDENTIAL="${SWIFT_CREDENTIAL:-swift_cred}"
SWIFT_AUTH_URL="${SWIFT_AUTH_URL:-https://pdx.s8k.io}"
SWIFT_AUTH_VERSION="${SWIFT_AUTH_VERSION:-1}"
SWIFT_USER="${SWIFT_USER:-team-isaac-lab}"
SWIFT_AUTH_KEY="${SWIFT_AUTH_KEY:-}"
SWIFT_CODE_CONTAINER="${SWIFT_CODE_CONTAINER:-isaaclab-nerd-code}"
SWIFT_CODE_OBJECT="${SWIFT_CODE_OBJECT:-IsaacLab-NeRD.tar.gz}"
SWIFT_DATA_CONTAINER="${SWIFT_DATA_CONTAINER:-isaaclab-nerd-datasets}"
SWIFT_OUTPUT_CONTAINER="${SWIFT_OUTPUT_CONTAINER:-isaaclab-nerd-output}"
NVDATASET_CREDENTIAL="${NVDATASET_CREDENTIAL:-nvdataset_cred}"
NVDATASET_CODE_DATASET="${NVDATASET_CODE_DATASET:-IsaacLab-NeRD-Code}"
NVDATASET_DATA_DATASET="${NVDATASET_DATA_DATASET:-IsaacLab-NeRD-Datasets}"
NVDATASET_OUTPUT_DATASET="${NVDATASET_OUTPUT_DATASET:-IsaacLab-NeRD-Output}"
NVDATASET_OUTPUT_DESCRIPTION="${NVDATASET_OUTPUT_DESCRIPTION:-IsaacLab-NeRD OSMO training outputs for $RUN_ID.}"
NVDATASET_INDEX_URL="${NVDATASET_INDEX_URL:-https://artifactory.pdx.nvidia.com/artifactory/api/pypi/sw-ngc-data-platform-pypi-local/simple}"
DATASET_CACHE_MODE="${DATASET_CACHE_MODE:-auto}"
AMLFS_DATA_ROOT="${AMLFS_DATA_ROOT:-}"
TRAIN_SEED="${TRAIN_SEED:-0}"
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
OSMO_PRIORITY_OVERRIDE=""

NGC_API_KEY="${NGC_API_KEY:-}"
NVDATASET_TENANTID="${NVDATASET_TENANTID:-}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_CREDENTIAL="${WANDB_CREDENTIAL:-wandb}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-nerd-newton}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

usage() {
    cat <<'EOF'
Usage: ./osmo_scripts/start.sh [options] [-- osmo-submit-options]

Common options:
  --preset NAME              Preset under osmo_scripts/presets/ (default: anymal_newton_native)
  --preset-file PATH         Use a custom preset YAML file
  --workflow-name NAME       Override preset workflow base name; run id is appended
  --dataset-subdir NAME      Override generated dataset cache subdirectory
  --amlfs-data-root PATH      Pool-wide Lustre directory used for datasets
  --train-seed N             Training random seed (default: 0)
  --storage-backend NAME     nvdataset (default), swift, or osmo_data
  --osmo-swift-base URL      Standalone Swift prefix used by osmo_data
  --osmo-data-credential N   Existing OSMO DATA credential for that prefix
  --code-only                Upload/replace code object without submitting OSMO
  --code-container NAME      Swift code container (default: isaaclab-nerd-code)
  --data-container NAME      Swift generated data container
  --output-container NAME    Swift output container
  --code-dataset NAME        NV-Datasets code dataset (default: IsaacLab-NeRD-Code)
  --data-dataset NAME        NV-Datasets generated data dataset
  --output-dataset NAME      NV-Datasets output dataset
  --num-gpu N                Override preset OSMO GPU resource
  --num-cpu N                Override preset OSMO CPU resource
  --memory SIZE              Override preset OSMO memory resource
  --storage SIZE             Override preset OSMO storage resource
  --platform NAME            Override preset OSMO platform
  --pool NAME                OSMO pool passed to validate and submit
  --priority LEVEL           Submit priority: HIGH, NORMAL, or LOW
  --enable-wandb             Enable Weights & Biases logging in the OSMO pod
  --wandb-project NAME       W&B project (default: nerd-newton)
  --wandb-exp-name NAME      W&B run name (default: workflow/dataset subdir)
  --wandb-entity NAME        Optional W&B entity or team
  --wandb-credential NAME    Existing OSMO GENERIC credential (default: wandb)
  --refresh-credential       Reset the OSMO generic credential before submit
  --workflow-file PATH       Override osmo_workflow.yaml path
  -- <osmo-options>          Forward extra OSMO options such as --pool to validate and submit

Credentials:
  OSMO DATA credential       Required for --storage-backend osmo_data
  SWIFT_AUTH_KEY             Required for --storage-backend swift
  NGC_API_KEY                Required for the default NV-Datasets backend
  NVDATASET_TENANTID         Required for the default NV-Datasets backend
  OSMO GENERIC credential    Required when using --enable-wandb
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
        --storage-backend)
            STORAGE_BACKEND="${2:?Missing value for --storage-backend}"
            shift 2
            ;;
        --osmo-swift-base)
            OSMO_SWIFT_BASE="${2:?Missing value for --osmo-swift-base}"
            shift 2
            ;;
        --osmo-data-credential)
            OSMO_DATA_CREDENTIAL="${2:?Missing value for --osmo-data-credential}"
            shift 2
            ;;
        --code-container)
            SWIFT_CODE_CONTAINER="${2:?Missing value for --code-container}"
            shift 2
            ;;
        --data-container)
            SWIFT_DATA_CONTAINER="${2:?Missing value for --data-container}"
            shift 2
            ;;
        --output-container)
            SWIFT_OUTPUT_CONTAINER="${2:?Missing value for --output-container}"
            shift 2
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
        --amlfs-data-root)
            AMLFS_DATA_ROOT="${2:?Missing value for --amlfs-data-root}"
            shift 2
            ;;
        --train-seed)
            TRAIN_SEED="${2:?Missing value for --train-seed}"
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
        --priority)
            OSMO_PRIORITY_OVERRIDE="${2:?Missing value for --priority}"
            shift 2
            ;;
        --enable-wandb)
            ENABLE_WANDB=1
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT_NAME="${2:?Missing value for --wandb-project}"
            shift 2
            ;;
        --wandb-exp-name)
            WANDB_EXP_NAME="${2:?Missing value for --wandb-exp-name}"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="${2:?Missing value for --wandb-entity}"
            shift 2
            ;;
        --wandb-credential)
            WANDB_CREDENTIAL="${2:?Missing value for --wandb-credential}"
            shift 2
            ;;
        --credential)
            STORAGE_CREDENTIAL="${2:?Missing value for --credential}"
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

OSMO_CREDENTIAL_LIST="$(osmo credential list 2>/dev/null || true)"
case "$STORAGE_BACKEND" in
    osmo_data)
        if [[ -z "$OSMO_SWIFT_BASE" ]]; then
            echo "[FATAL] Set OSMO_SWIFT_BASE for the OSMO DATA backend." >&2
            exit 2
        fi
        if [[ "$OSMO_CREDENTIAL_LIST" != *"$OSMO_DATA_CREDENTIAL"* ]]; then
            echo "[FATAL] OSMO DATA credential $OSMO_DATA_CREDENTIAL is unavailable." >&2
            exit 2
        fi
        echo "=== Verifying OSMO DATA access: ${OSMO_SWIFT_BASE%/}/ ==="
        osmo data check "${OSMO_SWIFT_BASE%/}/"
        ;;
    swift)
        STORAGE_CREDENTIAL="${STORAGE_CREDENTIAL:-$SWIFT_CREDENTIAL}"
        STORAGE_AUTH_URL_ENV="SWIFT_AUTH_URL"
        STORAGE_AUTH_URL_KEY="auth_url"
        STORAGE_AUTH_VERSION_ENV="SWIFT_AUTH_VERSION"
        STORAGE_AUTH_VERSION_KEY="auth_version"
        STORAGE_USER_ENV="SWIFT_USER"
        STORAGE_USER_KEY="user"
        STORAGE_KEY_ENV="SWIFT_AUTH_KEY"
        STORAGE_KEY_KEY="auth_key"
        STORAGE_TENANT_ENV="STORAGE_UNUSED_TENANT"
        STORAGE_TENANT_KEY="user"
        if [[ -z "$SWIFT_AUTH_KEY" ]]; then
            echo "[FATAL] Set SWIFT_AUTH_KEY before using the Swift storage backend." >&2
            exit 2
        fi
        ;;
    nvdataset)
        STORAGE_CREDENTIAL="${STORAGE_CREDENTIAL:-$NVDATASET_CREDENTIAL}"
        STORAGE_AUTH_URL_ENV="STORAGE_UNUSED_AUTH_URL"
        STORAGE_AUTH_URL_KEY="tenant_id"
        STORAGE_AUTH_VERSION_ENV="STORAGE_UNUSED_AUTH_VERSION"
        STORAGE_AUTH_VERSION_KEY="tenant_id"
        STORAGE_USER_ENV="STORAGE_UNUSED_USER"
        STORAGE_USER_KEY="tenant_id"
        STORAGE_KEY_ENV="NGC_API_KEY"
        STORAGE_KEY_KEY="nvapi_key"
        STORAGE_TENANT_ENV="NVDATASET_TENANTID"
        STORAGE_TENANT_KEY="tenant_id"
        if [[ -z "$NGC_API_KEY" ]]; then
            echo "[FATAL] Set NGC_API_KEY before using the NV-Datasets storage backend." >&2
            exit 2
        fi
        if [[ -z "$NVDATASET_TENANTID" ]]; then
            echo "[FATAL] Set NVDATASET_TENANTID before using the NV-Datasets storage backend." >&2
            exit 2
        fi
        ;;
    *)
        echo "[FATAL] Unsupported storage backend: $STORAGE_BACKEND (expected nvdataset, swift, or osmo_data)." >&2
        exit 2
        ;;
esac
if [[ "$ENABLE_WANDB" == "1" && "$OSMO_CREDENTIAL_LIST" != *"$WANDB_CREDENTIAL"* ]]; then
    echo "[FATAL] OSMO W&B credential $WANDB_CREDENTIAL is unavailable." >&2
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

if [[ "$STORAGE_BACKEND" == "osmo_data" ]]; then
    git_revision="$(git -C "$PROJECT_ROOT" rev-parse --short=12 HEAD)"
    osmo_base="${OSMO_SWIFT_BASE%/}"
    OSMO_CODE_URL="$osmo_base/code/${RUN_ID}-${git_revision}/"
    OSMO_DATASET_UPLOAD_URL="$osmo_base/data/datasets/${DATASET_SUBDIR}/"
    OSMO_OUTPUT_URL="$osmo_base/data/trained_models/"
    if [[ "$DATASET_CACHE_MODE" == "off" || "$DATASET_CACHE_MODE" == "local_require" ]]; then
        OSMO_DATASET_INPUT_URL=""
    elif osmo data check "$OSMO_DATASET_UPLOAD_URL" >/dev/null 2>&1; then
        OSMO_DATASET_INPUT_URL="$OSMO_DATASET_UPLOAD_URL"
    elif [[ "$DATASET_CACHE_MODE" == "require" ]]; then
        echo "[FATAL] Required OSMO DATA cache is unavailable: $OSMO_DATASET_UPLOAD_URL" >&2
        exit 2
    else
        echo "=== OSMO DATA cache is unavailable; this workflow will generate it. ==="
    fi
    echo "=== Immutable code URL: $OSMO_CODE_URL ==="
    echo "=== Dataset upload URL: $OSMO_DATASET_UPLOAD_URL ==="
    if [[ -n "$OSMO_DATASET_INPUT_URL" ]]; then
        echo "=== Dataset input URL: $OSMO_DATASET_INPUT_URL ==="
    fi
    echo "=== Output parent URL: $OSMO_OUTPUT_URL ==="
fi
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

if [[ "$STORAGE_BACKEND" == "osmo_data" ]]; then
    echo "=== Uploading code archive through OSMO DATA: $OSMO_CODE_URL ==="
    osmo data upload "$OSMO_CODE_URL" "$ARCHIVE_PATH"
    osmo data check "$OSMO_CODE_URL"
elif [[ "$STORAGE_BACKEND" == "swift" ]]; then
    echo "=== Ensuring local Swift client ==="
    export SWIFT_AUTH_URL SWIFT_AUTH_VERSION SWIFT_USER SWIFT_AUTH_KEY
    if ! python3 "$SCRIPT_DIR/lib/swift_io.py" check >/dev/null 2>&1; then
        python3 -m pip install --quiet -U python-swiftclient
        python3 "$SCRIPT_DIR/lib/swift_io.py" check
    fi
    echo "=== Uploading code archive to Swift: $SWIFT_CODE_CONTAINER/$SWIFT_CODE_OBJECT ==="
    python3 "$SCRIPT_DIR/lib/swift_io.py" upload-file \
        --container "$SWIFT_CODE_CONTAINER" \
        --source "$ARCHIVE_PATH" \
        --object-name "$SWIFT_CODE_OBJECT"
    SWIFT_VERIFY_PATH="$TMP_DIR/Swift-verify-${SWIFT_CODE_OBJECT##*/}"
    python3 "$SCRIPT_DIR/lib/swift_io.py" download-object \
        --container "$SWIFT_CODE_CONTAINER" \
        --object-name "$SWIFT_CODE_OBJECT" \
        --output "$SWIFT_VERIFY_PATH"
    if ! cmp --silent "$ARCHIVE_PATH" "$SWIFT_VERIFY_PATH"; then
        echo "[FATAL] Swift code upload verification failed: downloaded object differs from the archive." >&2
        exit 1
    fi
    echo "=== Swift code upload round-trip verification succeeded. ==="
else
    echo "=== Ensuring local nvdataset package ==="
    export NVDATASET_TENANTID NGC_API_KEY
    if ! python3 "$SCRIPT_DIR/lib/nvdataset_io.py" check >/dev/null 2>&1; then
        python3 -m pip install --quiet -U --extra-index-url "$NVDATASET_INDEX_URL" nvdataset
        python3 "$SCRIPT_DIR/lib/nvdataset_io.py" check
    fi
    echo "=== Uploading code archive to NV-Datasets: $NVDATASET_CODE_DATASET ==="
    python3 "$SCRIPT_DIR/lib/nvdataset_io.py" replace-files \
        --dataset "$NVDATASET_CODE_DATASET" \
        --description "IsaacLab-NeRD code snapshot archive for OSMO." \
        "$ARCHIVE_PATH"
fi

if [[ "$CODE_ONLY" == "1" ]]; then
    echo "=== Code-only mode complete. Skipping OSMO credential setup and workflow submit. ==="
    exit 0
fi

credential_exists() {
    local credential_list

    credential_list="$(osmo credential list 2>/dev/null || true)"
    [[ "$credential_list" == *"$STORAGE_CREDENTIAL"* ]]
}

if [[ "$STORAGE_BACKEND" == "osmo_data" ]]; then
    echo "=== Reusing existing OSMO DATA credential: $OSMO_DATA_CREDENTIAL ==="
elif [[ "$REFRESH_CREDENTIAL" != "1" ]] && credential_exists; then
    echo "=== Reusing existing OSMO storage credential: $STORAGE_CREDENTIAL ==="
else
    echo "=== Setting OSMO storage credential: $STORAGE_CREDENTIAL ==="
    credential_args=(osmo credential set "$STORAGE_CREDENTIAL" --type GENERIC)
    if [[ "$STORAGE_BACKEND" == "swift" ]]; then
        swift_auth_url_file="$TMP_DIR/swift-auth-url"
        swift_auth_version_file="$TMP_DIR/swift-auth-version"
        swift_user_file="$TMP_DIR/swift-user"
        swift_key_file="$TMP_DIR/swift-auth-key"
        printf '%s' "$SWIFT_AUTH_URL" > "$swift_auth_url_file"
        printf '%s' "$SWIFT_AUTH_VERSION" > "$swift_auth_version_file"
        printf '%s' "$SWIFT_USER" > "$swift_user_file"
        printf '%s' "$SWIFT_AUTH_KEY" > "$swift_key_file"
        chmod 600 "$swift_auth_url_file" "$swift_auth_version_file" "$swift_user_file" "$swift_key_file"
        credential_args+=(
            --payload-file
            auth_url="$swift_auth_url_file"
            auth_version="$swift_auth_version_file"
            user="$swift_user_file"
            auth_key="$swift_key_file"
        )
    else
        nvapi_key_file="$TMP_DIR/nvapi-key"
        nvdataset_tenant_file="$TMP_DIR/nvdataset-tenant"
        printf '%s' "$NGC_API_KEY" > "$nvapi_key_file"
        printf '%s' "$NVDATASET_TENANTID" > "$nvdataset_tenant_file"
        chmod 600 "$nvapi_key_file" "$nvdataset_tenant_file"
        credential_args+=(
            --payload-file
            nvapi_key="$nvapi_key_file"
            tenant_id="$nvdataset_tenant_file"
        )
    fi
    if ! credential_output="$("${credential_args[@]}" 2>&1)"; then
        if [[ "$credential_output" == *"duplicate key value"* || "$credential_output" == *"already exists"* ]]; then
            echo "Credential $STORAGE_CREDENTIAL already exists; reusing it."
        elif credential_exists; then
            echo "$credential_output" >&2
            echo "Credential $STORAGE_CREDENTIAL is present after set failed; reusing it."
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
    "amlfs_data_root=$AMLFS_DATA_ROOT"
    "train_seed=$TRAIN_SEED"
    "storage_backend=$STORAGE_BACKEND"
    "osmo_code_url=$OSMO_CODE_URL"
    "osmo_dataset_input_url=$OSMO_DATASET_INPUT_URL"
    "osmo_dataset_upload_url=$OSMO_DATASET_UPLOAD_URL"
    "osmo_output_url=$OSMO_OUTPUT_URL"
    "storage_credential=$STORAGE_CREDENTIAL"
    "storage_auth_url_env=$STORAGE_AUTH_URL_ENV"
    "storage_auth_url_key=$STORAGE_AUTH_URL_KEY"
    "storage_auth_version_env=$STORAGE_AUTH_VERSION_ENV"
    "storage_auth_version_key=$STORAGE_AUTH_VERSION_KEY"
    "storage_user_env=$STORAGE_USER_ENV"
    "storage_user_key=$STORAGE_USER_KEY"
    "storage_key_env=$STORAGE_KEY_ENV"
    "storage_key_key=$STORAGE_KEY_KEY"
    "storage_tenant_env=$STORAGE_TENANT_ENV"
    "storage_tenant_key=$STORAGE_TENANT_KEY"
    "osmo_experiment_preset=$OSMO_EXPERIMENT_PRESET"
    "memory=$OSMO_MEMORY"
    "storage=$OSMO_STORAGE"
    "resource_platform=$OSMO_PLATFORM"
    "swift_code_container=$SWIFT_CODE_CONTAINER"
    "swift_code_object=$SWIFT_CODE_OBJECT"
    "swift_data_container=$SWIFT_DATA_CONTAINER"
    "swift_output_container=$SWIFT_OUTPUT_CONTAINER"
    "nvdataset_credential=$NVDATASET_CREDENTIAL"
    "nvdataset_code_dataset=$NVDATASET_CODE_DATASET"
    "nvdataset_code_snapshot="
    "nvdataset_data_dataset=$NVDATASET_DATA_DATASET"
    "nvdataset_output_dataset=$NVDATASET_OUTPUT_DATASET"
    "nvdataset_output_description=$NVDATASET_OUTPUT_DESCRIPTION"
    "nvdataset_index_url=$NVDATASET_INDEX_URL"
)
if [[ "$ENABLE_WANDB" == "1" ]]; then
    SUBMIT_ARGS+=(
        "enable_wandb=true"
        "wandb_credential=$WANDB_CREDENTIAL"
        "wandb_project_name=$WANDB_PROJECT_NAME"
        "wandb_exp_name=$WANDB_EXP_NAME"
        "wandb_entity=$WANDB_ENTITY"
    )
fi
OSMO_RESOURCE_ARGS=()
if [[ -n "$OSMO_POOL_OVERRIDE" ]]; then
    OSMO_RESOURCE_ARGS+=(--pool "$OSMO_POOL_OVERRIDE")
fi
OSMO_SUBMIT_ONLY_ARGS=()
if [[ -n "$OSMO_PRIORITY_OVERRIDE" ]]; then
    OSMO_SUBMIT_ONLY_ARGS+=(--priority "$OSMO_PRIORITY_OVERRIDE")
fi

echo "=== Resolved OSMO resources: gpu=$OSMO_NUM_GPU cpu=$OSMO_NUM_CPU memory=$OSMO_MEMORY storage=$OSMO_STORAGE platform=$OSMO_PLATFORM pool=${OSMO_POOL_OVERRIDE:-<default>} train_seed=$TRAIN_SEED ==="

echo "=== Validating OSMO workflow ==="
osmo workflow validate "${SUBMIT_ARGS[@]}" "${OSMO_RESOURCE_ARGS[@]}" "${OSMO_ARGS[@]}" -- "$WORKFLOW_FILE"

echo "=== Submitting OSMO workflow ==="
osmo workflow submit \
    "${SUBMIT_ARGS[@]}" \
    "${OSMO_RESOURCE_ARGS[@]}" \
    "${OSMO_SUBMIT_ONLY_ARGS[@]}" \
    "${OSMO_ARGS[@]}" \
    -- "$WORKFLOW_FILE"
