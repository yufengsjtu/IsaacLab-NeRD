#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/source/isaaclab_neural:${REPO_ROOT}/source/isaaclab_newton:${REPO_ROOT}/source/isaaclab:${REPO_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${REPO_ROOT}/pretrained/control_policy/franka_reach/model_999.pt}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/actuator_compare}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_STEPS="${NUM_STEPS:-360}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}"

run_one() {
    local actuator="$1"
    echo "[compare] running actuator=${actuator}"
    "${REPO_ROOT}/isaaclab.sh" -p -m isaaclab_neural.eval.compare_franka_reach_actuators \
        --actuator "${actuator}" \
        --policy-checkpoint "${POLICY_CHECKPOINT}" \
        --num-envs "${NUM_ENVS}" \
        --num-steps "${NUM_STEPS}" \
        --seed "${SEED}" \
        --visualizer none \
        --metrics-json "${OUT_DIR}/${actuator}.json" \
        presets=newton_mjwarp
}

run_one implicit
run_one ideal_pd

python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
implicit = json.loads((out_dir / "implicit.json").read_text())
ideal = json.loads((out_dir / "ideal_pd.json").read_text())
keys = [
    "reward_mean",
    "position_error_m",
    "orientation_error_rad",
    "success_frac",
    "applied_effort_abs_nm",
    "success_resets",
    "timeouts",
]
print("\n[compare] implicit vs ideal_pd")
for key in keys:
    a = implicit[key]
    b = ideal[key]
    delta = b - a
    rel = delta / a if abs(a) > 1e-8 else float("inf")
    print(f"  {key:24s}  implicit={a:10.4f}  ideal_pd={b:10.4f}  delta={delta:+10.4f}  rel={rel:+.3f}")

pos_ok = abs(ideal["position_error_m"] - implicit["position_error_m"]) < max(0.02, 0.25 * implicit["position_error_m"])
ori_ok = abs(ideal["orientation_error_rad"] - implicit["orientation_error_rad"]) < max(0.05, 0.25 * implicit["orientation_error_rad"])
rew_ok = abs(ideal["reward_mean"] - implicit["reward_mean"]) < max(0.05, 0.25 * abs(implicit["reward_mean"]))
both_tracking = implicit["position_error_m"] < 0.15 and ideal["position_error_m"] < 0.15
passed = pos_ok and ori_ok and rew_ok and both_tracking
print(f"\n[compare] pass={passed} (pos_ok={pos_ok} ori_ok={ori_ok} rew_ok={rew_ok} both_tracking={both_tracking})")
if not passed:
    raise SystemExit(1)
PY
