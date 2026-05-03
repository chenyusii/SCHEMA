#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THOR_LITEFHE_NUM_LEVELS="${THOR_LITEFHE_NUM_LEVELS:-29}"
export THOR_LITEFHE_SCALE_BITS="${THOR_LITEFHE_SCALE_BITS:-45}"
export THOR_LITEFHE_FIRST_MOD="${THOR_LITEFHE_FIRST_MOD:-52}"
export THOR_LITEFHE_SECRET_KEY_DIST="${THOR_LITEFHE_SECRET_KEY_DIST:-SPARSE_TERNARY}"
export THOR_LITEFHE_RESCALE_TECH="${THOR_LITEFHE_RESCALE_TECH:-FLEXIBLEAUTO}"
export THOR_LITEFHE_THOR_P2_RECIPROCAL_MODE="${THOR_LITEFHE_THOR_P2_RECIPROCAL_MODE:-heinv}"
export THOR_LITEFHE_NONLINEAR_SCHEDULE_DIR="${THOR_LITEFHE_NONLINEAR_SCHEDULE_DIR:-${SCRIPT_DIR}/joint_refine_suite_thor_p2_user_config}"
export THOR_LITEFHE_AMP_SCHEDULE="${THOR_LITEFHE_AMP_SCHEDULE:-/home/yhh/PNP/THOR/litefhe_amp_schedules/sst2/amp_schedule_train_calib128_seed224.json}"

ARTIFACT_ROOT="${THOR_LITEFHE_ARTIFACT_ROOT:-/home/yhh/PNP/THOR}"
MODEL_ROOT="${THOR_LITEFHE_MODEL_ROOT:-${REPO_ROOT}/examples/sst-2-finetuned-bert-base-uncased}"
DATASET_ROOT="${THOR_LITEFHE_DATASET_ROOT:-/home/yhh/PNP/hf_assets/models/sst2}"
ENCODED_ROOT="${THOR_LITEFHE_ENCODED_ROOT:-${ARTIFACT_ROOT}/encoded_models_litefhe_ref/sst2}"
MIDDLE_ROOT="${THOR_LITEFHE_MIDDLE_ROOT:-${ARTIFACT_ROOT}/encoded_models_litefhe/sst2/middle_stage}"
PROGRESS_LOG="${THOR_LITEFHE_PROGRESS_LOG:-${ARTIFACT_ROOT}/litefhe_eval_logs/sst2/dev_current_schedule_progress.jsonl}"
SUMMARY_JSON="${THOR_LITEFHE_SUMMARY_JSON:-${ARTIFACT_ROOT}/litefhe_eval_logs/sst2/dev_current_schedule_summary.json}"
LIMIT="${THOR_LITEFHE_DEV_LIMIT:-872}"
START_IDX="${THOR_LITEFHE_DEV_START_IDX:-0}"
DEVICE="${THOR_LITEFHE_DEVICE:-cuda}"

mkdir -p "$(dirname "${PROGRESS_LOG}")"

echo "[sst2-dev] schedule: ${THOR_LITEFHE_NONLINEAR_SCHEDULE_DIR}"
echo "[sst2-dev] amp schedule: ${THOR_LITEFHE_AMP_SCHEDULE}"
echo "[sst2-dev] progress log: ${PROGRESS_LOG}"
echo "[sst2-dev] summary json: ${SUMMARY_JSON}"
echo "[sst2-dev] range: start=${START_IDX}, limit=${LIMIT}"

exec python "${SCRIPT_DIR}/eval_dev_litefhe.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-type sst2 \
  --model-root "${MODEL_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --encoded-root "${ENCODED_ROOT}" \
  --middle-root "${MIDDLE_ROOT}" \
  --device "${DEVICE}" \
  --start-idx "${START_IDX}" \
  --limit "${LIMIT}" \
  --resume-progress \
  --progress-log "${PROGRESS_LOG}" \
  --summary-json "${SUMMARY_JSON}" \
  "$@"
