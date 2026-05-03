#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export THOR_LITEFHE_NONLINEAR_SCHEDULE_DIR="${THOR_LITEFHE_NONLINEAR_SCHEDULE_DIR:-${SCRIPT_DIR}/joint_refine_suite_thor_p2_heinv_cost8_max360_gelu_ln_litefhe}"
export THOR_LITEFHE_THOR_P2_RECIPROCAL_MODE="${THOR_LITEFHE_THOR_P2_RECIPROCAL_MODE:-heinv}"
export THOR_LITEFHE_PROGRESS_LOG="${THOR_LITEFHE_PROGRESS_LOG:-/home/yhh/PNP/THOR/litefhe_eval_logs/sst2/dev_thor_p2_heinv_cost8_max360_gelu_ln_progress.jsonl}"
export THOR_LITEFHE_SUMMARY_JSON="${THOR_LITEFHE_SUMMARY_JSON:-/home/yhh/PNP/THOR/litefhe_eval_logs/sst2/dev_thor_p2_heinv_cost8_max360_gelu_ln_summary.json}"

exec "${SCRIPT_DIR}/run_sst2_dev_litefhe.sh" "$@"
