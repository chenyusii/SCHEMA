#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Stage 2 target:
#   - keep the THOR softmax schedule from stage 1 fixed
#   - search nonlinear approximations for GELU and LayerNorm only
MODEL_KEY="${MODEL_KEY:-bert-base}"
SOFTMAX_THOR_HEINV_COST="${SOFTMAX_THOR_HEINV_COST:-8}"
SOFTMAX_SCALE_SQUARE_PAIRS="${SOFTMAX_SCALE_SQUARE_PAIRS:-2:1,4:2,8:3}"
SOFTMAX_SCHEDULE="${SOFTMAX_SCHEDULE:-${REPO_ROOT}/examples/orion_dev/examples/site_schedule/bert_base_poly_signsoftmax/joint_refine_suite_thor_p2_heinv_cost8_softmax_max360_iters/softmax_only_budget_300/best_schedule_full.csv}"

# With the default cost model:
#   fixed softmax schedule = 300
#   GELU min/max = 12 * 5..7 = 60..84
#   LayerNorm min/max = 25 * 5..12 = 125..300
# so fixed-softmax GELU+LN feasible totals are 485..684.
BUDGET="${BUDGET:-550}"
BUDGET_BAND="${BUDGET_BAND:-10}"
OUT_SUBDIR="${OUT_SUBDIR:-joint_refine_suite_thor_p2_heinv_cost8_fixed_softmax_max360_gelu_ln}"

# Search-size knobs. Override from the environment for smoke tests or larger runs.
JOINT_FAST_N="${JOINT_FAST_N:-192}"
JOINT_FULL_N="${JOINT_FULL_N:-512}"
JOINT_MAX_STEPS="${JOINT_MAX_STEPS:-2}"
JOINT_MOVE_TRIALS="${JOINT_MOVE_TRIALS:-96}"
JOINT_CD_GELU_PASSES="${JOINT_CD_GELU_PASSES:-2}"
JOINT_CD_LAYERNORM_PASSES="${JOINT_CD_LAYERNORM_PASSES:-2}"
JOINT_CD_GELU_SITE_LIMIT="${JOINT_CD_GELU_SITE_LIMIT:-12}"
JOINT_CD_LAYERNORM_SITE_LIMIT="${JOINT_CD_LAYERNORM_SITE_LIMIT:-25}"
PROXY_STAGE1_N="${PROXY_STAGE1_N:-96}"
REUSE_PROXY="${REUSE_PROXY:-0}"

if [[ ! -f "${SOFTMAX_SCHEDULE}" ]]; then
  echo "[gelu-ln-search] missing softmax schedule: ${SOFTMAX_SCHEDULE}" >&2
  echo "[gelu-ln-search] run examples/THOR-main/run_bert_softmax_thor_search.sh first, or set SOFTMAX_SCHEDULE=/path/to/best_schedule_full.csv" >&2
  exit 1
fi

reuse_flag="--no-reuse-proxy-table"
if [[ "${REUSE_PROXY}" == "1" ]]; then
  reuse_flag="--reuse-proxy-table"
fi

echo "[gelu-ln-search] model=${MODEL_KEY}"
echo "[gelu-ln-search] fixed softmax schedule=${SOFTMAX_SCHEDULE}"
echo "[gelu-ln-search] budget=${BUDGET} band=${BUDGET_BAND}"
echo "[gelu-ln-search] output subdir=${OUT_SUBDIR}"
echo "[gelu-ln-search] thor he_inv cost=${SOFTMAX_THOR_HEINV_COST}"
echo "[gelu-ln-search] reuse proxy=${REUSE_PROXY}"

exec python "${SCRIPT_DIR}/bert_poly_joint_refine_thor_softmax_search.py" \
  --run-pipeline joint_refine \
  --model-key "${MODEL_KEY}" \
  --joint-task "${MODEL_KEY}:all:${BUDGET}:${BUDGET_BAND}:${SOFTMAX_SCHEDULE}" \
  --joint-root-subdir "${OUT_SUBDIR}" \
  --joint-mutable-kinds gelu,layernorm \
  --joint-proxy-kinds gelu,layernorm \
  --joint-cd-softmax-site-limit 0 \
  --joint-cd-gelu-passes "${JOINT_CD_GELU_PASSES}" \
  --joint-cd-layernorm-passes "${JOINT_CD_LAYERNORM_PASSES}" \
  --joint-cd-gelu-site-limit "${JOINT_CD_GELU_SITE_LIMIT}" \
  --joint-cd-layernorm-site-limit "${JOINT_CD_LAYERNORM_SITE_LIMIT}" \
  --joint-cd-skip-layernorm-if-nf-leq -1 \
  --softmax-shift-modes scaled \
  --softmax-exp-methods thor_p2 \
  --softmax-inv-init-methods thor_heinv \
  --softmax-thor-heinv-cost "${SOFTMAX_THOR_HEINV_COST}" \
  --softmax-scale-square-pairs "${SOFTMAX_SCALE_SQUARE_PAIRS}" \
  --gelu-degrees 19,29,45,59,69,89,119 \
  --gelu-range-mults 1.0,1.25 \
  --layernorm-init-methods chebyshev,taylor \
  --layernorm-init-degrees 3,5,7 \
  --layernorm-range-mults 1.0,1.25 \
  --layernorm-refines none,newton,goldschmidt \
  --layernorm-iters 0,1,2 \
  --search-profile strict \
  --post-eval-profiles strict \
  --joint-fast-n "${JOINT_FAST_N}" \
  --joint-full-n "${JOINT_FULL_N}" \
  --joint-max-steps "${JOINT_MAX_STEPS}" \
  --joint-move-trials "${JOINT_MOVE_TRIALS}" \
  --proxy-stage1-n "${PROXY_STAGE1_N}" \
  "${reuse_flag}" \
  "$@"
