#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Search target: keep GELU and LayerNorm exact, replace only attention softmax.
MODEL_KEY="${MODEL_KEY:-bert-base}"
# With the default he_inv cost below, all-iter1 costs 25 * 12 = 300.
# Search a constrained mixed band by default and keep total softmax cost <= 360.
BUDGET="${BUDGET:-300}"
BUDGET_BAND="${BUDGET_BAND:-0}"
OUT_SUBDIR="${OUT_SUBDIR:-joint_refine_suite_thor_p2_heinv_cost8_softmax_max360_iters}"

# THOR scaled softmax candidates. Each pair means:
#   p0 ~= softmax(x / SCALE), then run ITERS rounds of p <- p^2 / sum(p^2).
# Keeping SCALE=2^ITERS makes this an iteration-count search rather than an
# independent scale/iteration grid.
SOFTMAX_SCALE_SQUARE_PAIRS="${SOFTMAX_SCALE_SQUARE_PAIRS:-2:1,4:2,8:3}"
# Budget model for one THOR he_inv. The Python default is conservative (24);
# bert-base averages about 8.3 aSOR iterations per reciprocal call, because
# only layer 2 hits the 11-12 iteration case.
SOFTMAX_THOR_HEINV_COST="${SOFTMAX_THOR_HEINV_COST:-8}"

# Small search-size knobs. Override from the environment for faster smoke tests
# or larger final runs.
JOINT_FAST_N="${JOINT_FAST_N:-192}"
JOINT_FULL_N="${JOINT_FULL_N:-512}"
JOINT_MAX_STEPS="${JOINT_MAX_STEPS:-2}"
JOINT_MOVE_TRIALS="${JOINT_MOVE_TRIALS:-96}"
JOINT_CD_SOFTMAX_PASSES="${JOINT_CD_SOFTMAX_PASSES:-3}"
JOINT_CD_SOFTMAX_SITE_LIMIT="${JOINT_CD_SOFTMAX_SITE_LIMIT:-12}"
PROXY_STAGE1_N="${PROXY_STAGE1_N:-96}"
REUSE_PROXY="${REUSE_PROXY:-0}"

reuse_flag="--no-reuse-proxy-table"
if [[ "${REUSE_PROXY}" == "1" ]]; then
  reuse_flag="--reuse-proxy-table"
fi

echo "[thor-softmax-search] model=${MODEL_KEY}"
echo "[thor-softmax-search] budget=${BUDGET} band=${BUDGET_BAND}"
echo "[thor-softmax-search] output subdir=${OUT_SUBDIR}"
echo "[thor-softmax-search] softmax candidates=${SOFTMAX_SCALE_SQUARE_PAIRS}"
echo "[thor-softmax-search] thor he_inv cost=${SOFTMAX_THOR_HEINV_COST}"
echo "[thor-softmax-search] reuse proxy=${REUSE_PROXY}"

exec python "${SCRIPT_DIR}/bert_poly_joint_refine_thor_softmax_search.py" \
  --run-pipeline joint_refine \
  --model-key "${MODEL_KEY}" \
  --joint-task "${MODEL_KEY}:softmax_only:${BUDGET}:${BUDGET_BAND}" \
  --joint-root-subdir "${OUT_SUBDIR}" \
  --joint-mutable-kinds softmax \
  --joint-proxy-kinds softmax \
  --joint-cd-gelu-site-limit 0 \
  --joint-cd-layernorm-site-limit 0 \
  --joint-cd-softmax-passes "${JOINT_CD_SOFTMAX_PASSES}" \
  --joint-cd-softmax-site-limit "${JOINT_CD_SOFTMAX_SITE_LIMIT}" \
  --softmax-shift-modes scaled \
  --softmax-exp-methods thor_p2 \
  --softmax-inv-init-methods thor_heinv \
  --softmax-thor-heinv-cost "${SOFTMAX_THOR_HEINV_COST}" \
  --softmax-scale-square-pairs "${SOFTMAX_SCALE_SQUARE_PAIRS}" \
  --search-profile strict \
  --post-eval-profiles strict \
  --joint-fast-n "${JOINT_FAST_N}" \
  --joint-full-n "${JOINT_FULL_N}" \
  --joint-max-steps "${JOINT_MAX_STEPS}" \
  --joint-move-trials "${JOINT_MOVE_TRIALS}" \
  --proxy-stage1-n "${PROXY_STAGE1_N}" \
  "${reuse_flag}" \
  "$@"
