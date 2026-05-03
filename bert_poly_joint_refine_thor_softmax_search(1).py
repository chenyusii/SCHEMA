#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bert_poly_joint_refine_thor_softmax_search.py

Single-file simulator for staged bert-base-uncased approximation search, with
THOR-style P(z)^2 softmax variants integrated into the same v12 joint-refine
design space.  The default flow is softmax-first: keep GELU / LayerNorm exact,
search THOR softmax choices, then use the selected softmax schedule as the seed
for a later GELU / LayerNorm search.

What this script does
---------------------
1) Loads a *locally saved* Hugging Face sequence-classification checkpoint.
2) Instruments BERT GELU / attention Softmax / LayerNorm.
3) Replaces:
     - GELU        -> Chebyshev polynomial (same spirit as existing ReLU flow)
     - Softmax     -> approximate exp + approximate reciprocal(sum)
     - LayerNorm   -> approximate rsqrt(var + eps)
4) Collects calibration ranges on a train subset.
5) Builds a one-site proxy error table (all other sites exact).
6) Runs DP under a depth budget, optionally sweeping many budgets to get a Pareto curve.

Default target is SST-2 style local data + locally saved bert-tiny / bert-base checkpoints,
but the file layout is configurable from the top block.

Important assumptions / scope
-----------------------------
- This file is written specifically for BERT-family encoder models used for sequence classification.
- The attention patch targets standard encoder-only BERT self-attention.
- For bert-base-uncased / bert-tiny on SST-2 this is the intended path.
- This is a *simulator* for approximation impact in plaintext. It is not a full HE runtime.
- Positive-domain operations (1/x and 1/sqrt(x)) use a tiny epsilon floor by default to avoid NaNs.
  Set CLAMP_POSITIVE_DOMAIN=False if you want a stricter failure mode.

The overall structure intentionally mirrors the existing sweep_dp.py workflow: top config block,
single-file implementation, calibration -> proxy table -> DP -> budget sweep / final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import types
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

_THIS_FILE = Path(__file__).resolve()


def _find_repo_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    # This file normally lives at <repo>/examples/THOR-main/<script>.py.
    return path.parents[2] if len(path.parents) > 2 else path.parent


REPO_ROOT = _find_repo_root(_THIS_FILE)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
except Exception as exc:  # pragma: no cover - runtime dependency in user's env
    raise ImportError(
        "This script requires `transformers`. Install it in the target environment, e.g. `pip install transformers`."
    ) from exc


def _disable_transformers_torchvision_probe() -> None:
    """BERT text runs do not need torchvision; avoid mismatched torchvision ops."""
    try:
        from transformers.utils import import_utils as hf_import_utils

        hf_import_utils._torchvision_available = False
        hf_import_utils._torchvision_version = "disabled"
    except Exception:
        pass


_disable_transformers_torchvision_probe()

try:
    from datasets import load_from_disk as hf_load_from_disk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hf_load_from_disk = None


# =============================================================================
# User Config (edit here)
# =============================================================================
BASE_DIR = REPO_ROOT.parent

RUN_MODEL_KEYS = [
    "bert-base",
]

# Ablation modes: the listed component kinds are approximated; all other kinds stay exact.
# Supported presets:
#   all
#   gelu_only / softmax_only / layernorm_only
#   gelu_softmax / gelu_layernorm / softmax_layernorm
# You can also write custom combinations like "gelu+softmax".
RUN_ABLATION_MODES = [
    "softmax_only",
]
ABLATION_ROOT_SUBDIR = "ablation_suite_calibmax"
ABLATION_SUMMARY_CSV_NAME = "ablation_summary.csv"
ABLATION_SUMMARY_JSON_NAME = "ablation_summary.json"

MODEL_SPECS: Dict[str, Dict[str, Path]] = {
    "bert-tiny": {
        "model_dir": REPO_ROOT / "examples/sst-2-finetuned-bert-tiny",
        "tokenizer_dir": BASE_DIR / "hf_assets/models/bert-tiny",
        "output_dir": REPO_ROOT / "examples/orion_dev/examples/site_schedule/bert_tiny_poly_signsoftmax",
    },
    "bert-base": {
        "model_dir": REPO_ROOT / "examples/sst-2-finetuned-bert-base-uncased",
        "tokenizer_dir": BASE_DIR / "hf_assets/models/bert-base-uncased",
        "output_dir": REPO_ROOT / "examples/orion_dev/examples/site_schedule/bert_base_poly_signsoftmax",
    },
}

DATASET_DIR = BASE_DIR / "hf_assets/models/sst2"
DATASET_FORMAT = "auto"  # auto | hf_disk | tsv | csv | jsonl
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"
TRAIN_FILE_CANDIDATES = [
    "train.tsv", "train.csv", "train.jsonl",
]
VAL_FILE_CANDIDATES = [
    "dev.tsv", "validation.tsv", "dev.csv", "validation.csv", "dev.jsonl", "validation.jsonl",
]
TEST_FILE_CANDIDATES = [
    "test.tsv", "test.csv", "test.jsonl",
]

MAX_LENGTH = 128
LOWERCASE_TEXT = False

USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 8
NUM_WORKERS = 0

# Loader autotune / compute micro-batching.
# For BERT, extremely large loader batches can *hurt* because padding expands to the
# maximum sequence length inside the fetched batch. v6 therefore defaults to:
#   - fixed compute micro-batches (same semantics as before)
#   - optional loader autotune (OFF by default)
#   - deterministic length-bucketing / length-sorted loaders to reduce padding waste.
AUTO_TUNE_LOADER_BATCH = False
AUTO_LOADER_START_MULT = 8
AUTO_LOADER_MAX_MULT = 16
AUTO_LOADER_PREFETCH_FACTOR = 4
MOVE_BASELINE_LOGITS_TO_DEVICE = True
LOG_TUNED_LOADER_BATCHES = True
ENABLE_LENGTH_BUCKETING = True
LENGTH_BUCKET_ASCENDING = True

GLOBAL_SEED = 0

CALIB_N = 512
SWEEP_N = 256
VAL_N = 512
TEST_MAX_N: Optional[int] = None  # None => use full validation/test split

RUN_MODE = "budget_sweep"  # budget_sweep | single_budget
TOTAL_BUDGET = 996
BUDGET_SWEEP_LIST = [684, 736, 788, 840, 892, 944, 996, 1048, 1100, 1152, 1204, 1256, 1308]

# Guard: if a candidate drops validation accuracy too much vs the exact baseline, discard it.
ENABLE_VAL_ACC_GUARD = True
VAL_ACC_GUARD_DROP_PP = 3.0

# Depth cost heuristic. Reused from the ReLU script style.
DEPTH_COST_RANGES = [
    (1, 5, 3),
    (6, 13, 4),
    (14, 27, 5),
    (28, 59, 6),
    (60, 119, 7),
    (120, 247, 8),
    (248, 495, 9),
    (496, 1007, 10),
    (1008, 2031, 11),
]

# Approximation candidates -----------------------------------------------------
# GELU: pointwise, always Chebyshev (same spirit as the ReLU flow).
GELU_DEGREES = [19, 29, 45, 59, 69, 89, 119]
GELU_RANGE_MULTS = [1.0, 1.25]

# Softmax: approximate exp + approximate reciprocal(sum).
#
# HE-friendly max-shift replacements:
#   if shift_mode == "calib_max", calibration learns one public max-shift scalar per
#       softmax layer/site, then online forward subtracts that lookup-table scalar.
#   if shift_mode == "sign_max", we use a sign-based binary-tree reduction
#       max(a, b) ~= 0.5 * (a + b + (a - b) * sign_approx(a - b))
#   before the usual exp / sum / reciprocal pipeline.
#
# To keep the search space tractable, the sign polynomial configuration is GLOBAL
# (shared by all sign-max softmax sites) and the DP only searches the exp / inv parts.
#
# Supported shift modes: none | row_mean | row_max | sign_max | calib_max | scaled
# Default here uses a calibration-learned layerwise max lookup table, so online
# softmax does not compute a secret-dependent max.
SOFTMAX_SHIFT_MODES = ["scaled"]
SOFTMAX_EXP_METHODS = ["thor_p2"]
SOFTMAX_EXP_DEGREES = [13, 15, 17, 19]
SOFTMAX_EXP_RANGE_MULTS = [1.0, 1.25, 1.5]
SOFTMAX_INV_INIT_METHODS = ["thor_heinv"]
SOFTMAX_INV_INIT_DEGREES = [0]
SOFTMAX_INV_RANGE_MULTS = [1.0]
SOFTMAX_INV_REFINES = ["none"]
SOFTMAX_INV_ITERS = [0]
SOFTMAX_DENOM_EPS = 1e-6
SOFTMAX_OUTPUT_DROPOUT_IN_EVAL = False
SOFTMAX_MASK_THRESHOLD = -1e3  # additive attention-mask threshold (PUBLIC mask)
SOFTMAX_MASKED_FILL_FOR_DEN_CALIB = -80.0
SOFTMAX_CALIB_MAX_QUANTILE = 1.0
SOFTMAX_CALIB_MAX_MARGIN_MULT = 1.0
SOFTMAX_CALIB_MAX_MARGIN_ABS = 0.0
SOFTMAX_CALIB_MAX_SAMPLE_CAPACITY = 65536
SOFTMAX_CALIB_MAX_PER_UPDATE_LIMIT = 4096
# THOR fixed exp surrogate, copied from THOR softmax.py / he_exp1, he_exp2.
# It evaluates a fixed degree-15 polynomial P on a THOR-scaled input and returns P(z)^2.
SOFTMAX_THOR_EXP1_COEFFS_ASC: List[float] = list(
    reversed(
        [
            0.032855468333339584,
            0.05948672763856172,
            0.03881607331549499,
            0.0670090353368128,
            0.15202099984697098,
            0.20618261949210986,
            0.23721029007596767,
            0.26787311936472025,
            0.27220647178765545,
            0.2379982262906916,
            0.1780344447042791,
            0.11128698173597897,
            0.05566510463488879,
            0.020873931555133732,
            0.005218196900295354,
            0.0006522770224130905,
        ]
    )
)
SOFTMAX_THOR_EXP2_COEFFS_ASC: List[float] = list(
    reversed(
        [
            0.008201736399899691,
            0.014226972463907047,
            -0.008386712802267769,
            -0.009262268572236316,
            0.0397324053296174,
            0.04817928878801878,
            0.016604320800445653,
            0.02336452059478656,
            0.04217318306400685,
            0.03517495704921328,
            0.022268203231858744,
            0.014216636671807894,
            0.00749909544008294,
            0.0027930565779849003,
            0.0006877176070101981,
            8.615994668877663e-05,
        ]
    )
)
SOFTMAX_THOR_EXP1_MIN_X = -27.2493
SOFTMAX_THOR_EXP1_MAX_X = 21.72692
SOFTMAX_THOR_EXP1_MID_X = (SOFTMAX_THOR_EXP1_MIN_X + SOFTMAX_THOR_EXP1_MAX_X) / 2.0
SOFTMAX_THOR_EXP1_PLAIN_MIN = SOFTMAX_THOR_EXP1_MIN_X / 2.0
SOFTMAX_THOR_EXP1_PLAIN_MAX = SOFTMAX_THOR_EXP1_MAX_X / 2.0
SOFTMAX_THOR_EXP2_MIN_X = -70.0
SOFTMAX_THOR_EXP2_MAX_X = 70.0
SOFTMAX_THOR_EXP2_MID_X = (SOFTMAX_THOR_EXP2_MIN_X + SOFTMAX_THOR_EXP2_MAX_X) / 2.0
SOFTMAX_THOR_EXP2_PLAIN_MIN = SOFTMAX_THOR_EXP2_MIN_X / 4.0
SOFTMAX_THOR_EXP2_PLAIN_MAX = SOFTMAX_THOR_EXP2_MAX_X / 4.0
SOFTMAX_THOR_EXP_DEGREE = 15
SOFTMAX_THOR_EXP_LAYER2_INDEX = 2
SOFTMAX_THOR_EXP_CLAMP_INPUT = False
SOFTMAX_THOR_EXP_CLAMP_OUTPUT_MIN = False
# THOR reciprocal settings from thor_bert_base_plain_sim_heinv.py.  The
# estimator cost is intentionally kept in the v12 per-site depth band so these
# choices can compete with d11/d27 + Newton reciprocal schedules.
SOFTMAX_THOR_INV_EPS1 = 2.0 ** (-11)
SOFTMAX_THOR_INV_EPS2 = 2.0 ** (-18)
SOFTMAX_THOR_INTERNAL_ALPHA = 0.1
SOFTMAX_THOR_OUTPUT_ALPHA = 0.01
# he_inv runs 8-12 fixed aSOR iterations depending on layer/profile. Each
# iteration has serial multiplications, so use the conservative layer-2 cost for
# budget accounting; otherwise low-budget DP picks an unrealistically strong
# reciprocal everywhere.
SOFTMAX_THOR_HEINV_COST = 24
# THOR-style HE-friendly path:
#   p0 ~= softmax(x / scale), then repeatedly p <- p^2 / sum(p^2).
# When scale ~= 2^square_iters, the exact version recovers softmax(x) without
# computing a row-wise max.
SOFTMAX_INPUT_SCALES = [2.0, 4.0, 8.0]
SOFTMAX_SQUARE_ITERS = [1, 2, 3]
SOFTMAX_SCALE_SQUARE_MODE = "explicit"  # matched | explicit
SOFTMAX_SCALE_SQUARE_PAIRS: List[Tuple[float, int]] = [(2.0, 1), (4.0, 2), (8.0, 3)]
# Optional per-softmax depth-cost filter. Useful for keeping a scaled-softmax
# design space in a target HE depth band, e.g. 16..20 per attention layer.
SOFTMAX_COST_MIN: Optional[int] = None
SOFTMAX_COST_MAX: Optional[int] = None
SOFTMAX_LABEL_ALLOW_REGEX: Optional[str] = None

# Global sign polynomial config used when shift_mode == "sign_max".
SOFTMAX_SIGN_METHOD = "cubic"  # cubic | chebyshev
SOFTMAX_SIGN_CHEB_DEGREE = 15   # used only when method == "chebyshev"
SOFTMAX_SIGN_COMPOSE_STEPS = 5  # cubic: repeated self-composition count
SOFTMAX_SIGN_EPS = 0.05         # chebyshev fit excludes [-eps, +eps]
SOFTMAX_SIGN_SCALE_MULT = 1.0   # multiplier applied to calibrated sign-delta range
SOFTMAX_SIGN_CLIP_TO_UNIT = True
SOFTMAX_SIGN_FIT_POINTS = 4096
SOFTMAX_SIGN_COST_MODE = "flat"  # flat | tree
SOFTMAX_SIGN_TREE_LOG2_LEN = max(1, int(math.ceil(math.log2(max(2, MAX_LENGTH)))))

# LayerNorm: exact mean/variance, approximate rsqrt(var + eps).
LAYERNORM_INIT_METHODS = ["chebyshev", "taylor"]
LAYERNORM_INIT_DEGREES = [3, 5, 7]
LAYERNORM_RANGE_MULTS = [1.0, 1.25]
LAYERNORM_REFINES = ["none", "newton", "goldschmidt"]
LAYERNORM_ITERS = [0, 1, 2]

# Small numerical stabilization for positive-domain functions (1/x, 1/sqrt(x)).
CLAMP_POSITIVE_DOMAIN = True
POSITIVE_DOMAIN_EPS = 1e-6
# For strict forward-only evaluation we still allow *offline* positive-domain interval preservation
# when constructing reciprocal/rsqrt polynomials. This keeps coefficient construction well-defined
# without forcing an online clamp inside the simulated forward.
STRICT_OFFLINE_POSITIVE_PRESERVE = True

# Reuse proxy table if it already exists.
REUSE_PROXY_TABLE = True
PROXY_TABLE_NAME = "proxy_error_table_thor_selfcalib.csv"
SHARED_PROXY_TABLE_NAME = "proxy_error_table_all_sites_thor_selfcalib.csv"

# Proxy-stage timing / pruning.
# Stage-1: cheap prescreen on a smaller sweep subset over the full candidate space.
# Stage-2: full sweep proxy evaluation only on the top-K choices per (site, cost bucket).
# Dropped same-cost choices receive a surrogate error slightly worse than the best kept choice
# in that bucket, so DP will never prefer them over a kept alternative of equal cost.
ENABLE_PROXY_STAGE1_PRUNE = True
PROXY_STAGE1_N = 96
PROXY_STAGE1_KEEP_PER_COST = 2
# Kind-asymmetric stage1 survivor counts: keep more diversity for softmax/GELU, prune LayerNorm harder.
PROXY_STAGE1_KEEP_PER_COST_GELU = 3
PROXY_STAGE1_KEEP_PER_COST_SOFTMAX = 3
PROXY_STAGE1_KEEP_PER_COST_LAYERNORM = 2
PROXY_STAGE1_TABLE_NAME = "proxy_stage1_scores_calibmax.csv"
PROXY_DROP_SURROGATE_MARGIN = 1.0e-3
PHASE_TIMING_JSON_NAME = "phase_timing.json"
SOFTMAX_CALIB_MAX_LUT_CSV_NAME = "softmax_calib_max_lut.csv"

RESULT_JSON_NAME = "budget_sweep_results.json"
RESULT_CSV_NAME = "budget_sweep_results.csv"

# Optional post-DP greedy validation refinement.
ENABLE_GREEDY_VAL_REFINEMENT = False
GREEDY_MAX_PASSES = 2

# Local loading only.
LOCAL_FILES_ONLY = True
ATTN_IMPLEMENTATION = "eager"  # important: keep BERT attention on eager path for our patch.

# Range expansion.
RANGE_MARGIN_FRAC = 0.05

# Fallback ranges used if calibration failed for a site/channel.
RANGE_FALLBACKS = {
    "gelu::x": (-6.0, 6.0),
    "softmax_exp::none": (-8.0, 8.0),
    "softmax_exp::row_mean": (-8.0, 8.0),
    "softmax_exp::row_max": (-16.0, 0.0),
    "softmax_exp::sign_max": (-16.0, 0.0),
    "softmax_exp::calib_max": (-16.0, 0.0),
    "softmax_exp::scaled": (-8.0, 8.0),
    "softmax_den::none": (1e-4, 4096.0),
    "softmax_den::row_mean": (1e-4, 4096.0),
    "softmax_den::row_max": (1e-4, 4096.0),
    "softmax_den::sign_max": (1e-4, 4096.0),
    "softmax_den::calib_max": (SOFTMAX_DENOM_EPS, float(MAX_LENGTH)),
    "softmax_den::scaled": (1e-4, 4096.0),
    "softmax_den_square::scaled": (1.0 / float(MAX_LENGTH), 1.0),
    "softmax_sign::delta": (-32.0, 32.0),
    "softmax_calib_max::score": (-16.0, 16.0),
    "softmax_calib_max::row_max": (0.0, 16.0),
    "layernorm::var": (1e-6, 16.0),
}


# =============================================================================
# Joint refinement config (FAST -> VAL -> gated FULL), adapted from the ReLU joint
# optimizer style used in the attached reference script.
# =============================================================================

RUN_PIPELINE = "joint_refine"  # joint_refine | ablation_budget_sweep

# Target tasks. If schedule_csv is None, the script first tries the default ablation
# schedule path under MODEL_SPECS[model_key]["output_dir"]/ABLATION_ROOT_SUBDIR; if that
# file does not exist, it falls back to rebuilding the DP schedule from the shared proxy table.
JOINT_TASKS: List[Dict[str, Any]] = [
    {
        "model_key": "bert-base",
        "mode": "softmax_only",
        "budget": 996,
        "budget_band": 0,
        "schedule_csv": None,
    },
]

JOINT_ROOT_SUBDIR = "joint_refine_suite_thor_heinv_softmax_only_sq123"
JOINT_SHARED_PROXY_TABLE_PATH: Optional[Path] = None
JOINT_STAGE_CSV_NAME = "joint_stage_metrics.csv"
JOINT_STAGE_JSON_NAME = "joint_stage_metrics.json"
JOINT_BEST_SCHEDULE_NAME = "best_schedule.csv"
JOINT_BEST_SCHEDULE_FULL_NAME = "best_schedule_full.csv"

# Loader sizes for joint search
JOINT_FAST_N = 192
JOINT_FULL_N = None  # None => reuse CALIB_N examples from the calibration split
JOINT_TEST_N = None  # None => reuse TEST_MAX_N / full report split

# FAST/VAL/FULL search controls
JOINT_MAX_STEPS = 2
JOINT_MOVE_TRIALS_PER_STEP = 96
JOINT_TOPK_VAL = 8
JOINT_FAST_KEEP = 24

JOINT_ENABLE_SITE_PRIORITY_PROPOSAL = True
JOINT_PROPOSAL_TOPM_IMPORTANCE = 18
JOINT_PROPOSAL_POOL_MAX = 24
JOINT_PROPOSAL_POOL_PROB = 0.85

JOINT_SINGLE_MOVE_PROB = 0.30
JOINT_SWAP_MOVE_PROB = 0.48
JOINT_TRI_MOVE_PROB = 0.22

JOINT_ENABLE_TRI_MOVES = True

# Gated FULL guard
JOINT_FULL_GUARD_TOP_CANDS_PER_STEP = 1
JOINT_PARETO_MAX_TRY = 8
JOINT_FULL_GUARD_EVERY_K_ACCEPTS = 4
JOINT_FULL_GUARD_TRIGGER_REL_IMPROV = 0.03

JOINT_ENABLE_VAL_ACC_GUARD = True
JOINT_VAL_ACC_GUARD_DROP_PP = 3.0
JOINT_ENABLE_FULL_ACC_GUARD = True
JOINT_FULL_ACC_GUARD_DROP_PP = 4.0

# Use exact full-split max-abs-logit as a coarse numeric stability reference.
JOINT_LOGIT_ABS_SOFT_MULT = 8.0
JOINT_LOGIT_ABS_HARD_MULT = 50.0
JOINT_LOGIT_ABS_PENALTY_WEIGHT = 0.02

# Relaxed nonfinite handling for BERT joint refinement.
# The previous version treated any nonfinite logit on FAST/VAL/FULL as an immediate hard failure.
# That is too strict for the current BERT schedules, because the best ablation schedules often have
# rare nonfinite events but still strong end-to-end accuracy. We therefore:
#   - allow a *small* number/rate of nonfinite samples in FAST / VAL / FULL
#   - add a penalty term to the objective so cleaner candidates are still preferred
#   - let joint/global search accept "stability rescue" moves when they clearly reduce nonfinite
#     frequency or max-abs-logit, even if VAL MSE is only flat / slightly worse.
JOINT_NONFINITE_PENALTY_WEIGHT = 0.50
JOINT_NONFINITE_HARD_PENALTY = 1.00
JOINT_FAST_MAX_NONFINITE_SAMPLES = 2
JOINT_FAST_MAX_NONFINITE_RATE = 0.05
JOINT_VAL_MAX_NONFINITE_SAMPLES = 4
JOINT_VAL_MAX_NONFINITE_RATE = 0.02
# Soft-pass for VAL: let mildly dirty candidates reach FULL/rescue instead of hard-dropping
# them before the joint logic has a chance to trade off accuracy vs stability.
JOINT_ENABLE_VAL_NONFINITE_SOFT_PASS = True
JOINT_VAL_SOFT_MAX_NONFINITE_SAMPLES = 24
JOINT_VAL_SOFT_MAX_NONFINITE_RATE = 0.08
JOINT_FULL_MAX_NONFINITE_SAMPLES = 10
JOINT_FULL_MAX_NONFINITE_RATE = 0.01

# When the current schedule is not FULL-ok, allow "stability rescue" moves that improve FULL
# nonfinite-rate / max-abs-logit while keeping VAL accuracy roughly intact.
JOINT_ENABLE_STABILITY_RESCUE = True
JOINT_STABILITY_RESCUE_MAX_VAL_DROP_PP = 0.50
JOINT_STABILITY_RESCUE_MIN_NONFINITE_DROP = 0.002
JOINT_STABILITY_RESCUE_MIN_ABSLOGIT_REL_DROP = 0.05

# Allow deterministic fallback neighborhood scan when random proposals all get filtered.
JOINT_DETERMINISTIC_SAMECOST_FALLBACK = True
JOINT_DETERMINISTIC_TOP_SITES = 12

# Allow small per-site cost changes in the local neighborhood.
# The final schedule is still filtered by the task's search-budget band.
JOINT_MAX_LOCAL_COST_DELTA = 2

# Adaptive VAL-accuracy guard.
# If the current schedule is already below the baseline-relative VAL floor,
# switch to a current-relative floor so joint search is not dead-on-arrival.
JOINT_ENABLE_ADAPTIVE_VAL_ACC_GUARD = True
JOINT_VAL_ACC_CURRENT_DROP_PP = 0.50
JOINT_ENABLE_CURRENT_VAL_ACC_GUARD = False

# Search-budget band.
# For minimum-feasible-budget tasks, allowing a small upward budget band helps the
# local search escape same-cost dead zones. You can also override per task with
# budget_min / budget_max / budget_band in JOINT_TASKS.
JOINT_ENABLE_AUTO_MIN_BUDGET_BAND = True
JOINT_AUTO_MIN_BUDGET_BAND_WIDTH = 16

# Per-step / per-pass rejection logging.
JOINT_LOG_REJECTION_STATS = True

# Local eval caches inside joint / global-CD (per task, keyed by active-site schedule tuple).
JOINT_ENABLE_VAL_CACHE = True
JOINT_ENABLE_FULL_CACHE = True
JOINT_ENABLE_GLOBAL_CD_FAST_CACHE = True
JOINT_ENABLE_GLOBAL_CD_VAL_CACHE = True
JOINT_ENABLE_GLOBAL_CD_FULL_CACHE = True
JOINT_LOG_STEP_TIMING = True

# Allow acc-driven acceptance even if MSE objective is slightly worse.
JOINT_ENABLE_ACC_OBJ_TRADEOFF = True
JOINT_ACC_GAIN_MIN_PP = 0.20
JOINT_OBJ_WORSEN_MAX_RATIO = 1.20
JOINT_MIN_OBJ_IMPROV_ABS = 0.0
JOINT_MIN_OBJ_IMPROV_REL = 0.0
JOINT_FINAL_OBJ_EPS_ABS = 0.0
JOINT_FINAL_OBJ_EPS_REL = 0.0

# Optional refinement after the DP seed.
RUN_JOINT_SEARCH = True
RUN_JOINT_GLOBAL_REFINEMENT = True
JOINT_CD_MAX_PASSES = 3
JOINT_CD_FAST_TOPK_VAL = 5
JOINT_CD_MAX_ACCEPTS_PER_PASS = 12
JOINT_ENABLE_CD_SITE_SUBSET = True
JOINT_CD_TOPM_IMPORTANCE = 12
JOINT_CD_SITE_SUBSET_MAX = 12
# Faster/more targeted global-CD: kind-priority site ordering + FULL-lite pre-gating.
JOINT_CD_KIND_PRIORITY = ["softmax", "gelu", "layernorm"]
JOINT_CD_SOFTMAX_SITE_LIMIT = 12
JOINT_CD_GELU_SITE_LIMIT = 0
JOINT_CD_LAYERNORM_SITE_LIMIT = 0
JOINT_CD_SOFTMAX_PASSES = 3
JOINT_CD_GELU_PASSES = 1
JOINT_CD_LAYERNORM_PASSES = 1
# When the softmax stage has already pushed FULL nonfinite samples down to this level,
# move on to GELU instead of continuing to spend passes on softmax.
JOINT_CD_STOP_SOFTMAX_IF_FULL_NF_LEQ = -1
# Skip the LayerNorm stage once FULL nonfinite samples are already low. This avoids
# spending most of global-CD time on low-value LN polish in the bert-base/all regime.
# Set to -1 to disable.
JOINT_CD_SKIP_LAYERNORM_IF_FULL_NF_LEQ = 4
JOINT_FULL_LITE_N = 96
JOINT_CD_FULL_LITE_TOPK = 1

# Final schedule selection on VAL only. TEST is report-only.
JOINT_FINAL_ACC_TOL_PP = 0.30

# Runtime search/eval profiles for curriculum experiments
SEARCH_PROFILE = "strict"  # relaxed | no_pos_clamp | no_sign_clip | strict
POST_EVAL_PROFILES = ["strict"]  # extra profiles evaluated on the in-memory best schedule

# Local-polish mutable-site focus. Empty => all active sites can change.
JOINT_MUTABLE_KIND_FILTER = ["softmax"]
JOINT_MUTABLE_SITES: List[str] = []
JOINT_PROXY_KIND_FILTER: List[str] = []



# =============================================================================
# Utilities
# =============================================================================

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_like(path: Path, sep: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        return list(reader)


def sample_indices(n: int, k: Optional[int], seed: int) -> List[int]:
    n = int(n)
    if k is None or k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    idx = np.array(rng.choice(n, size=int(k), replace=False), dtype=np.int64)
    idx.sort()
    return idx.tolist()


def split_train_indices(n: int, calib_n: int, sweep_n: int, val_n: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    n = int(n)
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int64)
    rng.shuffle(perm)

    c = min(int(calib_n), n)
    s = min(int(sweep_n), max(0, n - c))
    v = min(int(val_n), max(0, n - c - s))

    calib = np.sort(perm[:c]).tolist()
    sweep = np.sort(perm[c:c + s]).tolist()
    val = np.sort(perm[c + s:c + s + v]).tolist()
    return calib, sweep, val


def expand_range(a: float, b: float, mult: float, margin_frac: float, *, positive: bool = False) -> Tuple[float, float]:
    a = float(a)
    b = float(b)
    if (not math.isfinite(a)) or (not math.isfinite(b)):
        raise ValueError(f"Non-finite range: {(a, b)}")
    if b < a:
        a, b = b, a
    if abs(b - a) < 1e-12:
        half = max(abs(a), abs(b), 1.0) * 0.05
        a, b = a - half, b + half
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    half = half * float(mult) * (1.0 + float(margin_frac))
    lo = mid - half
    hi = mid + half
    if positive and STRICT_OFFLINE_POSITIVE_PRESERVE:
        lo = max(lo, float(POSITIVE_DOMAIN_EPS))
        hi = max(hi, lo + max(1e-6, 0.01 * max(lo, 1.0)))
    return float(lo), float(hi)


def depth_cost_from_degree(degree: int) -> int:
    d = int(degree)
    for lo, hi, cost in DEPTH_COST_RANGES:
        if lo <= d <= hi:
            return int(cost)
    raise ValueError(f"degree {degree} out of supported range table")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pp(delta: float) -> str:
    return f"{delta:+.2f} pp"


def ensure_local_path(path: Path, *, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


# =============================================================================
# Dataset loading (local files only)
# =============================================================================

@dataclass
class TextExample:
    text: str
    label: int


class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[TextExample],
        tokenizer: Any,
        max_length: int,
        lowercase: bool = False,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.lowercase = bool(lowercase)

        texts = [ex.text.lower() if self.lowercase else ex.text for ex in self.examples]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        self.lengths: List[int] = [int(len(ids)) for ids in enc["input_ids"]]
        self.items: List[Dict[str, Any]] = []
        for i, ex in enumerate(self.examples):
            row: Dict[str, Any] = {k: enc[k][i] for k in enc.keys()}
            row["labels"] = int(ex.label)
            self.items.append(row)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return dict(self.items[idx])


class HFCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = [dict(x) for x in batch]
        labels = torch.tensor([int(x.pop("labels")) for x in batch], dtype=torch.long)
        padded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded


def _normalize_rows(rows: Iterable[MutableMapping[str, Any]], *, text_col: str, label_col: str) -> List[TextExample]:
    out: List[TextExample] = []
    for row in rows:
        if text_col not in row or label_col not in row:
            continue
        text = str(row[text_col])
        lab_raw = row[label_col]
        if lab_raw in (None, "", "-1"):
            continue
        label = int(lab_raw)
        out.append(TextExample(text=text, label=label))
    return out


def _pick_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def load_local_text_splits(
    dataset_dir: Path,
    *,
    dataset_format: str,
    text_col: str,
    label_col: str,
) -> Tuple[List[TextExample], List[TextExample]]:
    dataset_dir = ensure_local_path(dataset_dir, what="dataset_dir")
    mode = dataset_format.lower()

    if mode in ("auto", "hf_disk") and hf_load_from_disk is not None:
        try:
            obj = hf_load_from_disk(str(dataset_dir))
            if hasattr(obj, "keys") and ("train" in obj):
                train_rows = obj["train"]
                val_key = "validation" if "validation" in obj else ("dev" if "dev" in obj else None)
                if val_key is None:
                    raise KeyError("validation/dev split not found in load_from_disk dataset")
                val_rows = obj[val_key]
                train = _normalize_rows(train_rows, text_col=text_col, label_col=label_col)
                val = _normalize_rows(val_rows, text_col=text_col, label_col=label_col)
                if train and val:
                    return train, val
        except Exception:
            if mode == "hf_disk":
                raise

    if mode in ("auto", "tsv"):
        train_p = _pick_existing(dataset_dir, TRAIN_FILE_CANDIDATES)
        val_p = _pick_existing(dataset_dir, VAL_FILE_CANDIDATES)
        if train_p is not None and val_p is not None and train_p.suffix.lower() == ".tsv" and val_p.suffix.lower() == ".tsv":
            train = _normalize_rows(read_csv_like(train_p, "\t"), text_col=text_col, label_col=label_col)
            val = _normalize_rows(read_csv_like(val_p, "\t"), text_col=text_col, label_col=label_col)
            if train and val:
                return train, val
            if mode == "tsv":
                raise RuntimeError(f"Failed to parse TSV dataset under {dataset_dir}")

    if mode in ("auto", "csv"):
        train_p = _pick_existing(dataset_dir, TRAIN_FILE_CANDIDATES)
        val_p = _pick_existing(dataset_dir, VAL_FILE_CANDIDATES)
        if train_p is not None and val_p is not None and train_p.suffix.lower() == ".csv" and val_p.suffix.lower() == ".csv":
            train = _normalize_rows(read_csv_like(train_p, ","), text_col=text_col, label_col=label_col)
            val = _normalize_rows(read_csv_like(val_p, ","), text_col=text_col, label_col=label_col)
            if train and val:
                return train, val
            if mode == "csv":
                raise RuntimeError(f"Failed to parse CSV dataset under {dataset_dir}")

    if mode in ("auto", "jsonl"):
        train_p = _pick_existing(dataset_dir, TRAIN_FILE_CANDIDATES)
        val_p = _pick_existing(dataset_dir, VAL_FILE_CANDIDATES)
        if train_p is not None and val_p is not None and train_p.suffix.lower() == ".jsonl" and val_p.suffix.lower() == ".jsonl":
            train = _normalize_rows(read_jsonl(train_p), text_col=text_col, label_col=label_col)
            val = _normalize_rows(read_jsonl(val_p), text_col=text_col, label_col=label_col)
            if train and val:
                return train, val
            if mode == "jsonl":
                raise RuntimeError(f"Failed to parse JSONL dataset under {dataset_dir}")

    raise RuntimeError(
        f"Could not load dataset from {dataset_dir}. Supported inputs: load_from_disk dataset or train/dev TSV/CSV/JSONL."
    )


class StaticIndexSampler(torch.utils.data.Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self.indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def dataset_length_list(ds: Dataset) -> Optional[List[int]]:
    if hasattr(ds, "lengths"):
        return [int(x) for x in getattr(ds, "lengths")]
    if isinstance(ds, Subset):
        base = dataset_length_list(ds.dataset)
        if base is None:
            return None
        return [int(base[i]) for i in ds.indices]
    return None


def maybe_length_sorted_sampler(ds: Dataset) -> Optional[StaticIndexSampler]:
    if not ENABLE_LENGTH_BUCKETING:
        return None
    lengths = dataset_length_list(ds)
    if lengths is None or len(lengths) == 0:
        return None
    order = np.argsort(np.asarray(lengths, dtype=np.int32), kind="stable")
    if not bool(LENGTH_BUCKET_ASCENDING):
        order = order[::-1]
    return StaticIndexSampler(order.tolist())


def _loader_kwargs(tokenizer: Any, batch_size: int, device: torch.device, *, ds: Optional[Dataset] = None, num_workers: Optional[int] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(NUM_WORKERS if num_workers is None else num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": HFCollator(tokenizer),
    }
    sampler = maybe_length_sorted_sampler(ds) if ds is not None else None
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    if int(kwargs["num_workers"]) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(AUTO_LOADER_PREFETCH_FACTOR)
    return kwargs


def build_loader(ds: Dataset, tokenizer: Any, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(ds, **_loader_kwargs(tokenizer, int(batch_size), device, ds=ds))


def probe_loader_batch_size(ds: Dataset, tokenizer: Any, micro_batch: int, device: torch.device) -> int:
    if (not AUTO_TUNE_LOADER_BATCH) or int(micro_batch) <= 0:
        return int(micro_batch)
    n = int(len(ds))
    if n <= 0:
        return int(micro_batch)
    bs = min(n, int(micro_batch) * int(AUTO_LOADER_START_MULT), int(micro_batch) * int(AUTO_LOADER_MAX_MULT))
    bs = max(int(micro_batch), int(bs))
    while bs >= int(micro_batch):
        try:
            loader = DataLoader(ds, **_loader_kwargs(tokenizer, bs, device, ds=ds, num_workers=0))
            _ = next(iter(loader))
            return int(bs)
        except Exception:
            bs //= 2
    return int(micro_batch)


def build_tuned_loader(ds: Dataset, tokenizer: Any, micro_batch: int, device: torch.device) -> Tuple[DataLoader, int]:
    loader_batch = probe_loader_batch_size(ds, tokenizer, micro_batch, device)
    return build_loader(ds, tokenizer, loader_batch, device), int(loader_batch)


def slice_batch(batch: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {k: v[start:end] for k, v in batch.items()}


# =============================================================================
# Candidate choice space
# =============================================================================

@dataclass(frozen=True)
class ApproxChoice:
    kind: str
    choice_id: int
    label: str
    cost: int
    params: Dict[str, Any]


def reciprocal_refine_depth(refine: str, iters: int) -> int:
    refine = str(refine).lower()
    if refine == "none" or int(iters) <= 0:
        return 0
    # Heuristic:
    #   Newton / Goldschmidt for 1/x need roughly two serial multiplications per iteration.
    return 2 * int(iters)


def _is_thor_inv_init_method(method: str) -> bool:
    return str(method).strip().lower() in {"thor", "thor_inv", "thor_heinv", "heinv", "he_inv"}


def softmax_inv_grid() -> List[Tuple[str, int, float, str, int, int]]:
    configs: List[Tuple[str, int, float, str, int, int]] = []
    for inv_init_method in SOFTMAX_INV_INIT_METHODS:
        inv_init_method = str(inv_init_method)
        if _is_thor_inv_init_method(inv_init_method):
            configs.append(
                (
                    inv_init_method,
                    0,
                    1.0,
                    "thor_heinv",
                    0,
                    int(SOFTMAX_THOR_HEINV_COST),
                )
            )
            continue
        for inv_init_degree in SOFTMAX_INV_INIT_DEGREES:
            if int(inv_init_degree) <= 0:
                continue
            for inv_range_mult in SOFTMAX_INV_RANGE_MULTS:
                for refine in SOFTMAX_INV_REFINES:
                    refine = str(refine)
                    iter_list = [0] if refine.lower() == "none" else SOFTMAX_INV_ITERS
                    for iters in iter_list:
                        if refine.lower() != "none" and int(iters) <= 0:
                            continue
                        inv_cost = depth_cost_from_degree(int(inv_init_degree)) + reciprocal_refine_depth(refine, int(iters))
                        configs.append(
                            (
                                inv_init_method,
                                int(inv_init_degree),
                                float(inv_range_mult),
                                refine,
                                int(iters),
                                int(inv_cost),
                            )
                        )
    return configs


def rsqrt_refine_depth(refine: str, iters: int) -> int:
    refine = str(refine).lower()
    if refine == "none" or int(iters) <= 0:
        return 0
    # Heuristic:
    #   y <- 0.5*y*(3 - x*y*y)  has ~3 serial multiplications in a straightforward circuit.
    return 3 * int(iters)


def sign_poly_depth(method: str, degree: int, compose_steps: int) -> int:
    method = str(method).lower()
    steps = max(1, int(compose_steps))
    if method == "cubic":
        # p(x) = 1.5x - 0.5x^3 has depth ~2; composition multiplies serial depth.
        return 2 * steps
    if method == "chebyshev":
        return depth_cost_from_degree(int(degree)) * steps
    raise ValueError(f"Unknown sign method: {method}")


def softmax_sign_shift_cost() -> int:
    per_level = sign_poly_depth(SOFTMAX_SIGN_METHOD, SOFTMAX_SIGN_CHEB_DEGREE, SOFTMAX_SIGN_COMPOSE_STEPS) + 1
    mode = str(SOFTMAX_SIGN_COST_MODE).lower()
    if mode == "tree":
        return per_level * max(1, int(SOFTMAX_SIGN_TREE_LOG2_LEN))
    if mode == "flat":
        return per_level
    raise ValueError(f"Unknown SOFTMAX_SIGN_COST_MODE: {SOFTMAX_SIGN_COST_MODE}")


def softmax_sign_label_suffix() -> str:
    method = str(SOFTMAX_SIGN_METHOD).lower()
    clip = int(bool(SOFTMAX_SIGN_CLIP_TO_UNIT))
    if method == "cubic":
        return f"sgncub-cs{int(SOFTMAX_SIGN_COMPOSE_STEPS)}-sm{SOFTMAX_SIGN_SCALE_MULT}-clip{clip}"
    return (
        f"sgncheb-d{int(SOFTMAX_SIGN_CHEB_DEGREE)}-cs{int(SOFTMAX_SIGN_COMPOSE_STEPS)}-"
        f"eps{SOFTMAX_SIGN_EPS}-sm{SOFTMAX_SIGN_SCALE_MULT}-clip{clip}"
    )


def _scale_token(scale: float) -> str:
    text = f"{float(scale):g}"
    return text.replace("-", "m").replace(".", "p")


def _is_thor_p2_exp_method(method: str) -> bool:
    return str(method).strip().lower() in {"thor", "thor_p2", "thor-p2", "thor_fixed", "thor-fixed"}


def softmax_exp_grid(exp_method: str) -> List[Tuple[int, float]]:
    if _is_thor_p2_exp_method(exp_method):
        return [(int(SOFTMAX_THOR_EXP_DEGREE), 1.0)]
    return [(int(d), float(rm)) for d in SOFTMAX_EXP_DEGREES for rm in SOFTMAX_EXP_RANGE_MULTS]


def softmax_exp_cost(exp_method: str, degree: int) -> int:
    if _is_thor_p2_exp_method(exp_method):
        return depth_cost_from_degree(int(SOFTMAX_THOR_EXP_DEGREE)) + 1
    return depth_cost_from_degree(int(degree))


def softmax_exp_label(method: str, degree: int, range_mult: float) -> str:
    if _is_thor_p2_exp_method(method):
        return "expthorP2"
    return f"exp{str(method)[:4]}{int(degree)}-erm{float(range_mult)}"


def scaled_softmax_scale_square_pairs() -> List[Tuple[float, int]]:
    scales = [float(s) for s in SOFTMAX_INPUT_SCALES if float(s) > 0.0]
    square_iters = [max(0, int(v)) for v in SOFTMAX_SQUARE_ITERS]
    mode = str(SOFTMAX_SCALE_SQUARE_MODE).strip().lower()
    if not scales:
        raise ValueError("SOFTMAX_INPUT_SCALES must contain at least one positive scale for scaled softmax")
    if mode == "explicit":
        pairs = [(float(scale), max(0, int(iters))) for scale, iters in SOFTMAX_SCALE_SQUARE_PAIRS]
        pairs = [(scale, iters) for scale, iters in pairs if scale > 0.0]
        if not pairs:
            raise ValueError("SOFTMAX_SCALE_SQUARE_PAIRS must be non-empty when mode is explicit")
        return list(dict.fromkeys(pairs))
    if mode != "matched":
        raise ValueError(
            f"Unknown SOFTMAX_SCALE_SQUARE_MODE={SOFTMAX_SCALE_SQUARE_MODE!r}; "
            "valid modes are matched, explicit"
        )

    allowed_iters = set(square_iters)
    pairs: List[Tuple[float, int]] = []
    for scale in scales:
        log2_scale = math.log2(float(scale))
        matched_iters = int(round(log2_scale))
        if abs(log2_scale - matched_iters) > 1e-9:
            continue
        if matched_iters in allowed_iters:
            pairs.append((float(scale), matched_iters))
    if not pairs:
        raise ValueError(
            "No THOR-matched scaled-softmax pairs found. "
            "Use scale=2^square_iters, e.g. SOFTMAX_INPUT_SCALES=2,4,8 and "
            "SOFTMAX_SQUARE_ITERS=1,2,3, or set explicit SOFTMAX_SCALE_SQUARE_PAIRS."
        )
    return list(dict.fromkeys(pairs))


def build_choice_space() -> Dict[str, List[ApproxChoice]]:
    out: Dict[str, List[ApproxChoice]] = {}

    gelu_choices: List[ApproxChoice] = []
    cid = 0
    for degree in GELU_DEGREES:
        for range_mult in GELU_RANGE_MULTS:
            gelu_choices.append(
                ApproxChoice(
                    kind="gelu",
                    choice_id=cid,
                    label=f"gelu-cheb-d{degree}-rm{range_mult}",
                    cost=depth_cost_from_degree(int(degree)),
                    params={
                        "method": "chebyshev",
                        "degree": int(degree),
                        "range_mult": float(range_mult),
                    },
                )
            )
            cid += 1
    out["gelu"] = gelu_choices

    softmax_choices: List[ApproxChoice] = []
    cid = 0
    for shift_mode in SOFTMAX_SHIFT_MODES:
        shift_mode = str(shift_mode).lower()
        shift_cost = softmax_sign_shift_cost() if shift_mode == "sign_max" else 0
        sign_suffix = f"-{softmax_sign_label_suffix()}" if shift_mode == "sign_max" else ""
        scale_square_pairs = scaled_softmax_scale_square_pairs() if shift_mode == "scaled" else [(1.0, 0)]
        for exp_method in SOFTMAX_EXP_METHODS:
            exp_method = str(exp_method)
            for exp_degree, exp_range_mult in softmax_exp_grid(exp_method):
                for inv_init_method, inv_init_degree, inv_range_mult, refine, iters, inv_cost in softmax_inv_grid():
                    for input_scale, square_iters in scale_square_pairs:
                        scaled_cost = int(square_iters) * (inv_cost + 2)
                        cost = (
                            shift_cost
                            + softmax_exp_cost(exp_method, int(exp_degree))
                            + inv_cost
                            + 1  # initial numerator * inv(den)
                            + scaled_cost
                        )
                        choice_id = cid
                        cid += 1
                        if SOFTMAX_COST_MIN is not None and cost < int(SOFTMAX_COST_MIN):
                            continue
                        if SOFTMAX_COST_MAX is not None and cost > int(SOFTMAX_COST_MAX):
                            continue
                        scaled_suffix = ""
                        if shift_mode == "scaled":
                            scaled_suffix = f"-s{_scale_token(float(input_scale))}-sq{int(square_iters)}"
                        if _is_thor_inv_init_method(inv_init_method):
                            inv_label = "invthor-heinv"
                        else:
                            inv_label = f"inv{inv_init_method[:4]}{inv_init_degree}-{refine}{iters}-irm{inv_range_mult}"
                        label = (
                            f"smx-shift{shift_mode}{sign_suffix}{scaled_suffix}-{softmax_exp_label(exp_method, int(exp_degree), float(exp_range_mult))}-"
                            f"{inv_label}"
                        )
                        if SOFTMAX_LABEL_ALLOW_REGEX and re.search(SOFTMAX_LABEL_ALLOW_REGEX, label) is None:
                            continue
                        params = {
                            "shift_mode": str(shift_mode),
                            "exp_method": str(exp_method),
                            "exp_degree": int(exp_degree),
                            "exp_range_mult": float(exp_range_mult),
                            "inv_init_method": str(inv_init_method),
                            "inv_init_degree": int(inv_init_degree),
                            "inv_range_mult": float(inv_range_mult),
                            "inv_refine": str(refine),
                            "inv_iters": int(iters),
                        }
                        if shift_mode == "scaled":
                            params.update(
                                {
                                    "input_scale": float(input_scale),
                                    "square_iters": int(square_iters),
                                }
                            )
                        if shift_mode == "sign_max":
                            params.update(
                                {
                                    "sign_method": str(SOFTMAX_SIGN_METHOD),
                                    "sign_degree": int(SOFTMAX_SIGN_CHEB_DEGREE),
                                    "sign_eps": float(SOFTMAX_SIGN_EPS),
                                    "sign_scale_mult": float(SOFTMAX_SIGN_SCALE_MULT),
                                    "sign_compose_steps": int(SOFTMAX_SIGN_COMPOSE_STEPS),
                                    "sign_clip_to_unit": bool(SOFTMAX_SIGN_CLIP_TO_UNIT),
                                    "sign_fit_points": int(SOFTMAX_SIGN_FIT_POINTS),
                                }
                            )
                        softmax_choices.append(
                            ApproxChoice(
                                kind="softmax",
                                choice_id=choice_id,
                                label=label,
                                cost=cost,
                                params=params,
                            )
                        )
    if not softmax_choices:
        raise ValueError(
            "Softmax choice space is empty after applying SOFTMAX_COST_MIN/MAX and label filtering. "
            f"Current bounds: min={SOFTMAX_COST_MIN}, max={SOFTMAX_COST_MAX}, "
            f"label_allow_regex={SOFTMAX_LABEL_ALLOW_REGEX!r}."
        )
    out["softmax"] = softmax_choices

    ln_choices: List[ApproxChoice] = []
    cid = 0
    for init_method in LAYERNORM_INIT_METHODS:
        for init_degree in LAYERNORM_INIT_DEGREES:
            for range_mult in LAYERNORM_RANGE_MULTS:
                for refine in LAYERNORM_REFINES:
                    iter_list = [0] if refine == "none" else LAYERNORM_ITERS
                    for iters in iter_list:
                        if refine != "none" and int(iters) <= 0:
                            continue
                        cost = (
                            1  # square in variance
                            + depth_cost_from_degree(int(init_degree))
                            + rsqrt_refine_depth(refine, int(iters))
                            + 1  # centered_x * invstd
                        )
                        label = f"ln-{init_method[:4]}{init_degree}-{refine}{iters}-rm{range_mult}"
                        ln_choices.append(
                            ApproxChoice(
                                kind="layernorm",
                                choice_id=cid,
                                label=label,
                                cost=cost,
                                params={
                                    "init_method": str(init_method),
                                    "init_degree": int(init_degree),
                                    "range_mult": float(range_mult),
                                    "refine": str(refine),
                                    "iters": int(iters),
                                },
                            )
                        )
                        cid += 1
    out["layernorm"] = ln_choices

    return out


# =============================================================================
# Calibration store
# =============================================================================

@dataclass
class RunningMinMax:
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        x = tensor.detach().float()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return
        mn = float(x.amin().item())
        mx = float(x.amax().item())
        if mn < self.min_val:
            self.min_val = mn
        if mx > self.max_val:
            self.max_val = mx

    @property
    def is_valid(self) -> bool:
        return math.isfinite(self.min_val) and math.isfinite(self.max_val) and self.max_val >= self.min_val


@dataclass
class QuantileSampleCollector:
    capacity: int = SOFTMAX_CALIB_MAX_SAMPLE_CAPACITY
    per_update_limit: int = SOFTMAX_CALIB_MAX_PER_UPDATE_LIMIT
    values: List[np.ndarray] = field(default_factory=list)
    n_values: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        x = tensor.detach().float()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return
        arr = x.flatten().cpu().numpy().astype(np.float64, copy=False)
        if arr.size > int(self.per_update_limit):
            idx = np.random.choice(arr.size, size=int(self.per_update_limit), replace=False)
            arr = arr[idx]
        self.values.append(arr)
        self.n_values += int(arr.size)
        if self.n_values > int(self.capacity):
            merged = np.concatenate(self.values, axis=0)
            if merged.size > int(self.capacity):
                idx = np.random.choice(merged.size, size=int(self.capacity), replace=False)
                merged = merged[idx]
            self.values = [merged]
            self.n_values = int(merged.size)

    def quantile(self, q: float) -> Optional[float]:
        if self.n_values <= 0:
            return None
        merged = np.concatenate(self.values, axis=0)
        merged = merged[np.isfinite(merged)]
        if merged.size == 0:
            return None
        qq = min(max(float(q), 0.0), 1.0)
        return float(np.quantile(merged, qq))


class CalibrationBook:
    def __init__(self):
        self._store: Dict[str, Dict[str, RunningMinMax]] = defaultdict(dict)

    def update(self, site: str, channel: str, tensor: torch.Tensor) -> None:
        if channel not in self._store[site]:
            self._store[site][channel] = RunningMinMax()
        self._store[site][channel].update(tensor)

    def get(self, site: str, channel: str) -> Optional[RunningMinMax]:
        return self._store.get(site, {}).get(channel, None)

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for site, chans in self._store.items():
            for channel, st in chans.items():
                rows.append(
                    {
                        "site": site,
                        "channel": channel,
                        "min": st.min_val,
                        "max": st.max_val,
                    }
                )
        return rows


class ApproxController:
    def __init__(self, choices_by_kind: Dict[str, List[ApproxChoice]]):
        self.choices_by_kind = choices_by_kind
        self.site_kind: "OrderedDict[str, str]" = OrderedDict()
        self.schedule: Dict[str, int] = {}   # site -> local choice_id for its kind
        self.calibration = CalibrationBook()
        self._quantile_samples: Dict[Tuple[str, str], QuantileSampleCollector] = {}
        self.collect_calibration: bool = False

    def register_site(self, site: str, kind: str) -> None:
        if site not in self.site_kind:
            self.site_kind[site] = kind

    def ordered_sites(self) -> List[str]:
        return list(self.site_kind.keys())

    def set_exact(self) -> None:
        self.schedule = {}

    def set_schedule(self, schedule: Dict[str, int]) -> None:
        self.schedule = dict(schedule)

    def is_exact(self, site: str) -> bool:
        return site not in self.schedule

    def get_kind(self, site: str) -> str:
        return self.site_kind[site]

    def get_choice(self, site: str) -> Optional[ApproxChoice]:
        if self.is_exact(site):
            return None
        kind = self.get_kind(site)
        local_id = int(self.schedule[site])
        choices = self.choices_by_kind[kind]
        for ch in choices:
            if int(ch.choice_id) == local_id:
                return ch
        raise KeyError(f"choice_id={local_id} not found for site={site}, kind={kind}")

    def choices_for_site(self, site: str) -> List[ApproxChoice]:
        return self.choices_by_kind[self.get_kind(site)]

    def observe(self, site: str, channel: str, tensor: torch.Tensor) -> None:
        if self.collect_calibration:
            self.calibration.update(site, channel, tensor)

    def observe_quantile(self, site: str, channel: str, tensor: torch.Tensor) -> None:
        if not self.collect_calibration:
            return
        key = (str(site), str(channel))
        if key not in self._quantile_samples:
            self._quantile_samples[key] = QuantileSampleCollector()
        self._quantile_samples[key].update(tensor)

    def quantile_for(self, site: str, channel: str, q: float) -> Optional[float]:
        collector = self._quantile_samples.get((str(site), str(channel)))
        if collector is None:
            return None
        return collector.quantile(float(q))

    def range_for(self, site: str, channel: str, range_mult: float) -> Tuple[float, float]:
        if channel == "exp::calib_max":
            score_st = self.calibration.get(site, "calib_max::score")
            max_st = self.calibration.get(site, "calib_max::row_max")
            if score_st is not None and score_st.is_valid and max_st is not None and max_st.is_valid:
                shift = calibrated_layer_max_value(site, self)
                return expand_range(
                    float(score_st.min_val) - shift,
                    float(score_st.max_val) - shift,
                    range_mult,
                    RANGE_MARGIN_FRAC,
                    positive=False,
                )

        st = self.calibration.get(site, channel)
        positive = (channel.startswith("den::") or channel.startswith("den_square::") or channel == "var")
        if st is not None and st.is_valid:
            return expand_range(st.min_val, st.max_val, range_mult, RANGE_MARGIN_FRAC, positive=positive)

        if site in self.site_kind:
            kind = self.site_kind[site]
            if kind == "gelu":
                key = "gelu::x"
            elif kind == "softmax":
                if channel.startswith("sign::"):
                    key = "softmax_sign::delta"
                elif channel.startswith("calib_max::"):
                    key = f"softmax_calib_max::{channel.split('::', 1)[1]}"
                elif channel.startswith("den_square::"):
                    key = "softmax_den_square::scaled"
                elif channel.startswith("exp::scaled"):
                    key = "softmax_exp::scaled"
                elif channel.startswith("den::scaled"):
                    key = "softmax_den::scaled"
                else:
                    key = f"softmax_{'den' if channel.startswith('den::') else 'exp'}::{channel.split('::', 1)[1]}"
            elif kind == "layernorm":
                key = "layernorm::var"
            else:
                key = ""
            if key and key in RANGE_FALLBACKS:
                a, b = RANGE_FALLBACKS[key]
                return expand_range(a, b, range_mult, RANGE_MARGIN_FRAC, positive=positive)

        # final fallback
        if positive:
            return (POSITIVE_DOMAIN_EPS, 1.0)
        return (-1.0, 1.0)


# =============================================================================
# Polynomial approximations
# =============================================================================

_COEFF_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def scalar_gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def scalar_exp(x: float) -> float:
    return math.exp(x)


def scalar_reciprocal(x: float) -> float:
    return 1.0 / x


def scalar_rsqrt(x: float) -> float:
    return 1.0 / math.sqrt(x)


SCALAR_FN_REGISTRY: Dict[str, Callable[[float], float]] = {
    "gelu": scalar_gelu,
    "exp": scalar_exp,
    "reciprocal": scalar_reciprocal,
    "rsqrt": scalar_rsqrt,
}


def chebyshev_coefficients(func_name: str, a: float, b: float, degree: int) -> torch.Tensor:
    fn = SCALAR_FN_REGISTRY[func_name]
    n = int(degree) + 1
    if b <= a:
        raise ValueError(f"Invalid interval: {(a, b)}")
    ba = 0.5 * (b - a)
    aa = 0.5 * (b + a)
    pi_by_n = math.pi / n

    fvals = []
    for i in range(n):
        x = math.cos(pi_by_n * (i + 0.5)) * ba + aa
        fvals.append(fn(float(x)))

    coeffs = [0.0] * n
    mult = 2.0 / n
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += fvals[j] * math.cos(pi_by_n * k * (j + 0.5))
        coeffs[k] = s * mult
    return torch.tensor(coeffs, dtype=torch.float64)


def taylor_shifted_coefficients(func_name: str, center: float, degree: int) -> torch.Tensor:
    c = float(center)
    d = int(degree)
    if func_name == "exp":
        coeffs = [math.exp(c) / math.factorial(n) for n in range(d + 1)]
    elif func_name == "reciprocal":
        coeffs = [(((-1.0) ** n) / (c ** (n + 1))) for n in range(d + 1)]
    elif func_name == "rsqrt":
        coeffs = []
        binom_term = 1.0
        for n in range(d + 1):
            if n == 0:
                binom_term = 1.0
            else:
                binom_term *= -((2.0 * n - 1.0) / (2.0 * n))
            coeffs.append(binom_term * (c ** (-0.5 - n)))
    else:
        raise ValueError(f"Taylor is not supported for {func_name}")
    return torch.tensor(coeffs, dtype=torch.float64)


def get_coeffs_cached(key: Tuple[Any, ...], builder: Callable[[], torch.Tensor]) -> torch.Tensor:
    if key not in _COEFF_CACHE:
        _COEFF_CACHE[key] = builder()
    return _COEFF_CACHE[key]


def eval_chebyshev_series(x: torch.Tensor, coeffs: torch.Tensor, a: float, b: float) -> torch.Tensor:
    if b <= a:
        raise ValueError(f"Invalid interval: {(a, b)}")
    t = (2.0 * x - (a + b)) / (b - a)
    n = coeffs.numel() - 1
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for k in range(n, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0
    return t * b1 - b2 + 0.5 * coeffs[0]


def eval_shifted_power_series(x: torch.Tensor, coeffs: torch.Tensor, center: float) -> torch.Tensor:
    z = x - float(center)
    y = torch.zeros_like(x, dtype=x.dtype)
    for c in reversed(coeffs):
        y = y * z + c.to(dtype=x.dtype, device=x.device)
    return y


def approx_function(x: torch.Tensor, func_name: str, method: str, degree: int, a: float, b: float) -> torch.Tensor:
    method = str(method).lower()
    if method == "chebyshev":
        key = ("cheb", func_name, int(degree), round(float(a), 6), round(float(b), 6))
        coeffs = get_coeffs_cached(key, lambda: chebyshev_coefficients(func_name, float(a), float(b), int(degree)))
        coeffs = coeffs.to(device=x.device, dtype=x.dtype)
        return eval_chebyshev_series(x, coeffs, float(a), float(b))
    if method == "taylor":
        center = 0.5 * (float(a) + float(b))
        key = ("taylor", func_name, int(degree), round(float(center), 6))
        coeffs = get_coeffs_cached(key, lambda: taylor_shifted_coefficients(func_name, float(center), int(degree)))
        coeffs = coeffs.to(device=x.device, dtype=x.dtype)
        return eval_shifted_power_series(x, coeffs, float(center))
    raise ValueError(f"Unknown approx method: {method}")


def eval_power_series_ascending(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x, dtype=x.dtype)
    for c in reversed(coeffs):
        y = y * x + c.to(dtype=x.dtype, device=x.device)
    return y


def bert_layer_index_from_softmax_site(site_name: str) -> Optional[int]:
    m = re.search(r"bert\.encoder\.layer\.(\d+)\.attention\.self\.softmax$", str(site_name))
    if m is None:
        return None
    return int(m.group(1))


def thor_p2_softmax_exp(x: torch.Tensor, site_name: str) -> torch.Tensor:
    layer_idx = bert_layer_index_from_softmax_site(site_name)
    if layer_idx == int(SOFTMAX_THOR_EXP_LAYER2_INDEX):
        x0 = x
        if SOFTMAX_THOR_EXP_CLAMP_INPUT:
            x0 = torch.clamp(x0, min=float(SOFTMAX_THOR_EXP2_PLAIN_MIN), max=float(SOFTMAX_THOR_EXP2_PLAIN_MAX))
        z = (4.0 * x0 - float(SOFTMAX_THOR_EXP2_MID_X)) / 64.0
        coeffs = get_coeffs_cached(
            ("thor_p2_exp2",),
            lambda: torch.tensor(SOFTMAX_THOR_EXP2_COEFFS_ASC, dtype=torch.float64),
        )
    else:
        x0 = x
        if SOFTMAX_THOR_EXP_CLAMP_INPUT:
            x0 = torch.clamp(x0, min=float(SOFTMAX_THOR_EXP1_PLAIN_MIN), max=float(SOFTMAX_THOR_EXP1_PLAIN_MAX))
        z = (2.0 * x0 - float(SOFTMAX_THOR_EXP1_MID_X)) / 32.0
        coeffs = get_coeffs_cached(
            ("thor_p2_exp1",),
            lambda: torch.tensor(SOFTMAX_THOR_EXP1_COEFFS_ASC, dtype=torch.float64),
        )
    y = eval_power_series_ascending(z, coeffs.to(device=x.device, dtype=x.dtype))
    out = y * y
    if SOFTMAX_THOR_EXP_CLAMP_OUTPUT_MIN:
        out = torch.clamp(out, min=1e-30)
    return out


def maybe_positive_domain(x: torch.Tensor) -> torch.Tensor:
    if not CLAMP_POSITIVE_DOMAIN:
        return x
    return torch.clamp(x, min=float(POSITIVE_DOMAIN_EPS))


def approximate_reciprocal(
    x: torch.Tensor,
    *,
    init_method: str,
    init_degree: int,
    a: float,
    b: float,
    refine: str,
    iters: int,
) -> torch.Tensor:
    x0 = maybe_positive_domain(x)
    y = approx_function(x0, "reciprocal", init_method, int(init_degree), float(a), float(b))
    refine = str(refine).lower()
    if refine == "none" or int(iters) <= 0:
        return y
    if refine == "newton":
        for _ in range(int(iters)):
            y = y * (2.0 - x0 * y)
        return y
    if refine == "goldschmidt":
        for _ in range(int(iters)):
            g = x0 * y
            h = 2.0 - g
            y = y * h
        return y
    raise ValueError(f"Unknown reciprocal refine method: {refine}")


def thor_inv_epsilon_for_site(site_name: str) -> float:
    layer_idx = bert_layer_index_from_softmax_site(site_name)
    if layer_idx == int(SOFTMAX_THOR_EXP_LAYER2_INDEX):
        return float(SOFTMAX_THOR_INV_EPS2)
    return float(SOFTMAX_THOR_INV_EPS1)


def thor_he_inv_reciprocal(
    x: torch.Tensor,
    *,
    site_name: str,
    alpha: float,
    epsilon: Optional[float] = None,
) -> torch.Tensor:
    a = torch.ones_like(x, dtype=torch.float64)
    b = maybe_positive_domain(x).to(torch.float64)
    en = float(thor_inv_epsilon_for_site(site_name) if epsilon is None else epsilon)
    alpha = float(alpha)
    while en < 1.0 - alpha:
        kn = 2.0 / (en + 1.0)
        factor = kn * (2.0 - kn * b)
        a = a * factor
        b = b * factor
        en = kn * en * (2.0 - kn * en)
    return a.to(dtype=x.dtype, device=x.device)


def reciprocal_from_softmax_choice(
    site_name: str,
    controller: ApproxController,
    x: torch.Tensor,
    p: Dict[str, Any],
    *,
    den_channel: str,
    is_final_square: bool = False,
) -> torch.Tensor:
    if _is_thor_inv_init_method(str(p["inv_init_method"])):
        alpha_base = float(SOFTMAX_THOR_OUTPUT_ALPHA if is_final_square else SOFTMAX_THOR_INTERNAL_ALPHA)
        return thor_he_inv_reciprocal(
            x,
            site_name=site_name,
            alpha=alpha_base / 10.0,
        )
    inv_a, inv_b = controller.range_for(site_name, den_channel, float(p["inv_range_mult"]))
    return approximate_reciprocal(
        x,
        init_method=str(p["inv_init_method"]),
        init_degree=int(p["inv_init_degree"]),
        a=inv_a,
        b=inv_b,
        refine=str(p["inv_refine"]),
        iters=int(p["inv_iters"]),
    )


def approximate_rsqrt(
    x: torch.Tensor,
    *,
    init_method: str,
    init_degree: int,
    a: float,
    b: float,
    refine: str,
    iters: int,
) -> torch.Tensor:
    x0 = maybe_positive_domain(x)
    y = approx_function(x0, "rsqrt", init_method, int(init_degree), float(a), float(b))
    refine = str(refine).lower()
    if refine == "none" or int(iters) <= 0:
        return y
    if refine == "newton":
        for _ in range(int(iters)):
            y = 0.5 * y * (3.0 - x0 * y * y)
        return y
    if refine == "goldschmidt":
        for _ in range(int(iters)):
            g = x0 * y * y
            h = 0.5 * (3.0 - g)
            y = y * h
        return y
    raise ValueError(f"Unknown rsqrt refine method: {refine}")


# =============================================================================
# Sign-based HE-softmax helpers (plain simulator)
# =============================================================================

@dataclass(frozen=True)
class SignApproxConfig:
    method: str = "cubic"          # cubic | chebyshev
    degree: int = 15               # chebyshev degree
    eps: float = 0.05              # chebyshev fit excludes [-eps, +eps]
    scale: float = 8.0             # sign_approx(x / scale)
    compose_steps: int = 5         # repeated self-composition count
    clip_to_unit: bool = True
    fit_points: int = 4096


def sign_chebyshev_coefficients(degree: int, eps: float, fit_points: int) -> torch.Tensor:
    degree = int(degree)
    eps = float(eps)
    fit_points = int(fit_points)
    if degree < 1:
        raise ValueError("sign chebyshev degree must be >= 1")
    if not (0.0 < eps < 1.0):
        raise ValueError("sign chebyshev eps must lie in (0, 1)")
    n_each = max(256, fit_points // 2)
    xs_left = np.linspace(-1.0, -eps, n_each, endpoint=True)
    xs_right = np.linspace(eps, 1.0, n_each, endpoint=True)
    xs = np.concatenate([xs_left, xs_right], axis=0)
    ys = np.sign(xs)
    coeffs = np.polynomial.chebyshev.chebfit(xs, ys, deg=degree)
    coeffs[0::2] = 0.0  # enforce odd symmetry (including c0)
    return torch.tensor(coeffs, dtype=torch.float64)


def eval_chebyshev_series_unit(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    n = coeffs.numel() - 1
    b1 = torch.zeros_like(x)
    b2 = torch.zeros_like(x)
    for k in range(n, 0, -1):
        b0 = 2.0 * x * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0
    return x * b1 - b2 + 0.5 * coeffs[0]


def base_sign_poly(x: torch.Tensor, cfg: SignApproxConfig) -> torch.Tensor:
    z = torch.clamp(x, -1.0, 1.0) if bool(cfg.clip_to_unit) else x
    method = str(cfg.method).lower()
    if method == "cubic":
        return 1.5 * z - 0.5 * z * z * z
    if method == "chebyshev":
        key = ("sign-cheb", int(cfg.degree), round(float(cfg.eps), 6), int(cfg.fit_points))
        coeffs = get_coeffs_cached(key, lambda: sign_chebyshev_coefficients(int(cfg.degree), float(cfg.eps), int(cfg.fit_points)))
        coeffs = coeffs.to(device=z.device, dtype=z.dtype)
        return eval_chebyshev_series_unit(z, coeffs)
    raise ValueError(f"Unknown sign approximation method: {cfg.method}")


def sign_plain(x: torch.Tensor, cfg: SignApproxConfig) -> torch.Tensor:
    scale = max(float(cfg.scale), float(POSITIVE_DOMAIN_EPS))
    y = x / scale
    steps = max(1, int(cfg.compose_steps))
    y = base_sign_poly(y, cfg)
    for _ in range(steps - 1):
        y = base_sign_poly(y, cfg)
    return y


def pairwise_max_sign_plain(a: torch.Tensor, b: torch.Tensor, cfg: SignApproxConfig) -> torch.Tensor:
    s = sign_plain(a - b, cfg)
    return 0.5 * (a + b + (a - b) * s)


def normalize_mask_like(mask_like: Optional[torch.Tensor], scores: torch.Tensor) -> Optional[torch.Tensor]:
    if mask_like is None:
        return None
    mask = mask_like.to(device=scores.device, dtype=torch.bool)
    if mask.shape != scores.shape:
        mask = mask.expand_as(scores)
    return mask


def public_masked_softmax_eval_input(z: torch.Tensor, mask_like: Optional[torch.Tensor]) -> torch.Tensor:
    if mask_like is None:
        return z
    # The BERT attention mask is public. Select a benign public input before the
    # exp polynomial so masked slots do not hide huge intermediate values that
    # would be problematic in HE. The numerator is still zeroed after evaluation.
    return torch.where(mask_like, torch.zeros_like(z), z)


def exact_row_max_lastdim(scores: torch.Tensor, mask_like: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask_like = normalize_mask_like(mask_like, scores)
    if mask_like is None:
        return scores.amax(dim=-1, keepdim=True)
    masked = scores.masked_fill(mask_like, float("-inf"))
    maxv = masked.amax(dim=-1, keepdim=True)
    return torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))


def upper_margin_value(value: float, margin_mult: float, margin_abs: float) -> float:
    v = float(value)
    mult = max(float(margin_mult), float(POSITIVE_DOMAIN_EPS))
    if mult != 1.0:
        v = v * mult if v >= 0.0 else v / mult
    return float(v + max(float(margin_abs), 0.0))


def calibrated_layer_max_value(site_name: str, controller: ApproxController) -> float:
    qv = controller.quantile_for(site_name, "calib_max::row_max", float(SOFTMAX_CALIB_MAX_QUANTILE))
    if qv is not None:
        return upper_margin_value(qv, SOFTMAX_CALIB_MAX_MARGIN_MULT, SOFTMAX_CALIB_MAX_MARGIN_ABS)
    st = controller.calibration.get(site_name, "calib_max::row_max")
    if st is not None and st.is_valid:
        return upper_margin_value(float(st.max_val), SOFTMAX_CALIB_MAX_MARGIN_MULT, SOFTMAX_CALIB_MAX_MARGIN_ABS)
    return upper_margin_value(
        float(RANGE_FALLBACKS["softmax_calib_max::row_max"][1]),
        SOFTMAX_CALIB_MAX_MARGIN_MULT,
        SOFTMAX_CALIB_MAX_MARGIN_ABS,
    )


def apply_calibrated_layer_max_shift(
    site_name: str,
    controller: ApproxController,
    scores: torch.Tensor,
) -> torch.Tensor:
    maxv = torch.as_tensor(
        calibrated_layer_max_value(site_name, controller),
        device=scores.device,
        dtype=scores.dtype,
    )
    return scores - maxv


def scaled_softmax_exp_channel(scale: float) -> str:
    return f"exp::scaled::s{_scale_token(float(scale))}"


def softmax_num_family(exp_method: str) -> str:
    return "thor_p2" if _is_thor_p2_exp_method(exp_method) else "exact_exp"


def scaled_softmax_den_channel(scale: float, exp_method: str = "exact_exp") -> str:
    channel = f"den::scaled::s{_scale_token(float(scale))}"
    family = softmax_num_family(exp_method)
    if family == "exact_exp":
        return channel
    return f"{channel}::{family}"


def scaled_softmax_square_den_channel(scale: float, step: int, exp_method: str = "exact_exp") -> str:
    channel = f"den_square::scaled::s{_scale_token(float(scale))}::step{int(step)}"
    family = softmax_num_family(exp_method)
    if family == "exact_exp":
        return channel
    return f"{channel}::{family}"


def tree_max_sign_lastdim(
    x: torch.Tensor,
    *,
    sign_cfg: SignApproxConfig,
    valid_mask: Optional[torch.Tensor] = None,
    keepdim: bool = True,
) -> torch.Tensor:
    cur_x = x
    cur_valid = None if valid_mask is None else valid_mask.to(device=x.device, dtype=torch.bool)
    while cur_x.size(-1) > 1:
        n = int(cur_x.size(-1))
        pair_n = n // 2
        paired_x = cur_x[..., : 2 * pair_n].reshape(*cur_x.shape[:-1], pair_n, 2)
        a = paired_x[..., 0]
        b = paired_x[..., 1]
        approx = pairwise_max_sign_plain(a, b, sign_cfg)
        if cur_valid is None:
            reduced_x = approx
            reduced_valid = None
        else:
            paired_valid = cur_valid[..., : 2 * pair_n].reshape(*cur_valid.shape[:-1], pair_n, 2)
            va = paired_valid[..., 0]
            vb = paired_valid[..., 1]
            both = va & vb
            only_a = va & (~vb)
            only_b = vb & (~va)
            zero = torch.zeros_like(approx)
            reduced_x = torch.where(only_a, a, torch.where(only_b, b, torch.where(both, approx, zero)))
            reduced_valid = va | vb
        if n % 2 == 1:
            tail_x = cur_x[..., -1:]
            cur_x = torch.cat([reduced_x, tail_x], dim=-1)
            if cur_valid is not None:
                tail_valid = cur_valid[..., -1:]
                cur_valid = torch.cat([reduced_valid, tail_valid], dim=-1)
        else:
            cur_x = reduced_x
            if cur_valid is not None:
                cur_valid = reduced_valid
    return cur_x if keepdim else cur_x.squeeze(-1)


def sign_softmax_global_config_fallback() -> SignApproxConfig:
    scale = max(
        abs(float(RANGE_FALLBACKS["softmax_sign::delta"][0])),
        abs(float(RANGE_FALLBACKS["softmax_sign::delta"][1])),
        float(POSITIVE_DOMAIN_EPS),
    )
    return SignApproxConfig(
        method=str(SOFTMAX_SIGN_METHOD),
        degree=int(SOFTMAX_SIGN_CHEB_DEGREE),
        eps=float(SOFTMAX_SIGN_EPS),
        scale=float(scale),
        compose_steps=int(SOFTMAX_SIGN_COMPOSE_STEPS),
        clip_to_unit=bool(SOFTMAX_SIGN_CLIP_TO_UNIT),
        fit_points=int(SOFTMAX_SIGN_FIT_POINTS),
    )


def tree_max_exact_lastdim_collect_delta(
    site_name: str,
    controller: ApproxController,
    x: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    keepdim: bool = True,
) -> torch.Tensor:
    # Offline calibration helper:
    # collect sign::delta from exact pairwise differences and reduce with exact max.
    # This helper is not part of the simulated online HE forward.
    cur_x = x
    cur_valid = None if valid_mask is None else valid_mask.to(device=x.device, dtype=torch.bool)
    while cur_x.size(-1) > 1:
        n = int(cur_x.size(-1))
        pair_n = n // 2
        paired_x = cur_x[..., : 2 * pair_n].reshape(*cur_x.shape[:-1], pair_n, 2)
        a = paired_x[..., 0]
        b = paired_x[..., 1]
        diff = a - b
        exact_max = torch.maximum(a, b)
        if cur_valid is None:
            controller.observe(site_name, "sign::delta", diff)
            reduced_x = exact_max
            reduced_valid = None
        else:
            paired_valid = cur_valid[..., : 2 * pair_n].reshape(*cur_valid.shape[:-1], pair_n, 2)
            va = paired_valid[..., 0]
            vb = paired_valid[..., 1]
            both = va & vb
            only_a = va & (~vb)
            only_b = vb & (~va)
            controller.observe(site_name, "sign::delta", diff[both])
            zero = torch.zeros_like(exact_max)
            reduced_x = torch.where(only_a, a, torch.where(only_b, b, torch.where(both, exact_max, zero)))
            reduced_valid = va | vb
        if n % 2 == 1:
            tail_x = cur_x[..., -1:]
            cur_x = torch.cat([reduced_x, tail_x], dim=-1)
            if cur_valid is not None:
                tail_valid = cur_valid[..., -1:]
                cur_valid = torch.cat([reduced_valid, tail_valid], dim=-1)
        else:
            cur_x = reduced_x
            if cur_valid is not None:
                cur_valid = reduced_valid
    return cur_x if keepdim else cur_x.squeeze(-1)


def sign_softmax_config_from_choice(site_name: str, controller: ApproxController, choice: ApproxChoice) -> SignApproxConfig:
    p = choice.params
    a, b = controller.range_for(site_name, "sign::delta", float(p["sign_scale_mult"]))
    scale = max(abs(float(a)), abs(float(b)), float(POSITIVE_DOMAIN_EPS))
    return SignApproxConfig(
        method=str(p["sign_method"]),
        degree=int(p.get("sign_degree", 15)),
        eps=float(p.get("sign_eps", 0.05)),
        scale=float(scale),
        compose_steps=int(p.get("sign_compose_steps", 1)),
        # Runtime profiles must be able to override the value embedded in a saved choice.
        clip_to_unit=bool(SOFTMAX_SIGN_CLIP_TO_UNIT),
        fit_points=int(p.get("sign_fit_points", SOFTMAX_SIGN_FIT_POINTS)),
    )


# =============================================================================
# Approximation sites
# =============================================================================

class ApproxPointwiseActivation(nn.Module):
    def __init__(self, site_name: str, controller: ApproxController, func_name: str = "gelu"):
        super().__init__()
        self.site_name = site_name
        self.controller = controller
        self.func_name = func_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.controller.observe(self.site_name, "x", x)
        choice = self.controller.get_choice(self.site_name)
        if choice is None:
            if self.func_name == "gelu":
                return F.gelu(x)
            raise ValueError(f"Unsupported exact pointwise func: {self.func_name}")

        params = choice.params
        a, b = self.controller.range_for(self.site_name, "x", float(params["range_mult"]))
        return approx_function(
            x,
            func_name=self.func_name,
            method=str(params["method"]),
            degree=int(params["degree"]),
            a=a,
            b=b,
        )


class ApproxLayerNorm(nn.Module):
    def __init__(self, base: nn.LayerNorm, site_name: str, controller: ApproxController):
        super().__init__()
        self.site_name = site_name
        self.controller = controller
        self.normalized_shape = tuple(base.normalized_shape) if isinstance(base.normalized_shape, (tuple, list)) else (int(base.normalized_shape),)
        self.eps = float(base.eps)
        if base.elementwise_affine:
            self.weight = nn.Parameter(base.weight.detach().clone())
            self.bias = nn.Parameter(base.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _norm_dims(self, x: torch.Tensor) -> Tuple[int, ...]:
        return tuple(range(x.ndim - len(self.normalized_shape), x.ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = self._norm_dims(x)
        mean = x.mean(dim=dims, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(dim=dims, keepdim=True)
        var_eps = var + self.eps
        self.controller.observe(self.site_name, "var", var_eps)

        choice = self.controller.get_choice(self.site_name)
        if choice is None:
            invstd = torch.rsqrt(var_eps)
        else:
            p = choice.params
            a, b = self.controller.range_for(self.site_name, "var", float(p["range_mult"]))
            invstd = approximate_rsqrt(
                var_eps,
                init_method=str(p["init_method"]),
                init_degree=int(p["init_degree"]),
                a=a,
                b=b,
                refine=str(p["refine"]),
                iters=int(p["iters"]),
            )

        y = centered * invstd
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def apply_shift(scores: torch.Tensor, shift_mode: str, mask_like: Optional[torch.Tensor] = None) -> torch.Tensor:
    shift_mode = str(shift_mode).lower()
    mask_like = normalize_mask_like(mask_like, scores)
    if shift_mode == "none":
        return scores
    if shift_mode == "row_mean":
        if mask_like is None:
            mean = scores.mean(dim=-1, keepdim=True)
        else:
            valid = (~mask_like).to(dtype=scores.dtype)
            denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean = (scores * valid).sum(dim=-1, keepdim=True) / denom
        return scores - mean
    if shift_mode == "row_max":
        if mask_like is None:
            maxv = scores.amax(dim=-1, keepdim=True)
        else:
            masked = scores.masked_fill(mask_like, float("-inf"))
            maxv = masked.amax(dim=-1, keepdim=True)
            maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        return scores - maxv
    if shift_mode == "sign_max":
        valid_mask = None if mask_like is None else (~mask_like)
        cfg = SignApproxConfig(
            method=str(SOFTMAX_SIGN_METHOD),
            degree=int(SOFTMAX_SIGN_CHEB_DEGREE),
            eps=float(SOFTMAX_SIGN_EPS),
            scale=max(abs(RANGE_FALLBACKS["softmax_sign::delta"][0]), abs(RANGE_FALLBACKS["softmax_sign::delta"][1]), float(POSITIVE_DOMAIN_EPS)),
            compose_steps=int(SOFTMAX_SIGN_COMPOSE_STEPS),
            clip_to_unit=bool(SOFTMAX_SIGN_CLIP_TO_UNIT),
            fit_points=int(SOFTMAX_SIGN_FIT_POINTS),
        )
        maxv = tree_max_sign_lastdim(scores, sign_cfg=cfg, valid_mask=valid_mask, keepdim=True)
        return scores - maxv
    raise ValueError(f"Unknown softmax shift mode: {shift_mode}")


def collect_softmax_calibration(
    site_name: str,
    controller: ApproxController,
    scores: torch.Tensor,
    mask_like: Optional[torch.Tensor] = None,
) -> None:
    if not controller.collect_calibration:
        return
    raw = scores.detach()
    mask_like = normalize_mask_like(mask_like, raw)
    valid_mask = None if mask_like is None else (~mask_like)
    shift_modes = list(dict.fromkeys(str(m).lower() for m in SOFTMAX_SHIFT_MODES))
    for shift_mode in shift_modes:
        if shift_mode == "sign_max":
            maxv = tree_max_exact_lastdim_collect_delta(site_name, controller, raw, valid_mask=valid_mask, keepdim=True)
            z = raw - maxv
        elif shift_mode == "calib_max":
            if valid_mask is None:
                controller.observe(site_name, "calib_max::score", raw)
            else:
                controller.observe(site_name, "calib_max::score", raw[valid_mask])
            row_max = exact_row_max_lastdim(raw, mask_like)
            controller.observe(site_name, "calib_max::row_max", row_max)
            controller.observe_quantile(site_name, "calib_max::row_max", row_max)
            z = apply_calibrated_layer_max_shift(site_name, controller, raw)
        elif shift_mode == "scaled":
            max_square_iters_by_scale: Dict[float, int] = {}
            for input_scale, square_iters in scaled_softmax_scale_square_pairs():
                input_scale = float(input_scale)
                max_square_iters_by_scale[input_scale] = max(
                    int(square_iters),
                    max_square_iters_by_scale.get(input_scale, 0),
                )
            exp_families = list(
                dict.fromkeys(softmax_num_family(str(method)) for method in SOFTMAX_EXP_METHODS)
            )
            for input_scale, max_square_iters in max_square_iters_by_scale.items():
                z_scaled = raw / float(input_scale)
                z_scaled_eval = public_masked_softmax_eval_input(z_scaled, mask_like)
                exp_channel = scaled_softmax_exp_channel(input_scale)
                if valid_mask is None:
                    controller.observe(site_name, exp_channel, z_scaled)
                else:
                    controller.observe(site_name, exp_channel, z_scaled[valid_mask])
                for exp_family in exp_families:
                    if exp_family == "thor_p2":
                        num = thor_p2_softmax_exp(z_scaled_eval, site_name)
                    else:
                        num = torch.exp(torch.clamp(z_scaled_eval.float(), min=-80.0, max=80.0)).to(dtype=raw.dtype)
                    if valid_mask is not None:
                        num = torch.where(valid_mask, num, torch.zeros_like(num))
                    den_channel = scaled_softmax_den_channel(input_scale, exp_family)
                    denom = num.sum(dim=-1, keepdim=True) + float(SOFTMAX_DENOM_EPS)
                    controller.observe(site_name, den_channel, denom)
                    if exp_family == "thor_p2":
                        inv = thor_he_inv_reciprocal(
                            denom,
                            site_name=site_name,
                            alpha=float(SOFTMAX_THOR_INTERNAL_ALPHA) / 10.0,
                        )
                        probs = num * inv
                    else:
                        probs = num / denom
                    if mask_like is not None:
                        probs = torch.where(mask_like, torch.zeros_like(probs), probs)
                    for step in range(1, max_square_iters + 1):
                        sq = probs * probs
                        if mask_like is not None:
                            sq = torch.where(mask_like, torch.zeros_like(sq), sq)
                        sq_denom = sq.sum(dim=-1, keepdim=True) + float(SOFTMAX_DENOM_EPS)
                        controller.observe(
                            site_name,
                            scaled_softmax_square_den_channel(input_scale, step, exp_family),
                            sq_denom,
                        )
                        if exp_family == "thor_p2":
                            inv = thor_he_inv_reciprocal(
                                sq_denom,
                                site_name=site_name,
                                alpha=(
                                    float(SOFTMAX_THOR_OUTPUT_ALPHA)
                                    if step == int(max_square_iters)
                                    else float(SOFTMAX_THOR_INTERNAL_ALPHA)
                                )
                                / 10.0,
                            )
                            probs = sq * inv
                        else:
                            probs = sq / sq_denom
                        if mask_like is not None:
                            probs = torch.where(mask_like, torch.zeros_like(probs), probs)
            continue
        else:
            z = apply_shift(raw, shift_mode, mask_like)
        if valid_mask is None:
            controller.observe(site_name, f"exp::{shift_mode}", z)
            z_for_denom = z
        else:
            controller.observe(site_name, f"exp::{shift_mode}", z[valid_mask])
            z_for_denom = torch.where(valid_mask, z, torch.full_like(z, float(SOFTMAX_MASKED_FILL_FOR_DEN_CALIB)))
        exp_families = list(
            dict.fromkeys(softmax_num_family(str(method)) for method in SOFTMAX_EXP_METHODS)
        )
        for exp_family in exp_families:
            if exp_family == "thor_p2":
                num = thor_p2_softmax_exp(public_masked_softmax_eval_input(z, mask_like), site_name)
                if valid_mask is not None:
                    num = torch.where(valid_mask, num, torch.zeros_like(num))
                den_channel = f"den::{shift_mode}::thor_p2"
            else:
                num = torch.exp(torch.clamp(z_for_denom.float(), min=-80.0, max=80.0)).to(dtype=raw.dtype)
                den_channel = f"den::{shift_mode}"
            denom = num.sum(dim=-1, keepdim=True)
            controller.observe(site_name, den_channel, denom)
        if shift_mode == "calib_max":
            den_bounds = torch.tensor(
                [float(SOFTMAX_DENOM_EPS), float(MAX_LENGTH)],
                device=raw.device,
                dtype=raw.dtype,
            )
            controller.observe(site_name, f"den::{shift_mode}", den_bounds)


def approximate_softmax(
    site_name: str,
    controller: ApproxController,
    scores: torch.Tensor,
    mask_like: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask_like = normalize_mask_like(mask_like, scores)
    collect_softmax_calibration(site_name, controller, scores, mask_like=mask_like)
    choice = controller.get_choice(site_name)
    if choice is None:
        return F.softmax(scores, dim=-1)

    p = choice.params
    shift_mode = str(p["shift_mode"]).lower()
    valid_mask = None if mask_like is None else (~mask_like)
    if shift_mode == "sign_max":
        sign_cfg = sign_softmax_config_from_choice(site_name, controller, choice)
        maxv = tree_max_sign_lastdim(scores, sign_cfg=sign_cfg, valid_mask=valid_mask, keepdim=True)
        z = scores - maxv
        exp_channel = f"exp::{shift_mode}"
        den_channel = f"den::{shift_mode}"
    elif shift_mode == "calib_max":
        z = apply_calibrated_layer_max_shift(site_name, controller, scores)
        exp_channel = f"exp::{shift_mode}"
        den_channel = f"den::{shift_mode}"
    elif shift_mode == "scaled":
        input_scale = max(float(p.get("input_scale", 1.0)), float(POSITIVE_DOMAIN_EPS))
        z = scores / input_scale
        exp_channel = scaled_softmax_exp_channel(input_scale)
        den_channel = scaled_softmax_den_channel(input_scale, str(p["exp_method"]))
    else:
        z = apply_shift(scores, shift_mode, mask_like)
        exp_channel = f"exp::{shift_mode}"
        den_channel = f"den::{shift_mode}"
    if shift_mode != "scaled" and _is_thor_p2_exp_method(str(p["exp_method"])):
        den_channel = f"{den_channel}::thor_p2"
    z_eval = public_masked_softmax_eval_input(z, mask_like)
    if _is_thor_p2_exp_method(str(p["exp_method"])):
        num = thor_p2_softmax_exp(z_eval, site_name)
    else:
        exp_a, exp_b = controller.range_for(site_name, exp_channel, float(p["exp_range_mult"]))
        num = approx_function(
            z_eval,
            func_name="exp",
            method=str(p["exp_method"]),
            degree=int(p["exp_degree"]),
            a=exp_a,
            b=exp_b,
        )
    if mask_like is not None:
        num = torch.where(mask_like, torch.zeros_like(num), num)
    denom = num.sum(dim=-1, keepdim=True) + float(SOFTMAX_DENOM_EPS)
    inv = reciprocal_from_softmax_choice(
        site_name,
        controller,
        denom,
        p,
        den_channel=den_channel,
        is_final_square=(shift_mode != "scaled" or int(p.get("square_iters", 0)) <= 0),
    )
    probs = num * inv
    if mask_like is not None:
        probs = torch.where(mask_like, torch.zeros_like(probs), probs)
    if shift_mode == "scaled":
        square_iters = max(0, int(p.get("square_iters", 0)))
        for step in range(1, square_iters + 1):
            sq = probs * probs
            if mask_like is not None:
                sq = torch.where(mask_like, torch.zeros_like(sq), sq)
            sq_denom = sq.sum(dim=-1, keepdim=True) + float(SOFTMAX_DENOM_EPS)
            sq_inv = reciprocal_from_softmax_choice(
                site_name,
                controller,
                sq_denom,
                p,
                den_channel=scaled_softmax_square_den_channel(float(p["input_scale"]), step, str(p["exp_method"])),
                is_final_square=(step == square_iters),
            )
            probs = sq * sq_inv
            if mask_like is not None:
                probs = torch.where(mask_like, torch.zeros_like(probs), probs)
    return probs


# =============================================================================
# Instrument BERT
# =============================================================================


def _is_gelu_like(obj: Any) -> bool:
    name = getattr(obj, "__name__", obj.__class__.__name__).lower()
    return "gelu" in name


def _replace_child(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = new_child  # type: ignore[index]
    else:
        setattr(parent, child_name, new_child)


def replace_layernorm_modules(module: nn.Module, controller: ApproxController, prefix: str = "") -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.LayerNorm):
            site_name = f"{full_name}.layernorm"
            controller.register_site(site_name, "layernorm")
            _replace_child(module, child_name, ApproxLayerNorm(child, site_name, controller))
        else:
            replace_layernorm_modules(child, controller, prefix=full_name)


def patch_gelu_sites(model: nn.Module, controller: ApproxController) -> None:
    for name, module in model.named_modules():
        if hasattr(module, "intermediate_act_fn") and _is_gelu_like(getattr(module, "intermediate_act_fn")):
            site_name = f"{name}.gelu"
            controller.register_site(site_name, "gelu")
            setattr(module, "intermediate_act_fn", ApproxPointwiseActivation(site_name, controller, func_name="gelu"))
        if hasattr(module, "transform_act_fn") and _is_gelu_like(getattr(module, "transform_act_fn")):
            site_name = f"{name}.gelu"
            controller.register_site(site_name, "gelu")
            setattr(module, "transform_act_fn", ApproxPointwiseActivation(site_name, controller, func_name="gelu"))


def build_self_attention_patch(site_name: str, controller: ApproxController, original_forward: Callable[..., Any]):
    def _patched_forward(self: nn.Module, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        # Keep a safe fallback for any unsupported path.
        if args:
            return original_forward(hidden_states, attention_mask, *args, **kwargs)
        if kwargs.get("encoder_hidden_states", None) is not None:
            return original_forward(hidden_states, attention_mask, *args, **kwargs)
        if kwargs.get("encoder_attention_mask", None) is not None:
            return original_forward(hidden_states, attention_mask, *args, **kwargs)
        if kwargs.get("past_key_value", None) is not None or kwargs.get("past_key_values", None) is not None:
            return original_forward(hidden_states, attention_mask, *args, **kwargs)
        if getattr(self, "is_decoder", False):
            return original_forward(hidden_states, attention_mask, *args, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        scaling = float(getattr(self, "scaling", self.attention_head_size ** -0.5))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * scaling
        mask_like = None
        if attention_mask is not None:
            mask_like = attention_mask <= float(SOFTMAX_MASK_THRESHOLD)
            attention_scores = attention_scores + attention_mask

        attention_probs = approximate_softmax(site_name, controller, attention_scores, mask_like=mask_like)
        drop_p = getattr(getattr(self, "dropout", None), "p", 0.0)
        if self.training:
            attention_probs = F.dropout(attention_probs, p=float(drop_p), training=True)
        elif SOFTMAX_OUTPUT_DROPOUT_IN_EVAL and float(drop_p) > 0.0:
            attention_probs = F.dropout(attention_probs, p=float(drop_p), training=False)

        head_mask = kwargs.get("head_mask", None)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attn_output = torch.matmul(attention_probs, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return attn_output, attention_probs

    return _patched_forward


def patch_bert_attention(model: nn.Module, controller: ApproxController) -> None:
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BertSelfAttention":
            site_name = f"{name}.softmax"
            controller.register_site(site_name, "softmax")
            orig = module.forward
            module._poly_original_forward = orig  # type: ignore[attr-defined]
            module.forward = types.MethodType(build_self_attention_patch(site_name, controller, orig), module)


def instrument_bert_model(model: nn.Module, controller: ApproxController) -> nn.Module:
    replace_layernorm_modules(model, controller)
    patch_gelu_sites(model, controller)
    patch_bert_attention(model, controller)
    return model


def eval_with_cache(
    cache: Optional[Dict[Tuple[int, ...], "EvalResult"]],
    key: Tuple[int, ...],
    fn: Callable[[], "EvalResult"],
) -> "EvalResult":
    if cache is not None and key in cache:
        return cache[key]
    ev = fn()
    if cache is not None:
        cache[key] = ev
    return ev


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EvalResult:
    logit_mse: float
    acc: float
    max_abs_logit: float
    nonfinite_any: bool
    nonfinite_sample_count: int
    n_total: int


def eval_nonfinite_rate(ev: "EvalResult") -> float:
    return float(ev.nonfinite_sample_count) / max(1, int(ev.n_total))


@torch.inference_mode()
def compute_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, float]:
    outs: List[torch.Tensor] = []
    total = 0
    micro_batch = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    non_blocking = (device.type == "cuda")
    correct_t = torch.zeros((), device=device, dtype=torch.int64)
    for big_batch in loader:
        big_bs = int(big_batch["labels"].shape[0])
        for start in range(0, big_bs, int(micro_batch)):
            sub = slice_batch(big_batch, start, start + int(micro_batch))
            labels = sub["labels"].to(device, non_blocking=non_blocking)
            inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in sub.items() if k != "labels"}
            logits = model(**inputs).logits.detach().float()
            outs.append(logits.cpu())
            correct_t += (logits.argmax(dim=-1) == labels).sum(dtype=torch.int64)
            total += int(labels.numel())
    acc = float(correct_t.item()) / max(1, total)
    return torch.cat(outs, dim=0), float(acc)


@torch.inference_mode()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline_logits: Optional[torch.Tensor] = None,
) -> EvalResult:
    total = 0
    offset = 0
    micro_batch = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    non_blocking = (device.type == "cuda")

    correct_t = torch.zeros((), device=device, dtype=torch.int64)
    mse_sum_t = torch.zeros((), device=device, dtype=torch.float64)
    nonfinite_any_t = torch.zeros((), device=device, dtype=torch.bool)
    nonfinite_sample_count_t = torch.zeros((), device=device, dtype=torch.int64)
    max_abs_logit_t = torch.zeros((), device=device, dtype=torch.float32)

    for big_batch in loader:
        big_bs = int(big_batch["labels"].shape[0])
        for start in range(0, big_bs, int(micro_batch)):
            sub = slice_batch(big_batch, start, start + int(micro_batch))
            labels = sub["labels"].to(device, non_blocking=non_blocking)
            inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in sub.items() if k != "labels"}
            logits_raw = model(**inputs).logits.detach().float()

            bad_sample_mask = (~torch.isfinite(logits_raw)).any(dim=-1)
            nonfinite_any_t |= bad_sample_mask.any()
            nonfinite_sample_count_t += bad_sample_mask.sum(dtype=torch.int64)

            logits = torch.nan_to_num(logits_raw, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs_logit_t = torch.maximum(max_abs_logit_t, logits.abs().amax())

            correct_t += (logits.argmax(dim=-1) == labels).sum(dtype=torch.int64)
            bs = int(labels.numel())
            total += bs

            if baseline_logits is not None:
                base_src = baseline_logits[offset:offset + bs]
                if base_src.device == device and base_src.dtype == logits.dtype:
                    base = base_src
                else:
                    base = base_src.to(device=device, dtype=logits.dtype, non_blocking=non_blocking)
                diff = logits - base
                mse_sum_t += ((diff * diff).mean().to(torch.float64) * bs)
                offset += bs

    return EvalResult(
        logit_mse=(float(mse_sum_t.item()) / max(1, total)) if baseline_logits is not None else float("nan"),
        acc=float(correct_t.item() / max(1, total)),
        max_abs_logit=float(max_abs_logit_t.item()),
        nonfinite_any=bool(nonfinite_any_t.item()),
        nonfinite_sample_count=int(nonfinite_sample_count_t.item()),
        n_total=int(total),
    )


# =============================================================================
# Proxy table + DP
# =============================================================================



def schedule_cost(schedule: Dict[str, int], controller: ApproxController) -> int:
    total = 0
    for site, local_choice_id in schedule.items():
        kind = controller.get_kind(site)
        found = False
        for ch in controller.choices_by_kind[kind]:
            if int(ch.choice_id) == int(local_choice_id):
                total += int(ch.cost)
                found = True
                break
        if not found:
            raise KeyError(f"Missing choice_id={local_choice_id} for site={site}, kind={kind}")
    return int(total)


def schedule_to_rows(
    schedule: Dict[str, int],
    controller: ApproxController,
    *,
    sites: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    iter_sites = list(sites) if sites is not None else controller.ordered_sites()
    for site in iter_sites:
        if site not in schedule:
            rows.append({
                "site": site,
                "kind": controller.get_kind(site),
                "choice_id": "exact",
                "label": "exact",
                "cost": 0,
            })
            continue
        kind = controller.get_kind(site)
        local_id = int(schedule[site])
        choice = None
        for ch in controller.choices_by_kind[kind]:
            if int(ch.choice_id) == local_id:
                choice = ch
                break
        if choice is None:
            raise KeyError(f"Missing choice_id={local_id} for site={site}, kind={kind}")
        rows.append({
            "site": site,
            "kind": choice.kind,
            "choice_id": int(choice.choice_id),
            "label": choice.label,
            "cost": int(choice.cost),
        })
    return rows


def budget_bounds(
    controller: ApproxController,
    *,
    sites: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    min_b = 0
    max_b = 0
    iter_sites = list(sites) if sites is not None else controller.ordered_sites()
    for site in iter_sites:
        costs = [int(ch.cost) for ch in controller.choices_for_site(site)]
        min_b += min(costs)
        max_b += max(costs)
    return int(min_b), int(max_b)


def load_proxy_table(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_proxy_error_table(
    model: nn.Module,
    controller: ApproxController,
    sweep_loader: DataLoader,
    baseline_logits: torch.Tensor,
    device: torch.device,
    out_path: Path,
    *,
    sites: Optional[Sequence[str]] = None,
    prescreen_loader: Optional[DataLoader] = None,
    prescreen_baseline_logits: Optional[torch.Tensor] = None,
    stage1_keep_per_cost: Optional[int] = None,
    stage1_out_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Any]]:
    target_sites = list(sites) if sites is not None else controller.ordered_sites()
    t_total0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "reuse": False,
        "stage1_enabled": False,
        "stage1_seconds": 0.0,
        "stage2_seconds": 0.0,
        "total_seconds": 0.0,
        "total_candidates": int(sum(len(controller.choices_for_site(s)) for s in target_sites)),
        "stage2_evals": 0,
        "survivors_total": 0,
    }

    if REUSE_PROXY_TABLE and out_path.exists():
        rows = load_proxy_table(out_path)
        if rows is not None:
            proxy: Dict[str, Dict[int, float]] = defaultdict(dict)
            for row in rows:
                site = str(row.get("site", ""))
                if site not in target_sites:
                    continue
                proxy[site][int(row["choice_id"])] = float(row["mse"])
            expected = {
                (site, int(ch.choice_id), str(ch.label), int(ch.cost))
                for site in target_sites
                for ch in controller.choices_for_site(site)
            }
            found = {
                (
                    str(row.get("site", "")),
                    int(row.get("choice_id", -1)),
                    str(row.get("label", "")),
                    int(row.get("cost", -1)),
                )
                for row in rows
                if str(row.get("site", "")) in target_sites
            }
            if expected.issubset(found):
                meta["reuse"] = True
                print(f"[proxy] reuse table from {out_path}")
                return proxy, meta
            print(f"[proxy] existing table mismatches current config, rebuilding: {out_path}")

    total_evals = int(sum(len(controller.choices_for_site(s)) for s in target_sites))
    keep_per_cost = int(PROXY_STAGE1_KEEP_PER_COST if stage1_keep_per_cost is None else stage1_keep_per_cost)
    use_stage1 = bool(
        ENABLE_PROXY_STAGE1_PRUNE
        and prescreen_loader is not None
        and prescreen_baseline_logits is not None
        and keep_per_cost > 0
    )

    stage1_scores: Dict[str, Dict[int, float]] = defaultdict(dict)
    survivors: Dict[str, set[int]] = defaultdict(set)

    if use_stage1:
        meta["stage1_enabled"] = True
        t1 = time.perf_counter()
        done1 = 0
        stage1_rows: List[Dict[str, Any]] = []
        print(f"[proxy-stage1] prescreen all choices on smaller subset: {len(target_sites)} sites, {total_evals} evals")
        for site in target_sites:
            kind = controller.get_kind(site)
            for choice in controller.choices_for_site(site):
                controller.set_exact()
                controller.schedule[site] = int(choice.choice_id)
                ev = eval_model(model, prescreen_loader, device, baseline_logits=prescreen_baseline_logits)
                mse = float(ev.logit_mse)
                if ev.nonfinite_any:
                    mse = mse + 1.0e6
                cid = int(choice.choice_id)
                stage1_scores[site][cid] = mse
                stage1_rows.append({
                    "site": site,
                    "kind": kind,
                    "choice_id": cid,
                    "label": choice.label,
                    "cost": int(choice.cost),
                    "stage1_mse": float(mse),
                    "nonfinite_any": int(ev.nonfinite_any),
                })
                done1 += 1
                if done1 % 100 == 0 or done1 == total_evals:
                    print(f"[proxy-stage1] {done1}/{total_evals}")

        for site in target_sites:
            site_keep = proxy_stage1_keep_for_site(controller, site, keep_per_cost)
            buckets: Dict[int, List[ApproxChoice]] = defaultdict(list)
            for ch in controller.choices_for_site(site):
                buckets[int(ch.cost)].append(ch)
            for cost, bucket in buckets.items():
                bucket_sorted = sorted(bucket, key=lambda ch: (float(stage1_scores[site][int(ch.choice_id)]), int(ch.choice_id)))
                for ch in bucket_sorted[:site_keep]:
                    survivors[site].add(int(ch.choice_id))
        meta["survivors_total"] = int(sum(len(v) for v in survivors.values()))
        meta["stage1_seconds"] = float(time.perf_counter() - t1)
        if stage1_out_path is not None:
            save_csv(stage1_rows, stage1_out_path)
            print(f"[proxy-stage1] saved prescreen table to {stage1_out_path}")
        kept = int(meta["survivors_total"])
        ratio = 100.0 * kept / max(1, total_evals)
        surv_by_kind: Dict[str, int] = defaultdict(int)
        total_by_kind: Dict[str, int] = defaultdict(int)
        for site in target_sites:
            kind = controller.get_kind(site)
            surv_by_kind[kind] += len(survivors[site])
            total_by_kind[kind] += len(controller.choices_for_site(site))
        kind_msg = ", ".join(f"{k}:{surv_by_kind[k]}/{total_by_kind[k]}" for k in sorted(total_by_kind))
        print(f"[proxy-stage1] keep asymmetric top-k per (site,cost): survivors={kept}/{total_evals} ({ratio:.1f}%) | {kind_msg}")
    else:
        for site in target_sites:
            survivors[site] = {int(ch.choice_id) for ch in controller.choices_for_site(site)}
        meta["survivors_total"] = int(sum(len(v) for v in survivors.values()))

    rows_to_save: List[Dict[str, Any]] = []
    proxy: Dict[str, Dict[int, float]] = defaultdict(dict)
    full_scores_kept: Dict[str, Dict[int, float]] = defaultdict(dict)
    t2 = time.perf_counter()
    total_stage2 = int(sum(len(v) for v in survivors.values()))
    meta["stage2_evals"] = total_stage2
    done2 = 0
    print(f"[proxy-stage2] full proxy on survivors: {total_stage2} evals")

    for site in target_sites:
        kind = controller.get_kind(site)
        for choice in controller.choices_for_site(site):
            cid = int(choice.choice_id)
            if cid not in survivors[site]:
                continue
            controller.set_exact()
            controller.schedule[site] = cid
            ev = eval_model(model, sweep_loader, device, baseline_logits=baseline_logits)
            mse = float(ev.logit_mse)
            if ev.nonfinite_any:
                mse = mse + 1.0e6
            full_scores_kept[site][cid] = mse
            proxy[site][cid] = mse
            rows_to_save.append({
                "site": site,
                "kind": kind,
                "choice_id": cid,
                "label": choice.label,
                "cost": int(choice.cost),
                "mse": float(mse),
                "nonfinite_any": int(ev.nonfinite_any),
                "proxy_source": "stage2",
                "stage1_mse": float(stage1_scores.get(site, {}).get(cid, mse)),
            })
            done2 += 1
            if done2 % 50 == 0 or done2 == total_stage2:
                print(f"[proxy-stage2] {done2}/{total_stage2}")

    for site in target_sites:
        kind = controller.get_kind(site)
        buckets: Dict[int, List[ApproxChoice]] = defaultdict(list)
        for ch in controller.choices_for_site(site):
            buckets[int(ch.cost)].append(ch)
        for cost, bucket in buckets.items():
            kept_bucket = [ch for ch in bucket if int(ch.choice_id) in survivors[site]]
            if not kept_bucket:
                bucket_sorted = sorted(bucket, key=lambda ch: (float(stage1_scores.get(site, {}).get(int(ch.choice_id), 1.0e30)), int(ch.choice_id)))
                kept_bucket = bucket_sorted[:1]
            best_kept_mse = min(float(full_scores_kept[site][int(ch.choice_id)]) for ch in kept_bucket)
            bucket_sorted_stage1 = sorted(bucket, key=lambda ch: (float(stage1_scores.get(site, {}).get(int(ch.choice_id), 1.0e30)), int(ch.choice_id)))
            rank = {int(ch.choice_id): i for i, ch in enumerate(bucket_sorted_stage1)}
            for ch in bucket:
                cid = int(ch.choice_id)
                if cid in proxy[site]:
                    continue
                surrogate = float(best_kept_mse + float(PROXY_DROP_SURROGATE_MARGIN) * (1 + rank.get(cid, 0)))
                proxy[site][cid] = surrogate
                rows_to_save.append({
                    "site": site,
                    "kind": kind,
                    "choice_id": cid,
                    "label": ch.label,
                    "cost": int(ch.cost),
                    "mse": surrogate,
                    "nonfinite_any": 0,
                    "proxy_source": "surrogate",
                    "stage1_mse": float(stage1_scores.get(site, {}).get(cid, surrogate)),
                })

    meta["stage2_seconds"] = float(time.perf_counter() - t2)
    meta["total_seconds"] = float(time.perf_counter() - t_total0)
    save_csv(rows_to_save, out_path)
    print(f"[proxy] saved to {out_path}")
    print(f"[proxy] timing | stage1={meta['stage1_seconds']:.2f}s stage2={meta['stage2_seconds']:.2f}s total={meta['total_seconds']:.2f}s")
    controller.set_exact()
    return proxy, meta

def filter_proxy_errors(
    proxy_errors: Dict[str, Dict[int, float]],
    sites: Sequence[str],
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = defaultdict(dict)
    for site in sites:
        if site in proxy_errors:
            out[site] = dict(proxy_errors[site])
    return out



def fill_missing_schedule_sites_with_dp(
    schedule: Dict[str, int],
    *,
    active_sites: Sequence[str],
    controller: ApproxController,
    proxy_errors: Dict[str, Dict[int, float]],
    search_budget_lo: int,
    search_budget_hi: int,
) -> Dict[str, int]:
    active_sites = list(active_sites)
    out = {site: int(cid) for site, cid in schedule.items() if site in active_sites}
    missing = [site for site in active_sites if site not in out]
    if not missing:
        return out

    fixed_cost = schedule_cost(out, controller)
    rem_lo = max(0, int(search_budget_lo) - int(fixed_cost))
    rem_hi = int(search_budget_hi) - int(fixed_cost)
    if rem_hi < 0:
        raise ValueError(
            f"Partial seed cost={fixed_cost} exceeds search budget upper bound={search_budget_hi}; "
            f"cannot fill missing sites={len(missing)}"
        )
    print(
        f"[joint-task] partial seed missing {len(missing)} active sites; "
        f"fixed_cost={fixed_cost}, filling missing budget_band={rem_lo}..{rem_hi} via DP"
    )
    missing_proxy = filter_proxy_errors(proxy_errors, missing)
    missing_schedule = dp_allocate_schedule(
        list(missing),
        controller,
        missing_proxy,
        int(rem_hi),
        min_budget=int(rem_lo),
    )
    out.update(missing_schedule)
    return out




def proxy_stage1_keep_for_site(controller: ApproxController, site: str, default_keep: int) -> int:
    kind = controller.get_kind(site)
    if kind == "softmax":
        return max(1, int(PROXY_STAGE1_KEEP_PER_COST_SOFTMAX))
    if kind == "gelu":
        return max(1, int(PROXY_STAGE1_KEEP_PER_COST_GELU))
    if kind == "layernorm":
        return max(1, int(PROXY_STAGE1_KEEP_PER_COST_LAYERNORM))
    return max(1, int(default_keep))


def build_cd_site_subset(
    controller: ApproxController,
    active_sites: Sequence[str],
    importance_rank: Sequence[str],
) -> List[str]:
    active_set = set(active_sites)
    ordered = [s for s in importance_rank if s in active_set]
    ordered.extend([s for s in active_sites if s not in set(ordered)])
    if not JOINT_ENABLE_CD_SITE_SUBSET:
        ordered = list(active_sites)
    else:
        ordered = ordered[: int(JOINT_CD_TOPM_IMPORTANCE)]
        if len(ordered) > int(JOINT_CD_SITE_SUBSET_MAX):
            ordered = ordered[: int(JOINT_CD_SITE_SUBSET_MAX)]
        if not ordered:
            ordered = list(active_sites)

    priority = [str(k).lower() for k in JOINT_CD_KIND_PRIORITY]
    grouped: Dict[str, List[str]] = defaultdict(list)
    for s in ordered:
        grouped[controller.get_kind(s)].append(s)

    out: List[str] = []
    for kind in priority:
        arr = list(grouped.get(kind, []))
        if kind == "softmax":
            arr = arr[: int(JOINT_CD_SOFTMAX_SITE_LIMIT)]
        elif kind == "gelu":
            arr = arr[: int(JOINT_CD_GELU_SITE_LIMIT)]
        elif kind == "layernorm":
            arr = arr[: int(JOINT_CD_LAYERNORM_SITE_LIMIT)]
        out.extend(arr)

    # any leftover kinds not listed in priority
    for s in ordered:
        if s not in set(out):
            out.append(s)
    return out




def cd_stage_pass_limit(kind: str) -> int:
    kind = str(kind).lower()
    if kind == "softmax":
        return max(1, int(JOINT_CD_SOFTMAX_PASSES))
    if kind == "gelu":
        return max(1, int(JOINT_CD_GELU_PASSES))
    if kind == "layernorm":
        return max(1, int(JOINT_CD_LAYERNORM_PASSES))
    return max(1, int(JOINT_CD_MAX_PASSES))


def build_cd_stage_groups(
    controller: ApproxController,
    ordered_sites: Sequence[str],
) -> List[Tuple[str, List[str], int]]:
    ordered_sites = list(ordered_sites)
    priority = [str(k).lower() for k in JOINT_CD_KIND_PRIORITY]
    groups: Dict[str, List[str]] = defaultdict(list)
    for s in ordered_sites:
        groups[controller.get_kind(s)].append(s)

    out: List[Tuple[str, List[str], int]] = []
    seen: set[str] = set()
    for kind in priority:
        arr = list(groups.get(kind, []))
        if not arr:
            continue
        out.append((kind, arr, cd_stage_pass_limit(kind)))
        seen.update(arr)

    leftovers = [s for s in ordered_sites if s not in seen]
    if leftovers:
        out.append(("other", leftovers, max(1, int(JOINT_CD_MAX_PASSES))))
    return out

def full_lite_guard_ok(
    ev: EvalResult,
    *,
    base_full_lite_acc: float,
    base_full_lite_abs_logit: float,
) -> bool:
    return joint_full_guard_ok(ev, base_full_acc=base_full_lite_acc, base_full_abs_logit=base_full_lite_abs_logit)

def save_proxy_subset(
    controller: ApproxController,
    proxy_errors: Dict[str, Dict[int, float]],
    sites: Sequence[str],
    out_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for site in sites:
        kind = controller.get_kind(site)
        site_proxy = proxy_errors.get(site, {})
        for ch in controller.choices_for_site(site):
            if int(ch.choice_id) not in site_proxy:
                continue
            rows.append({
                "site": site,
                "kind": kind,
                "choice_id": int(ch.choice_id),
                "label": ch.label,
                "cost": int(ch.cost),
                "mse": float(site_proxy[int(ch.choice_id)]),
            })
    save_csv(rows, out_path)


def dp_allocate_schedule(
    sites: List[str],
    controller: ApproxController,
    proxy_errors: Dict[str, Dict[int, float]],
    budget: int,
    *,
    min_budget: int = 0,
) -> Dict[str, int]:
    INF = 1e30
    B = int(budget)
    lo = max(0, int(min_budget))
    n = len(sites)

    if n == 0:
        return {}
    if lo > B:
        raise RuntimeError(f"DP failed: invalid budget range {lo}..{B}")

    dp = [[INF] * (B + 1) for _ in range(n + 1)]
    back_choice: List[List[Optional[int]]] = [[None] * (B + 1) for _ in range(n + 1)]
    back_prevb: List[List[Optional[int]]] = [[None] * (B + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i, site in enumerate(sites, start=1):
        choices = controller.choices_for_site(site)
        for b in range(B + 1):
            prev = dp[i - 1][b]
            if prev >= INF:
                continue
            for ch in choices:
                c = int(ch.cost)
                nb = b + c
                if nb > B:
                    continue
                e = float(proxy_errors[site][int(ch.choice_id)])
                val = prev + e
                if val < dp[i][nb]:
                    dp[i][nb] = val
                    back_choice[i][nb] = int(ch.choice_id)
                    back_prevb[i][nb] = b

    feasible_band = [b for b in range(lo, B + 1) if dp[n][b] < INF]
    if feasible_band:
        best_b = min(feasible_band, key=lambda x: (dp[n][x], -x))
    else:
        feasible_any = [b for b in range(B + 1) if dp[n][b] < INF]
        if not feasible_any:
            raise RuntimeError(f"DP failed: no feasible schedule under budget={budget}")
        raise RuntimeError(
            f"DP failed: no feasible schedule inside budget range {lo}..{B}; "
            f"reachable costs span {min(feasible_any)}..{max(feasible_any)}"
        )

    sched: Dict[str, int] = {}
    cur_b = best_b
    for i in range(n, 0, -1):
        site = sites[i - 1]
        ch = back_choice[i][cur_b]
        pb = back_prevb[i][cur_b]
        if ch is None or pb is None:
            raise RuntimeError("DP backtracking failed")
        sched[site] = int(ch)
        cur_b = int(pb)
    return sched


def greedy_val_refine(
    model: nn.Module,
    controller: ApproxController,
    active_sites: Sequence[str],
    schedule: Dict[str, int],
    budget: int,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    base_val_acc: float,
    device: torch.device,
) -> Tuple[Dict[str, int], EvalResult]:
    controller.set_schedule(schedule)
    best_ev = eval_model(model, val_loader, device, baseline_logits=val_base_logits)
    best_sched = dict(schedule)

    for pass_i in range(1, GREEDY_MAX_PASSES + 1):
        improved = False
        for site in active_sites:
            cur_choice_id = int(best_sched[site])
            for ch in controller.choices_for_site(site):
                if int(ch.choice_id) == cur_choice_id:
                    continue
                trial = dict(best_sched)
                trial[site] = int(ch.choice_id)
                if schedule_cost(trial, controller) > int(budget):
                    continue
                controller.set_schedule(trial)
                ev = eval_model(model, val_loader, device, baseline_logits=val_base_logits)
                if ev.nonfinite_any:
                    continue
                if ENABLE_VAL_ACC_GUARD and (ev.acc < (base_val_acc - VAL_ACC_GUARD_DROP_PP / 100.0)):
                    continue
                better = ev.logit_mse < best_ev.logit_mse - 1e-12
                tie = abs(ev.logit_mse - best_ev.logit_mse) <= 1e-12 and ev.acc > best_ev.acc + 1e-12
                if better or tie:
                    best_sched = trial
                    best_ev = ev
                    improved = True
                    print(
                        f"[greedy] pass{pass_i}: accept {site} -> {ch.label} | "
                        f"val_mse={best_ev.logit_mse:.6e} val_acc={_fmt_pct(best_ev.acc)}"
                    )
                    break
        if not improved:
            break

    controller.set_schedule(best_sched)
    return best_sched, best_ev


# =============================================================================
# Pareto helper
# =============================================================================


def mark_pareto_rows(rows: List[Dict[str, Any]], *, flag_key: str = "is_pareto") -> None:
    # Dominance: lower/equal cost and higher/equal val_acc, at least one strict.
    for row in rows:
        row[flag_key] = bool(int(row.get("is_valid", 1)))
    for i, a in enumerate(rows):
        if not bool(int(a.get("is_valid", 1))):
            a[flag_key] = False
            continue
        for j, b in enumerate(rows):
            if i == j or (not bool(int(b.get("is_valid", 1)))):
                continue
            cost_a = int(a["cost"])
            cost_b = int(b["cost"])
            acc_a = float(a["val_acc"])
            acc_b = float(b["val_acc"])
            if (cost_b <= cost_a and acc_b >= acc_a) and (cost_b < cost_a or acc_b > acc_a):
                a[flag_key] = False
                break


# =============================================================================
# Ablation helpers
# =============================================================================


ALL_APPROX_KINDS: Tuple[str, ...] = ("gelu", "softmax", "layernorm")
ABLATION_MODE_PRESETS: Dict[str, Tuple[str, ...]] = {
    "all": ALL_APPROX_KINDS,
    "gelu_only": ("gelu",),
    "softmax_only": ("softmax",),
    "layernorm_only": ("layernorm",),
    "gelu_softmax": ("gelu", "softmax"),
    "gelu_layernorm": ("gelu", "layernorm"),
    "softmax_layernorm": ("softmax", "layernorm"),
    "exact": tuple(),
}
KIND_TOKEN_ALIASES = {
    "gelu": "gelu",
    "smx": "softmax",
    "softmax": "softmax",
    "ln": "layernorm",
    "layernorm": "layernorm",
    "layer_norm": "layernorm",
}


@dataclass(frozen=True)
class AblationModeSpec:
    input_name: str
    slug: str
    active_kinds: Tuple[str, ...]

    @property
    def display_name(self) -> str:
        return self.slug



def _canonical_kind_token(token: str) -> str:
    key = token.strip().lower().replace("-", "_")
    if key not in KIND_TOKEN_ALIASES:
        raise KeyError(f"Unknown ablation kind token={token!r}. Supported tokens: {sorted(KIND_TOKEN_ALIASES)}")
    return KIND_TOKEN_ALIASES[key]



def _canonical_mode_slug(active_kinds: Sequence[str]) -> str:
    kinds = tuple(k for k in ALL_APPROX_KINDS if k in set(active_kinds))
    if len(kinds) == 0:
        return "exact"
    if len(kinds) == len(ALL_APPROX_KINDS):
        return "all"
    if len(kinds) == 1:
        return f"{kinds[0]}_only"
    return "_".join(kinds)



def resolve_ablation_mode(mode_name: str) -> AblationModeSpec:
    raw = str(mode_name).strip()
    if not raw:
        raise ValueError("Empty ablation mode name")
    key = raw.lower().replace("-", "_").replace(" ", "")
    if key in ABLATION_MODE_PRESETS:
        active = tuple(ABLATION_MODE_PRESETS[key])
    elif "+" in raw or "," in raw or "/" in raw:
        tokens = re.split(r"[+,/]", raw)
        active = tuple(k for k in ALL_APPROX_KINDS if k in {_canonical_kind_token(tok) for tok in tokens if tok.strip()})
    elif key in KIND_TOKEN_ALIASES:
        active = (_canonical_kind_token(key),)
    else:
        raise KeyError(
            f"Unknown ablation mode={mode_name!r}. Supported presets: {sorted(ABLATION_MODE_PRESETS)}"
        )
    return AblationModeSpec(input_name=raw, slug=_canonical_mode_slug(active), active_kinds=active)



def resolve_ablation_modes(mode_names: Sequence[str]) -> List[AblationModeSpec]:
    modes: List[AblationModeSpec] = []
    seen = set()
    for name in mode_names:
        spec = resolve_ablation_mode(name)
        if spec.slug in seen:
            continue
        seen.add(spec.slug)
        modes.append(spec)
    return modes



def active_sites_for_mode(controller: ApproxController, mode_spec: AblationModeSpec) -> List[str]:
    active_kind_set = set(mode_spec.active_kinds)
    return [site for site in controller.ordered_sites() if controller.get_kind(site) in active_kind_set]


def resolve_joint_mutable_sites(controller: ApproxController, active_sites: Sequence[str]) -> List[str]:
    sites = list(active_sites)
    kinds = {str(k).lower() for k in JOINT_MUTABLE_KIND_FILTER}
    if kinds:
        sites = [s for s in sites if controller.get_kind(s).lower() in kinds]
    focus = [str(s) for s in JOINT_MUTABLE_SITES]
    if focus:
        focus_set = set(focus)
        sites = [s for s in sites if s in focus_set]
    return sites if sites else list(active_sites)



def kind_counts_for_sites(controller: ApproxController, sites: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for site in sites:
        counts[controller.get_kind(site)] += 1
    return dict(counts)



def select_budgets(min_budget: int, max_budget: int) -> List[int]:
    min_budget = int(min_budget)
    max_budget = int(max_budget)
    if min_budget == 0 and max_budget == 0:
        return [0]
    if RUN_MODE == "single_budget":
        if int(TOTAL_BUDGET) < int(min_budget):
            raise ValueError(
                f"TOTAL_BUDGET={TOTAL_BUDGET} is infeasible for this ablation; minimum feasible budget is {min_budget}."
            )
        return [min(int(TOTAL_BUDGET), int(max_budget))]

    budgets = [int(b) for b in BUDGET_SWEEP_LIST if int(min_budget) <= int(b) <= int(max_budget)]
    if not budgets:
        span = max(1, int(max_budget) - int(min_budget))
        step = max(1, span // 8)
        budgets = list(range(int(min_budget), int(max_budget) + 1, int(step)))
        if budgets[-1] != int(max_budget):
            budgets.append(int(max_budget))
    elif budgets[0] != int(min_budget):
        budgets = [int(min_budget)] + budgets
    budgets = sorted(set(int(b) for b in budgets))
    return budgets


# =============================================================================
# Model loading + site summary
# =============================================================================


def load_local_model_and_tokenizer(model_dir: Path, tokenizer_dir: Optional[Path]) -> Tuple[Any, nn.Module]:
    model_dir = ensure_local_path(model_dir, what="model_dir")
    tok_dir = tokenizer_dir if tokenizer_dir is not None and tokenizer_dir.exists() else model_dir

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=LOCAL_FILES_ONLY)
    try:
        setattr(config, "_attn_implementation", ATTN_IMPLEMENTATION)
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=LOCAL_FILES_ONLY, use_fast=True)
    except (ImportError, ValueError):
        print(f"[tokenizer] fast tokenizer unavailable for {tok_dir}; falling back to slow tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=LOCAL_FILES_ONLY, use_fast=False)
        except (ImportError, ValueError):
            tokenizer = BertTokenizer.from_pretrained(str(tok_dir), local_files_only=LOCAL_FILES_ONLY)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=LOCAL_FILES_ONLY,
    )
    try:
        setattr(model.config, "_attn_implementation", ATTN_IMPLEMENTATION)
    except Exception:
        pass
    model.eval()
    return tokenizer, model



def site_summary_rows(
    controller: ApproxController,
    *,
    sites: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counts = defaultdict(int)
    iter_sites = list(sites) if sites is not None else controller.ordered_sites()
    for site in iter_sites:
        kind = controller.get_kind(site)
        counts[kind] += 1
        rows.append({"site": site, "kind": kind})
    rows.append({"site": "__TOTAL__", "kind": json.dumps(dict(counts), ensure_ascii=False)})
    return rows


def softmax_calib_max_lut_rows(controller: ApproxController) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for site in controller.ordered_sites():
        if controller.get_kind(site) != "softmax":
            continue
        st = controller.calibration.get(site, "calib_max::row_max")
        if st is None or not st.is_valid:
            continue
        rows.append(
            {
                "site": site,
                "shift_mode": "calib_max",
                "quantile": float(SOFTMAX_CALIB_MAX_QUANTILE),
                "margin_mult": float(SOFTMAX_CALIB_MAX_MARGIN_MULT),
                "margin_abs": float(SOFTMAX_CALIB_MAX_MARGIN_ABS),
                "min_row_max": float(st.min_val),
                "observed_max": float(st.max_val),
                "quantile_value": controller.quantile_for(site, "calib_max::row_max", float(SOFTMAX_CALIB_MAX_QUANTILE)),
                "shift_value": calibrated_layer_max_value(site, controller),
            }
        )
    return rows


# =============================================================================
# Main runner for one model
# =============================================================================


def run_one_ablation_mode(
    *,
    model_key: str,
    mode_spec: AblationModeSpec,
    model: nn.Module,
    controller: ApproxController,
    base_out_dir: Path,
    active_sites: Sequence[str],
    sweep_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    base_sweep_logits: torch.Tensor,
    base_val_logits: torch.Tensor,
    base_test_logits: torch.Tensor,
    base_val_acc: float,
    base_test_acc: float,
    device: torch.device,
    shared_proxy: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    mode_out_dir = base_out_dir / f"ablation_{mode_spec.slug}"
    mode_out_dir.mkdir(parents=True, exist_ok=True)

    active_sites = list(active_sites)
    active_kind_counts = kind_counts_for_sites(controller, active_sites)
    save_csv(site_summary_rows(controller, sites=active_sites), mode_out_dir / "active_site_summary.csv")
    save_csv(site_summary_rows(controller), mode_out_dir / "site_summary_all.csv")

    min_budget, max_budget = budget_bounds(controller, sites=active_sites)
    budgets = select_budgets(min_budget, max_budget)

    print("\n" + "-" * 120)
    print(f"[ablation] mode={mode_spec.slug} | active_kinds={list(mode_spec.active_kinds)}")
    print(f"[ablation] active_sites={len(active_sites)} | active_kind_counts={active_kind_counts}")
    print(f"[ablation] budget-range feasible min={min_budget} max={max_budget}")
    print(f"[budgets] {budgets}")

    mode_proxy = filter_proxy_errors(shared_proxy, active_sites)
    if active_sites:
        save_proxy_subset(controller, mode_proxy, active_sites, mode_out_dir / PROXY_TABLE_NAME)

    result_rows: List[Dict[str, Any]] = []

    for budget in budgets:
        print("\n" + "-" * 120)
        print(f"[budget] {budget}")
        print("-" * 120)

        if active_sites:
            dp_schedule = dp_allocate_schedule(active_sites, controller, mode_proxy, int(budget))
            controller.set_schedule(dp_schedule)
            dp_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
            best_schedule = dict(dp_schedule)
            best_val_ev = dp_val_ev
            if ENABLE_GREEDY_VAL_REFINEMENT:
                best_schedule, best_val_ev = greedy_val_refine(
                    model,
                    controller,
                    active_sites,
                    dp_schedule,
                    budget,
                    val_loader,
                    base_val_logits,
                    float(base_val_acc),
                    device,
                )
        else:
            controller.set_exact()
            best_schedule = {}
            best_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)

        controller.set_schedule(best_schedule)
        final_cost = schedule_cost(best_schedule, controller)
        final_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
        final_test_ev = eval_model(model, test_loader, device, baseline_logits=base_test_logits)

        acc_drop_pp = (float(base_val_acc) - float(final_val_ev.acc)) * 100.0
        is_valid = (not final_val_ev.nonfinite_any)
        if ENABLE_VAL_ACC_GUARD and acc_drop_pp > float(VAL_ACC_GUARD_DROP_PP):
            is_valid = False

        schedule_dir = mode_out_dir / f"budget_{int(budget)}"
        schedule_dir.mkdir(parents=True, exist_ok=True)
        save_csv(schedule_to_rows(best_schedule, controller, sites=active_sites), schedule_dir / "schedule.csv")
        save_csv(schedule_to_rows(best_schedule, controller), schedule_dir / "schedule_full.csv")

        row = {
            "model_key": model_key,
            "ablation_mode": mode_spec.slug,
            "active_kinds": json.dumps(list(mode_spec.active_kinds), ensure_ascii=False),
            "num_active_sites": int(len(active_sites)),
            "budget": int(budget),
            "mode_min_budget": int(min_budget),
            "mode_max_budget": int(max_budget),
            "cost": int(final_cost),
            "val_acc": float(final_val_ev.acc),
            "val_acc_drop_pp": float(acc_drop_pp),
            "val_logit_mse": float(final_val_ev.logit_mse),
            "val_max_abs_logit": float(final_val_ev.max_abs_logit),
            "test_acc": float(final_test_ev.acc),
            "test_acc_drop_pp": float((float(base_test_acc) - float(final_test_ev.acc)) * 100.0),
            "test_logit_mse": float(final_test_ev.logit_mse),
            "test_max_abs_logit": float(final_test_ev.max_abs_logit),
            "nonfinite_any": int(final_val_ev.nonfinite_any or final_test_ev.nonfinite_any),
            "is_valid": int(is_valid),
        }
        result_rows.append(row)

        print(
            f"[result] mode={mode_spec.slug} | budget={budget:>4d} | cost={final_cost:>4d} | "
            f"val_acc={_fmt_pct(final_val_ev.acc)} ({_fmt_pp((final_val_ev.acc - base_val_acc) * 100.0)}) | "
            f"test_acc={_fmt_pct(final_test_ev.acc)} ({_fmt_pp((final_test_ev.acc - base_test_acc) * 100.0)}) | "
            f"val_mse={final_val_ev.logit_mse:.6e} | test_mse={final_test_ev.logit_mse:.6e} | "
            f"nonfinite={(final_val_ev.nonfinite_any or final_test_ev.nonfinite_any)}"
        )

    mark_pareto_rows(result_rows)
    save_csv(result_rows, mode_out_dir / RESULT_CSV_NAME)
    with (mode_out_dir / RESULT_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": model_key,
                "ablation_mode": mode_spec.slug,
                "active_kinds": list(mode_spec.active_kinds),
                "num_active_sites": len(active_sites),
                "site_kind_counts": active_kind_counts,
                "baseline_val_acc": float(base_val_acc),
                "baseline_test_acc": float(base_test_acc),
                "min_budget": int(min_budget),
                "max_budget": int(max_budget),
                "budgets": result_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[saved] {mode_out_dir / RESULT_CSV_NAME}")
    print(f"[saved] {mode_out_dir / RESULT_JSON_NAME}")

    return {
        "mode": mode_spec.slug,
        "active_kinds": list(mode_spec.active_kinds),
        "active_sites": active_sites,
        "site_kind_counts": active_kind_counts,
        "min_budget": int(min_budget),
        "max_budget": int(max_budget),
        "result_rows": result_rows,
        "out_dir": mode_out_dir,
    }



def run_one_model(model_key: str) -> None:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model_key={model_key}. Available: {sorted(MODEL_SPECS)}")

    spec = MODEL_SPECS[model_key]
    model_dir = ensure_local_path(spec["model_dir"], what=f"model_dir[{model_key}]")
    tokenizer_dir = spec.get("tokenizer_dir", model_dir)
    legacy_out_dir = spec["output_dir"]
    out_dir = legacy_out_dir / ABLATION_ROOT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    compute_batch = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    print("\n" + "=" * 120)
    print(f"[model] {model_key}")
    print("=" * 120)
    print(f"[device] {device} | compute_batch={compute_batch}")
    print(f"[model_dir] {model_dir}")
    print(f"[tokenizer_dir] {tokenizer_dir}")
    print(f"[dataset_dir] {DATASET_DIR}")
    print(f"[output_dir] {out_dir}")

    set_runtime_profile(SEARCH_PROFILE)
    print(f"[search_profile] {SEARCH_PROFILE} | post_eval_profiles={POST_EVAL_PROFILES}")

    tokenizer, model = load_local_model_and_tokenizer(model_dir, tokenizer_dir)

    train_examples, dev_examples = load_local_text_splits(
        DATASET_DIR,
        dataset_format=DATASET_FORMAT,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN,
    )
    print(f"[dataset] train={len(train_examples)} dev/report={len(dev_examples)}")

    calib_idx, sweep_idx, val_idx = split_train_indices(
        len(train_examples),
        CALIB_N,
        SWEEP_N,
        VAL_N,
        GLOBAL_SEED,
    )
    if not sweep_idx:
        sweep_idx = list(calib_idx[: max(1, min(len(calib_idx), 64))])
    if not val_idx:
        val_idx = list(sweep_idx)
    test_idx = sample_indices(
        len(dev_examples),
        TEST_MAX_N if TEST_MAX_N is not None else len(dev_examples),
        GLOBAL_SEED + 1,
    )

    calib_ds = TokenizedTextDataset([train_examples[i] for i in calib_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    sweep_ds = TokenizedTextDataset([train_examples[i] for i in sweep_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    val_ds = TokenizedTextDataset([train_examples[i] for i in val_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    test_ds = TokenizedTextDataset([dev_examples[i] for i in test_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)

    calib_loader, calib_loader_batch = build_tuned_loader(calib_ds, tokenizer, compute_batch, device)
    sweep_loader, sweep_loader_batch = build_tuned_loader(sweep_ds, tokenizer, compute_batch, device)
    val_loader, val_loader_batch = build_tuned_loader(val_ds, tokenizer, compute_batch, device)
    test_loader, test_loader_batch = build_tuned_loader(test_ds, tokenizer, compute_batch, device)
    proxy_stage1_loader = None
    proxy_stage1_loader_batch = 0
    if ENABLE_PROXY_STAGE1_PRUNE:
        proxy_stage1_take = min(int(PROXY_STAGE1_N), len(sweep_ds))
        proxy_stage1_idx = sample_indices(len(sweep_ds), proxy_stage1_take, GLOBAL_SEED + 21)
        proxy_stage1_ds = Subset(sweep_ds, proxy_stage1_idx)
        proxy_stage1_loader, proxy_stage1_loader_batch = build_tuned_loader(proxy_stage1_ds, tokenizer, compute_batch, device)
    if LOG_TUNED_LOADER_BATCHES:
        msg = f"[loader-batch] calib={calib_loader_batch} sweep={sweep_loader_batch} val={val_loader_batch} test={test_loader_batch}"
        if proxy_stage1_loader is not None:
            msg += f" proxy1={proxy_stage1_loader_batch}"
        print(msg)

    choices_by_kind = build_choice_space()
    controller = ApproxController(choices_by_kind)
    model = instrument_bert_model(model, controller).to(device).eval()

    all_sites = controller.ordered_sites()
    save_csv(site_summary_rows(controller), out_dir / "site_summary.csv")
    print(f"[sites] total={len(all_sites)}")
    kind_counts = kind_counts_for_sites(controller, all_sites)
    print(f"[sites] by kind={kind_counts}")
    min_budget, max_budget = budget_bounds(controller, sites=all_sites)
    print(f"[budget-range] all-sites feasible min={min_budget} max={max_budget}")

    phase_timing: Dict[str, float] = {}
    t_phase = time.perf_counter()
    controller.set_exact()
    controller.collect_calibration = True
    _ = eval_model(model, calib_loader, device, baseline_logits=None)
    controller.collect_calibration = False
    phase_timing["calibration_s"] = float(time.perf_counter() - t_phase)
    save_csv(controller.calibration.rows(), out_dir / "calibration_ranges.csv")
    print(f"[calib] saved calibration ranges to {out_dir / 'calibration_ranges.csv'}")
    save_csv(softmax_calib_max_lut_rows(controller), out_dir / SOFTMAX_CALIB_MAX_LUT_CSV_NAME)
    print(f"[calib] saved softmax calib-max LUT to {out_dir / SOFTMAX_CALIB_MAX_LUT_CSV_NAME}")

    t_phase = time.perf_counter()
    controller.set_exact()
    base_sweep_logits, _ = compute_logits(model, sweep_loader, device)
    base_proxy1_logits = None
    if proxy_stage1_loader is not None:
        base_proxy1_logits, _ = compute_logits(model, proxy_stage1_loader, device)
    base_val_logits, base_val_acc = compute_logits(model, val_loader, device)
    base_test_logits, base_test_acc = compute_logits(model, test_loader, device)
    if MOVE_BASELINE_LOGITS_TO_DEVICE and device.type == "cuda":
        base_sweep_logits = base_sweep_logits.to(device, non_blocking=True)
        if base_proxy1_logits is not None:
            base_proxy1_logits = base_proxy1_logits.to(device, non_blocking=True)
        base_val_logits = base_val_logits.to(device, non_blocking=True)
        base_test_logits = base_test_logits.to(device, non_blocking=True)
    phase_timing["baseline_s"] = float(time.perf_counter() - t_phase)
    print(f"[baseline] exact internal VAL acc={_fmt_pct(base_val_acc)}")
    print(f"[baseline] exact report TEST acc={_fmt_pct(base_test_acc)}")

    shared_proxy_path = Path(JOINT_SHARED_PROXY_TABLE_PATH) if JOINT_SHARED_PROXY_TABLE_PATH is not None else (out_dir / SHARED_PROXY_TABLE_NAME)
    legacy_proxy_path = spec["output_dir"] / ABLATION_ROOT_SUBDIR / SHARED_PROXY_TABLE_NAME
    if JOINT_SHARED_PROXY_TABLE_PATH is None and legacy_proxy_path.exists() and not shared_proxy_path.exists():
        shared_proxy_path = legacy_proxy_path
    t_phase = time.perf_counter()
    shared_proxy, proxy_meta = build_proxy_error_table(
        model,
        controller,
        sweep_loader,
        base_sweep_logits,
        device,
        shared_proxy_path,
        sites=all_sites,
        prescreen_loader=proxy_stage1_loader,
        prescreen_baseline_logits=base_proxy1_logits,
        stage1_out_path=(out_dir / PROXY_STAGE1_TABLE_NAME) if ENABLE_PROXY_STAGE1_PRUNE else None,
    )
    phase_timing["proxy_total_s"] = float(time.perf_counter() - t_phase)
    phase_timing["proxy_stage1_s"] = float(proxy_meta.get("stage1_seconds", 0.0))
    phase_timing["proxy_stage2_s"] = float(proxy_meta.get("stage2_seconds", 0.0))
    phase_timing["proxy_total_candidates"] = float(proxy_meta.get("total_candidates", 0))
    phase_timing["proxy_stage2_evals"] = float(proxy_meta.get("stage2_evals", 0))
    phase_timing["proxy_survivors_total"] = float(proxy_meta.get("survivors_total", 0))

    mode_specs = resolve_ablation_modes(RUN_ABLATION_MODES)
    print(f"[ablation-modes] {[m.slug for m in mode_specs]}")

    aggregate_rows: List[Dict[str, Any]] = []
    mode_payloads: List[Dict[str, Any]] = []
    for mode_spec in mode_specs:
        active_sites = active_sites_for_mode(controller, mode_spec)
        _mode_t0 = time.perf_counter()
        mode_payload = run_one_ablation_mode(
            model_key=model_key,
            mode_spec=mode_spec,
            model=model,
            controller=controller,
            base_out_dir=out_dir,
            active_sites=active_sites,
            sweep_loader=sweep_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            base_sweep_logits=base_sweep_logits,
            base_val_logits=base_val_logits,
            base_test_logits=base_test_logits,
            base_val_acc=float(base_val_acc),
            base_test_acc=float(base_test_acc),
            device=device,
            shared_proxy=shared_proxy,
        )
        mode_payload["wall_s"] = float(time.perf_counter() - _mode_t0)
        mode_payloads.append({
            "mode": mode_payload["mode"],
            "active_kinds": mode_payload["active_kinds"],
            "num_active_sites": len(mode_payload["active_sites"]),
            "site_kind_counts": mode_payload["site_kind_counts"],
            "min_budget": mode_payload["min_budget"],
            "max_budget": mode_payload["max_budget"],
            "out_dir": str(mode_payload["out_dir"]),
            "wall_s": float(mode_payload.get("wall_s", 0.0)),
        })
        aggregate_rows.extend(mode_payload["result_rows"])

    mark_pareto_rows(aggregate_rows, flag_key="is_pareto_global")
    save_csv(aggregate_rows, out_dir / ABLATION_SUMMARY_CSV_NAME)
    with (out_dir / ABLATION_SUMMARY_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": model_key,
                "model_dir": str(model_dir),
                "tokenizer_dir": str(tokenizer_dir),
                "dataset_dir": str(DATASET_DIR),
                "output_dir": str(out_dir),
                "baseline_val_acc": float(base_val_acc),
                "baseline_test_acc": float(base_test_acc),
                "num_sites": len(all_sites),
                "site_kind_counts": kind_counts,
                "shared_proxy_table": str(shared_proxy_path),
                "modes": mode_payloads,
                "rows": aggregate_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    phase_timing["total_setup_s"] = float(phase_timing.get("calibration_s", 0.0) + phase_timing.get("baseline_s", 0.0) + phase_timing.get("proxy_total_s", 0.0))
    phase_timing["ablation_modes_total_s"] = float(sum(float(m.get("wall_s", 0.0)) for m in mode_payloads))
    with (out_dir / PHASE_TIMING_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key,
            "output_dir": str(out_dir),
            "phase_timing": phase_timing,
            "modes": mode_payloads,
        }, f, ensure_ascii=False, indent=2)
    print(f"[time] calib={phase_timing['calibration_s']:.2f}s baseline={phase_timing['baseline_s']:.2f}s proxy(stage1={phase_timing['proxy_stage1_s']:.2f}s stage2={phase_timing['proxy_stage2_s']:.2f}s total={phase_timing['proxy_total_s']:.2f}s)")
    print(f"[saved] {out_dir / ABLATION_SUMMARY_CSV_NAME}")
    print(f"[saved] {out_dir / ABLATION_SUMMARY_JSON_NAME}")
    print(f"[saved] {out_dir / PHASE_TIMING_JSON_NAME}")


# =============================================================================
# Joint refinement (FAST -> VAL -> gated FULL)
# =============================================================================

@dataclass
class JointTaskSpec:
    model_key: str
    mode: str
    budget: int
    schedule_csv: Optional[Path] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    budget_band: Optional[int] = None


@dataclass
class JointStageRecord:
    stage: str
    budget: int
    cost: int
    val_acc: float
    val_acc_drop_pp: float
    val_logit_mse: float
    val_max_abs_logit: float
    val_nonfinite_samples: int
    full_acc: float
    full_acc_drop_pp: float
    full_logit_mse: float
    full_max_abs_logit: float
    full_nonfinite_samples: int
    test_acc: float
    test_acc_drop_pp: float
    test_logit_mse: float
    test_max_abs_logit: float
    test_nonfinite_samples: int
    full_ok: bool
    nonfinite_any: bool
    val_obj: float


def resolve_joint_tasks(task_dicts: Sequence[Dict[str, Any]]) -> List[JointTaskSpec]:
    tasks: List[JointTaskSpec] = []
    for raw in task_dicts:
        if "model_key" not in raw or "mode" not in raw or "budget" not in raw:
            raise KeyError(f"Each JOINT_TASKS item must contain model_key/mode/budget. Got: {raw}")
        schedule_csv = raw.get("schedule_csv", None)
        tasks.append(
            JointTaskSpec(
                model_key=str(raw["model_key"]),
                mode=str(raw["mode"]),
                budget=int(raw["budget"]),
                schedule_csv=(Path(schedule_csv).expanduser() if schedule_csv else None),
                budget_min=(int(raw["budget_min"]) if raw.get("budget_min", None) is not None else None),
                budget_max=(int(raw["budget_max"]) if raw.get("budget_max", None) is not None else None),
                budget_band=(int(raw["budget_band"]) if raw.get("budget_band", None) is not None else None),
            )
        )
    return tasks


def format_budget_band(lo: int, hi: int) -> str:
    lo = int(lo)
    hi = int(hi)
    return str(lo) if lo == hi else f"{lo}..{hi}"


def resolve_joint_budget_band(task: JointTaskSpec, min_budget: int, max_budget: int) -> Tuple[int, int]:
    report_budget = int(task.budget)
    lo = report_budget
    hi = report_budget

    if task.budget_min is not None:
        lo = int(task.budget_min)
    if task.budget_max is not None:
        hi = int(task.budget_max)
    elif task.budget_band is not None:
        hi = report_budget + int(task.budget_band)
    elif JOINT_ENABLE_AUTO_MIN_BUDGET_BAND and report_budget <= int(min_budget):
        hi = report_budget + int(JOINT_AUTO_MIN_BUDGET_BAND_WIDTH)

    lo = max(int(min_budget), int(lo))
    hi = min(int(max_budget), int(hi))
    if hi < lo:
        hi = lo
    return int(lo), int(hi)


def default_joint_schedule_csv(model_key: str, mode_slug: str, budget: int) -> Optional[Path]:
    if model_key not in MODEL_SPECS:
        return None
    root = MODEL_SPECS[model_key]["output_dir"] / ABLATION_ROOT_SUBDIR / f"ablation_{mode_slug}" / f"budget_{int(budget)}"
    cand1 = root / "schedule.csv"
    cand2 = root / "schedule_full.csv"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def load_schedule_csv(path: Path, controller: ApproxController) -> Dict[str, int]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"schedule csv not found: {path}")
    sched: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            site = str(row.get("site", "")).strip()
            choice_id_raw = str(row.get("choice_id", "")).strip()
            label_raw = str(row.get("label", "")).strip()
            if not site:
                continue
            if site not in controller.site_kind:
                # allow legacy rows like __TOTAL__
                continue
            if choice_id_raw.lower() == "exact" or choice_id_raw == "":
                if label_raw.lower() == "exact" or label_raw == "":
                    continue

            choice: Optional[ApproxChoice] = None
            if label_raw and label_raw.lower() != "exact":
                for ch in controller.choices_for_site(site):
                    if str(ch.label) == label_raw:
                        choice = ch
                        break
            if choice is None and choice_id_raw and choice_id_raw.lower() != "exact":
                try:
                    cid = int(choice_id_raw)
                    for ch in controller.choices_for_site(site):
                        if int(ch.choice_id) == cid:
                            choice = ch
                            break
                except Exception:
                    choice = None
            if choice is None:
                raise KeyError(
                    f"Cannot resolve schedule row for site={site}: choice_id={choice_id_raw!r}, label={label_raw!r}"
                )
            sched[site] = int(choice.choice_id)
    return sched


def apply_move(schedule: Dict[str, int], move: Dict[str, int]) -> Dict[str, int]:
    out = dict(schedule)
    out.update({k: int(v) for k, v in move.items()})
    return out


def schedule_key_for_sites(schedule: Dict[str, int], sites: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(schedule.get(site, -1)) for site in sites)


def build_site_choice_maps(
    controller: ApproxController,
    active_sites: Sequence[str],
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, List[int]]]]:
    choice_costs: Dict[str, Dict[int, int]] = {}
    cost_buckets: Dict[str, Dict[int, List[int]]] = {}
    for site in active_sites:
        c2c: Dict[int, int] = {}
        buckets: Dict[int, List[int]] = defaultdict(list)
        for ch in controller.choices_for_site(site):
            cid = int(ch.choice_id)
            cc = int(ch.cost)
            c2c[cid] = cc
            buckets[cc].append(cid)
        choice_costs[site] = c2c
        cost_buckets[site] = dict(buckets)
    return choice_costs, cost_buckets


def compute_site_importance_rank_from_proxy(
    active_sites: Sequence[str],
    start_schedule: Dict[str, int],
    proxy_errors: Dict[str, Dict[int, float]],
) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for site in active_sites:
        per = proxy_errors.get(site, {})
        if not per:
            scored.append((0.0, site))
            continue
        cur_id = int(start_schedule.get(site, next(iter(per.keys()))))
        cur_e = float(per.get(cur_id, min(per.values())))
        best_e = float(min(per.values()))
        span_e = float(max(per.values()) - best_e)
        score = max(cur_e - best_e, 0.0) + 0.5 * span_e
        scored.append((float(score), site))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [site for _score, site in scored]


def joint_proposal_pool(active_sites: Sequence[str], importance_rank: Sequence[str]) -> List[str]:
    if not JOINT_ENABLE_SITE_PRIORITY_PROPOSAL:
        return list(active_sites)
    pool: List[str] = []
    seen = set()
    for site in list(importance_rank)[: int(JOINT_PROPOSAL_TOPM_IMPORTANCE)]:
        if site in seen:
            continue
        if site in active_sites:
            pool.append(site)
            seen.add(site)
        if len(pool) >= int(JOINT_PROPOSAL_POOL_MAX):
            break
    return pool if pool else list(active_sites)


def get_near_cost_choice_ids(
    site: str,
    cur_id: int,
    site_choice_costs: Dict[str, Dict[int, int]],
    site_cost_buckets: Dict[str, Dict[int, List[int]]],
    max_cost_delta: int,
) -> List[int]:
    cur_cost = int(site_choice_costs[site][int(cur_id)])
    delta = max(0, int(max_cost_delta))
    out: List[int] = []
    for cost, ids in site_cost_buckets[site].items():
        if abs(int(cost) - cur_cost) > delta:
            continue
        for cid in ids:
            cid = int(cid)
            if cid == int(cur_id):
                continue
            out.append(cid)
    out = sorted(
        set(out),
        key=lambda cid: (abs(int(site_choice_costs[site][int(cid)]) - cur_cost), int(site_choice_costs[site][int(cid)]), int(cid)),
    )
    return out


def random_joint_local_single_move(
    rng: np.random.Generator,
    site_pool: Sequence[str],
    schedule: Dict[str, int],
    site_cost_buckets: Dict[str, Dict[int, List[int]]],
    site_choice_costs: Dict[str, Dict[int, int]],
    max_cost_delta: int,
    max_tries: int = 8,
) -> Optional[Dict[str, int]]:
    if not site_pool:
        return None
    pool = list(site_pool)
    for _ in range(max(1, int(max_tries))):
        site = str(rng.choice(np.array(pool, dtype=object)))
        cur = int(schedule[site])
        options = get_near_cost_choice_ids(site, cur, site_choice_costs, site_cost_buckets, int(max_cost_delta))
        if not options:
            continue
        new_cid = int(rng.choice(np.array(options, dtype=np.int64)))
        return {site: new_cid}
    return None


def random_joint_cost_neutral_swap_move(
    rng: np.random.Generator,
    site_pool: Sequence[str],
    active_sites: Sequence[str],
    schedule: Dict[str, int],
    site_cost_buckets: Dict[str, Dict[int, List[int]]],
    site_choice_costs: Dict[str, Dict[int, int]],
) -> Optional[Dict[str, int]]:
    if len(active_sites) < 2 or not site_pool:
        return None
    s1 = str(rng.choice(np.array(list(site_pool), dtype=object)))
    s2_pool = list(site_pool) if rng.random() < 0.7 else list(active_sites)
    s2 = str(rng.choice(np.array(list(s2_pool), dtype=object)))
    if s1 == s2:
        return None

    cur1 = int(schedule[s1]); cur2 = int(schedule[s2])
    c1 = int(site_choice_costs[s1][cur1]); c2 = int(site_choice_costs[s2][cur2])

    all_ids_1 = [cid for cids in site_cost_buckets[s1].values() for cid in cids if int(cid) != cur1]
    if not all_ids_1:
        return None
    new1 = int(rng.choice(np.array(all_ids_1, dtype=np.int64)))
    nc1 = int(site_choice_costs[s1][new1])
    delta = nc1 - c1
    target_c2 = c2 - delta
    options2 = [cid for cid in site_cost_buckets[s2].get(target_c2, []) if int(cid) != cur2]
    if not options2:
        return None
    new2 = int(rng.choice(np.array(options2, dtype=np.int64)))
    return {s1: new1, s2: new2}


def random_joint_cost_neutral_tri_move(
    rng: np.random.Generator,
    site_pool: Sequence[str],
    active_sites: Sequence[str],
    schedule: Dict[str, int],
    site_cost_buckets: Dict[str, Dict[int, List[int]]],
    site_choice_costs: Dict[str, Dict[int, int]],
    max_tries: int = 20,
) -> Optional[Dict[str, int]]:
    if (not JOINT_ENABLE_TRI_MOVES) or len(active_sites) < 3 or not site_pool:
        return None
    for _ in range(int(max_tries)):
        s1 = str(rng.choice(np.array(list(site_pool), dtype=object)))
        s2 = str(rng.choice(np.array(list(site_pool), dtype=object)))
        s3_pool = list(site_pool) if rng.random() < 0.6 else list(active_sites)
        s3 = str(rng.choice(np.array(list(s3_pool), dtype=object)))
        if len({s1, s2, s3}) != 3:
            continue

        cur1 = int(schedule[s1]); cur2 = int(schedule[s2]); cur3 = int(schedule[s3])
        c1 = int(site_choice_costs[s1][cur1]); c2 = int(site_choice_costs[s2][cur2]); c3 = int(site_choice_costs[s3][cur3])

        all_ids_1 = [cid for cids in site_cost_buckets[s1].values() for cid in cids if int(cid) != cur1]
        all_ids_2 = [cid for cids in site_cost_buckets[s2].values() for cid in cids if int(cid) != cur2]
        if not all_ids_1 or not all_ids_2:
            continue

        new1 = int(rng.choice(np.array(all_ids_1, dtype=np.int64)))
        new2 = int(rng.choice(np.array(all_ids_2, dtype=np.int64)))
        nc1 = int(site_choice_costs[s1][new1]); nc2 = int(site_choice_costs[s2][new2])
        delta = (nc1 - c1) + (nc2 - c2)
        target_c3 = c3 - delta
        options3 = [cid for cid in site_cost_buckets[s3].get(target_c3, []) if int(cid) != cur3]
        if not options3:
            continue
        new3 = int(rng.choice(np.array(options3, dtype=np.int64)))
        return {s1: new1, s2: new2, s3: new3}
    return None


def joint_val_acc_floor(*, current_acc: float, base_val_acc: float) -> float:
    if not JOINT_ENABLE_VAL_ACC_GUARD:
        return -1.0e18
    base_floor = float(base_val_acc) - float(JOINT_VAL_ACC_GUARD_DROP_PP) / 100.0
    if JOINT_ENABLE_CURRENT_VAL_ACC_GUARD:
        cur_floor = float(current_acc) - float(JOINT_VAL_ACC_CURRENT_DROP_PP) / 100.0
        return float(max(0.0, base_floor, cur_floor))
    if (not JOINT_ENABLE_ADAPTIVE_VAL_ACC_GUARD) or (float(current_acc) >= base_floor):
        return float(base_floor)
    cur_floor = float(current_acc) - float(JOINT_VAL_ACC_CURRENT_DROP_PP) / 100.0
    return float(max(0.0, cur_floor))


def format_joint_reject_stats(stats: Dict[str, int]) -> str:
    if not stats:
        return "none"
    items = [(str(k), int(v)) for k, v in stats.items() if int(v) != 0]
    if not items:
        return "none"
    items.sort(key=lambda kv: kv[0])
    return " ".join(f"{k}={v}" for k, v in items)


def joint_nonfinite_ok(ev: EvalResult, *, split: str) -> bool:
    split = str(split).lower()
    rate = eval_nonfinite_rate(ev)
    count = int(ev.nonfinite_sample_count)
    if split == "fast":
        return (count <= int(JOINT_FAST_MAX_NONFINITE_SAMPLES)) or (rate <= float(JOINT_FAST_MAX_NONFINITE_RATE))
    if split == "val":
        return (count <= int(JOINT_VAL_MAX_NONFINITE_SAMPLES)) or (rate <= float(JOINT_VAL_MAX_NONFINITE_RATE))
    if split == "full":
        return (count <= int(JOINT_FULL_MAX_NONFINITE_SAMPLES)) or (rate <= float(JOINT_FULL_MAX_NONFINITE_RATE))
    raise ValueError(f"Unknown split for joint_nonfinite_ok: {split}")


def joint_nonfinite_soft_ok(ev: EvalResult, *, split: str) -> bool:
    split = str(split).lower()
    if split != "val":
        return joint_nonfinite_ok(ev, split=split)
    rate = eval_nonfinite_rate(ev)
    count = int(ev.nonfinite_sample_count)
    return (count <= int(JOINT_VAL_SOFT_MAX_NONFINITE_SAMPLES)) or (rate <= float(JOINT_VAL_SOFT_MAX_NONFINITE_RATE))


def joint_val_objective(ev: EvalResult, base_abs_logit: float) -> float:
    base_abs = max(float(base_abs_logit), 1e-6)
    p_abs = 0.0
    if ev.max_abs_logit > float(JOINT_LOGIT_ABS_SOFT_MULT) * base_abs:
        p_abs = ((ev.max_abs_logit - float(JOINT_LOGIT_ABS_SOFT_MULT) * base_abs) / (base_abs + 1e-12)) ** 2
    nf_rate = eval_nonfinite_rate(ev)
    p_nf = float(JOINT_NONFINITE_PENALTY_WEIGHT) * nf_rate
    if ev.nonfinite_any:
        # keep rare-nonfinite schedules searchable, but strongly discourage widespread nonfinites
        p_nf += float(JOINT_NONFINITE_HARD_PENALTY) * max(0.0, nf_rate - float(JOINT_FULL_MAX_NONFINITE_RATE))
    return float(ev.logit_mse + float(JOINT_LOGIT_ABS_PENALTY_WEIGHT) * p_abs + p_nf)


def candidate_admissible_joint(
    *,
    cur_obj: float,
    cur_acc: float,
    cand_obj: float,
    cand_acc: float,
) -> bool:
    improv_eps = max(
        1e-12,
        float(JOINT_MIN_OBJ_IMPROV_ABS),
        abs(float(cur_obj)) * float(JOINT_MIN_OBJ_IMPROV_REL),
    )
    if cand_obj <= cur_obj - improv_eps:
        return True
    if not JOINT_ENABLE_ACC_OBJ_TRADEOFF:
        return False
    min_gain = float(JOINT_ACC_GAIN_MIN_PP) / 100.0
    if cand_acc >= cur_acc + min_gain:
        if cand_obj <= cur_obj * float(JOINT_OBJ_WORSEN_MAX_RATIO) + 1e-18:
            return True
    return False


def should_run_joint_full_guard(accepts_so_far: int, rel_improv: float) -> bool:
    if int(JOINT_FULL_GUARD_EVERY_K_ACCEPTS) > 0 and (accepts_so_far % int(JOINT_FULL_GUARD_EVERY_K_ACCEPTS) == 0):
        return True
    if rel_improv >= float(JOINT_FULL_GUARD_TRIGGER_REL_IMPROV):
        return True
    return False


def joint_full_guard_ok(
    ev_full: EvalResult,
    *,
    base_full_acc: float,
    base_full_abs_logit: float,
) -> bool:
    if not joint_nonfinite_ok(ev_full, split="full"):
        return False
    if JOINT_ENABLE_FULL_ACC_GUARD and (ev_full.acc < (float(base_full_acc) - float(JOINT_FULL_ACC_GUARD_DROP_PP) / 100.0)):
        return False
    base_abs = max(float(base_full_abs_logit), 1e-6)
    if ev_full.max_abs_logit > float(JOINT_LOGIT_ABS_HARD_MULT) * base_abs:
        return False
    return True


def stability_rescue_accept(
    *,
    cur_full_ev: EvalResult,
    cand_full_ev: EvalResult,
    cur_val_ev: EvalResult,
    cand_val_ev: EvalResult,
    base_full_acc: float,
    base_full_abs_logit: float,
) -> bool:
    if not JOINT_ENABLE_STABILITY_RESCUE:
        return False

    max_val_drop = float(JOINT_STABILITY_RESCUE_MAX_VAL_DROP_PP) / 100.0
    if float(cand_val_ev.acc) < float(cur_val_ev.acc) - max_val_drop:
        return False

    cur_ok = joint_full_guard_ok(cur_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
    cand_ok = joint_full_guard_ok(cand_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
    if (not cur_ok) and cand_ok:
        return True

    cur_nf = eval_nonfinite_rate(cur_full_ev)
    cand_nf = eval_nonfinite_rate(cand_full_ev)
    if cand_nf <= cur_nf - float(JOINT_STABILITY_RESCUE_MIN_NONFINITE_DROP):
        return True

    cur_abs = max(float(cur_full_ev.max_abs_logit), 1e-12)
    cand_abs = float(cand_full_ev.max_abs_logit)
    if cand_abs <= cur_abs * (1.0 - float(JOINT_STABILITY_RESCUE_MIN_ABSLOGIT_REL_DROP)):
        return True

    return False


def evaluate_stage_record(
    *,
    stage: str,
    budget: int,
    controller: ApproxController,
    schedule: Dict[str, int],
    model: nn.Module,
    val_loader: DataLoader,
    full_loader: DataLoader,
    test_loader: DataLoader,
    base_val_logits: torch.Tensor,
    base_full_logits: torch.Tensor,
    base_test_logits: torch.Tensor,
    base_val_acc: float,
    base_full_acc: float,
    base_test_acc: float,
    base_val_abs_logit: float,
    base_full_abs_logit: float,
    device: torch.device,
) -> Tuple[JointStageRecord, EvalResult, EvalResult, EvalResult]:
    controller.set_schedule(schedule)
    val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
    full_ev = eval_model(model, full_loader, device, baseline_logits=base_full_logits)
    test_ev = eval_model(model, test_loader, device, baseline_logits=base_test_logits)
    rec = JointStageRecord(
        stage=str(stage),
        budget=int(budget),
        cost=int(schedule_cost(schedule, controller)),
        val_acc=float(val_ev.acc),
        val_acc_drop_pp=float((float(base_val_acc) - float(val_ev.acc)) * 100.0),
        val_logit_mse=float(val_ev.logit_mse),
        val_max_abs_logit=float(val_ev.max_abs_logit),
        val_nonfinite_samples=int(val_ev.nonfinite_sample_count),
        full_acc=float(full_ev.acc),
        full_acc_drop_pp=float((float(base_full_acc) - float(full_ev.acc)) * 100.0),
        full_logit_mse=float(full_ev.logit_mse),
        full_max_abs_logit=float(full_ev.max_abs_logit),
        full_nonfinite_samples=int(full_ev.nonfinite_sample_count),
        test_acc=float(test_ev.acc),
        test_acc_drop_pp=float((float(base_test_acc) - float(test_ev.acc)) * 100.0),
        test_logit_mse=float(test_ev.logit_mse),
        test_max_abs_logit=float(test_ev.max_abs_logit),
        test_nonfinite_samples=int(test_ev.nonfinite_sample_count),
        full_ok=bool(joint_full_guard_ok(full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)),
        nonfinite_any=bool(val_ev.nonfinite_any or full_ev.nonfinite_any or test_ev.nonfinite_any),
        val_obj=float(joint_val_objective(val_ev, base_val_abs_logit)),
    )
    return rec, val_ev, full_ev, test_ev


def select_best_joint_stage(records: Sequence[JointStageRecord]) -> JointStageRecord:
    if not records:
        raise ValueError("No joint stage records to choose from")
    strict = [r for r in records if r.full_ok]
    pool = strict if strict else list(records)
    best_acc = max(float(r.val_acc) for r in pool)
    tol = float(JOINT_FINAL_ACC_TOL_PP) / 100.0
    acc_pool = [r for r in pool if float(r.val_acc) >= best_acc - tol]
    if strict:
        best_obj = min(float(r.val_obj) for r in acc_pool)
        obj_eps = max(
            float(JOINT_FINAL_OBJ_EPS_ABS),
            abs(float(best_obj)) * float(JOINT_FINAL_OBJ_EPS_REL),
        )
        if obj_eps > 0.0:
            stage_rank = {"initial": 0, "joint": 1, "global_cd": 2}
            near_obj_pool = [r for r in acc_pool if float(r.val_obj) <= best_obj + obj_eps]
            near_obj_pool.sort(
                key=lambda r: (
                    float(r.full_nonfinite_samples),
                    float(r.full_max_abs_logit),
                    float(r.val_nonfinite_samples),
                    int(stage_rank.get(str(r.stage), 99)),
                    float(r.val_obj),
                    -float(r.val_acc),
                    float(r.cost),
                )
            )
            return near_obj_pool[0]
    if strict:
        acc_pool.sort(key=lambda r: (float(r.val_obj), float(r.full_nonfinite_samples), float(r.full_max_abs_logit), -float(r.val_acc), float(r.cost)))
    else:
        acc_pool.sort(key=lambda r: (float(r.full_nonfinite_samples), float(r.full_max_abs_logit), float(r.val_obj), -float(r.val_acc), float(r.cost)))
    return acc_pool[0]


def run_joint_search(
    *,
    model: nn.Module,
    controller: ApproxController,
    active_sites: Sequence[str],
    start_schedule: Dict[str, int],
    budget: int,
    search_budget_lo: int,
    search_budget_hi: int,
    fast_loader: DataLoader,
    val_loader: DataLoader,
    full_lite_loader: DataLoader,
    full_loader: DataLoader,
    base_fast_logits: torch.Tensor,
    base_val_logits: torch.Tensor,
    base_full_lite_logits: torch.Tensor,
    base_full_logits: torch.Tensor,
    base_val_acc: float,
    base_full_lite_acc: float,
    base_full_acc: float,
    base_fast_abs_logit: float,
    base_val_abs_logit: float,
    base_full_lite_abs_logit: float,
    base_full_abs_logit: float,
    device: torch.device,
    rng: np.random.Generator,
    importance_rank: Sequence[str],
) -> Tuple[Dict[str, int], EvalResult, EvalResult]:
    active_sites = list(active_sites)
    current = dict(start_schedule)
    controller.set_schedule(current)
    current_key = schedule_key_for_sites(current, active_sites)
    current_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
    current_full_lite_ev = eval_model(model, full_lite_loader, device, baseline_logits=base_full_lite_logits)
    current_full_ev = eval_model(model, full_loader, device, baseline_logits=base_full_logits)
    current_obj = float(joint_val_objective(current_val_ev, base_val_abs_logit))
    current_acc = float(current_val_ev.acc)

    site_choice_costs, site_cost_buckets = build_site_choice_maps(controller, active_sites)
    prop_pool = joint_proposal_pool(active_sites, importance_rank)
    fast_cache: Dict[Tuple[int, ...], EvalResult] = {}
    val_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_VAL_CACHE else None
    full_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_FULL_CACHE else None
    if val_cache is not None:
        val_cache[current_key] = current_val_ev
    if full_cache is not None:
        full_cache[current_key] = current_full_ev

    band_str = format_budget_band(int(search_budget_lo), int(search_budget_hi))
    print(
        f"[joint] init: cost={schedule_cost(current, controller)}/{band_str} [report={int(budget)}] | "
        f"val_obj={current_obj:.6e} val_acc={_fmt_pct(current_acc)} | "
        f"full_acc={_fmt_pct(current_full_ev.acc)} full_mse={current_full_ev.logit_mse:.6e} "
        f"full_abs_logit={current_full_ev.max_abs_logit:.3e} full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total} "
        f"full_ok={joint_full_guard_ok(current_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)}"
    )

    accepts = 0
    for step in range(1, int(JOINT_MAX_STEPS) + 1):
        step_t0 = time.perf_counter()
        cand_moves: List[Dict[str, int]] = []
        for _ in range(int(JOINT_MOVE_TRIALS_PER_STEP)):
            use_pool = JOINT_ENABLE_SITE_PRIORITY_PROPOSAL and (rng.random() < float(JOINT_PROPOSAL_POOL_PROB))
            site_pool = prop_pool if (use_pool and prop_pool) else active_sites
            r = float(rng.random())
            if r < float(JOINT_SINGLE_MOVE_PROB):
                mv = random_joint_local_single_move(
                    rng,
                    site_pool,
                    current,
                    site_cost_buckets,
                    site_choice_costs,
                    max_cost_delta=int(JOINT_MAX_LOCAL_COST_DELTA),
                )
            elif r < float(JOINT_SINGLE_MOVE_PROB) + float(JOINT_SWAP_MOVE_PROB):
                mv = random_joint_cost_neutral_swap_move(rng, site_pool, active_sites, current, site_cost_buckets, site_choice_costs)
            else:
                mv = random_joint_cost_neutral_tri_move(rng, site_pool, active_sites, current, site_cost_buckets, site_choice_costs)
            if mv is not None:
                cand_moves.append(mv)

        if JOINT_DETERMINISTIC_SAMECOST_FALLBACK:
            for site in list(importance_rank)[: max(1, int(JOINT_DETERMINISTIC_TOP_SITES))]:
                if site not in active_sites:
                    continue
                cur_id = int(current[site])
                for cid in get_near_cost_choice_ids(site, cur_id, site_choice_costs, site_cost_buckets, int(JOINT_MAX_LOCAL_COST_DELTA)):
                    cand_moves.append({site: int(cid)})

        uniq_moves: List[Dict[str, int]] = []
        seen = set()
        for mv in cand_moves:
            key = tuple(sorted((str(k), int(v)) for k, v in mv.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq_moves.append(mv)
        if not uniq_moves:
            print(f"[joint] stop@step{step:03d}: no candidates generated.")
            break

        fast_stats: Dict[str, int] = defaultdict(int)
        fast_stats["generated"] = int(len(cand_moves))
        fast_stats["uniq"] = int(len(uniq_moves))
        fast_scored: List[Tuple[float, Dict[str, int], EvalResult, Tuple[int, ...]]] = []
        for mv in uniq_moves:
            trial = apply_move(current, mv)
            trial_cost = int(schedule_cost(trial, controller))
            if trial_cost < int(search_budget_lo) or trial_cost > int(search_budget_hi):
                fast_stats["budget"] += 1
                continue
            key = schedule_key_for_sites(trial, active_sites)
            ev_fast = eval_with_cache(
                fast_cache,
                key,
                lambda trial=trial: (controller.set_schedule(trial), eval_model(model, fast_loader, device, baseline_logits=base_fast_logits))[1],
            )
            if ev_fast.nonfinite_any and (not joint_nonfinite_ok(ev_fast, split="fast")):
                fast_stats["fast_nonfinite"] += 1
                continue
            score = float(joint_val_objective(ev_fast, base_fast_abs_logit))
            fast_scored.append((score, mv, ev_fast, key))
        fast_stats["fast_pass"] = int(len(fast_scored))
        if not fast_scored:
            if JOINT_LOG_REJECTION_STATS:
                elapsed = time.perf_counter() - step_t0
                print(f"[joint] step{step:03d} rejects | time={elapsed:.2f}s | fast{{{format_joint_reject_stats(fast_stats)}}}")
            print(f"[joint] stop@step{step:03d}: no FAST-feasible candidates.")
            break

        fast_scored.sort(key=lambda x: float(x[0]))
        top_fast = fast_scored[: max(int(JOINT_FAST_KEEP), int(JOINT_TOPK_VAL))]

        val_stats: Dict[str, int] = defaultdict(int)
        val_floor = float(joint_val_acc_floor(current_acc=current_acc, base_val_acc=base_val_acc))
        val_cands: List[Tuple[int, float, float, Dict[str, int], EvalResult, bool, Tuple[int, ...]]] = []
        for _score_fast, mv, _fast_ev, key in top_fast[: int(JOINT_TOPK_VAL)]:
            trial = apply_move(current, mv)
            ev_val = eval_with_cache(
                val_cache,
                key,
                lambda trial=trial: (controller.set_schedule(trial), eval_model(model, val_loader, device, baseline_logits=base_val_logits))[1],
            )
            val_nf_strict = (not ev_val.nonfinite_any) or joint_nonfinite_ok(ev_val, split="val")
            val_nf_soft = val_nf_strict or (
                bool(JOINT_ENABLE_VAL_NONFINITE_SOFT_PASS) and joint_nonfinite_soft_ok(ev_val, split="val")
            )
            if not val_nf_soft:
                val_stats["val_nonfinite_hard"] += 1
                continue
            if (not val_nf_strict) and ev_val.nonfinite_any:
                val_stats["val_nonfinite_soft"] += 1
            if JOINT_ENABLE_VAL_ACC_GUARD and (ev_val.acc < val_floor):
                val_stats["val_acc"] += 1
                continue
            obj = float(joint_val_objective(ev_val, base_val_abs_logit))
            nf_rank = 0 if val_nf_strict else 1
            val_cands.append((nf_rank, obj, -float(ev_val.acc), mv, ev_val, val_nf_strict, key))
        val_stats["val_pass"] = int(len(val_cands))

        if not val_cands:
            if JOINT_LOG_REJECTION_STATS:
                elapsed = time.perf_counter() - step_t0
                print(
                    f"[joint] step{step:03d} rejects | time={elapsed:.2f}s | val_floor={_fmt_pct(val_floor)} | "
                    f"fast{{{format_joint_reject_stats(fast_stats)}}} val{{{format_joint_reject_stats(val_stats)}}}"
                )
            print(f"[joint] stop@step{step:03d}: no VAL-feasible candidates.")
            break

        val_cands.sort(key=lambda x: (int(x[0]), float(x[1]), float(x[2]), len(x[3])))
        accepted = False
        cur_full_ok = joint_full_guard_ok(current_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
        dec_stats: Dict[str, int] = defaultdict(int)

        for rank, (nf_rank, obj, _neg_acc, mv, ev_val, val_nf_strict, key) in enumerate(val_cands[: max(int(JOINT_PARETO_MAX_TRY), int(JOINT_FULL_GUARD_TOP_CANDS_PER_STEP), 1)]):
            trial = apply_move(current, mv)
            rel_improv = max(0.0, (current_obj - float(obj)) / max(current_obj, 1e-12))
            run_full = (not cur_full_ok) or (not bool(val_nf_strict)) or should_run_joint_full_guard(accepts + 1, rel_improv) or (rank < int(JOINT_FULL_GUARD_TOP_CANDS_PER_STEP))
            full_msg = "nofull"
            ev_full = current_full_ev
            cand_adm = candidate_admissible_joint(cur_obj=current_obj, cur_acc=current_acc, cand_obj=float(obj), cand_acc=float(ev_val.acc))

            if run_full:
                ev_full = eval_with_cache(
                    full_cache,
                    key,
                    lambda trial=trial: (controller.set_schedule(trial), eval_model(model, full_loader, device, baseline_logits=base_full_logits))[1],
                )
                full_ok = joint_full_guard_ok(ev_full, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
                rescue_ok = stability_rescue_accept(
                    cur_full_ev=current_full_ev,
                    cand_full_ev=ev_full,
                    cur_val_ev=current_val_ev,
                    cand_val_ev=ev_val,
                    base_full_acc=base_full_acc,
                    base_full_abs_logit=base_full_abs_logit,
                )
                if not ((cand_adm and full_ok) or rescue_ok):
                    if not cand_adm:
                        dec_stats["obj"] += 1
                    if not full_ok:
                        dec_stats["full"] += 1
                    if (not rescue_ok) and (not full_ok):
                        dec_stats["rescue"] += 1
                    continue
                if not bool(val_nf_strict):
                    dec_stats["val_soft_used"] += 1
                full_msg = "full" if full_ok else "rescue"
            else:
                if not cand_adm:
                    dec_stats["obj"] += 1
                    continue

            current = trial
            controller.set_schedule(current)
            current_key = key
            current_val_ev = ev_val
            current_full_ev = ev_full
            current_obj = float(obj)
            current_acc = float(ev_val.acc)
            accepts += 1
            mv_str = ", ".join(f"{k}->{v}" for k, v in mv.items())
            print(
                f"[joint] step{step:03d}: accept({full_msg}) {{{mv_str}}} | "
                f"cost={schedule_cost(current, controller)}/{band_str} | "
                f"val_obj={current_obj:.6e} val_acc={_fmt_pct(current_acc)} | "
                f"full_acc={_fmt_pct(current_full_ev.acc)} full_mse={current_full_ev.logit_mse:.6e} "
                f"full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total}"
            )
            accepted = True
            break

        if JOINT_LOG_REJECTION_STATS:
            elapsed = time.perf_counter() - step_t0
            print(
                f"[joint] step{step:03d} stats | time={elapsed:.2f}s | val_floor={_fmt_pct(val_floor)} | "
                f"fast{{{format_joint_reject_stats(fast_stats)}}} "
                f"val{{{format_joint_reject_stats(val_stats)}}} "
                f"dec{{{format_joint_reject_stats(dec_stats)}}}"
            )

        if not accepted:
            print(f"[joint] stop@step{step:03d}: top VAL candidates failed FULL/stability checks.")
            break

    controller.set_schedule(current)
    final_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
    final_full_ev = eval_model(model, full_loader, device, baseline_logits=base_full_logits)
    return current, final_val_ev, final_full_ev



def run_joint_global_cd(
    *,
    model: nn.Module,
    controller: ApproxController,
    active_sites: Sequence[str],
    start_schedule: Dict[str, int],
    budget: int,
    search_budget_lo: int,
    search_budget_hi: int,
    fast_loader: DataLoader,
    val_loader: DataLoader,
    full_lite_loader: DataLoader,
    full_loader: DataLoader,
    base_fast_logits: torch.Tensor,
    base_val_logits: torch.Tensor,
    base_full_lite_logits: torch.Tensor,
    base_full_logits: torch.Tensor,
    base_val_acc: float,
    base_full_lite_acc: float,
    base_full_acc: float,
    base_fast_abs_logit: float,
    base_val_abs_logit: float,
    base_full_lite_abs_logit: float,
    base_full_abs_logit: float,
    device: torch.device,
    rng: np.random.Generator,
    importance_rank: Sequence[str],
) -> Tuple[Dict[str, int], EvalResult, EvalResult]:
    active_sites = list(active_sites)
    current = dict(start_schedule)
    controller.set_schedule(current)
    current_key = schedule_key_for_sites(current, active_sites)
    current_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
    current_full_lite_ev = eval_model(model, full_lite_loader, device, baseline_logits=base_full_lite_logits)
    current_full_ev = eval_model(model, full_loader, device, baseline_logits=base_full_logits)
    current_obj = float(joint_val_objective(current_val_ev, base_val_abs_logit))
    current_acc = float(current_val_ev.acc)

    choice_costs, cost_buckets = build_site_choice_maps(controller, active_sites)
    fast_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_GLOBAL_CD_FAST_CACHE else None
    val_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_GLOBAL_CD_VAL_CACHE else None
    full_lite_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_GLOBAL_CD_FULL_CACHE else None
    full_cache: Optional[Dict[Tuple[int, ...], EvalResult]] = {} if JOINT_ENABLE_GLOBAL_CD_FULL_CACHE else None
    if val_cache is not None:
        val_cache[current_key] = current_val_ev
    if full_lite_cache is not None:
        full_lite_cache[current_key] = current_full_lite_ev
    if full_cache is not None:
        full_cache[current_key] = current_full_ev

    site_subset = build_cd_site_subset(controller, active_sites, importance_rank)
    stage_groups = build_cd_stage_groups(controller, site_subset)

    band_str = format_budget_band(int(search_budget_lo), int(search_budget_hi))
    print(f"[global-cd] start: cost={schedule_cost(current, controller)} | budget={band_str} [report={int(budget)}] | val_obj={current_obj:.6e} acc={_fmt_pct(current_acc)}")

    stage_stop_softmax_nf = int(JOINT_CD_STOP_SOFTMAX_IF_FULL_NF_LEQ)
    stage_skip_ln_nf = int(JOINT_CD_SKIP_LAYERNORM_IF_FULL_NF_LEQ)

    for stage_name, stage_sites, stage_max_passes in stage_groups:
        if not stage_sites:
            continue
        if stage_name == "layernorm" and stage_skip_ln_nf >= 0 and int(current_full_ev.nonfinite_sample_count) <= stage_skip_ln_nf:
            print(
                f"[global-cd] skip stage={stage_name} | full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total} <= {stage_skip_ln_nf}"
            )
            continue

        print(
            f"[global-cd] stage={stage_name} | sites={len(stage_sites)} | max_passes={int(stage_max_passes)} | "
            f"cur_full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total}"
        )

        for pass_i in range(1, int(stage_max_passes) + 1):
            pass_t0 = time.perf_counter()
            improved_any = False
            accepts_pass = 0
            ordered_sites = list(stage_sites)
            cur_full_ok = joint_full_guard_ok(current_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
            val_floor = float(joint_val_acc_floor(current_acc=current_acc, base_val_acc=base_val_acc))
            pass_stats: Dict[str, int] = defaultdict(int)

            for site in ordered_sites:
                val_floor = float(joint_val_acc_floor(current_acc=current_acc, base_val_acc=base_val_acc))
                cur_id = int(current[site])
                cand_ids = get_near_cost_choice_ids(site, cur_id, choice_costs, cost_buckets, int(JOINT_MAX_LOCAL_COST_DELTA))
                if not cand_ids:
                    pass_stats["no_local"] += 1
                    continue

                fast_scored: List[Tuple[float, int, Tuple[int, ...], Dict[str, int]]] = []
                for cid in cand_ids:
                    trial = dict(current)
                    trial[site] = int(cid)
                    trial_cost = int(schedule_cost(trial, controller))
                    if trial_cost < int(search_budget_lo) or trial_cost > int(search_budget_hi):
                        pass_stats["budget"] += 1
                        continue
                    key = schedule_key_for_sites(trial, active_sites)
                    ev_fast = eval_with_cache(
                        fast_cache,
                        key,
                        lambda trial=trial: (controller.set_schedule(trial), eval_model(model, fast_loader, device, baseline_logits=base_fast_logits))[1],
                    )
                    if ev_fast.nonfinite_any and (not joint_nonfinite_ok(ev_fast, split="fast")):
                        pass_stats["fast_nonfinite"] += 1
                        continue
                    fast_scored.append((float(joint_val_objective(ev_fast, base_fast_abs_logit)), int(cid), key, trial))
                if not fast_scored:
                    pass_stats["site_fast_empty"] += 1
                    continue
                fast_scored.sort(key=lambda x: float(x[0]))
                top_rows = fast_scored[: max(1, int(JOINT_CD_FAST_TOPK_VAL))]

                val_cands: List[Tuple[int, float, float, int, EvalResult, bool, Tuple[int, ...], Dict[str, int]]] = []
                for _fast_obj, cid, key, trial in top_rows:
                    ev_val = eval_with_cache(
                        val_cache,
                        key,
                        lambda trial=trial: (controller.set_schedule(trial), eval_model(model, val_loader, device, baseline_logits=base_val_logits))[1],
                    )
                    val_nf_strict = (not ev_val.nonfinite_any) or joint_nonfinite_ok(ev_val, split="val")
                    val_nf_soft = val_nf_strict or (
                        bool(JOINT_ENABLE_VAL_NONFINITE_SOFT_PASS) and joint_nonfinite_soft_ok(ev_val, split="val")
                    )
                    if not val_nf_soft:
                        pass_stats["val_nonfinite_hard"] += 1
                        continue
                    if (not val_nf_strict) and ev_val.nonfinite_any:
                        pass_stats["val_nonfinite_soft"] += 1
                    if JOINT_ENABLE_VAL_ACC_GUARD and (ev_val.acc < val_floor):
                        pass_stats["val_acc"] += 1
                        continue
                    obj = float(joint_val_objective(ev_val, base_val_abs_logit))
                    nf_rank = 0 if val_nf_strict else 1
                    val_cands.append((nf_rank, obj, -float(ev_val.acc), int(cid), ev_val, val_nf_strict, key, trial))
                if not val_cands:
                    pass_stats["site_val_empty"] += 1
                    continue
                val_cands.sort(key=lambda x: (int(x[0]), float(x[1]), float(x[2])))

                accepted_site = False
                lite_rows: List[Tuple[int, float, float, int, EvalResult, bool, Tuple[int, ...], Dict[str, int], EvalResult, bool, bool]] = []
                for nf_rank, obj, _neg_acc, cid, ev_val, val_nf_strict, key, trial in val_cands[: max(1, int(JOINT_CD_FULL_LITE_TOPK))]:
                    cand_adm = candidate_admissible_joint(cur_obj=current_obj, cur_acc=current_acc, cand_obj=float(obj), cand_acc=float(ev_val.acc))
                    ev_full_lite = eval_with_cache(
                        full_lite_cache,
                        key,
                        lambda trial=trial: (controller.set_schedule(trial), eval_model(model, full_lite_loader, device, baseline_logits=base_full_lite_logits))[1],
                    )
                    lite_ok = full_lite_guard_ok(
                        ev_full_lite,
                        base_full_lite_acc=base_full_lite_acc,
                        base_full_lite_abs_logit=base_full_lite_abs_logit,
                    )
                    lite_rescue_ok = stability_rescue_accept(
                        cur_full_ev=current_full_lite_ev,
                        cand_full_ev=ev_full_lite,
                        cur_val_ev=current_val_ev,
                        cand_val_ev=ev_val,
                        base_full_acc=base_full_lite_acc,
                        base_full_abs_logit=base_full_lite_abs_logit,
                    )
                    if not ((cand_adm and lite_ok) or ((not cur_full_ok) and lite_rescue_ok)):
                        if not cand_adm:
                            pass_stats["decision_obj"] += 1
                        else:
                            pass_stats["decision_full_lite"] += 1
                        continue
                    lite_rows.append((nf_rank, float(obj), -float(ev_val.acc), int(cid), ev_val, val_nf_strict, key, trial, ev_full_lite, lite_ok, lite_rescue_ok))

                if not lite_rows:
                    pass_stats["site_lite_empty"] += 1
                    continue

                lite_rows.sort(key=lambda x: (int(x[0]), float(x[1]), float(joint_val_objective(x[8], base_full_lite_abs_logit)), float(x[2])))
                for nf_rank, obj, _neg_acc, cid, ev_val, val_nf_strict, key, trial, ev_full_lite, lite_ok, lite_rescue_ok in lite_rows[: max(1, int(JOINT_CD_FULL_LITE_TOPK))]:
                    ev_full = eval_with_cache(
                        full_cache,
                        key,
                        lambda trial=trial: (controller.set_schedule(trial), eval_model(model, full_loader, device, baseline_logits=base_full_logits))[1],
                    )
                    full_ok = joint_full_guard_ok(ev_full, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
                    cand_adm = candidate_admissible_joint(cur_obj=current_obj, cur_acc=current_acc, cand_obj=float(obj), cand_acc=float(ev_val.acc))
                    rescue_ok = stability_rescue_accept(
                        cur_full_ev=current_full_ev,
                        cand_full_ev=ev_full,
                        cur_val_ev=current_val_ev,
                        cand_val_ev=ev_val,
                        base_full_acc=base_full_acc,
                        base_full_abs_logit=base_full_abs_logit,
                    )
                    if not ((cand_adm and full_ok) or ((not cur_full_ok) and rescue_ok)):
                        if not cand_adm:
                            pass_stats["decision_obj"] += 1
                        if not full_ok:
                            pass_stats["decision_full"] += 1
                        if (not rescue_ok) and (not full_ok):
                            pass_stats["decision_rescue"] += 1
                        continue
                    if not bool(val_nf_strict):
                        pass_stats["decision_val_soft"] += 1

                    current = dict(trial)
                    controller.set_schedule(current)
                    current_key = key
                    current_val_ev = ev_val
                    current_full_lite_ev = ev_full_lite
                    current_full_ev = ev_full
                    current_obj = float(obj)
                    current_acc = float(ev_val.acc)
                    cur_full_ok = joint_full_guard_ok(current_full_ev, base_full_acc=base_full_acc, base_full_abs_logit=base_full_abs_logit)
                    improved_any = True
                    accepts_pass += 1
                    accepted_site = True
                    pass_stats["accept"] += 1
                    tag = "full" if full_ok else "rescue"
                    print(
                        f"[global-cd] {stage_name}/pass{pass_i}: accept({tag}) {site}->{cid} | "
                        f"cost={schedule_cost(current, controller)}/{band_str} | val_obj={current_obj:.6e} val_acc={_fmt_pct(current_acc)} | "
                        f"full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total}"
                    )
                    break

                if not accepted_site:
                    pass_stats["site_no_accept"] += 1

                if accepted_site and accepts_pass >= int(JOINT_CD_MAX_ACCEPTS_PER_PASS):
                    break

            if JOINT_LOG_REJECTION_STATS:
                elapsed = time.perf_counter() - pass_t0
                print(
                    f"[global-cd] {stage_name}/pass{pass_i} stats | time={elapsed:.2f}s | val_floor={_fmt_pct(val_floor)} | "
                    f"{format_joint_reject_stats(pass_stats)}"
                )

            if stage_name == "softmax" and stage_stop_softmax_nf >= 0 and int(current_full_ev.nonfinite_sample_count) <= stage_stop_softmax_nf:
                print(
                    f"[global-cd] stage={stage_name} reached full_nf={current_full_ev.nonfinite_sample_count}/{current_full_ev.n_total} <= {stage_stop_softmax_nf}; moving to next stage."
                )
                break

            if not improved_any:
                print(f"[global-cd] stage={stage_name} stop after pass{pass_i}: no improvement.")
                break

    controller.set_schedule(current)
    final_val_ev = eval_model(model, val_loader, device, baseline_logits=base_val_logits)
    final_full_ev = eval_model(model, full_loader, device, baseline_logits=base_full_logits)
    return current, final_val_ev, final_full_ev


def run_one_joint_task(

    *,
    task: JointTaskSpec,
    model: nn.Module,
    controller: ApproxController,
    shared_proxy: Dict[str, Dict[int, float]],
    base_out_dir: Path,
    fast_loader: DataLoader,
    val_loader: DataLoader,
    full_lite_loader: DataLoader,
    full_loader: DataLoader,
    test_loader: DataLoader,
    base_fast_logits: torch.Tensor,
    base_val_logits: torch.Tensor,
    base_full_lite_logits: torch.Tensor,
    base_full_logits: torch.Tensor,
    base_test_logits: torch.Tensor,
    base_fast_acc: float,
    base_val_acc: float,
    base_full_lite_acc: float,
    base_full_acc: float,
    base_test_acc: float,
    base_fast_abs_logit: float,
    base_val_abs_logit: float,
    base_full_lite_abs_logit: float,
    base_full_abs_logit: float,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    mode_spec = resolve_ablation_mode(task.mode)
    active_sites = active_sites_for_mode(controller, mode_spec)
    if not active_sites:
        raise ValueError(f"Joint refinement requires at least one active site. mode={task.mode}")

    min_budget, max_budget = budget_bounds(controller, sites=active_sites)
    search_budget_lo, search_budget_hi = resolve_joint_budget_band(task, min_budget, max_budget)
    band_str = format_budget_band(search_budget_lo, search_budget_hi)
    report_budget = int(task.budget)

    if search_budget_lo == search_budget_hi == report_budget:
        task_dir_name = f"{mode_spec.slug}_budget_{report_budget}"
    else:
        task_dir_name = f"{mode_spec.slug}_budget_{report_budget}_search_{search_budget_lo}_{search_budget_hi}"
    task_out_dir = base_out_dir / task_dir_name
    task_out_dir.mkdir(parents=True, exist_ok=True)

    schedule_csv = task.schedule_csv
    if schedule_csv is None:
        schedule_csv = default_joint_schedule_csv(task.model_key, mode_spec.slug, int(task.budget))

    init_budget_for_dp = int(search_budget_hi)
    mode_proxy = filter_proxy_errors(shared_proxy, active_sites)
    if schedule_csv is not None and schedule_csv.exists():
        init_schedule = load_schedule_csv(schedule_csv, controller)
        print(f"[joint-task] init schedule from {schedule_csv}")
        init_schedule = fill_missing_schedule_sites_with_dp(
            init_schedule,
            active_sites=active_sites,
            controller=controller,
            proxy_errors=mode_proxy,
            search_budget_lo=int(search_budget_lo),
            search_budget_hi=int(search_budget_hi),
        )
    else:
        print(
            f"[joint-task] schedule csv missing for mode={mode_spec.slug} budget={task.budget}; "
            f"falling back to DP from shared proxy (init_budget_band={search_budget_lo}..{init_budget_for_dp})"
        )
        init_schedule = dp_allocate_schedule(
            list(active_sites),
            controller,
            mode_proxy,
            int(init_budget_for_dp),
            min_budget=int(search_budget_lo),
        )

    # keep only active sites and validate
    init_schedule = {site: int(cid) for site, cid in init_schedule.items() if site in active_sites}
    for site in active_sites:
        if site not in init_schedule:
            raise KeyError(f"Initial schedule missing active site={site}. mode={mode_spec.slug}")

    init_cost = schedule_cost(init_schedule, controller)
    if init_cost < int(search_budget_lo) or init_cost > int(search_budget_hi):
        print(
            f"[joint-task] init schedule cost={init_cost} is outside search band {search_budget_lo}..{search_budget_hi}; "
            f"rebuilding seed from shared proxy inside band."
        )
        init_schedule = dp_allocate_schedule(
            list(active_sites),
            controller,
            mode_proxy,
            int(init_budget_for_dp),
            min_budget=int(search_budget_lo),
        )
        init_cost = schedule_cost(init_schedule, controller)
    if init_cost < int(search_budget_lo) or init_cost > int(search_budget_hi):
        raise ValueError(
            f"Initial schedule cost={init_cost} is outside search budget band {search_budget_lo}..{search_budget_hi}"
        )

    print(
        f"[joint-task] budget-band | mode={mode_spec.slug} | feasible={min_budget}..{max_budget} | "
        f"search={band_str} | report={report_budget}"
    )

    save_csv(schedule_to_rows(init_schedule, controller, sites=active_sites), task_out_dir / "init_schedule.csv")
    save_csv(schedule_to_rows(init_schedule, controller), task_out_dir / "init_schedule_full.csv")

    mutable_sites = resolve_joint_mutable_sites(controller, active_sites)
    print(f"[joint-task] mutable_sites={len(mutable_sites)} / active_sites={len(active_sites)}")
    importance_rank = compute_site_importance_rank_from_proxy(mutable_sites, init_schedule, mode_proxy)

    init_record, _init_val_ev, _init_full_ev, _init_test_ev = evaluate_stage_record(
        stage="initial",
        budget=int(task.budget),
        controller=controller,
        schedule=init_schedule,
        model=model,
        val_loader=val_loader,
        full_loader=full_loader,
        test_loader=test_loader,
        base_val_logits=base_val_logits,
        base_full_logits=base_full_logits,
        base_test_logits=base_test_logits,
        base_val_acc=base_val_acc,
        base_full_acc=base_full_acc,
        base_test_acc=base_test_acc,
        base_val_abs_logit=base_val_abs_logit,
        base_full_abs_logit=base_full_abs_logit,
        device=device,
    )
    print(
        f"[joint-task] initial | mode={mode_spec.slug} budget={task.budget} | "
        f"cost={init_record.cost} | val_acc={_fmt_pct(init_record.val_acc)} | "
        f"full_acc={_fmt_pct(init_record.full_acc)} | test_acc={_fmt_pct(init_record.test_acc)} | "
        f"val_mse={init_record.val_logit_mse:.6e} | full_ok={init_record.full_ok}"
    )

    joint_schedule = dict(init_schedule)
    joint_record = init_record
    if RUN_JOINT_SEARCH:
        joint_schedule, _joint_val_ev, _joint_full_ev = run_joint_search(
            model=model,
            controller=controller,
            active_sites=mutable_sites,
            start_schedule=init_schedule,
            budget=int(task.budget),
            search_budget_lo=int(search_budget_lo),
            search_budget_hi=int(search_budget_hi),
            fast_loader=fast_loader,
            val_loader=val_loader,
            full_lite_loader=full_lite_loader,
            full_loader=full_loader,
            base_fast_logits=base_fast_logits,
            base_val_logits=base_val_logits,
            base_full_lite_logits=base_full_lite_logits,
            base_full_logits=base_full_logits,
            base_val_acc=base_val_acc,
            base_full_lite_acc=base_full_lite_acc,
            base_full_acc=base_full_acc,
            base_fast_abs_logit=base_fast_abs_logit,
            base_val_abs_logit=base_val_abs_logit,
            base_full_lite_abs_logit=base_full_lite_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
            rng=rng,
            importance_rank=importance_rank,
        )
        save_csv(schedule_to_rows(joint_schedule, controller, sites=active_sites), task_out_dir / "joint_schedule.csv")
        save_csv(schedule_to_rows(joint_schedule, controller), task_out_dir / "joint_schedule_full.csv")

        joint_record, _, _, _ = evaluate_stage_record(
            stage="joint",
            budget=int(task.budget),
            controller=controller,
            schedule=joint_schedule,
            model=model,
            val_loader=val_loader,
            full_loader=full_loader,
            test_loader=test_loader,
            base_val_logits=base_val_logits,
            base_full_logits=base_full_logits,
            base_test_logits=base_test_logits,
            base_val_acc=base_val_acc,
            base_full_acc=base_full_acc,
            base_test_acc=base_test_acc,
            base_val_abs_logit=base_val_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
        )
    else:
        print("[joint-task] DP-only mode: skipping local joint search and global CD.")

    global_record = joint_record
    global_schedule = joint_schedule
    if RUN_JOINT_SEARCH and RUN_JOINT_GLOBAL_REFINEMENT:
        global_schedule, _, _ = run_joint_global_cd(
            model=model,
            controller=controller,
            active_sites=mutable_sites,
            start_schedule=joint_schedule,
            budget=int(task.budget),
            search_budget_lo=int(search_budget_lo),
            search_budget_hi=int(search_budget_hi),
            fast_loader=fast_loader,
            val_loader=val_loader,
            full_lite_loader=full_lite_loader,
            full_loader=full_loader,
            base_fast_logits=base_fast_logits,
            base_val_logits=base_val_logits,
            base_full_lite_logits=base_full_lite_logits,
            base_full_logits=base_full_logits,
            base_val_acc=base_val_acc,
            base_full_lite_acc=base_full_lite_acc,
            base_full_acc=base_full_acc,
            base_fast_abs_logit=base_fast_abs_logit,
            base_val_abs_logit=base_val_abs_logit,
            base_full_lite_abs_logit=base_full_lite_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
            rng=rng,
            importance_rank=importance_rank,
        )
        save_csv(schedule_to_rows(global_schedule, controller, sites=active_sites), task_out_dir / "global_schedule.csv")
        save_csv(schedule_to_rows(global_schedule, controller), task_out_dir / "global_schedule_full.csv")
        global_record, _, _, _ = evaluate_stage_record(
            stage="global_cd",
            budget=int(task.budget),
            controller=controller,
            schedule=global_schedule,
            model=model,
            val_loader=val_loader,
            full_loader=full_loader,
            test_loader=test_loader,
            base_val_logits=base_val_logits,
            base_full_logits=base_full_logits,
            base_test_logits=base_test_logits,
            base_val_acc=base_val_acc,
            base_full_acc=base_full_acc,
            base_test_acc=base_test_acc,
            base_val_abs_logit=base_val_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
        )

    stage_records = [init_record]
    if RUN_JOINT_SEARCH:
        stage_records.append(joint_record)
    if RUN_JOINT_SEARCH and RUN_JOINT_GLOBAL_REFINEMENT:
        stage_records.append(global_record)

    best_record = select_best_joint_stage(stage_records)
    if best_record.stage == "initial":
        best_schedule = init_schedule
    elif best_record.stage == "joint":
        best_schedule = joint_schedule
    else:
        best_schedule = global_schedule

    save_csv(schedule_to_rows(best_schedule, controller, sites=active_sites), task_out_dir / JOINT_BEST_SCHEDULE_NAME)
    save_csv(schedule_to_rows(best_schedule, controller), task_out_dir / JOINT_BEST_SCHEDULE_FULL_NAME)
    save_csv([r.__dict__ for r in stage_records], task_out_dir / JOINT_STAGE_CSV_NAME)
    with (task_out_dir / JOINT_STAGE_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": {
                    "model_key": task.model_key,
                    "mode": mode_spec.slug,
                    "budget": int(task.budget),
                    "search_budget_lo": int(search_budget_lo),
                    "search_budget_hi": int(search_budget_hi),
                    "feasible_budget_min": int(min_budget),
                    "feasible_budget_max": int(max_budget),
                    "schedule_csv": str(schedule_csv) if schedule_csv is not None else None,
                },
                "active_sites": list(active_sites),
                "importance_rank": list(importance_rank),
                "best_stage": best_record.stage,
                "stages": [r.__dict__ for r in stage_records],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[joint-task] BEST | mode={mode_spec.slug} budget={task.budget} | search={band_str} | "
        f"stage={best_record.stage} | cost={best_record.cost} | "
        f"val_acc={_fmt_pct(best_record.val_acc)} | test_acc={_fmt_pct(best_record.test_acc)} | "
        f"val_mse={best_record.val_logit_mse:.6e} | full_ok={best_record.full_ok}"
    )

    post_profile_records: List[JointStageRecord] = []
    seen_profiles: set[str] = set()
    for profile in POST_EVAL_PROFILES:
        p = str(profile).lower()
        if not p or p in seen_profiles:
            continue
        seen_profiles.add(p)
        rec = evaluate_schedule_under_profile(
            profile=p,
            task=task,
            controller=controller,
            schedule=best_schedule,
            model=model,
            val_loader=val_loader,
            full_loader=full_loader,
            test_loader=test_loader,
            base_val_logits=base_val_logits,
            base_full_logits=base_full_logits,
            base_test_logits=base_test_logits,
            base_val_acc=base_val_acc,
            base_full_acc=base_full_acc,
            base_test_acc=base_test_acc,
            base_val_abs_logit=base_val_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
        )
        post_profile_records.append(rec)
        print(
            f"[post-eval:{p}] cost={rec.cost} | val_acc={_fmt_pct(rec.val_acc)} | test_acc={_fmt_pct(rec.test_acc)} | "
            f"val_mse={rec.val_logit_mse:.6e} | full_ok={rec.full_ok}"
        )
    save_post_profile_records(post_profile_records, task_out_dir)

    return {
        "task": task,
        "mode_spec": mode_spec,
        "active_sites": list(active_sites),
        "search_budget_lo": int(search_budget_lo),
        "search_budget_hi": int(search_budget_hi),
        "feasible_budget_min": int(min_budget),
        "feasible_budget_max": int(max_budget),
        "init_schedule": init_schedule,
        "joint_schedule": joint_schedule,
        "global_schedule": global_schedule,
        "best_schedule": best_schedule,
        "stage_records": stage_records,
        "best_record": best_record,
        "out_dir": task_out_dir,
        "post_profile_records": post_profile_records,
    }


def run_joint_tasks_for_model(model_key: str, model_tasks: Sequence[JointTaskSpec]) -> None:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model_key={model_key}. Available: {sorted(MODEL_SPECS)}")
    if not model_tasks:
        return

    spec = MODEL_SPECS[model_key]
    model_dir = ensure_local_path(spec["model_dir"], what=f"model_dir[{model_key}]")
    tokenizer_dir = spec.get("tokenizer_dir", model_dir)
    out_dir = spec["output_dir"] / JOINT_ROOT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    compute_batch = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    print("\n" + "=" * 120)
    print(f"[model] {model_key}")
    print("=" * 120)
    print(f"[device] {device} | compute_batch={compute_batch}")
    print(f"[model_dir] {model_dir}")
    print(f"[tokenizer_dir] {tokenizer_dir}")
    print(f"[dataset_dir] {DATASET_DIR}")
    print(f"[output_dir] {out_dir}")

    set_runtime_profile(SEARCH_PROFILE)
    print(f"[search_profile] {SEARCH_PROFILE} | post_eval_profiles={POST_EVAL_PROFILES}")

    tokenizer, model = load_local_model_and_tokenizer(model_dir, tokenizer_dir)

    train_examples, dev_examples = load_local_text_splits(
        DATASET_DIR,
        dataset_format=DATASET_FORMAT,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN,
    )
    print(f"[dataset] train={len(train_examples)} dev/report={len(dev_examples)}")

    calib_idx, sweep_idx, val_idx = split_train_indices(
        len(train_examples),
        CALIB_N,
        SWEEP_N,
        VAL_N,
        GLOBAL_SEED,
    )
    if not sweep_idx:
        sweep_idx = list(calib_idx[: max(1, min(len(calib_idx), 64))])
    if not val_idx:
        val_idx = list(sweep_idx)

    fast_idx = sample_indices(len(val_idx), min(int(JOINT_FAST_N), len(val_idx)), GLOBAL_SEED + 11)
    fast_src_examples = [train_examples[val_idx[i]] for i in fast_idx]

    full_n = CALIB_N if JOINT_FULL_N is None else JOINT_FULL_N
    full_idx = sample_indices(len(calib_idx), min(int(full_n), len(calib_idx)), GLOBAL_SEED + 12)
    full_src_examples = [train_examples[calib_idx[i]] for i in full_idx]
    full_lite_n = min(int(JOINT_FULL_LITE_N), len(calib_idx))
    full_lite_idx = sample_indices(len(calib_idx), full_lite_n, GLOBAL_SEED + 13)
    full_lite_src_examples = [train_examples[calib_idx[i]] for i in full_lite_idx]

    test_take = TEST_MAX_N if JOINT_TEST_N is None else JOINT_TEST_N
    test_idx = sample_indices(len(dev_examples), test_take if test_take is not None else len(dev_examples), GLOBAL_SEED + 1)

    fast_ds = TokenizedTextDataset(fast_src_examples, tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    sweep_ds = TokenizedTextDataset([train_examples[i] for i in sweep_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    val_ds = TokenizedTextDataset([train_examples[i] for i in val_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    full_lite_ds = TokenizedTextDataset(full_lite_src_examples, tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    full_ds = TokenizedTextDataset(full_src_examples, tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)
    test_ds = TokenizedTextDataset([dev_examples[i] for i in test_idx], tokenizer, MAX_LENGTH, lowercase=LOWERCASE_TEXT)

    fast_loader, fast_loader_batch = build_tuned_loader(fast_ds, tokenizer, compute_batch, device)
    sweep_loader, sweep_loader_batch = build_tuned_loader(sweep_ds, tokenizer, compute_batch, device)
    val_loader, val_loader_batch = build_tuned_loader(val_ds, tokenizer, compute_batch, device)
    full_lite_loader, full_lite_loader_batch = build_tuned_loader(full_lite_ds, tokenizer, compute_batch, device)
    full_loader, full_loader_batch = build_tuned_loader(full_ds, tokenizer, compute_batch, device)
    test_loader, test_loader_batch = build_tuned_loader(test_ds, tokenizer, compute_batch, device)
    proxy_stage1_loader = None
    proxy_stage1_loader_batch = 0
    if ENABLE_PROXY_STAGE1_PRUNE:
        proxy_stage1_take = min(int(PROXY_STAGE1_N), len(sweep_ds))
        proxy_stage1_idx = sample_indices(len(sweep_ds), proxy_stage1_take, GLOBAL_SEED + 31)
        proxy_stage1_ds = Subset(sweep_ds, proxy_stage1_idx)
        proxy_stage1_loader, proxy_stage1_loader_batch = build_tuned_loader(proxy_stage1_ds, tokenizer, compute_batch, device)
    if LOG_TUNED_LOADER_BATCHES:
        msg = f"[loader-batch] fast={fast_loader_batch} sweep={sweep_loader_batch} val={val_loader_batch} full_lite={full_lite_loader_batch} full={full_loader_batch} test={test_loader_batch}"
        if proxy_stage1_loader is not None:
            msg += f" proxy1={proxy_stage1_loader_batch}"
        print(msg)

    choices_by_kind = build_choice_space()
    controller = ApproxController(choices_by_kind)
    model = instrument_bert_model(model, controller).to(device).eval()

    all_sites = controller.ordered_sites()
    save_csv(site_summary_rows(controller), out_dir / "site_summary.csv")
    print(f"[sites] total={len(all_sites)}")
    print(f"[sites] by kind={kind_counts_for_sites(controller, all_sites)}")
    proxy_site_set: set[str] = set()
    for task in model_tasks:
        task_mode_spec = resolve_ablation_mode(task.mode)
        proxy_site_set.update(active_sites_for_mode(controller, task_mode_spec))
    proxy_sites = [site for site in all_sites if site in proxy_site_set]
    proxy_kinds = {str(k).lower() for k in JOINT_PROXY_KIND_FILTER}
    if proxy_kinds:
        proxy_sites = [site for site in proxy_sites if controller.get_kind(site).lower() in proxy_kinds]
    if not proxy_sites:
        proxy_sites = list(all_sites)
    print(f"[proxy-sites] total={len(proxy_sites)}")
    print(f"[proxy-sites] by kind={kind_counts_for_sites(controller, proxy_sites)}")

    phase_timing: Dict[str, float] = {}
    t_phase = time.perf_counter()
    controller.set_exact()
    controller.collect_calibration = True
    _ = eval_model(model, full_loader, device, baseline_logits=None)
    controller.collect_calibration = False
    phase_timing["calibration_s"] = float(time.perf_counter() - t_phase)
    save_csv(controller.calibration.rows(), out_dir / "calibration_ranges.csv")
    print(f"[calib] saved calibration ranges to {out_dir / 'calibration_ranges.csv'}")
    save_csv(softmax_calib_max_lut_rows(controller), out_dir / SOFTMAX_CALIB_MAX_LUT_CSV_NAME)
    print(f"[calib] saved softmax calib-max LUT to {out_dir / SOFTMAX_CALIB_MAX_LUT_CSV_NAME}")

    t_phase = time.perf_counter()
    controller.set_exact()
    base_fast_logits, base_fast_acc = compute_logits(model, fast_loader, device)
    base_sweep_logits, _ = compute_logits(model, sweep_loader, device)
    base_proxy1_logits = None
    if proxy_stage1_loader is not None:
        base_proxy1_logits, _ = compute_logits(model, proxy_stage1_loader, device)
    base_val_logits, base_val_acc = compute_logits(model, val_loader, device)
    base_full_lite_logits, base_full_lite_acc = compute_logits(model, full_lite_loader, device)
    base_full_logits, base_full_acc = compute_logits(model, full_loader, device)
    base_test_logits, base_test_acc = compute_logits(model, test_loader, device)
    if MOVE_BASELINE_LOGITS_TO_DEVICE and device.type == "cuda":
        base_fast_logits = base_fast_logits.to(device, non_blocking=True)
        base_sweep_logits = base_sweep_logits.to(device, non_blocking=True)
        if base_proxy1_logits is not None:
            base_proxy1_logits = base_proxy1_logits.to(device, non_blocking=True)
        base_val_logits = base_val_logits.to(device, non_blocking=True)
        base_full_lite_logits = base_full_lite_logits.to(device, non_blocking=True)
        base_full_logits = base_full_logits.to(device, non_blocking=True)
        base_test_logits = base_test_logits.to(device, non_blocking=True)
    phase_timing["baseline_s"] = float(time.perf_counter() - t_phase)

    base_fast_ev = eval_model(model, fast_loader, device, baseline_logits=None)
    base_val_ev = eval_model(model, val_loader, device, baseline_logits=None)
    base_full_lite_ev = eval_model(model, full_lite_loader, device, baseline_logits=None)
    base_full_ev = eval_model(model, full_loader, device, baseline_logits=None)
    base_test_ev = eval_model(model, test_loader, device, baseline_logits=None)

    print(f"[baseline] fast_acc={_fmt_pct(base_fast_acc)}")
    print(f"[baseline] val_acc={_fmt_pct(base_val_acc)}")
    print(f"[baseline] full_acc={_fmt_pct(base_full_acc)}")
    print(f"[baseline] test_acc={_fmt_pct(base_test_acc)}")

    shared_proxy_path = Path(JOINT_SHARED_PROXY_TABLE_PATH) if JOINT_SHARED_PROXY_TABLE_PATH is not None else (out_dir / SHARED_PROXY_TABLE_NAME)
    t_phase = time.perf_counter()
    shared_proxy, proxy_meta = build_proxy_error_table(
        model,
        controller,
        sweep_loader,
        base_sweep_logits,
        device,
        shared_proxy_path,
        sites=proxy_sites,
        prescreen_loader=proxy_stage1_loader,
        prescreen_baseline_logits=base_proxy1_logits,
        stage1_out_path=(out_dir / PROXY_STAGE1_TABLE_NAME) if ENABLE_PROXY_STAGE1_PRUNE else None,
    )
    phase_timing["proxy_total_s"] = float(time.perf_counter() - t_phase)
    phase_timing["proxy_stage1_s"] = float(proxy_meta.get("stage1_seconds", 0.0))
    phase_timing["proxy_stage2_s"] = float(proxy_meta.get("stage2_seconds", 0.0))
    phase_timing["proxy_total_candidates"] = float(proxy_meta.get("total_candidates", 0))
    phase_timing["proxy_stage2_evals"] = float(proxy_meta.get("stage2_evals", 0))
    phase_timing["proxy_survivors_total"] = float(proxy_meta.get("survivors_total", 0))

    task_rows: List[Dict[str, Any]] = []
    for idx, task in enumerate(model_tasks, start=1):
        print("\n" + "-" * 120)
        print(f"[joint-task] {idx}/{len(model_tasks)} | mode={task.mode} | budget={task.budget}")
        print("-" * 120)
        _task_t0 = time.perf_counter()
        payload = run_one_joint_task(
            task=task,
            model=model,
            controller=controller,
            shared_proxy=shared_proxy,
            base_out_dir=out_dir,
            fast_loader=fast_loader,
            val_loader=val_loader,
            full_lite_loader=full_lite_loader,
            full_loader=full_loader,
            test_loader=test_loader,
            base_fast_logits=base_fast_logits,
            base_val_logits=base_val_logits,
            base_full_lite_logits=base_full_lite_logits,
            base_full_logits=base_full_logits,
            base_test_logits=base_test_logits,
            base_fast_acc=base_fast_acc,
            base_val_acc=base_val_acc,
            base_full_lite_acc=base_full_lite_acc,
            base_full_acc=base_full_acc,
            base_test_acc=base_test_acc,
            base_fast_abs_logit=base_fast_ev.max_abs_logit,
            base_val_abs_logit=base_val_ev.max_abs_logit,
            base_full_lite_abs_logit=base_full_lite_ev.max_abs_logit,
            base_full_abs_logit=base_full_ev.max_abs_logit,
            device=device,
            rng=np.random.default_rng(GLOBAL_SEED + 1000 + idx),
        )
        payload["wall_s"] = float(time.perf_counter() - _task_t0)
        best = payload["best_record"]
        task_rows.append({
            "model_key": model_key,
            "mode": payload["mode_spec"].slug,
            "budget": int(task.budget),
            "search_budget_lo": int(payload["search_budget_lo"]),
            "search_budget_hi": int(payload["search_budget_hi"]),
            "feasible_budget_min": int(payload["feasible_budget_min"]),
            "feasible_budget_max": int(payload["feasible_budget_max"]),
            "best_stage": str(best.stage),
            "cost": int(best.cost),
            "val_acc": float(best.val_acc),
            "val_acc_drop_pp": float(best.val_acc_drop_pp),
            "val_logit_mse": float(best.val_logit_mse),
            "full_acc": float(best.full_acc),
            "full_logit_mse": float(best.full_logit_mse),
            "test_acc": float(best.test_acc),
            "test_acc_drop_pp": float(best.test_acc_drop_pp),
            "test_logit_mse": float(best.test_logit_mse),
            "full_ok": int(best.full_ok),
            "nonfinite_any": int(best.nonfinite_any),
            "out_dir": str(payload["out_dir"]),
            "wall_s": float(payload.get("wall_s", 0.0)),
        })

    save_csv(task_rows, out_dir / "joint_summary.csv")
    with (out_dir / "joint_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": model_key,
                "model_dir": str(model_dir),
                "tokenizer_dir": str(tokenizer_dir),
                "dataset_dir": str(DATASET_DIR),
                "output_dir": str(out_dir),
                "baseline_fast_acc": float(base_fast_acc),
                "baseline_val_acc": float(base_val_acc),
                "baseline_full_acc": float(base_full_acc),
                "baseline_test_acc": float(base_test_acc),
                "shared_proxy_table": str(shared_proxy_path),
                "tasks": task_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    phase_timing["total_setup_s"] = float(phase_timing.get("calibration_s", 0.0) + phase_timing.get("baseline_s", 0.0) + phase_timing.get("proxy_total_s", 0.0))
    phase_timing["joint_tasks_total_s"] = float(sum(float(r.get("wall_s", 0.0)) for r in task_rows))
    with (out_dir / PHASE_TIMING_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key,
            "output_dir": str(out_dir),
            "phase_timing": phase_timing,
            "tasks": task_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"[time] calib={phase_timing['calibration_s']:.2f}s baseline={phase_timing['baseline_s']:.2f}s proxy(stage1={phase_timing['proxy_stage1_s']:.2f}s stage2={phase_timing['proxy_stage2_s']:.2f}s total={phase_timing['proxy_total_s']:.2f}s)")
    print(f"[saved] {out_dir / 'joint_summary.csv'}")
    print(f"[saved] {out_dir / 'joint_summary.json'}")
    print(f"[saved] {out_dir / PHASE_TIMING_JSON_NAME}")



# =============================================================================
# Runtime profile helpers (curriculum experiments)
# =============================================================================

def set_runtime_profile(profile: str) -> None:
    global CLAMP_POSITIVE_DOMAIN
    global SOFTMAX_SIGN_CLIP_TO_UNIT
    profile = str(profile).lower()
    if profile == "relaxed":
        CLAMP_POSITIVE_DOMAIN = True
        SOFTMAX_SIGN_CLIP_TO_UNIT = True
    elif profile == "no_pos_clamp":
        CLAMP_POSITIVE_DOMAIN = False
        SOFTMAX_SIGN_CLIP_TO_UNIT = True
    elif profile == "no_sign_clip":
        CLAMP_POSITIVE_DOMAIN = True
        SOFTMAX_SIGN_CLIP_TO_UNIT = False
    elif profile == "strict":
        CLAMP_POSITIVE_DOMAIN = False
        SOFTMAX_SIGN_CLIP_TO_UNIT = False
    else:
        raise ValueError(f"Unknown runtime profile: {profile}")


def parse_profile_list(text: str) -> List[str]:
    out: List[str] = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


def parse_string_list(text: str) -> List[str]:
    out: List[str] = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


def parse_mutable_kind_list(text: str) -> List[str]:
    toks = parse_string_list(text)
    if any(tok.strip().lower() in {"*", "all"} for tok in toks):
        return []
    return [_canonical_kind_token(tok) for tok in toks]


def parse_mutable_site_list(text: str) -> List[str]:
    toks = parse_string_list(text)
    if any(tok.strip().lower() in {"*", "all"} for tok in toks):
        return []
    return toks


def parse_softmax_shift_modes(text: str) -> List[str]:
    valid = {"none", "row_mean", "row_max", "sign_max", "calib_max", "scaled"}
    out = [tok.lower() for tok in parse_string_list(text)]
    bad = [tok for tok in out if tok not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown softmax shift mode(s): {','.join(bad)}. Valid: {','.join(sorted(valid))}"
        )
    if not out:
        raise argparse.ArgumentTypeError("--softmax-shift-modes cannot be empty")
    return list(dict.fromkeys(out))


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def parse_scale_square_pairs(text: str) -> List[Tuple[float, int]]:
    out: List[Tuple[float, int]] = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' not in tok:
            raise argparse.ArgumentTypeError(
                f"Invalid softmax scale/square pair {tok!r}; expected SCALE:ITERS, e.g. 4:2"
            )
        scale_text, iters_text = tok.split(':', 1)
        scale = float(scale_text.strip())
        iters = int(iters_text.strip())
        if scale <= 0.0:
            raise argparse.ArgumentTypeError(f"Scale must be positive in pair {tok!r}")
        if iters < 0:
            raise argparse.ArgumentTypeError(f"Square iters must be non-negative in pair {tok!r}")
        out.append((scale, iters))
    if not out:
        raise argparse.ArgumentTypeError("--softmax-scale-square-pairs cannot be empty")
    return list(dict.fromkeys(out))


def evaluate_schedule_under_profile(
    *,
    profile: str,
    task: JointTaskSpec,
    controller: ApproxController,
    schedule: Dict[str, int],
    model: nn.Module,
    val_loader: DataLoader,
    full_loader: DataLoader,
    test_loader: DataLoader,
    base_val_logits: torch.Tensor,
    base_full_logits: torch.Tensor,
    base_test_logits: torch.Tensor,
    base_val_acc: float,
    base_full_acc: float,
    base_test_acc: float,
    base_val_abs_logit: float,
    base_full_abs_logit: float,
    device: torch.device,
) -> JointStageRecord:
    global CLAMP_POSITIVE_DOMAIN
    global SOFTMAX_SIGN_CLIP_TO_UNIT
    prev_clamp = bool(CLAMP_POSITIVE_DOMAIN)
    prev_clip = bool(SOFTMAX_SIGN_CLIP_TO_UNIT)
    set_runtime_profile(profile)
    try:
        rec, _, _, _ = evaluate_stage_record(
            stage=f"post_eval_{str(profile).lower()}",
            budget=int(task.budget),
            controller=controller,
            schedule=schedule,
            model=model,
            val_loader=val_loader,
            full_loader=full_loader,
            test_loader=test_loader,
            base_val_logits=base_val_logits,
            base_full_logits=base_full_logits,
            base_test_logits=base_test_logits,
            base_val_acc=base_val_acc,
            base_full_acc=base_full_acc,
            base_test_acc=base_test_acc,
            base_val_abs_logit=base_val_abs_logit,
            base_full_abs_logit=base_full_abs_logit,
            device=device,
        )
    finally:
        CLAMP_POSITIVE_DOMAIN = prev_clamp
        SOFTMAX_SIGN_CLIP_TO_UNIT = prev_clip
    return rec


def save_post_profile_records(records: List[JointStageRecord], out_dir: Path) -> None:
    if not records:
        return
    save_csv([r.__dict__ for r in records], out_dir / "post_profile_compare.csv")
    with (out_dir / "post_profile_compare.json").open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in records], f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI overrides
# =============================================================================


def parse_joint_task_arg(text: str) -> Dict[str, Any]:
    parts = [p for p in str(text).split(":")]
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "--joint-task must be model_key:mode:budget[:budget_band[:schedule_csv]]"
        )
    out: Dict[str, Any] = {
        "model_key": parts[0],
        "mode": parts[1],
        "budget": int(parts[2]),
        "schedule_csv": None,
    }
    if len(parts) >= 4 and parts[3] != "":
        out["budget_band"] = int(parts[3])
    if len(parts) >= 5 and parts[4] != "":
        out["schedule_csv"] = parts[4]
    return out


def apply_cli_overrides() -> None:
    parser = argparse.ArgumentParser(description="BERT polynomial joint-refine simulator with sign-softmax (v12 curriculum + v17 focus-fast non-binary changes).")
    parser.add_argument("--run-pipeline", choices=["joint_refine", "ablation_budget_sweep"], default=None)
    parser.add_argument("--model-key", action="append", default=None, help="Repeatable model key override.")
    parser.add_argument("--joint-task", action="append", default=None, help="model_key:mode:budget[:budget_band[:schedule_csv]]")
    parser.add_argument("--joint-root-subdir", default=None)
    parser.add_argument("--shared-proxy-table-path", default=None, help="Existing shared proxy table CSV to reuse, even when writing results to another output subdir.")
    parser.add_argument("--dp-only", action="store_true", help="Only build/evaluate the DP seed schedule; skip local joint search and global CD.")
    parser.add_argument("--disable-joint-global-refinement", action="store_true", help="Run local joint search but skip the global coordinate-descent refinement.")
    parser.add_argument("--joint-mutable-kinds", default=None, help="Comma-separated mutable kinds for refine, e.g. softmax,gelu. Use all or * for all active kinds.")
    parser.add_argument("--joint-mutable-sites", default=None, help="Comma-separated mutable site names for refine. Use all or * for all active sites.")
    parser.add_argument("--joint-proxy-kinds", default=None, help="Comma-separated kinds to build the shared proxy table for. Defaults to all active kinds.")
    parser.add_argument("--batch-size-gpu", type=int, default=None)
    parser.add_argument("--batch-size-cpu", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--auto-loader-start-mult", type=int, default=None)
    parser.add_argument("--auto-loader-max-mult", type=int, default=None)
    parser.add_argument("--auto-loader-prefetch-factor", type=int, default=None)
    parser.add_argument("--joint-fast-n", type=int, default=None)
    parser.add_argument("--joint-full-n", type=int, default=None)
    parser.add_argument("--joint-max-steps", type=int, default=None)
    parser.add_argument("--joint-move-trials", type=int, default=None)
    parser.add_argument("--joint-topk-val", type=int, default=None)
    parser.add_argument("--joint-fast-keep", type=int, default=None)
    parser.add_argument("--joint-acc-gain-min-pp", type=float, default=None)
    parser.add_argument("--joint-obj-worsen-max-ratio", type=float, default=None)
    parser.add_argument("--joint-max-local-cost-delta", type=int, default=None)
    parser.add_argument("--joint-current-val-acc-guard", action="store_true", help="Require candidates to stay within joint-val-acc-current-drop-pp of the current VAL accuracy.")
    parser.add_argument("--joint-val-acc-current-drop-pp", type=float, default=None)
    parser.add_argument("--joint-min-obj-improv-abs", type=float, default=None)
    parser.add_argument("--joint-min-obj-improv-rel", type=float, default=None)
    parser.add_argument("--joint-final-obj-eps-abs", type=float, default=None)
    parser.add_argument("--joint-final-obj-eps-rel", type=float, default=None)
    parser.add_argument("--joint-final-acc-tol-pp", type=float, default=None)
    parser.add_argument("--joint-cd-max-passes", type=int, default=None)
    parser.add_argument("--joint-cd-fast-topk-val", type=int, default=None)
    parser.add_argument("--joint-full-lite-n", type=int, default=None)
    parser.add_argument("--joint-cd-full-lite-topk", type=int, default=None)
    parser.add_argument("--joint-cd-topm-importance", type=int, default=None)
    parser.add_argument("--joint-cd-site-subset-max", type=int, default=None)
    parser.add_argument("--joint-cd-softmax-passes", type=int, default=None)
    parser.add_argument("--joint-cd-gelu-passes", type=int, default=None)
    parser.add_argument("--joint-cd-layernorm-passes", type=int, default=None)
    parser.add_argument("--joint-cd-stop-softmax-if-nf-leq", type=int, default=None)
    parser.add_argument("--joint-cd-skip-layernorm-if-nf-leq", type=int, default=None)
    parser.add_argument("--joint-cd-softmax-site-limit", type=int, default=None)
    parser.add_argument("--joint-cd-gelu-site-limit", type=int, default=None)
    parser.add_argument("--joint-cd-layernorm-site-limit", type=int, default=None)
    parser.add_argument("--proxy-stage1-n", type=int, default=None)
    parser.add_argument("--proxy-stage1-keep-per-cost", type=int, default=None)
    parser.add_argument("--proxy-stage1-keep-gelu", type=int, default=None)
    parser.add_argument("--proxy-stage1-keep-softmax", type=int, default=None)
    parser.add_argument("--proxy-stage1-keep-layernorm", type=int, default=None)
    parser.add_argument("--disable-proxy-stage1-prune", action="store_true")
    parser.add_argument("--disable-auto-tune-loader-batch", action="store_true")
    parser.add_argument("--disable-move-baseline-logits-to-device", action="store_true")
    parser.add_argument("--disable-length-bucketing", action="store_true")
    parser.add_argument("--search-profile", choices=["relaxed", "no_pos_clamp", "no_sign_clip", "strict"], default=None)
    parser.add_argument("--gelu-degrees", default=None, help="Comma-separated GELU Chebyshev degrees, e.g. 45,59,89,119")
    parser.add_argument("--gelu-range-mults", default=None, help="Comma-separated GELU range multipliers, e.g. 1.0,1.25")
    parser.add_argument("--softmax-shift-modes", default=None, help="Comma-separated softmax shift modes: none,row_mean,row_max,sign_max,calib_max,scaled")
    parser.add_argument("--softmax-exp-methods", default=None, help="Comma-separated exp methods, e.g. chebyshev,thor_p2")
    parser.add_argument("--softmax-exp-degrees", default=None, help="Comma-separated exp degrees, e.g. 13,15")
    parser.add_argument("--softmax-exp-range-mults", default=None, help="Comma-separated exp range multipliers, e.g. 1.0,1.25")
    parser.add_argument("--softmax-inv-init-methods", default=None, help="Comma-separated reciprocal init methods, e.g. thor_heinv,chebyshev,taylor")
    parser.add_argument("--softmax-inv-init-degrees", default=None, help="Comma-separated reciprocal init degrees, e.g. 3,5,7")
    parser.add_argument("--softmax-inv-range-mults", default=None, help="Comma-separated reciprocal range multipliers")
    parser.add_argument("--softmax-inv-refines", default=None, help="Comma-separated reciprocal refine methods, e.g. none,newton,goldschmidt")
    parser.add_argument("--softmax-inv-iters", default=None, help="Comma-separated reciprocal refine iterations, e.g. 1,2,3")
    parser.add_argument("--softmax-input-scales", default=None, help="Comma-separated x/scale factors for scaled softmax, e.g. 4,8,16")
    parser.add_argument("--softmax-square-iters", default=None, help="Comma-separated square-normalize iterations for scaled softmax, e.g. 2,3,4")
    parser.add_argument(
        "--softmax-scale-square-mode",
        choices=["matched", "explicit"],
        default=None,
        help="How to combine scaled-softmax scales and square iters. matched keeps only scale=2^iters.",
    )
    parser.add_argument(
        "--softmax-scale-square-pairs",
        default=None,
        help="Comma-separated explicit scaled-softmax SCALE:ITERS pairs, e.g. 2:1,4:2,8:3. Implies explicit mode.",
    )
    parser.add_argument("--softmax-cost-min", type=int, default=None, help="Keep only softmax choices with per-site depth cost >= this value.")
    parser.add_argument("--softmax-cost-max", type=int, default=None, help="Keep only softmax choices with per-site depth cost <= this value.")
    parser.add_argument("--softmax-label-allow-regex", default=None, help="Keep only softmax choices whose generated label matches this regex.")
    parser.add_argument("--softmax-thor-heinv-cost", type=int, default=None, help="Per reciprocal depth-cost heuristic for thor_heinv choices.")
    parser.add_argument("--softmax-calib-max-quantile", type=float, default=None)
    parser.add_argument("--softmax-calib-max-margin-mult", type=float, default=None)
    parser.add_argument("--softmax-calib-max-margin-abs", type=float, default=None)
    parser.add_argument("--layernorm-init-methods", default=None, help="Comma-separated LayerNorm rsqrt init methods, e.g. chebyshev,taylor")
    parser.add_argument("--layernorm-init-degrees", default=None, help="Comma-separated LayerNorm rsqrt init degrees, e.g. 5,7,11,15")
    parser.add_argument("--layernorm-range-mults", default=None, help="Comma-separated LayerNorm rsqrt range multipliers")
    parser.add_argument("--layernorm-refines", default=None, help="Comma-separated LayerNorm refine methods, e.g. none,newton,goldschmidt")
    parser.add_argument("--layernorm-iters", default=None, help="Comma-separated LayerNorm refine iterations, e.g. 1,2,3")
    parser.add_argument("--reuse-proxy-table", action="store_true", help="Reuse proxy table if compatible")
    parser.add_argument("--no-reuse-proxy-table", action="store_true", help="Force proxy rebuild even if cache exists")
    parser.add_argument("--post-eval-profiles", default=None, help="Comma-separated extra profiles to evaluate on the in-memory best schedule, e.g. strict or relaxed,strict")
    args = parser.parse_args()

    global RUN_PIPELINE, RUN_MODEL_KEYS, JOINT_TASKS, JOINT_ROOT_SUBDIR, JOINT_SHARED_PROXY_TABLE_PATH
    global BATCH_SIZE_GPU, BATCH_SIZE_CPU, NUM_WORKERS
    global AUTO_TUNE_LOADER_BATCH, AUTO_LOADER_START_MULT, AUTO_LOADER_MAX_MULT, AUTO_LOADER_PREFETCH_FACTOR
    global JOINT_FAST_N, JOINT_FULL_N, JOINT_MAX_STEPS, JOINT_MOVE_TRIALS_PER_STEP, JOINT_TOPK_VAL, JOINT_FAST_KEEP
    global JOINT_ACC_GAIN_MIN_PP, JOINT_OBJ_WORSEN_MAX_RATIO, JOINT_MAX_LOCAL_COST_DELTA
    global JOINT_ENABLE_CURRENT_VAL_ACC_GUARD, JOINT_VAL_ACC_CURRENT_DROP_PP
    global JOINT_MIN_OBJ_IMPROV_ABS, JOINT_MIN_OBJ_IMPROV_REL, JOINT_FINAL_OBJ_EPS_ABS, JOINT_FINAL_OBJ_EPS_REL
    global JOINT_FINAL_ACC_TOL_PP
    global JOINT_CD_MAX_PASSES, JOINT_CD_FAST_TOPK_VAL, MOVE_BASELINE_LOGITS_TO_DEVICE, ENABLE_LENGTH_BUCKETING
    global RUN_JOINT_SEARCH, RUN_JOINT_GLOBAL_REFINEMENT
    global JOINT_FULL_LITE_N, JOINT_CD_FULL_LITE_TOPK
    global JOINT_CD_TOPM_IMPORTANCE, JOINT_CD_SITE_SUBSET_MAX
    global JOINT_CD_SOFTMAX_PASSES, JOINT_CD_GELU_PASSES, JOINT_CD_LAYERNORM_PASSES
    global JOINT_CD_STOP_SOFTMAX_IF_FULL_NF_LEQ, JOINT_CD_SKIP_LAYERNORM_IF_FULL_NF_LEQ
    global JOINT_CD_SOFTMAX_SITE_LIMIT, JOINT_CD_GELU_SITE_LIMIT, JOINT_CD_LAYERNORM_SITE_LIMIT
    global JOINT_MUTABLE_KIND_FILTER, JOINT_MUTABLE_SITES, JOINT_PROXY_KIND_FILTER
    global PROXY_STAGE1_N, PROXY_STAGE1_KEEP_PER_COST, ENABLE_PROXY_STAGE1_PRUNE
    global PROXY_STAGE1_KEEP_PER_COST_GELU, PROXY_STAGE1_KEEP_PER_COST_SOFTMAX, PROXY_STAGE1_KEEP_PER_COST_LAYERNORM
    global SEARCH_PROFILE, POST_EVAL_PROFILES
    global GELU_DEGREES, GELU_RANGE_MULTS
    global SOFTMAX_SHIFT_MODES
    global SOFTMAX_EXP_METHODS, SOFTMAX_EXP_DEGREES, SOFTMAX_EXP_RANGE_MULTS
    global SOFTMAX_INV_INIT_METHODS, SOFTMAX_INV_INIT_DEGREES, SOFTMAX_INV_RANGE_MULTS, SOFTMAX_INV_REFINES, SOFTMAX_INV_ITERS
    global SOFTMAX_INPUT_SCALES, SOFTMAX_SQUARE_ITERS, SOFTMAX_SCALE_SQUARE_MODE, SOFTMAX_SCALE_SQUARE_PAIRS
    global SOFTMAX_COST_MIN, SOFTMAX_COST_MAX, SOFTMAX_LABEL_ALLOW_REGEX, SOFTMAX_THOR_HEINV_COST
    global SOFTMAX_CALIB_MAX_QUANTILE, SOFTMAX_CALIB_MAX_MARGIN_MULT, SOFTMAX_CALIB_MAX_MARGIN_ABS
    global LAYERNORM_INIT_METHODS, LAYERNORM_INIT_DEGREES, LAYERNORM_RANGE_MULTS, LAYERNORM_REFINES, LAYERNORM_ITERS
    global REUSE_PROXY_TABLE

    if args.run_pipeline is not None:
        RUN_PIPELINE = str(args.run_pipeline)
    if args.model_key:
        RUN_MODEL_KEYS = list(args.model_key)
    if args.joint_task:
        JOINT_TASKS = [parse_joint_task_arg(x) for x in args.joint_task]
    if args.joint_root_subdir is not None:
        JOINT_ROOT_SUBDIR = str(args.joint_root_subdir)
    if args.shared_proxy_table_path is not None:
        JOINT_SHARED_PROXY_TABLE_PATH = Path(args.shared_proxy_table_path)
    if args.dp_only:
        RUN_JOINT_SEARCH = False
        RUN_JOINT_GLOBAL_REFINEMENT = False
    if args.disable_joint_global_refinement:
        RUN_JOINT_GLOBAL_REFINEMENT = False
    if args.joint_mutable_kinds is not None:
        JOINT_MUTABLE_KIND_FILTER = parse_mutable_kind_list(args.joint_mutable_kinds)
    if args.joint_mutable_sites is not None:
        JOINT_MUTABLE_SITES = parse_mutable_site_list(args.joint_mutable_sites)
    if args.joint_proxy_kinds is not None:
        JOINT_PROXY_KIND_FILTER = parse_mutable_kind_list(args.joint_proxy_kinds)
    if args.batch_size_gpu is not None:
        BATCH_SIZE_GPU = int(args.batch_size_gpu)
    if args.batch_size_cpu is not None:
        BATCH_SIZE_CPU = int(args.batch_size_cpu)
    if args.num_workers is not None:
        NUM_WORKERS = int(args.num_workers)
    if args.auto_loader_start_mult is not None:
        AUTO_LOADER_START_MULT = int(args.auto_loader_start_mult)
    if args.auto_loader_max_mult is not None:
        AUTO_LOADER_MAX_MULT = int(args.auto_loader_max_mult)
    if args.auto_loader_prefetch_factor is not None:
        AUTO_LOADER_PREFETCH_FACTOR = int(args.auto_loader_prefetch_factor)
    if args.joint_fast_n is not None:
        JOINT_FAST_N = int(args.joint_fast_n)
    if args.joint_full_n is not None:
        JOINT_FULL_N = int(args.joint_full_n)
    if args.joint_max_steps is not None:
        JOINT_MAX_STEPS = int(args.joint_max_steps)
    if args.joint_move_trials is not None:
        JOINT_MOVE_TRIALS_PER_STEP = int(args.joint_move_trials)
    if args.joint_topk_val is not None:
        JOINT_TOPK_VAL = int(args.joint_topk_val)
    if args.joint_fast_keep is not None:
        JOINT_FAST_KEEP = int(args.joint_fast_keep)
    if args.joint_acc_gain_min_pp is not None:
        JOINT_ACC_GAIN_MIN_PP = float(args.joint_acc_gain_min_pp)
    if args.joint_obj_worsen_max_ratio is not None:
        JOINT_OBJ_WORSEN_MAX_RATIO = float(args.joint_obj_worsen_max_ratio)
    if args.joint_max_local_cost_delta is not None:
        JOINT_MAX_LOCAL_COST_DELTA = int(args.joint_max_local_cost_delta)
    if args.joint_current_val_acc_guard:
        JOINT_ENABLE_CURRENT_VAL_ACC_GUARD = True
    if args.joint_val_acc_current_drop_pp is not None:
        JOINT_VAL_ACC_CURRENT_DROP_PP = float(args.joint_val_acc_current_drop_pp)
    if args.joint_min_obj_improv_abs is not None:
        JOINT_MIN_OBJ_IMPROV_ABS = float(args.joint_min_obj_improv_abs)
    if args.joint_min_obj_improv_rel is not None:
        JOINT_MIN_OBJ_IMPROV_REL = float(args.joint_min_obj_improv_rel)
    if args.joint_final_obj_eps_abs is not None:
        JOINT_FINAL_OBJ_EPS_ABS = float(args.joint_final_obj_eps_abs)
    if args.joint_final_obj_eps_rel is not None:
        JOINT_FINAL_OBJ_EPS_REL = float(args.joint_final_obj_eps_rel)
    if args.joint_final_acc_tol_pp is not None:
        JOINT_FINAL_ACC_TOL_PP = float(args.joint_final_acc_tol_pp)
    if args.joint_cd_max_passes is not None:
        JOINT_CD_MAX_PASSES = int(args.joint_cd_max_passes)
    if args.joint_cd_fast_topk_val is not None:
        JOINT_CD_FAST_TOPK_VAL = int(args.joint_cd_fast_topk_val)
    if args.joint_full_lite_n is not None:
        JOINT_FULL_LITE_N = int(args.joint_full_lite_n)
    if args.joint_cd_full_lite_topk is not None:
        JOINT_CD_FULL_LITE_TOPK = int(args.joint_cd_full_lite_topk)
    if args.joint_cd_topm_importance is not None:
        JOINT_CD_TOPM_IMPORTANCE = int(args.joint_cd_topm_importance)
    if args.joint_cd_site_subset_max is not None:
        JOINT_CD_SITE_SUBSET_MAX = int(args.joint_cd_site_subset_max)
    if args.joint_cd_softmax_passes is not None:
        JOINT_CD_SOFTMAX_PASSES = int(args.joint_cd_softmax_passes)
    if args.joint_cd_gelu_passes is not None:
        JOINT_CD_GELU_PASSES = int(args.joint_cd_gelu_passes)
    if args.joint_cd_layernorm_passes is not None:
        JOINT_CD_LAYERNORM_PASSES = int(args.joint_cd_layernorm_passes)
    if args.joint_cd_stop_softmax_if_nf_leq is not None:
        JOINT_CD_STOP_SOFTMAX_IF_FULL_NF_LEQ = int(args.joint_cd_stop_softmax_if_nf_leq)
    if args.joint_cd_skip_layernorm_if_nf_leq is not None:
        JOINT_CD_SKIP_LAYERNORM_IF_FULL_NF_LEQ = int(args.joint_cd_skip_layernorm_if_nf_leq)
    if args.joint_cd_softmax_site_limit is not None:
        JOINT_CD_SOFTMAX_SITE_LIMIT = int(args.joint_cd_softmax_site_limit)
    if args.joint_cd_gelu_site_limit is not None:
        JOINT_CD_GELU_SITE_LIMIT = int(args.joint_cd_gelu_site_limit)
    if args.joint_cd_layernorm_site_limit is not None:
        JOINT_CD_LAYERNORM_SITE_LIMIT = int(args.joint_cd_layernorm_site_limit)
    if args.proxy_stage1_n is not None:
        PROXY_STAGE1_N = int(args.proxy_stage1_n)
    if args.proxy_stage1_keep_per_cost is not None:
        PROXY_STAGE1_KEEP_PER_COST = int(args.proxy_stage1_keep_per_cost)
    if args.proxy_stage1_keep_gelu is not None:
        PROXY_STAGE1_KEEP_PER_COST_GELU = int(args.proxy_stage1_keep_gelu)
    if args.proxy_stage1_keep_softmax is not None:
        PROXY_STAGE1_KEEP_PER_COST_SOFTMAX = int(args.proxy_stage1_keep_softmax)
    if args.proxy_stage1_keep_layernorm is not None:
        PROXY_STAGE1_KEEP_PER_COST_LAYERNORM = int(args.proxy_stage1_keep_layernorm)
    if args.disable_proxy_stage1_prune:
        ENABLE_PROXY_STAGE1_PRUNE = False
    if args.disable_auto_tune_loader_batch:
        AUTO_TUNE_LOADER_BATCH = False
    if args.disable_move_baseline_logits_to_device:
        MOVE_BASELINE_LOGITS_TO_DEVICE = False
    if args.disable_length_bucketing:
        ENABLE_LENGTH_BUCKETING = False
    if args.search_profile is not None:
        SEARCH_PROFILE = str(args.search_profile)
    if args.gelu_degrees is not None:
        GELU_DEGREES = parse_int_list(args.gelu_degrees)
    if args.gelu_range_mults is not None:
        GELU_RANGE_MULTS = parse_float_list(args.gelu_range_mults)
    if args.softmax_shift_modes is not None:
        SOFTMAX_SHIFT_MODES = parse_softmax_shift_modes(args.softmax_shift_modes)
    if args.softmax_exp_methods is not None:
        SOFTMAX_EXP_METHODS = parse_string_list(args.softmax_exp_methods)
    if args.softmax_exp_degrees is not None:
        SOFTMAX_EXP_DEGREES = parse_int_list(args.softmax_exp_degrees)
    if args.softmax_exp_range_mults is not None:
        SOFTMAX_EXP_RANGE_MULTS = parse_float_list(args.softmax_exp_range_mults)
    if args.softmax_inv_init_methods is not None:
        SOFTMAX_INV_INIT_METHODS = parse_string_list(args.softmax_inv_init_methods)
    if args.softmax_inv_init_degrees is not None:
        SOFTMAX_INV_INIT_DEGREES = parse_int_list(args.softmax_inv_init_degrees)
    if args.softmax_inv_range_mults is not None:
        SOFTMAX_INV_RANGE_MULTS = parse_float_list(args.softmax_inv_range_mults)
    if args.softmax_inv_refines is not None:
        SOFTMAX_INV_REFINES = parse_string_list(args.softmax_inv_refines)
    if args.softmax_inv_iters is not None:
        SOFTMAX_INV_ITERS = parse_int_list(args.softmax_inv_iters)
    if args.softmax_input_scales is not None:
        SOFTMAX_INPUT_SCALES = parse_float_list(args.softmax_input_scales)
    if args.softmax_square_iters is not None:
        SOFTMAX_SQUARE_ITERS = parse_int_list(args.softmax_square_iters)
    if args.softmax_scale_square_mode is not None:
        SOFTMAX_SCALE_SQUARE_MODE = str(args.softmax_scale_square_mode)
    if args.softmax_scale_square_pairs is not None:
        SOFTMAX_SCALE_SQUARE_PAIRS = parse_scale_square_pairs(args.softmax_scale_square_pairs)
        if args.softmax_scale_square_mode is None:
            SOFTMAX_SCALE_SQUARE_MODE = "explicit"
    if args.softmax_cost_min is not None:
        SOFTMAX_COST_MIN = int(args.softmax_cost_min)
    if args.softmax_cost_max is not None:
        SOFTMAX_COST_MAX = int(args.softmax_cost_max)
    if args.softmax_label_allow_regex is not None:
        SOFTMAX_LABEL_ALLOW_REGEX = str(args.softmax_label_allow_regex)
    if args.softmax_thor_heinv_cost is not None:
        SOFTMAX_THOR_HEINV_COST = int(args.softmax_thor_heinv_cost)
    if args.softmax_calib_max_quantile is not None:
        SOFTMAX_CALIB_MAX_QUANTILE = float(args.softmax_calib_max_quantile)
    if args.softmax_calib_max_margin_mult is not None:
        SOFTMAX_CALIB_MAX_MARGIN_MULT = float(args.softmax_calib_max_margin_mult)
    if args.softmax_calib_max_margin_abs is not None:
        SOFTMAX_CALIB_MAX_MARGIN_ABS = float(args.softmax_calib_max_margin_abs)
    if args.layernorm_init_methods is not None:
        LAYERNORM_INIT_METHODS = parse_string_list(args.layernorm_init_methods)
    if args.layernorm_init_degrees is not None:
        LAYERNORM_INIT_DEGREES = parse_int_list(args.layernorm_init_degrees)
    if args.layernorm_range_mults is not None:
        LAYERNORM_RANGE_MULTS = parse_float_list(args.layernorm_range_mults)
    if args.layernorm_refines is not None:
        LAYERNORM_REFINES = parse_string_list(args.layernorm_refines)
    if args.layernorm_iters is not None:
        LAYERNORM_ITERS = parse_int_list(args.layernorm_iters)
    if args.reuse_proxy_table and args.no_reuse_proxy_table:
        raise ValueError("Cannot set both --reuse-proxy-table and --no-reuse-proxy-table")
    if args.reuse_proxy_table:
        REUSE_PROXY_TABLE = True
    if args.no_reuse_proxy_table:
        REUSE_PROXY_TABLE = False
    if args.post_eval_profiles is not None:
        POST_EVAL_PROFILES = parse_profile_list(args.post_eval_profiles)


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    apply_cli_overrides()
    set_seeds(GLOBAL_SEED)
    if RUN_PIPELINE == "ablation_budget_sweep":
        for model_key in RUN_MODEL_KEYS:
            run_one_model(model_key)
        return
    if RUN_PIPELINE == "joint_refine":
        tasks = resolve_joint_tasks(JOINT_TASKS)
        grouped: Dict[str, List[JointTaskSpec]] = defaultdict(list)
        for task in tasks:
            grouped[task.model_key].append(task)
        for model_key, model_tasks in grouped.items():
            run_joint_tasks_for_model(model_key, model_tasks)
        return
    raise ValueError(f"Unknown RUN_PIPELINE={RUN_PIPELINE!r}. Expected 'joint_refine' or 'ablation_budget_sweep'.")


if __name__ == "__main__":
    main()
