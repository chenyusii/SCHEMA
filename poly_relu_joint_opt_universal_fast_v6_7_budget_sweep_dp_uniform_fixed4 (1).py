#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poly_relu_joint_opt_universal_fast_v2.py

A faster, universal, self-contained (single-file) optimizer for per-ReLU Chebyshev degrees
under a depth/budget constraint, designed to approximate FHE plaintext behavior:

  - NO clip(x, [a,b])
  - NO clamp(t, [-1,1])

  1) FAST stage uses MSE-only eval (no hooks) (already), and we now compute baseline logits
     for FAST loader directly (no slicing assumptions).
  2) FULL-guard is **gated** everywhere (joint + global CD + beam) and configurable.
  3) Global coordinate descent (CD) is sped up by:
       - Only scanning a **subset** of sites (importance/risk pool), not all sites.
       - Using FAST pre-screen for per-site candidate degrees; only evaluate top-K on VAL.
       - Computing current VAL objective once per pass (not per site).
  4) Beam refinement full-guards only top-N candidates per step (configurable).
  5) Optional evaluation cache for MSE-only evaluations (LRU).

Config is a top block (no CLI). Edit "User Config".

"""

from __future__ import annotations

# =============================================================================
# Torch + minimal image utilities (NO torchvision dependency)
# =============================================================================
import torch
from PIL import Image

import copy
import csv
import math
import itertools
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# =============================================================================
# User Config (edit here)
# =============================================================================

# -------- Paths (repo-relative by default) --------
# WEIGHTS_REL = Path("examples/resnet/gen_aespa_weights/ResNet20.pth")
WEIGHTS_REL = Path("examples/resnet/gen_aespa_weights/resnet32_cifar10_from_txt.pth")
# WEIGHTS_REL = Path("./examples/resnet/gen_aespa_weights/resnet18-f37072fd.pth")

DATASET_NAME = "cifar10_bin"  # "cifar10_bin","cifar100_bin","cifar10","cifar100","imagenet","tinyimagenet"
# DATASET_NAME = "imagenet"  # "cifar10_bin","cifar100_bin","cifar10","cifar100","imagenet","tinyimagenet"

CIFAR10_BIN_DIR_REL = Path("examples/resnet/cifar10")
CIFAR10_BIN_TRAIN_RELS = [CIFAR10_BIN_DIR_REL / f"data_batch_{i}.bin" for i in range(1, 6)]
CIFAR10_BIN_TEST_REL = CIFAR10_BIN_DIR_REL / "test_batch.bin"

CIFAR100_BIN_DIR_REL = Path("examples/resnet/cifar100")
CIFAR100_BIN_TRAIN_RELS = [CIFAR100_BIN_DIR_REL / "train.bin"]  # edit if needed
CIFAR100_BIN_TEST_REL = CIFAR100_BIN_DIR_REL / "test.bin"  # edit if needed

TORCHVISION_CIFAR_ROOT_REL = Path("~/.torch_datasets").expanduser()

IMAGENET_ROOT = Path("./examples/resnet/imagenet_1k")          # must have train/ and val/
TINYIMAGENET_ROOT = Path("/path/to/tinyimagenet")  # must have train/ and val/

OUTPUT_DIR_REL = Path("examples/resnet/gen_aespa_weights/poly_opt_outputs_fast_v2_res32_cifar10_dp")

# -------- Model choice --------
MODEL_NAME = "auto"  # "auto", "resnet56_cifar", "resnet18", "vgg16_bn", ...

# -------- Device / batching --------
USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE_GPU = 1024
BATCH_SIZE_CPU = 256
NUM_WORKERS = 8


# -------- Auto-tune DataLoader batch size (results unchanged) --------
# Goal: reduce Python/DataLoader iteration overhead WITHOUT changing any numeric results.
# We keep the *compute* micro-batch size the same as before (BATCH_SIZE_GPU/CPU) by
# splitting each fetched loader batch into fixed-size microbatches inside eval loops.
# This preserves evaluation reduction order and thus keeps schedule/accept decisions identical.
AUTO_TUNE_LOADER_BATCH = True
# Try at most (micro_batch * START_MULT), then shrink by /2 until a single batch can be fetched.
AUTO_LOADER_START_MULT = 8
# Cap the tuned loader batch size to (micro_batch * MAX_MULT) to avoid excessive host RAM.
AUTO_LOADER_MAX_MULT = 32
# DataLoader prefetching (only used when NUM_WORKERS>0). Does not change ordering or results.
AUTO_LOADER_PREFETCH_FACTOR = 4

# -------- Subset sizes --------
CALIB_N = 5000
DP_SWEEP_N = 2000
JOINT_SEARCH_N = 2000
JOINT_VAL_N = 2000
JOINT_FAST_N = 512
TEST_MAX_N = None  # None => full test

GLOBAL_SEED = 0

# -------- Bounds --------
BOUND_MODE = "minmax"  # "minmax" or "quantile"
Q_LOW = 0.001
Q_HIGH = 0.999
RANGE_MARGIN_FRAC = 0.02

# ---- Backward-compatible alias names (for older code / IDE diagnostics) ----
BOUNDS_MODE = BOUND_MODE
BOUNDS_MARGIN_FRAC = RANGE_MARGIN_FRAC
BOUNDS_Q_LOW = Q_LOW
BOUNDS_Q_HIGH = Q_HIGH

# -------- DP mixed scaling (minmax vs symmetric) --------
# If True: when building the DP one-layer error table, we evaluate BOTH minmax scaling
# (bounds = [a,b]) and symmetric scaling (bounds = [-max(|a|,|b|), +max(|a|,|b|)]) for each
# (site, degree), and use the better one. This allows mixing the two scaling modes across sites.
ENABLE_MIXED_SCALING_IN_DP = False

# -------- Approx (HE/FHE faithful) --------
USE_CLIP_X = False
CLAMP_T_TO_UNIT = False
RANGE_EPS = 1e-12

# -------- Candidate degrees + budget --------
CANDIDATE_DEGREES = [3, 5, 7, 9, 11, 13, 17, 21, 27, 39, 59, 79, 89, 119, 129, 139, 149, 179, 249]

# Auto-expand: ensure degrees {13,15,17,...,59} are included.
AUTO_EXPAND_CANDIDATE_DEGREES_13_TO_59_STEP2 = True
if AUTO_EXPAND_CANDIDATE_DEGREES_13_TO_59_STEP2:
    _extra = list(range(13, 60, 2))
    CANDIDATE_DEGREES = sorted(set(int(x) for x in (list(CANDIDATE_DEGREES) + _extra)))
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
TOTAL_BUDGET = 91  # for ResNet56 typical; adjust if desired

# -------- Budget sweep (DP vs UNIFORM) --------
# If enabled, we will sweep TOTAL_BUDGET in [BUDGET_SWEEP_START, BUDGET_SWEEP_END]
# and record DP schedule metrics + best UNIFORM schedule metrics into a JSON file.
ENABLE_BUDGET_SWEEP = True
BUDGET_SWEEP_START = 120
BUDGET_SWEEP_END = 160
BUDGET_SWEEP_JSON_NAME = "budget_sweep_dp_uniform_metrics.json"
SAVE_ALL_UNIFORM_PER_BUDGET = True  # if True, also store metrics for every feasible uniform degree (can be large)

# -------- Reuse DP one-layer error table --------
REUSE_DP_ONE_LAYER_ERROR_TABLE = True
DP_ONE_LAYER_ERROR_TABLE_BEST_NAME = "dp_one_layer_error_table_no_clip.csv"
DP_ONE_LAYER_ERROR_TABLE_MIXED_NAME = "dp_one_layer_error_table_no_clip_mixed_scaling.csv"

# -------- Uniform candidate schedules (helps when DP becomes unstable at high budgets) --------
# We will optionally evaluate uniform schedules (same degree for all sites) and:
#   - use a feasible uniform schedule as the START if DP violates FULL guard
#   - include feasible uniform schedules in FINAL candidate comparison
ENABLE_UNIFORM_CANDIDATES = True
UNIFORM_CANDIDATE_DEGS = []  # will be overwritten to match CANDIDATE_DEGREES below
UNIFORM_CANDIDATE_DEGS = list(CANDIDATE_DEGREES)
# will be filtered by budget; add more if you like
AUTO_ADD_MAX_UNIFORM_DEG_UNDER_BUDGET = True
PREFER_FEASIBLE_UNIFORM_START_IF_DP_FAILS = True

# -------- Penalized objective & guards --------
LAMBDA_ABS = 0.02
LAMBDA_T = 0.02
LAMBDA_OOR = 0.05

ABS_MULT_SOFT = 8.0
ABS_MULT_HARD = 50.0
T_MAX_SOFT = 1.8
T_MAX_HARD = 10.0
OOR_SOFT = 0.001  # 0.1%
OOR_HARD = 0.05  # 5%

ENABLE_ACC_GUARD = True
ACC_GUARD_DROP_PP = 1.0

# -------- SPEEDUPS --------
# FAST stage should NOT collect stats (hooks). Rank by MSE only.
FAST_COLLECT_STATS = False

# Full-guard gating:
FULL_GUARD_TOP_CANDS_PER_STEP = 1  # in joint: run full-guard only for top-1 VAL-improving candidate(s)
FULL_GUARD_EVERY_K_ACCEPTS = 5  # after every K accepts, force a full-guard check on the current schedule
FULL_GUARD_TRIGGER_REL_IMPROV = 0.03  # if candidate improves VAL obj by >= 3%, run full-guard before accepting

# In beam refinement:
BEAM_FULL_GUARD_TOP = 1  # full-guard only top-N beam candidates per step (others filtered by VAL only)

# Site-priority proposal:
ENABLE_SITE_PRIORITY_PROPOSAL = True
PROPOSAL_TOPM_IMPORTANCE = 14
PROPOSAL_TOPM_RISK = 10
PROPOSAL_POOL_MAX = 18
PROPOSAL_POOL_PROB = 0.85

# MSE-only evaluation cache (LRU). Helps when schedules repeat.
ENABLE_EVAL_CACHE = True
EVAL_CACHE_MAX_ITEMS = 6000

# -------- Search breadth --------
JOINT_MAX_STEPS = 10
JOINT_MOVE_TRIALS_PER_STEP = 256
JOINT_TOPK_VAL = 8

RUN_GLOBAL_REFINEMENT = False
GLOBAL_REFINEMENT_RESTARTS = 0
GLOBAL_REFINEMENT_KICK_SITES = 4
GLOBAL_REFINEMENT_KICK_TRIES = 80

# Faster CD:
ENABLE_CD_SITE_SUBSET = False
CD_TOPM_IMPORTANCE = 28
CD_TOPM_RISK = 16
CD_SITE_SUBSET_MAX = 34
CD_FAST_TOPK_VAL = 2  # per-site: after FAST pre-screen, evaluate top-K candidates on VAL
CD_MAX_ACCEPTS_PER_PASS = 8  # early stop pass after this many accepts (speed)
CD_MAX_PASSES = 4

# Beam refinement:
RUN_GLOBAL_BEAM = False
GLOBAL_BEAM_WIDTH = 4
GLOBAL_BEAM_STEPS = 20
GLOBAL_BEAM_RANDOM_MOVES_PER_PARENT = 120
GLOBAL_BEAM_TOPK_FAST = 160
GLOBAL_BEAM_TOPK_VAL = 60

# Stronger moves
ENABLE_TRI_MOVES = True

# Optional local window refinement (can be expensive)
ENABLE_LOCAL_WINDOW_SEARCH = False
LOCAL_WINDOW_SIZE = 6
LOCAL_WINDOW_TOPK_SITES = 4
LOCAL_DEG_NEIGHBOR_RADIUS = 1
LOCAL_PROXY_TOPK = 600
LOCAL_FAST_TOPK = 200
LOCAL_VAL_TOPK = 50
LOCAL_FULL_TOPK = 6
LOCAL_IMPROVE_EPS = 1e-9

# Optional outer loop (bounds<->degree coupling)
ENABLE_OUTER_RECALIB = True
OUTER_RECALIB_ITERS = 1
OUTER_RECALIB_BOUND_MODE = "minmax"
OUTER_RECALIB_MARGIN_FRAC = 0.02

# -------- Coeff cache --------
COEFF_CACHE_ROUND_DECIMALS = 6

# -------- Diagnostic: exact ReLU on 1–3 sites (PLAINTEXT ONLY; for diagnosis) --------
# This does NOT represent an HE-compatible model; it is a probe to see if a few sites dominate the accuracy drop.
RUN_EXACT_RELU_DIAG = True
EXACT_RELU_DIAG_BASE = "final"  # "final" | "dp" | "joint" | "uniform"
EXACT_RELU_DIAG_UNIFORM_DEG = 27

# Evaluate all single-site overrides, then enumerate pairs/triples among top-K singles.
EXACT_RELU_DIAG_TOPK_SINGLE = 12  # top-K sites (by Δacc on VAL) used for pair/triple enumeration
EXACT_RELU_DIAG_PRINT_TOP = 15  # how many to print in each section
EXACT_RELU_DIAG_EVAL_ON_TEST = False  # optional: evaluate best 1/2/3-site combos on TEST (report only)

# Note: selection is ALWAYS done on VAL; TEST is never used for choosing.
# -------- Full-guard diagnostics: per-image explosion statistics (PLAINTEXT ONLY) --------
# Motivation: sometimes a FULL-guard failure (huge abs_x / t) is caused by only a handful of "pathological" samples.
# This report tells you how many images are responsible, and which activation sites they first blow up at.
ENABLE_SAMPLE_GUARD_REPORT = True
SAMPLE_GUARD_ONLY_ON_FAIL = True  # if True: only run when FULL guard fails; else always run
SAMPLE_GUARD_REPORT_AT_DP_FULL = True  # run after initial DP FULL-guard check
SAMPLE_GUARD_REPORT_AT_FINAL = True  # run after final schedule is chosen
SAMPLE_GUARD_MAX_OFFENDERS_PRINT = 20  # print top-K offending samples
SAMPLE_GUARD_SAVE_CSV = True  # save per-sample & per-site CSVs under out_dir
SAMPLE_GUARD_REPORT_AT_FINAL_FULL = SAMPLE_GUARD_REPORT_AT_FINAL  # alias

# OOR diagnostics: counts samples that have ANY out-of-range element (min<a or max>b).
# This is informative but NOT the same as the guard's element-wise oor_rate threshold.
SAMPLE_GUARD_COUNT_OOR_ANY = True
SAMPLE_GUARD_INCLUDE_OOR_IN_EXPLODE = False  # if True, treat oor_any as "explode" too (usually keep False)

# -------- Relaxed FULL-guard: allow rare outlier explosions (PLAINTEXT ONLY) --------
# You said: a tiny fraction of images may blow up (e.g., 10 / 5000), but NOT "everyone slightly overflows".
ALLOW_RARE_OUTLIER_EXPLOSIONS = True
ENFORCE_SAMPLE_GUARD_ON_ACCEPT = False

# Rates are measured on FULL calib loader (size ~= CALIB_N).
MAX_CALIB_EXPLODE_RATE = 0.002  # 0.2%  -> 10/5000
MAX_CALIB_NONFINITE_RATE = 0.002
# MAX_CALIB_OOR_ANY_RATE   = 0.20    # forbid "every image has some OOR somewhere"
MAX_CALIB_OOR_ANY_RATE = 1.1  # forbid "every image has some OOR somewhere"

# -------- Targeted repair inside polynomial eval (PLAINTEXT ONLY) --------
# This is NOT FHE-faithful, but you explicitly allowed targeted repair at explosion sites.
ENABLE_TARGETED_REPAIR = True
REPAIR_T_CLAMP = 50.0  # clamp |t| for bad elements only
REPAIR_Y_CLAMP = 1.0e6  # clamp |y| or fallback for bad elements only
REPAIR_FALLBACK_TO_RELU = False  # for bad elements, use exact relu(x) as a repair

# -------- Pareto schedule selection (VAL) --------
# We treat (val_obj ↓, val_acc ↑) as a Pareto problem; selection prefers high-acc but keeps uniform randomness.
ENABLE_PARETO_SELECTION = True
PARETO_UNIFORM_PICK_PROB = 0.20
PARETO_ACC_EPS = 0.0  # absolute epsilon for acc dominance
PARETO_OBJ_EPS_REL = 0.0  # relative epsilon for obj dominance
PARETO_PICK_MODE = "acc_then_obj"  # "acc_then_obj" or "score"
PARETO_SCORE_ALPHA = 0.05  # only used when mode=="score": score = acc - alpha*log(obj)
PARETO_MAX_CANDS_TRY = 6  # try up to K Pareto candidates when FULL-guard is triggered

# -------- Final schedule selection heuristic (VAL-only; avoid overfitting to tiny acc deltas) --------
FINAL_RANDOM_PICK_PROB = 0.0  # randomness in final pick (0 = deterministic)
FINAL_ACC_TOL_PP = 0.30  # within this many pp of best VAL acc => prefer lower obj / more stable schedule
FINAL_PREFER_STRICT_WITHIN_TOL = True  # if any strict-ok schedules exist within tol, prefer them
FINAL_TIEBREAK_BY_STABILITY = True  # tie-break with FULL abs_x / t / worst_oor (computed on CALIB)

# -------- Bounded acc-guided acceptance (avoid obj astronomical while acc tiny improves) --------
# Allow accepting a candidate even if val_obj is worse, only when acc gain is meaningful and obj doesn't worsen too much.
ENABLE_ACC_OBJ_TRADEOFF = True
ACC_GAIN_MIN_PP = 0.20  # require at least +0.20 percentage points acc gain
OBJ_WORSEN_MAX_RATIO = 1.20  # allow obj to worsen by at most 20% when taking an acc-driven move

# -------- Robust logit-MSE objective (matches "rare outliers OK") --------
# When baseline_logits are provided, compute per-sample MSE and aggregate robustly to reduce domination by a few outliers.
USE_ROBUST_MSE = True
ROBUST_MSE_MODE = "cap"  # "cap" or "trim"
ROBUST_MSE_CAP_Q = 0.995  # cap per-sample MSE at this quantile (computed per-batch)
ROBUST_MSE_TRIM_TOP_FRAC = 0.002  # trim top fraction of samples in each batch when mode=="trim"

# -------- Multi-start global refinement (DP / JOINT / UNIFORM starts) --------
ENABLE_MULTI_START_GLOBAL = False
MULTI_START_INCLUDE_DP = True
MULTI_START_INCLUDE_JOINT = True
MULTI_START_INCLUDE_UNIFORM = True

# -------- Test eval insertion (report-only; NEVER used for schedule decisions) --------
ENABLE_STAGE_TEST_EVAL = True
TEST_FAST_N = 512


# =============================================================================
# Utilities
# =============================================================================

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(40):
        if (cur / "examples").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Cannot find repo root: no 'examples/' directory found upwards.")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pp(x: float) -> str:
    return f"{x:+.2f} pp"


def _isfinite(x: float) -> bool:
    return math.isfinite(x) and (not math.isnan(x))


def _round_key(a: float, b: float, dec: int) -> Tuple[float, float]:
    return (round(float(a), dec), round(float(b), dec))


def expand_range(a: float, b: float, margin_frac: float) -> Tuple[float, float]:
    if margin_frac <= 0.0 or not (_isfinite(a) and _isfinite(b)) or b <= a:
        return a, b
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    half *= (1.0 + margin_frac)
    return (mid - half, mid + half)


def save_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def load_dp_one_layer_error_table_mixed_csv(
    best_path: Path,
    mixed_path: Path,
    sites: List[str],
    degrees: List[int],
) -> tuple[Optional[Dict[str, Dict[int, float]]], Optional[Dict[str, Dict[int, str]]]]:
    """
    Try to reuse previously saved DP one-layer error tables.

    Returns:
      - per_site_error[site][degree] = mse_best
      - per_site_best_scale[site][degree] = {"minmax"|"symmetric"}  (if mixed table exists)
    If any required entry is missing, returns (None, None) to force regeneration.
    """
    sites_set = set(sites)
    deg_set = set(int(d) for d in degrees)

    def _init_error():
        return {s: {} for s in sites}

    # Prefer mixed table because it also stores best_scale
    if mixed_path.exists():
        per_site_error = _init_error()
        per_site_best_scale: Dict[str, Dict[int, str]] = {s: {} for s in sites}
        try:
            with mixed_path.open("r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                cols = set(rdr.fieldnames or [])
                need = {"site", "degree", "mse_best", "best_scale"}
                if not need.issubset(cols):
                    raise ValueError(f"Missing columns in {mixed_path}: need {need}, got {cols}")
                for row in rdr:
                    s = row.get("site", "")
                    if s not in sites_set:
                        continue
                    d = int(row["degree"])
                    if d not in deg_set:
                        continue
                    per_site_error[s][d] = float(row["mse_best"])
                    per_site_best_scale[s][d] = str(row.get("best_scale", "minmax"))
        except Exception:
            return None, None

        for s in sites:
            for d in deg_set:
                if d not in per_site_error[s]:
                    return None, None
                if d not in per_site_best_scale[s]:
                    per_site_best_scale[s][d] = "minmax"
        return per_site_error, per_site_best_scale

    # Fallback: best-only table (no scale info)
    if best_path.exists():
        per_site_error = _init_error()
        per_site_best_scale: Dict[str, Dict[int, str]] = {s: {} for s in sites}
        try:
            with best_path.open("r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                cols = set(rdr.fieldnames or [])
                need = {"site", "degree", "mse"}
                if not need.issubset(cols):
                    raise ValueError(f"Missing columns in {best_path}: need {need}, got {cols}")
                for row in rdr:
                    s = row.get("site", "")
                    if s not in sites_set:
                        continue
                    d = int(row["degree"])
                    if d not in deg_set:
                        continue
                    per_site_error[s][d] = float(row["mse"])
                    per_site_best_scale[s][d] = "minmax"
        except Exception:
            return None, None

        for s in sites:
            for d in deg_set:
                if d not in per_site_error[s]:
                    return None, None
                if d not in per_site_best_scale[s]:
                    per_site_best_scale[s][d] = "minmax"
        return per_site_error, per_site_best_scale

    return None, None


def apply_schedule_and_scaling(
    *,
    model_fx: nn.Module,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    schedule: Dict[str, int],
    ref_bounds: Dict[str, Tuple[float, float]],
    sym_bounds: Dict[str, Tuple[float, float]],
    per_site_best_scale: Dict[str, Dict[int, str]],
    scale_mode_ref: Dict[str, str],
) -> None:
    """Apply schedule + (optional) per-site mixed scaling, then rebuild all SiteActivation coeffs."""
    schedule_ref.update(schedule)
    n_sym = 0
    for s in sites:
        d = int(schedule.get(s, 0))
        mode = str(per_site_best_scale.get(s, {}).get(d, "minmax"))
        scale_mode_ref[s] = mode
        if ENABLE_MIXED_SCALING_IN_DP and (mode == "symmetric"):
            used_bounds_ref[s] = sym_bounds.get(s, ref_bounds.get(s, (-1.0, 1.0)))
            n_sym += 1
        else:
            used_bounds_ref[s] = ref_bounds.get(s, (-1.0, 1.0))
    rebuild_all_site_coeffs(model_fx)
    if ENABLE_MIXED_SCALING_IN_DP:
        print(f"[scale] mixed scaling applied: symmetric_sites={n_sym}/{len(sites)}")


def depth_cost_from_ranges(deg: int) -> int:
    d = int(deg)
    for lo, hi, cost in DEPTH_COST_RANGES:
        if lo <= d <= hi:
            return int(cost)
    raise ValueError(f"degree {d} out of supported range table")


def schedule_cost(schedule: Dict[str, int]) -> int:
    return sum(depth_cost_from_ranges(d) for d in schedule.values())


def build_cost_buckets(cands: List[int]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for d in sorted(set(int(x) for x in cands)):
        buckets.setdefault(depth_cost_from_ranges(d), []).append(d)
    return buckets


def build_uniform_schedule(sites: List[str], deg: int) -> Dict[str, int]:
    d = int(deg)
    return {s: d for s in sites}


def max_uniform_deg_under_budget(sites: List[str], candidates: List[int], budget: int) -> Optional[int]:
    """Return the largest degree in candidates such that uniform schedule fits budget.

    Budget check uses depth_cost_from_ranges(deg) * num_sites <= budget.
    """
    n = len(sites)
    best = None
    for d in sorted({int(x) for x in candidates}):
        c = depth_cost_from_ranges(d)
        if c * n <= budget:
            best = d
    return best


def uniform_candidate_list(sites: List[str], candidates: List[int], budget: int) -> List[int]:
    """Return unique uniform degrees to try (filtered by budget)."""
    degs = []
    if ENABLE_UNIFORM_CANDIDATES:
        for d in UNIFORM_CANDIDATE_DEGS:
            if d is None:
                continue
            d = int(d)
            if depth_cost_from_ranges(d) * len(sites) == budget:
                degs.append(d)
        if AUTO_ADD_MAX_UNIFORM_DEG_UNDER_BUDGET:
            best = max_uniform_deg_under_budget(sites, candidates, budget)
            if best is not None:
                degs.append(int(best))
    # unique + stable order
    out = []
    seen = set()
    for d in degs:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


# =============================================================================
# Dataset support (NO torchvision)
# =============================================================================

# Normalization constants (match torchvision defaults)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name in ("cifar10", "cifar10_bin"): return 10
    if name in ("cifar100", "cifar100_bin"): return 100
    if name == "tinyimagenet": return 200
    if name == "imagenet": return 1000
    raise ValueError(f"Unknown dataset {name}")


# -------- Minimal transform stack (subset of torchvision.transforms) --------
class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    """If size is int: resize shorter side to size (keep aspect ratio), like torchvision."""

    def __init__(self, size: int | Tuple[int, int], resample=Image.BILINEAR):
        self.size = size
        self.resample = resample

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("Resize expects a PIL.Image")
        if isinstance(self.size, int):
            w, h = img.size
            if w == 0 or h == 0:
                return img
            if w < h:
                new_w = self.size
                new_h = int(round(h * (self.size / w)))
            else:
                new_h = self.size
                new_w = int(round(w * (self.size / h)))
            return img.resize((new_w, new_h), resample=self.resample)
        else:
            return img.resize((int(self.size[0]), int(self.size[1])), resample=self.resample)


class CenterCrop:
    def __init__(self, size: int | Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("CenterCrop expects a PIL.Image")
        if isinstance(self.size, int):
            th, tw = self.size, self.size
        else:
            tw, th = int(self.size[0]), int(self.size[1])
        w, h = img.size
        if tw > w or th > h:
            # if requested crop is bigger, just return (or pad if you prefer)
            return img
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return img.crop((j, i, j + tw, i + th))


class ToTensor:
    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, Image.Image):
            arr = np.array(img, dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            raise TypeError("ToTensor expects PIL.Image or np.ndarray")
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # HWC uint8 -> CHW float32 in [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t.to(torch.float32).div_(255.0)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            raise TypeError("Normalize expects a torch.Tensor")
        return (t - self.mean.to(t.device)) / self.std.to(t.device)


def make_transforms(ds: str, split: str) -> Callable:
    ds = ds.lower()
    if ds in ("cifar10", "cifar10_bin"):
        return Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if ds in ("cifar100", "cifar100_bin"):
        return Compose([ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    if ds in ("imagenet", "tinyimagenet"):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(ds)


class CIFARBinaryDataset(Dataset):
    def __init__(self, bin_files: Sequence[Path], is_cifar100: bool, transform: Optional[Callable] = None):
        self.bin_files = [Path(p) for p in bin_files]
        self.is_cifar100 = bool(is_cifar100)
        self.transform = transform
        self.record_len = 3074 if self.is_cifar100 else 3073

        self._raws = []
        self._sizes = []
        for p in self.bin_files:
            raw = np.fromfile(str(p), dtype=np.uint8)
            if raw.size % self.record_len != 0:
                raise ValueError(f"Bad bin size for {p}: {raw.size} not divisible by {self.record_len}")
            n = raw.size // self.record_len
            self._raws.append(raw.reshape(n, self.record_len))
            self._sizes.append(n)
        self._cum = np.cumsum(np.array(self._sizes, dtype=np.int64))

    def __len__(self) -> int:
        return int(self._cum[-1]) if len(self._cum) else 0

    def _map_index(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        file_id = int(np.searchsorted(self._cum, idx, side="right"))
        prev = int(self._cum[file_id - 1]) if file_id > 0 else 0
        return file_id, idx - prev

    def __getitem__(self, idx: int):
        fid, li = self._map_index(idx)
        row = self._raws[fid][li]
        if self.is_cifar100:
            y = int(row[1])
            img = row[2:].reshape(3, 32, 32).transpose(1, 2, 0)
        else:
            y = int(row[0])
            img = row[1:].reshape(3, 32, 32).transpose(1, 2, 0)

        im = Image.fromarray(img)
        if self.transform is not None:
            im = self.transform(im)
        return im, y


class CIFARPickleDataset(Dataset):
    """Read CIFAR-10/100 python-format batches without torchvision.
    Expects extracted dirs:
      - CIFAR-10:  <root>/cifar-10-batches-py/
      - CIFAR-100: <root>/cifar-100-python/
    """

    def __init__(self, root: Path, train: bool, is_cifar100: bool, transform: Optional[Callable] = None):
        self.root = Path(root).expanduser()
        self.train = bool(train)
        self.is_cifar100 = bool(is_cifar100)
        self.transform = transform

        import pickle

        if self.is_cifar100:
            base = self.root / "cifar-100-python"
            fpath = base / ("train" if self.train else "test")
            if not fpath.exists():
                raise FileNotFoundError(f"Missing CIFAR-100 file: {fpath}")
            with open(fpath, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            data = d[b"data"]
            labels = d[b"fine_labels"]
        else:
            base = self.root / "cifar-10-batches-py"
            if self.train:
                files = [base / f"data_batch_{i}" for i in range(1, 6)]
            else:
                files = [base / "test_batch"]
            for fp in files:
                if not fp.exists():
                    raise FileNotFoundError(f"Missing CIFAR-10 batch: {fp}")
            data_list, labels_list = [], []
            for fp in files:
                with open(fp, "rb") as f:
                    d = pickle.load(f, encoding="bytes")
                data_list.append(d[b"data"])
                labels_list.extend(d[b"labels"])
            data = np.concatenate(data_list, axis=0)
            labels = labels_list

        data = np.array(data, dtype=np.uint8)
        self.data = data.reshape(-1, 3, 32, 32)
        self.targets = np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int):
        x = self.data[idx].transpose(1, 2, 0)  # HWC
        y = int(self.targets[idx])
        im = Image.fromarray(x, mode="RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, y


class ImageFolderNoTV(Dataset):
    """A minimal ImageFolder replacement (class subfolders)."""
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"ImageFolder root missing: {self.root}")

        classes = [d.name for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not classes:
            raise FileNotFoundError(f"No class subfolders found under: {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        samples: List[Tuple[Path, int]] = []
        for c in classes:
            cdir = self.root / c
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS:
                    samples.append((p, self.class_to_idx[c]))
        if not samples:
            raise FileNotFoundError(f"No images found under: {self.root}")

        # deterministic order
        samples.sort(key=lambda x: str(x[0]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        im = Image.open(p).convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, int(y)


def build_dataset(name: str, split: str, repo_root: Path) -> Dataset:
    name = name.lower()
    tfm = make_transforms(name, split)

    if name in ("cifar10", "cifar100"):
        root = TORCHVISION_CIFAR_ROOT_REL  # kept name for backward compat
        is_c100 = (name == "cifar100")
        return CIFARPickleDataset(root=root, train=(split != "test"), is_cifar100=is_c100, transform=tfm)

    if name in ("cifar10_bin", "cifar100_bin"):
        is_c100 = (name == "cifar100_bin")
        if name == "cifar10_bin":
            train_bins = [repo_root / p for p in CIFAR10_BIN_TRAIN_RELS]
            test_bin = repo_root / CIFAR10_BIN_TEST_REL
        else:
            train_bins = [repo_root / p for p in CIFAR100_BIN_TRAIN_RELS]
            test_bin = repo_root / CIFAR100_BIN_TEST_REL

        if split == "test":
            if not test_bin.exists():
                raise FileNotFoundError(f"Missing test bin: {test_bin}")
            return CIFARBinaryDataset([test_bin], is_cifar100=is_c100, transform=tfm)

        train_bins = [p for p in train_bins if p.exists()]
        if not train_bins:
            raise FileNotFoundError("No train bins found. Edit CIFAR*_BIN_TRAIN_RELS in config.")
        return CIFARBinaryDataset(train_bins, is_cifar100=is_c100, transform=tfm)

    if name in ("imagenet", "tinyimagenet"):
        root = IMAGENET_ROOT if name == "imagenet" else TINYIMAGENET_ROOT
        if not root.exists():
            root = (repo_root / root).resolve()
        subdir = "val" if split == "test" else "train"
        dirp = root / subdir
        if not dirp.exists():
            raise FileNotFoundError(f"ImageFolder dir missing: {dirp} (expected train/val subfolders)")
        return ImageFolderNoTV(root=dirp, transform=tfm)

    raise ValueError(f"Unknown dataset {name}")


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    k = min(int(k), int(n))
    idx = rng.choice(int(n), size=k, replace=False)
    idx = np.array(idx, dtype=np.int64)
    idx.sort()
    return idx.tolist()


def build_loader(ds: Dataset, indices: Optional[List[int]], batch_size: int, device: torch.device) -> DataLoader:
    subset: Dataset = ds if indices is None else Subset(ds, indices)
    pin = (device.type == "cuda")
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        pin_memory=pin,
    )
    # Prefetching only applies when using workers; it doesn't change order or results.
    if int(NUM_WORKERS) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(AUTO_LOADER_PREFETCH_FACTOR)
    return DataLoader(subset, **kwargs)
def _probe_loader_batch_size(
    ds: Dataset,
    indices: Optional[List[int]],
    *,
    micro_batch: int,
    device: torch.device,
    start_mult: int,
    max_mult: int,
) -> int:
    """Probe a DataLoader batch_size that can be *fetched* (host-side) without OOM/crash.

    Important: This is for loader throughput only. Compute still runs in fixed-size microbatches
    (micro_batch) inside eval loops to keep numeric results identical.
    """
    if (not AUTO_TUNE_LOADER_BATCH) or micro_batch <= 0:
        return int(micro_batch)

    subset: Dataset = ds if indices is None else Subset(ds, indices)
    n = len(subset)
    if n <= 0:
        return int(micro_batch)

    pin = (device.type == "cuda")
    # Start from a capped multiple of micro_batch.
    bs0 = int(min(n, max(1, micro_batch * int(start_mult))))
    bs_cap = int(min(n, max(1, micro_batch * int(max_mult))))
    bs = int(min(bs0, bs_cap))

    # Build a tiny loader and try to fetch a single batch.
    while bs >= micro_batch:
        try:
            kwargs = dict(batch_size=bs, shuffle=False, num_workers=int(NUM_WORKERS), pin_memory=pin)
            if int(NUM_WORKERS) > 0:
                kwargs["persistent_workers"] = True
                kwargs["prefetch_factor"] = int(AUTO_LOADER_PREFETCH_FACTOR)
            loader = DataLoader(subset, **kwargs)
            _ = next(iter(loader))
            return int(bs)
        except Exception:
            bs = bs // 2

    return int(micro_batch)

# =============================================================================
# Model support (auto-infer) incl. CIFAR ResNet Option-A shortcut (FX-friendly)
# =============================================================================

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                sd = {k: v for k, v in obj[key].items() if torch.is_tensor(v)}
                if sd:
                    return sd
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if sd:
            return sd
    raise ValueError("Unrecognized checkpoint format (cannot find tensor state_dict).")


def load_checkpoint(weights_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(weights_path), map_location="cpu")
    return _strip_module_prefix(_extract_state_dict(ckpt))


def infer_num_classes_from_sd(sd: Dict[str, torch.Tensor], fallback: int) -> int:
    if "fc.weight" in sd and sd["fc.weight"].ndim == 2:
        return int(sd["fc.weight"].shape[0])
    for k, v in sd.items():
        if k.endswith("classifier.6.weight") and v.ndim == 2:
            return int(v.shape[0])
    return int(fallback)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, proj_mode: str, proj_name: Optional[str]):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)

        # static ints for FX-friendly Option-A control flow
        self.in_planes = int(in_planes)
        self.planes = int(planes)
        self.stride = int(stride)
        self.proj_mode = str(proj_mode)  # "conv" or "optionA"
        self._proj_name = proj_name

        needs_proj = (stride != 1) or (in_planes != planes)
        if needs_proj and self.proj_mode == "conv" and self._proj_name is not None:
            proj = nn.Sequential(conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes))
            setattr(self, self._proj_name, proj)

    # # for 菠萝权重
    # def _optionA_identity(self, x: torch.Tensor) -> torch.Tensor:
    #     # FX-friendly parameter-free shortcut (CIFAR ResNet Option A)
    #     y = x
    #     st = int(self.stride)
    #     if st != 1:
    #         y = y[:, :, ::st, ::st]
    #     c_in = int(self.in_planes)   # static int
    #     c_out = int(self.planes)
    #     if c_out == c_in:
    #         return y
    #     if c_out > c_in:
    #         n = y.size(0); h = y.size(2); w = y.size(3)
    #         pad = y.new_zeros((n, c_out - c_in, h, w))
    #         return torch.cat([y, pad], dim=1)
    #     return y[:, :c_out, :, :]

    # for fhe-mp-cnn 权重
    def _optionA_identity(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        st = int(self.stride)
        if st != 1:
            y = y[:, :, ::st, ::st]
        c_in = int(self.in_planes)
        c_out = int(self.planes)
        if c_out == c_in:
            return y
        if c_out > c_in:
            n, _, h, w = y.size()
            pad_c = c_out - c_in
            pad_left = pad_c // 2
            pad_right = pad_c - pad_left
            padL = y.new_zeros((n, pad_left, h, w))
            padR = y.new_zeros((n, pad_right, h, w))
            return torch.cat([padL, y, padR], dim=1)
        # (一般 CIFAR ResNet 不会走到这里，但给完整性)
        cut = c_in - c_out
        cut_left = cut // 2
        return y[:, cut_left:cut_left + c_out, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj_mode == "conv" and self._proj_name is not None and hasattr(self, self._proj_name):
            identity = getattr(self, self._proj_name)(identity)
        elif self.proj_mode == "optionA":
            if (self.stride != 1) or (self.in_planes != self.planes):
                identity = self._optionA_identity(identity)

        out = self.relu2(out + identity)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, blocks: Tuple[int, int, int], num_classes: int, proj_mode: str, proj_name: Optional[str]):
        super().__init__()
        self.in_planes = 16
        self.proj_mode = str(proj_mode)
        self.proj_name = proj_name

        self.conv1 = conv3x3(3, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU(inplace=False)

        n1, n2, n3 = blocks
        self.layer1 = self._make_layer(16, n1, stride=1)
        self.layer2 = self._make_layer(32, n2, stride=2)
        self.layer3 = self._make_layer(64, n3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(
                BasicBlockCIFAR(self.in_planes, planes, st, proj_mode=self.proj_mode, proj_name=self.proj_name))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu0(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def infer_proj_name(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    keys = sd.keys()
    if any(".shortcut." in k for k in keys):
        return "shortcut"
    if any(".downsample." in k for k in keys):
        return "downsample"
    return None


def infer_cifar_6n2_blocks(sd: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    pat = re.compile(r"layer([123])\.(\d+)\.conv1\.weight")
    counts = {1: set(), 2: set(), 3: set()}
    for k in sd.keys():
        m = pat.match(k)
        if m:
            li = int(m.group(1));
            bi = int(m.group(2))
            counts[li].add(bi)
    if all(len(counts[i]) > 0 for i in (1, 2, 3)):
        return (max(counts[1]) + 1, max(counts[2]) + 1, max(counts[3]) + 1)
    return None


def build_torchvision_resnet(name: str, num_classes: int, cifar_stem: bool) -> nn.Module:
    """DEPRECATED name kept for backward-compat: builds a ResNet without torchvision."""
    return build_resnet(name=name, num_classes=num_classes, cifar_stem=cifar_stem)


def build_torchvision_vgg(name: str, num_classes: int) -> nn.Module:
    """DEPRECATED name kept for backward-compat: builds a VGG without torchvision."""
    return build_vgg(name=name, num_classes=num_classes)


# ---- ImageNet ResNet (torchvision-compatible module/key structure) ----
def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockIM(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckIM(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetIM(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * getattr(block, "expansion", 1), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckIM):
                    nn.init.constant_(m.bn3.weight, 0.0)
                elif isinstance(m, BasicBlockIM):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * getattr(block, "expansion", 1):
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * getattr(block, "expansion", 1), stride),
                norm_layer(planes * getattr(block, "expansion", 1)),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * getattr(block, "expansion", 1)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_im(num_classes: int) -> nn.Module:
    return ResNetIM(BasicBlockIM, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_im(num_classes: int) -> nn.Module:
    return ResNetIM(BasicBlockIM, [3, 4, 6, 3], num_classes=num_classes)


def resnet50_im(num_classes: int) -> nn.Module:
    return ResNetIM(BottleneckIM, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_im(num_classes: int) -> nn.Module:
    return ResNetIM(BottleneckIM, [3, 4, 23, 3], num_classes=num_classes)


def resnet152_im(num_classes: int) -> nn.Module:
    return ResNetIM(BottleneckIM, [3, 8, 36, 3], num_classes=num_classes)


def build_resnet(name: str, num_classes: int, cifar_stem: bool) -> nn.Module:
    name = name.lower()
    fn_map = {
        "resnet18": resnet18_im,
        "resnet34": resnet34_im,
        "resnet50": resnet50_im,
        "resnet101": resnet101_im,
        "resnet152": resnet152_im,
    }
    if name not in fn_map:
        raise ValueError(f"Unknown resnet family model: {name}")
    m: nn.Module = fn_map[name](num_classes=num_classes)
    if cifar_stem and hasattr(m, "conv1") and isinstance(getattr(m, "conv1"), nn.Conv2d):
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(m, "maxpool"):
            m.maxpool = nn.Identity()
    if hasattr(m, "fc") and isinstance(getattr(m, "fc"), nn.Linear):
        fc: nn.Linear = getattr(m, "fc")
        if fc.out_features != num_classes:
            m.fc = nn.Linear(fc.in_features, num_classes)
    return m


# ---- VGG (torchvision-compatible structure) ----
_VGG_CFGS: Dict[str, List[Any]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_vgg_layers(cfg: List[Any], batch_norm: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGGNoTV(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_vgg(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    bn = name.endswith("_bn")
    base = name.replace("_bn", "")
    if base not in _VGG_CFGS:
        raise ValueError(f"Unknown vgg family model: {name}")
    features = _make_vgg_layers(_VGG_CFGS[base], batch_norm=bn)
    m = VGGNoTV(features, num_classes=num_classes)
    # ensure classifier out matches
    last = m.classifier[-1]
    if isinstance(last, nn.Linear) and last.out_features != num_classes:
        m.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return m


def load_match_score(m: nn.Module, sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        missing, unexpected = m.load_state_dict(sd, strict=False)
        return len(missing) + len(unexpected), 0
    except Exception:
        return 10 ** 9, 1


def build_model_auto(sd: Dict[str, torch.Tensor], dataset_num_classes: int) -> nn.Module:
    inferred_classes = infer_num_classes_from_sd(sd, dataset_num_classes)

    candidates: List[Tuple[str, Callable[[], nn.Module]]] = []

    blocks = infer_cifar_6n2_blocks(sd)
    if blocks is not None:
        proj_name = infer_proj_name(sd)
        proj_mode = "conv" if proj_name is not None else "optionA"
        candidates.append(
            (
                f"resnet_cifar_6n2_{blocks}_proj={proj_name}_mode={proj_mode}",
                lambda blocks=blocks, proj_name=proj_name, proj_mode=proj_mode: ResNetCIFAR(
                    blocks=blocks, num_classes=inferred_classes, proj_mode=proj_mode, proj_name=proj_name
                ),
            )
        )

    for rn in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        candidates.append((f"{rn}_imagenetstem", lambda rn=rn: build_resnet(rn, inferred_classes, cifar_stem=False)))
        candidates.append((f"{rn}_cifarstem", lambda rn=rn: build_resnet(rn, inferred_classes, cifar_stem=True)))

    for vn in ("vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"):
        candidates.append((vn, lambda vn=vn: build_vgg(vn, inferred_classes)))

    best_name, best_model, best_score = None, None, 10 ** 9
    for name, fn in candidates:
        m = fn()
        score, exc = load_match_score(m, sd)
        if exc:
            continue
        if score < best_score:
            best_score = score
            best_name = name
            best_model = m

    if best_model is None:
        raise RuntimeError("Auto model build failed: no candidate architecture could load the checkpoint.")
    print(f"[model-auto] selected: {best_name} | load_score(missing+unexpected)={best_score}")
    return best_model


def build_model_explicit(model_name: str, sd: Dict[str, torch.Tensor], dataset_num_classes: int) -> nn.Module:
    inferred_classes = infer_num_classes_from_sd(sd, dataset_num_classes)
    name = model_name.lower()

    m = re.match(r"resnet(\d+)_cifar", name)
    if m:
        depth = int(m.group(1))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        blocks = (n, n, n)
        proj_name = infer_proj_name(sd)
        proj_mode = "conv" if proj_name is not None else "optionA"
        model = ResNetCIFAR(blocks=blocks, num_classes=inferred_classes, proj_mode=proj_mode, proj_name=proj_name)
        model.load_state_dict(sd, strict=False)
        return model

    if name.startswith("resnet") and name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        # try both stems
        m1 = build_resnet(name, inferred_classes, cifar_stem=False)
        s1, e1 = load_match_score(m1, sd)
        if e1 == 0 and s1 < 2000:
            return m1
        m2 = build_resnet(name, inferred_classes, cifar_stem=True)
        s2, e2 = load_match_score(m2, sd)
        if e2 == 0:
            return m2
        raise RuntimeError(f"Could not load sd into {name} (imagenet/cifar stems tried).")

    if name.startswith("vgg") and name in (
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"):
        m = build_vgg(name, inferred_classes)
        m.load_state_dict(sd, strict=False)
        return m

    raise ValueError(f"Unknown MODEL_NAME={model_name}")


def build_model(model_name: str, sd: Dict[str, torch.Tensor], dataset_num_classes: int) -> nn.Module:
    if model_name.lower() == "auto":
        return build_model_auto(sd, dataset_num_classes)
    return build_model_explicit(model_name, sd, dataset_num_classes)


# =============================================================================
# Chebyshev approximation with coefficient cache
# =============================================================================

_COEFF_CACHE: Dict[Tuple[int, float, float], torch.Tensor] = {}


def eval_chebyshev_coefficients_relu(a: float, b: float, degree: int) -> torch.Tensor:
    degree = int(degree)
    if degree <= 0:
        raise ValueError("degree must be >= 1")
    if not (_isfinite(a) and _isfinite(b)) or (b - a) <= RANGE_EPS:
        return torch.zeros(degree + 1, dtype=torch.float64)

    def relu_scalar(z: float) -> float:
        return 0.0 if z < 0.0 else z

    n = degree + 1
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    pi_by_n = math.pi / n

    fvals = []
    for i in range(n):
        x = math.cos(pi_by_n * (i + 0.5)) * b_minus_a + b_plus_a
        fvals.append(relu_scalar(x))

    mult = 2.0 / n
    coeffs = [0.0] * n
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += fvals[j] * math.cos(pi_by_n * k * (j + 0.5))
        coeffs[k] = s * mult

    return torch.tensor(coeffs, dtype=torch.float64)


def get_coeffs_cached(degree: int, a: float, b: float) -> torch.Tensor:
    deg = int(degree)
    ar, br = _round_key(a, b, COEFF_CACHE_ROUND_DECIMALS)
    key = (deg, ar, br)
    if key in _COEFF_CACHE:
        return _COEFF_CACHE[key]
    coeffs = eval_chebyshev_coefficients_relu(a, b, deg)
    _COEFF_CACHE[key] = coeffs
    return coeffs


def chebyshev_series_clenshaw(t: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    n = coeffs.numel() - 1
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for k in range(n, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0
    return t * b1 - b2 + 0.5 * coeffs[0]


# =============================================================================
# SiteActivation + FX instrumentation
# =============================================================================

class SiteActivation(nn.Module):
    def __init__(
        self,
        site_name: str,
        candidate_degrees: Sequence[int],
        schedule_ref: Dict[str, int],
        used_bounds_ref: Dict[str, Tuple[float, float]],
    ):
        super().__init__()
        self.site_name = str(site_name)
        self.cand = sorted(set(int(d) for d in candidate_degrees))
        self.schedule_ref = schedule_ref
        self.used_bounds_ref = used_bounds_ref

        self._max_len = max(self.cand) + 1

        # Keep coefficient matrices as registered buffers so they move with .to(device).
        # Important: rebuild_coeffs() MUST NOT reassign these buffers (would drop them back to CPU).
        coeff64 = torch.zeros((len(self.cand), self._max_len), dtype=torch.float64)
        self.register_buffer("coeff_mat", coeff64, persistent=True)
        self.register_buffer("coeff_mat_f32", torch.zeros_like(coeff64, dtype=torch.float32), persistent=False)
        self.register_buffer("coeff_mat_f16", torch.zeros_like(coeff64, dtype=torch.float16), persistent=False)

        self.deg2idx = {d: i for i, d in enumerate(self.cand)}
        self.is_site_activation = True

        self.register_buffer("repair_t_cnt", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("repair_y_cnt", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("total_cnt", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("fallback_cnt", torch.zeros((), dtype=torch.int64), persistent=False)

        # 可选：按样本统计
        self.register_buffer("bad_sample_cnt_t", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("bad_sample_cnt_y", torch.zeros((), dtype=torch.int64), persistent=False)

    def reset_repair_stats(self):
        self.repair_t_cnt.zero_()
        self.repair_y_cnt.zero_()
        self.total_cnt.zero_()
        self.fallback_cnt.zero_()
        self.bad_sample_cnt_t.zero_()
        self.bad_sample_cnt_y.zero_()


    def rebuild_coeffs(self) -> None:
        """Recompute Chebyshev coeffs for THIS site using current used_bounds_ref.

        This is called after bounds update / scaling-mode changes. Must be deterministic.
        """
        a, b = self.used_bounds_ref.get(self.site_name, (-1.0, 1.0))
        a = float(a)
        b = float(b)

        mat64 = self.coeff_mat
        mat64.zero_()
        dev = mat64.device

        for i, d in enumerate(self.cand):
            coeffs = get_coeffs_cached(d, a, b)
            if coeffs.device != dev:
                coeffs = coeffs.to(device=dev)
            mat64[i, : coeffs.numel()] = coeffs

        # Cache commonly-used dtypes to avoid per-forward .to() overhead.
        # copy_ will cast if needed.
        self.coeff_mat_f32.copy_(mat64)
        self.coeff_mat_f16.copy_(mat64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deg = int(self.schedule_ref.get(self.site_name, 0))
        if deg <= 0:
            return torch.relu(x)

        a, b = self.used_bounds_ref.get(self.site_name, (-1.0, 1.0))
        a = float(a)
        b = float(b)
        if (not (_isfinite(a) and _isfinite(b))) or (b - a) <= RANGE_EPS:
            return torch.relu(x)

        # Minmax scaling maps [a,b] -> [-1,1]
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        t = (x - mid) / half

        # Targeted repair: branchless (no GPU->CPU sync). Only affects truly bad elements.
        if ENABLE_TARGETED_REPAIR:
            clamp_t = float(REPAIR_T_CLAMP)
            if clamp_t > 0:
                bad_t = (~torch.isfinite(t)) | (t.abs() > clamp_t)
                # 在 forward 里，在你构造 bad_t / bad_y 后加：
                self.total_cnt += t.numel()
                self.repair_t_cnt += bad_t.sum().to(torch.int64)
                # 可选：按样本统计“是否出现过坏点”
                B = t.shape[0]
                self.bad_sample_cnt_t += bad_t.view(B, -1).any(dim=1).sum().to(torch.int64)
                t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                t_safe = torch.clamp(t_safe, -clamp_t, clamp_t)
                t = torch.where(bad_t, t_safe, t)

        idx = self.deg2idx.get(deg, None)
        if idx is None:
            return torch.relu(x)

        # Fast coeff selection without per-forward device transfers.
        if t.dtype == torch.float32:
            coeffs = self.coeff_mat_f32[idx, : deg + 1]
        elif t.dtype == torch.float16:
            coeffs = self.coeff_mat_f16[idx, : deg + 1]
        else:
            coeffs = self.coeff_mat[idx, : deg + 1].to(dtype=t.dtype)

        y = chebyshev_series_clenshaw(t, coeffs)

        if ENABLE_TARGETED_REPAIR:
            clamp_y = float(REPAIR_Y_CLAMP)
            if clamp_y > 0:
                bad_y = (~torch.isfinite(y)) | (y.abs() > clamp_y)
                # 在 forward 里，在你构造 bad_t / bad_y 后加：
                self.total_cnt += t.numel()
                self.fallback_cnt += bad_y.sum().to(torch.int64)  # fallback 就是 bad_y 的子集/等价
                B = y.shape[0]
                self.bad_sample_cnt_y += bad_y.view(B, -1).any(dim=1).sum().to(torch.int64)
                if REPAIR_FALLBACK_TO_RELU:
                    # branchless: compute relu(x) unconditionally to avoid sync
                    y_safe = torch.relu(x)
                else:
                    y_safe = torch.nan_to_num(y, nan=0.0, posinf=clamp_y, neginf=-clamp_y)
                    y_safe = torch.clamp(y_safe, -clamp_y, clamp_y)
                y = torch.where(bad_y, y_safe, y)

        return y


def instrument_relu_sites_fx(model: nn.Module, candidate_degrees: Sequence[int],
                             schedule_ref: Dict[str, int], used_bounds_ref: Dict[str, Tuple[float, float]]) -> Tuple[
    nn.Module, List[str]]:
    from torch.fx import symbolic_trace
    model = copy.deepcopy(model).eval().to("cpu")
    gm = symbolic_trace(model)

    sites: List[str] = []
    site_idx = 0
    g = gm.graph
    nodes = list(g.nodes)

    def sanitize(s: str) -> str:
        s = s.replace(".", "_").replace("/", "_")
        s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
        return s

    def new_site_name(base: str) -> str:
        nonlocal site_idx
        nm = f"actsite_{site_idx:04d}_{sanitize(base)}"
        site_idx += 1
        return nm

    def add_site_module(nm: str) -> None:
        gm.add_module(nm, SiteActivation(nm, candidate_degrees, schedule_ref, used_bounds_ref))

    for node in nodes:
        if node.op == "call_module":
            target_mod = gm.get_submodule(node.target)
            if isinstance(target_mod, (nn.ReLU, nn.ReLU6)):
                nm = new_site_name(node.target)
                add_site_module(nm)
                with g.inserting_after(node):
                    new_node = g.call_module(nm, args=node.args, kwargs={})
                node.replace_all_uses_with(new_node)
                g.erase_node(node)
                sites.append(nm)
        elif node.op == "call_function":
            if node.target in (torch.relu, F.relu):
                nm = new_site_name(getattr(node.target, "__name__", "relu"))
                add_site_module(nm)
                with g.inserting_after(node):
                    new_node = g.call_module(nm, args=node.args, kwargs={})
                node.replace_all_uses_with(new_node)
                g.erase_node(node)
                sites.append(nm)

    g.lint()
    gm.recompile()
    return gm, sites


def list_site_modules(model: nn.Module) -> List[str]:
    out = []
    for nm, m in model.named_modules():
        if hasattr(m, "is_site_activation") and getattr(m, "is_site_activation"):
            out.append(nm)
    out.sort()
    return out


def rebuild_all_site_coeffs(model: nn.Module) -> None:
    for _nm, m in model.named_modules():
        if isinstance(m, SiteActivation):
            m.rebuild_coeffs()


def set_schedule(schedule_ref: Dict[str, int], sites: List[str], value: int) -> None:
    for s in sites:
        schedule_ref[s] = int(value)


def apply_move(schedule: Dict[str, int], move: Dict[str, int]) -> Dict[str, int]:
    out = dict(schedule)
    out.update({k: int(v) for k, v in move.items()})
    return out


def schedule_key(schedule: Dict[str, int], sites: List[str]) -> Tuple[int, ...]:
    return tuple(int(schedule[s]) for s in sites)


# =============================================================================
# Eval with optional stats; LRU cache for MSE-only
# =============================================================================

@dataclass
class RangeStats:
    min_obs: float
    max_obs: float
    abs_max: float
    t_max: float
    oor_rate: float
    nonfinite: bool


@dataclass
class EvalResult:
    logit_mse: float
    acc: float
    stats: Dict[str, RangeStats]
    abs_x_global: float
    t_global: float
    worst_oor: float
    nonfinite_any: bool


class LRUCache:
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self.od: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.od:
            return None
        val = self.od.pop(key)
        self.od[key] = val
        return val

    def put(self, key: Any, val: Any) -> None:
        if key in self.od:
            self.od.pop(key)
        self.od[key] = val
        if len(self.od) > self.max_items:
            self.od.popitem(last=False)


_eval_cache_fast = {
    "fast": LRUCache(EVAL_CACHE_MAX_ITEMS),
    "sweep": LRUCache(EVAL_CACHE_MAX_ITEMS),
    "test": LRUCache(EVAL_CACHE_MAX_ITEMS),
}


def _prepare_hooks_for_range_stats(
    model: torch.nn.Module,
    sites: List[str],
    used_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
):
    """
    Register lightweight forward-pre-hooks that collect per-site activation *input* range statistics.

    Important: This implementation is designed to be **numerically identical** to the previous one,
    while being much faster on GPU by avoiding per-batch `.item()` synchronizations. All reductions
    happen on-device; we convert to Python scalars only once at the end of `eval_model`.
    """
    handles: List[Any] = []

    # Accumulators live on the same device as the model activations to avoid CPU-GPU sync.
    # We keep cnt as a Python int (no sync, cheap).
    acc: Dict[str, Dict[str, Any]] = {}
    for s in sites:
        acc[s] = {
            "min": torch.tensor(float("inf"), device=device, dtype=torch.float32),
            "max": torch.tensor(float("-inf"), device=device, dtype=torch.float32),
            "abs": torch.tensor(0.0, device=device, dtype=torch.float32),
            "tmax": torch.tensor(0.0, device=device, dtype=torch.float32),
            "oor": torch.zeros((), device=device, dtype=torch.int64),
            "cnt": 0,
            "nonfinite": torch.zeros((), device=device, dtype=torch.bool),
        }

    module_by_name = dict(model.named_modules())

    def hook(name: str):
        a, b = used_bounds[name]
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a) if (b - a) > RANGE_EPS else 1.0

        def _fn(mod, inputs):
            # NOTE: inputs is a tuple; we track the activation *input* (pre-ReLU).
            x = inputs[0].detach()

            # Track non-finite without synchronizing to CPU.
            finite = torch.isfinite(x).all()
            acc[name]["nonfinite"] = acc[name]["nonfinite"] | (~finite)

            # Match previous behavior: if any non-finite, replace with zeros before range stats.
            # We do it unconditionally after recording 'finite' to avoid a Python branch + sync.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-batch stats (all on-device scalars)
            mn = x.amin()
            mx = x.amax()
            absmax = x.abs().amax()

            # Normalize to unit domain and track the maximum |t|.
            tmax = ((x - mid) / half).abs().amax()

            # Out-of-range count relative to *used* bounds.
            oor = ((x < a) | (x > b)).sum(dtype=torch.int64)

            # Update accumulators (still on-device; no `.item()` syncs).
            acc[name]["min"] = torch.minimum(acc[name]["min"], mn)
            acc[name]["max"] = torch.maximum(acc[name]["max"], mx)
            acc[name]["abs"] = torch.maximum(acc[name]["abs"], absmax)
            acc[name]["tmax"] = torch.maximum(acc[name]["tmax"], tmax)
            acc[name]["oor"] = acc[name]["oor"] + oor
            acc[name]["cnt"] += x.numel()

        return _fn

    for name in sites:
        if name not in module_by_name:
            raise KeyError(f"Site module not found: {name}")
        handles.append(module_by_name[name].register_forward_pre_hook(hook(name)))

    return handles, acc


@torch.inference_mode()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    used_bounds: Dict[str, Tuple[float, float]],
    baseline_logits_cpu: Optional[torch.Tensor] = None,
    collect_stats: bool = True,
) -> EvalResult:
    model.eval()
    handles, acc = ([], {}) if not collect_stats else _prepare_hooks_for_range_stats(model, sites, used_bounds, device)

    mse_sum = 0.0
    n_total = 0
    correct_t = torch.zeros((), device=device, dtype=torch.int64)
    offset = 0
    nonfinite_any_t = torch.zeros((), device=device, dtype=torch.bool)

    micro_bs = int(BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU)

    for xb_big, yb_big in loader:
        # Keep sample order identical; split into fixed-size microbatches to preserve numeric reductions.
        big_bs = int(yb_big.numel())
        for j in range(0, big_bs, micro_bs):
            xb = xb_big[j:j + micro_bs].to(device, non_blocking=True)
            yb = yb_big[j:j + micro_bs].to(device, non_blocking=True)

        logits = model(xb).detach().float()
        nonfinite_any_t |= (~torch.isfinite(logits)).any()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        pred = logits.argmax(dim=1)
        correct_t += (pred == yb).sum()
        bs = int(yb.numel())
        n_total += bs

        if baseline_logits_cpu is not None:
            base_src = baseline_logits_cpu[offset:offset + bs]
            if base_src.device == device and base_src.dtype == logits.dtype:
                base = base_src
            else:
                base = base_src.to(device=device, dtype=logits.dtype, non_blocking=True)
            diff = logits - base
            # Robust objective (optional): aggregate per-sample MSE to tolerate rare extreme outliers.
            if USE_ROBUST_MSE:
                per = (diff * diff).mean(dim=1)  # (B,)
                if ROBUST_MSE_MODE == "trim":
                    top_frac = float(ROBUST_MSE_TRIM_TOP_FRAC)
                    if top_frac > 0.0 and per.numel() >= 8:
                        k = int(math.floor(per.numel() * top_frac))
                        if k > 0:
                            per_sorted, _ = torch.sort(per)
                            per = per_sorted[: max(1, per_sorted.numel() - k)]
                else:
                    q = float(ROBUST_MSE_CAP_Q)
                    q = min(max(q, 0.0), 1.0)
                    if per.numel() >= 8 and q < 1.0:
                        cap = torch.quantile(per, q)
                        per = torch.minimum(per, cap)
                mse_sum += float(per.mean().item()) * bs
            else:
                mse_sum += float((diff * diff).mean().item()) * bs
            offset += bs

    for h in handles:
        h.remove()

    stats: Dict[str, RangeStats] = {}
    abs_x_global = 0.0
    t_global = 0.0
    worst_oor = 0.0
    nonfinite_any_stats = False
    if collect_stats:
        for s in sites:
            d = acc[s]
            cnt = float(d["cnt"]) if d["cnt"] > 0 else 1.0
            oor_rate = float(d["oor"]) / cnt
            st = RangeStats(
                min_obs=float(d["min"]),
                max_obs=float(d["max"]),
                abs_max=float(d["abs"]),
                t_max=float(d["tmax"]),
                oor_rate=oor_rate,
                nonfinite=bool(d["nonfinite"]),
            )
            stats[s] = st
            abs_x_global = max(abs_x_global, st.abs_max)
            t_global = max(t_global, st.t_max)
            worst_oor = max(worst_oor, st.oor_rate)
            nonfinite_any_stats = nonfinite_any_stats or st.nonfinite

    logit_mse = (mse_sum / max(1, n_total)) if baseline_logits_cpu is not None else float("nan")
    correct = int(correct_t.item())
    accv = correct / max(1, n_total)
    nonfinite_any = bool(nonfinite_any_t.item()) or bool(nonfinite_any_stats)
    return EvalResult(
        logit_mse=float(logit_mse),
        acc=float(accv),
        stats=stats,
        abs_x_global=float(abs_x_global),
        t_global=float(t_global),
        worst_oor=float(worst_oor),
        nonfinite_any=bool(nonfinite_any),
    )


@torch.no_grad()
def eval_mse_cached(
    tag: str,
    schedule: Dict[str, int],
    sites: List[str],
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    used_bounds: Dict[str, Tuple[float, float]],
    baseline_logits_cpu: torch.Tensor,
) -> float:
    """MSE-only eval with optional LRU cache (no hooks)."""
    if not ENABLE_EVAL_CACHE:
        ev = eval_model(model, loader, device, sites, used_bounds, baseline_logits_cpu=baseline_logits_cpu,
                        collect_stats=False)
        return float(ev.logit_mse)

    cache = _eval_cache_fast.get(tag, None)
    if cache is None:
        cache = LRUCache(EVAL_CACHE_MAX_ITEMS)
        _eval_cache_fast[tag] = cache

    key = schedule_key(schedule, sites)
    got = cache.get(key)
    if got is not None:
        return float(got)

    ev = eval_model(model, loader, device, sites, used_bounds, baseline_logits_cpu=baseline_logits_cpu,
                    collect_stats=False)
    cache.put(key, float(ev.logit_mse))
    return float(ev.logit_mse)


# =============================================================================
# Bounds collection
# =============================================================================

@torch.no_grad()
def collect_bounds_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    mode: str,
    margin_frac: float,
    q_low: float = 0.001,
    q_high: float = 0.999,
    samples_per_batch: int = 2048,
    max_stored: int = 200000,
) -> Dict[str, Tuple[float, float]]:
    mode = mode.lower()
    if mode not in ("minmax", "quantile"):
        raise ValueError(mode)

    site_set = set(sites)
    vmin = {s: float("inf") for s in sites}
    vmax = {s: float("-inf") for s in sites}
    samples: Dict[str, List[torch.Tensor]] = {s: [] for s in sites}
    handles = []

    def make_hook_minmax(name: str):
        def hook(_mod, inputs):
            x = inputs[0].detach()
            mn = float(x.amin().item())
            mx = float(x.amax().item())
            if mn < vmin[name]:
                vmin[name] = mn
            if mx > vmax[name]:
                vmax[name] = mx

        return hook

    def make_hook_quant(name: str):
        def hook(_mod, inputs):
            x = inputs[0].detach()
            mn = float(x.amin().item())
            mx = float(x.amax().item())
            if mn < vmin[name]:
                vmin[name] = mn
            if mx > vmax[name]:
                vmax[name] = mx
            flat = x.reshape(-1)
            n = flat.numel()
            k = min(samples_per_batch, n)
            if n <= k:
                samp = flat
            else:
                idx = torch.randint(0, n, (k,), device=flat.device)
                samp = flat[idx]
            samples[name].append(samp.to("cpu", dtype=torch.float32))

        return hook

    for nm, m in model.named_modules():
        if nm in site_set:
            if mode == "minmax":
                handles.append(m.register_forward_pre_hook(make_hook_minmax(nm)))
            else:
                handles.append(m.register_forward_pre_hook(make_hook_quant(nm)))

    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)

    for h in handles:
        h.remove()

    bounds: Dict[str, Tuple[float, float]] = {}
    for s in sites:
        if mode == "minmax":
            a, b = float(vmin[s]), float(vmax[s])
        else:
            if not samples[s]:
                a, b = float(vmin[s]), float(vmax[s])
            else:
                cat = torch.cat(samples[s], dim=0)
                if max_stored > 0 and cat.numel() > max_stored:
                    idx = torch.randperm(cat.numel())[:max_stored]
                    cat = cat[idx]
                a = float(torch.quantile(cat, q_low).item())
                b = float(torch.quantile(cat, q_high).item())
                if not (_isfinite(a) and _isfinite(b)) or b <= a:
                    a, b = float(vmin[s]), float(vmax[s])
        a, b = expand_range(a, b, margin_frac)
        bounds[s] = (a, b)
    return bounds


def make_symmetric_bounds(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Derive symmetric bounds [-m, +m] from existing bounds [a,b].

    m = max(|a|,|b|). This is used to evaluate symmetric scaling as an alternative to minmax.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for s, (a, b) in bounds.items():
        a = float(a);
        b = float(b)
        m = max(abs(a), abs(b))
        if not _isfinite(m) or m <= 0.0:
            m = 1.0
        out[s] = (-m, +m)
    return out


# =============================================================================
# Penalized objective & guard
# =============================================================================

@dataclass
class PenaltyResult:
    obj: float
    base: float
    p_abs: float
    p_t: float
    p_oor: float
    ok: bool
    worst_abs_site: str
    worst_t_site: str
    worst_oor_site: str


def penalize_and_guard(ev: EvalResult, ref_bounds: Dict[str, Tuple[float, float]]) -> PenaltyResult:
    if not ev.stats:
        return PenaltyResult(
            obj=float(ev.logit_mse),
            base=float(ev.logit_mse),
            p_abs=0.0, p_t=0.0, p_oor=0.0,
            ok=not ev.nonfinite_any,
            worst_abs_site="", worst_t_site="", worst_oor_site=""
        )

    p_abs = 0.0
    p_t = 0.0
    p_oor = 0.0
    ok = (not ev.nonfinite_any)

    worst_abs_site = ""
    worst_t_site = ""
    worst_oor_site = ""

    worst_abs_ratio = -1.0
    worst_t_val = -1.0
    worst_oor_val = -1.0

    for s, st in ev.stats.items():
        a_ref, b_ref = ref_bounds.get(s, (-1.0, 1.0))
        abs_ref = max(abs(a_ref), abs(b_ref), 1e-6)

        soft_abs = ABS_MULT_SOFT * abs_ref
        hard_abs = ABS_MULT_HARD * abs_ref
        if st.abs_max > hard_abs:
            ok = False
        if st.abs_max > soft_abs:
            p_abs += ((st.abs_max - soft_abs) / (soft_abs + 1e-12)) ** 2

        if st.t_max > T_MAX_HARD:
            ok = False
        if st.t_max > T_MAX_SOFT:
            p_t += ((st.t_max - T_MAX_SOFT) / (T_MAX_SOFT + 1e-12)) ** 2

        if st.oor_rate > OOR_HARD:
            ok = False
        if st.oor_rate > OOR_SOFT:
            p_oor += ((st.oor_rate - OOR_SOFT) / (OOR_SOFT + 1e-12)) ** 2

        abs_ratio = st.abs_max / abs_ref
        if abs_ratio > worst_abs_ratio:
            worst_abs_ratio = abs_ratio
            worst_abs_site = s
        if st.t_max > worst_t_val:
            worst_t_val = st.t_max
            worst_t_site = s
        if st.oor_rate > worst_oor_val:
            worst_oor_val = st.oor_rate
            worst_oor_site = s

    base = float(ev.logit_mse)
    obj = base + LAMBDA_ABS * p_abs + LAMBDA_T * p_t + LAMBDA_OOR * p_oor
    return PenaltyResult(
        obj=float(obj), base=float(base),
        p_abs=float(p_abs), p_t=float(p_t), p_oor=float(p_oor),
        ok=bool(ok),
        worst_abs_site=worst_abs_site, worst_t_site=worst_t_site, worst_oor_site=worst_oor_site
    )


# =============================================================================
# Relaxed FULL-guard helpers (sample-level rates)  [PLAINTEXT ONLY]
# =============================================================================

@torch.no_grad()
def sample_guard_rates_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    used_bounds: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    *,
    count_oor_any: bool = True,
    include_oor_in_explode: bool = False,
) -> Dict[str, float]:
    """
    Compute sample-level rates (no CSV, no per-site details).

    Rates are measured over the given loader (typically CALIB):
      - explode_rate: fraction of samples that have ANY of:
            nonfinite OR abs_hard OR t_hard OR (optional) oor_any
      - nonfinite_rate
      - oor_any_rate  (if enabled)

    IMPORTANT:
      - This is a *plaintext* diagnostic/guard helper. It does not affect HE meaning.
      - Implementation keeps the per-sample flags on the same device as the forward pass
        to avoid GPU->CPU async copy quirks (previous versions could miscount).
    """
    model.eval()

    named = dict(model.named_modules())
    modules: List[nn.Module] = []
    active_sites: List[str] = []
    for s in sites:
        m = named.get(s, None)
        if m is None:
            continue
        if hasattr(m, "is_site_activation") and getattr(m, "is_site_activation"):
            modules.append(m)
            active_sites.append(s)

    n_total = int(len(loader.dataset))
    if n_total <= 0:
        return {"n": 0.0, "explode_rate": 0.0, "nonfinite_rate": 0.0, "oor_any_rate": 0.0}

    # Keep flags on device for correctness + speed.
    dev = device
    explode_any = torch.zeros((n_total,), dtype=torch.bool, device=dev)
    nonfinite_any = torch.zeros((n_total,), dtype=torch.bool, device=dev)
    oor_any = torch.zeros((n_total,), dtype=torch.bool, device=dev)

    state: Dict[str, Any] = {"sl": slice(0, 0)}
    hooks: List[Any] = []

    def make_pre_hook(site_name: str):
        ua, ub = used_bounds.get(site_name, (-1.0, 1.0))
        ua = float(ua)
        ub = float(ub)
        mid = 0.5 * (ua + ub)
        half = 0.5 * (ub - ua)

        ra, rb = ref_bounds.get(site_name, (ua, ub))
        ra = float(ra)
        rb = float(rb)
        abs_ref = max(abs(ra), abs(rb), 1e-6)
        abs_thr = float(ABS_MULT_HARD) * abs_ref

        def _hook(_mod: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach()
            if x.ndim == 0:
                return
            B = int(x.shape[0])
            if B <= 0:
                return

            sl = state["sl"]

            # Flatten for per-sample reductions
            xf = x.flatten(1)

            # nonfinite per sample
            nf = ~torch.isfinite(xf).all(dim=1)

            # Avoid NaN propagation in reductions (only when needed)
            if nf.any():
                x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                xf = x_safe.flatten(1)

            # abs hard
            abs_h = xf.abs().amax(dim=1) > abs_thr

            # t hard
            if half <= RANGE_EPS:
                th = torch.zeros((B,), dtype=torch.bool, device=xf.device)
            else:
                tmax = ((xf - mid).abs() / half).amax(dim=1)
                th = tmax > float(T_MAX_HARD)

            # oor_any
            if count_oor_any and _isfinite(ua) and _isfinite(ub) and (ub - ua) > RANGE_EPS:
                mn = xf.amin(dim=1)
                mx = xf.amax(dim=1)
                oor = (mn < ua) | (mx > ub)
            else:
                oor = torch.zeros((B,), dtype=torch.bool, device=xf.device)

            exp = nf | abs_h | th
            if include_oor_in_explode:
                exp = exp | oor

            # Ensure device match (xf.device should equal dev, but be safe)
            exp = exp.to(dev)
            nf = nf.to(dev)
            oor = oor.to(dev)

            explode_any[sl] |= exp
            nonfinite_any[sl] |= nf
            oor_any[sl] |= oor

        return _hook

    for s, m in zip(active_sites, modules):
        hooks.append(m.register_forward_pre_hook(make_pre_hook(s)))

    seen = 0
    try:
        for batch in loader:
            xb = batch[0]
            xb = xb.to(device, non_blocking=True)
            B = int(xb.shape[0])
            state["sl"] = slice(seen, min(seen + B, n_total))
            _ = model(xb)
            seen += B
            if seen >= n_total:
                break
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    n = int(min(seen, n_total))
    if n <= 0:
        return {"n": 0.0, "explode_rate": 0.0, "nonfinite_rate": 0.0, "oor_any_rate": 0.0}

    explode_rate = float(explode_any[:n].float().mean().item())
    nonfinite_rate = float(nonfinite_any[:n].float().mean().item())
    oor_any_rate = float(oor_any[:n].float().mean().item())

    return {
        "n": float(n),
        "explode_rate": explode_rate,
        "nonfinite_rate": nonfinite_rate,
        "oor_any_rate": oor_any_rate,
    }


def full_guard_check(
    *,
    pen_full: PenaltyResult,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    used_bounds: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
) -> Tuple[bool, Optional[Dict[str, float]]]:
    """
    Decide whether FULL-guard is acceptable, with the intended behavior:

      - If strict FULL-guard passes: ACCEPT.
        (Optionally compute sample-guard stats for logging; only enforce them if
         ENFORCE_SAMPLE_GUARD_ON_ACCEPT=True.)

      - If strict FULL-guard fails: may still ACCEPT if ALLOW_RARE_OUTLIER_EXPLOSIONS=True
        AND sample-level rates are below thresholds.

    This avoids the regression where strict-ok schedules were rejected due to
    sample-guard miscounts or overly strict enforcement.
    """
    sg: Optional[Dict[str, float]] = None

    need_sg = ENFORCE_SAMPLE_GUARD_ON_ACCEPT or (ALLOW_RARE_OUTLIER_EXPLOSIONS and (not pen_full.ok))
    if need_sg:
        sg = sample_guard_rates_only(
            model=model,
            loader=loader,
            device=device,
            sites=sites,
            used_bounds=used_bounds,
            ref_bounds=ref_bounds,
            count_oor_any=SAMPLE_GUARD_COUNT_OOR_ANY,
            include_oor_in_explode=SAMPLE_GUARD_INCLUDE_OOR_IN_EXPLODE,
        )
        sg_ok = (
            (sg["explode_rate"] <= float(MAX_CALIB_EXPLODE_RATE)) and
            (sg["nonfinite_rate"] <= float(MAX_CALIB_NONFINITE_RATE)) and
            (sg["oor_any_rate"] <= float(MAX_CALIB_OOR_ANY_RATE))
        )
    else:
        sg_ok = True

    # Strict pass: accept (unless explicitly enforcing sample-guard)
    if pen_full.ok:
        if ENFORCE_SAMPLE_GUARD_ON_ACCEPT and (not sg_ok):
            return False, sg
        return True, sg

    # Strict fail: allow rare outliers if sample-guard deems it ok
    if (not pen_full.ok) and ALLOW_RARE_OUTLIER_EXPLOSIONS and sg_ok:
        return True, sg

    return False, sg


# =============================================================================
# FULL-guard sample-level explosion diagnostics (PLAINTEXT ONLY)
# =============================================================================

@torch.no_grad()
def sample_guard_explosion_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    used_bounds: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    *,
    tag: str,
    out_dir: Optional[Path],
    max_print: int = 20,
    save_csv_flag: bool = True,
    count_oor_any: bool = True,
    include_oor_in_explode: bool = False,
) -> Dict[str, Any]:
    '''
    Run a *plaintext-only* diagnostic that answers:
      - How many samples "explode" under current schedule? (hard abs/t/nonfinite; optional oor_any)
      - Is the explosion concentrated on a few images or widespread?
      - Which site is the *first* explosion site per sample (forward order)?
      - What are the top offending samples (by max abs_ratio or max t)?

    NOTE:
      - This is a diagnostic utility; it has no HE meaning.
      - Thresholds:
          abs_hard: |x| > ABS_MULT_HARD * max(|ref_a|,|ref_b|)
          t_hard  : max(|(x-mid)/half|) > T_MAX_HARD   (mid/half from used_bounds)
          nonfinite: any NaN/Inf in activation input
          oor_any : min(x)<a or max(x)>b  (a,b from used_bounds) [optional]
    '''
    model.eval()

    # Resolve dataset indices (useful for debugging which images trigger blow-ups)
    ds = loader.dataset
    n = int(len(ds))
    subset_indices: Optional[List[int]] = None
    base_ds: Optional[Dataset] = None
    if isinstance(ds, Subset):
        subset_indices = list(ds.indices)  # type: ignore[attr-defined]
        base_ds = ds.dataset  # type: ignore[assignment]
    else:
        base_ds = ds

    def _orig_index(i: int) -> int:
        if subset_indices is None:
            return i
        return int(subset_indices[i])

    def _maybe_path(i: int) -> str:
        # For ImageFolder-like datasets
        if base_ds is None:
            return ""
        oi = _orig_index(i)
        if hasattr(base_ds, "samples"):
            try:
                return str(base_ds.samples[oi][0])  # type: ignore[index]
            except Exception:
                return ""
        return ""

    # Per-sample arrays (kept on `device` for in-hook updates; moved to CPU at the end)
    first_expl_site = torch.full((n,), -1, dtype=torch.int32, device=device)
    expl_any = torch.zeros((n,), dtype=torch.bool, device=device)
    abs_hard_any = torch.zeros((n,), dtype=torch.bool, device=device)
    t_hard_any = torch.zeros((n,), dtype=torch.bool, device=device)
    nonfinite_any = torch.zeros((n,), dtype=torch.bool, device=device)
    oor_any = torch.zeros((n,), dtype=torch.bool, device=device)

    max_abs_ratio = torch.zeros((n,), dtype=torch.float32, device=device)
    max_abs_val = torch.zeros((n,), dtype=torch.float32, device=device)
    max_abs_site = torch.full((n,), -1, dtype=torch.int32, device=device)

    max_t_val = torch.zeros((n,), dtype=torch.float32, device=device)
    max_t_site = torch.full((n,), -1, dtype=torch.int32, device=device)

    labels_cpu = torch.full((n,), -1, dtype=torch.int64)

    # Per-site counters (counts of images where *this* site triggers condition; not necessarily "first")
    site_counts: Dict[str, Dict[str, int]] = {
        s: {"explode": 0, "abs_hard": 0, "t_hard": 0, "nonfinite": 0, "oor_any": 0} for s in sites
    }

    # Precompute site parameters for speed
    site_to_idx = {s: i for i, s in enumerate(sites)}
    params: Dict[str, Dict[str, float]] = {}
    for s in sites:
        a, b = used_bounds[s]
        ra, rb = ref_bounds[s]
        abs_ref = max(abs(float(ra)), abs(float(rb)), 1e-6)
        hard_abs = float(ABS_MULT_HARD) * abs_ref
        mid = 0.5 * (float(a) + float(b))
        half = 0.5 * (float(b) - float(a))
        if (not _isfinite(half)) or half < RANGE_EPS:
            half = float(RANGE_EPS)
        params[s] = {
            "a": float(a), "b": float(b),
            "mid": float(mid), "half": float(half),
            "abs_ref": float(abs_ref), "hard_abs": float(hard_abs),
        }

    # Batch offset state shared between main loop and hooks
    state = {"offset": 0}

    hooks = []
    for name, mod in model.named_modules():
        if name not in site_to_idx:
            continue
        p = params[name]
        site_idx = int(site_to_idx[name])
        a = p["a"];
        b = p["b"]
        mid = p["mid"];
        half = p["half"]
        abs_ref = p["abs_ref"];
        hard_abs = p["hard_abs"]

        def _make_hook(_site_name: str, _site_idx: int, _a: float, _b: float, _mid: float, _half: float,
                       _abs_ref: float, _hard_abs: float):
            def _hook(_m: nn.Module, _inp: Tuple[torch.Tensor, ...]) -> None:
                x = _inp[0].detach()
                if x.ndim == 0:
                    return
                if x.shape[0] == 0:
                    return
                B = int(x.shape[0])
                off = int(state["offset"])
                sl = slice(off, off + B)

                # nonfinite (fast path)
                if not torch.isfinite(x).all():
                    fin = torch.isfinite(x).flatten(1).all(dim=1)
                    nonfin = ~fin
                    # avoid NaN propagation in later reductions
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    nonfin = torch.zeros((B,), dtype=torch.bool, device=x.device)

                xf = x.flatten(1)
                absmax = xf.abs().amax(dim=1)
                abs_ratio = absmax / float(_abs_ref)

                # Track per-sample max abs ratio and where it occurs
                cur = max_abs_ratio[sl]
                upd = abs_ratio > cur
                if upd.any():
                    max_abs_ratio[sl] = torch.where(upd, abs_ratio, cur)
                    max_abs_val[sl] = torch.where(upd, absmax, max_abs_val[sl])
                    max_abs_site[sl] = torch.where(upd, torch.full_like(max_abs_site[sl], int(_site_idx)),
                                                   max_abs_site[sl])

                # tmax per sample
                tmax = ((xf - float(_mid)).abs() / float(_half)).amax(dim=1)
                cur_t = max_t_val[sl]
                upd_t = tmax > cur_t
                if upd_t.any():
                    max_t_val[sl] = torch.where(upd_t, tmax, cur_t)
                    max_t_site[sl] = torch.where(upd_t, torch.full_like(max_t_site[sl], int(_site_idx)), max_t_site[sl])

                # Conditions
                abs_hard = absmax > float(_hard_abs)
                t_hard = tmax > float(T_MAX_HARD)

                if count_oor_any:
                    mn = xf.amin(dim=1)
                    mx = xf.amax(dim=1)
                    oor = (mn < float(_a)) | (mx > float(_b))
                else:
                    oor = torch.zeros((B,), dtype=torch.bool, device=x.device)

                expl = nonfin | abs_hard | t_hard
                if include_oor_in_explode:
                    expl = expl | oor

                # Update global per-sample flags
                if nonfin.any():
                    nonfinite_any[sl] |= nonfin
                if abs_hard.any():
                    abs_hard_any[sl] |= abs_hard
                if t_hard.any():
                    t_hard_any[sl] |= t_hard
                if oor.any():
                    oor_any[sl] |= oor
                if expl.any():
                    expl_any[sl] |= expl
                    # First exploding site (forward order)
                    cur_first = first_expl_site[sl]
                    upd_first = expl & (cur_first < 0)
                    if upd_first.any():
                        first_expl_site[sl] = torch.where(upd_first, torch.full_like(cur_first, int(_site_idx)),
                                                          cur_first)

                # Per-site counters (sync scalars only when needed)
                if expl.any():
                    site_counts[_site_name]["explode"] += int(expl.sum().item())
                if abs_hard.any():
                    site_counts[_site_name]["abs_hard"] += int(abs_hard.sum().item())
                if t_hard.any():
                    site_counts[_site_name]["t_hard"] += int(t_hard.sum().item())
                if nonfin.any():
                    site_counts[_site_name]["nonfinite"] += int(nonfin.sum().item())
                if oor.any():
                    site_counts[_site_name]["oor_any"] += int(oor.sum().item())

            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name, site_idx, a, b, mid, half, abs_ref, hard_abs)))

    # Run forward pass over loader
    offset = 0
    for xb, yb in loader:
        B = int(xb.shape[0])
        if B == 0:
            continue
        labels_cpu[offset:offset + B] = yb.to(torch.int64).cpu()
        state["offset"] = offset
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        offset += B
        if offset >= n:
            break

    for h in hooks:
        h.remove()

    # Move arrays to CPU for analysis
    expl_cpu = expl_any.detach().cpu()
    abs_cpu = abs_hard_any.detach().cpu()
    t_cpu = t_hard_any.detach().cpu()
    nonfin_cpu = nonfinite_any.detach().cpu()
    oor_cpu = oor_any.detach().cpu()
    first_cpu = first_expl_site.detach().cpu()
    max_abs_ratio_cpu = max_abs_ratio.detach().cpu()
    max_abs_val_cpu = max_abs_val.detach().cpu()
    max_abs_site_cpu = max_abs_site.detach().cpu()
    max_t_val_cpu = max_t_val.detach().cpu()
    max_t_site_cpu = max_t_site.detach().cpu()

    explode_n = int(expl_cpu.sum().item())
    abs_n = int(abs_cpu.sum().item())
    t_n = int(t_cpu.sum().item())
    nonfin_n = int(nonfin_cpu.sum().item())
    oor_n = int(oor_cpu.sum().item())
    explode_rate = float(explode_n) / float(max(1, n))
    oor_rate = float(oor_n) / float(max(1, n))

    # Histogram of first exploding site
    first_mask = first_cpu >= 0
    hist = torch.zeros((len(sites),), dtype=torch.int64)
    if int(first_mask.sum().item()) > 0:
        hist = torch.bincount(first_cpu[first_mask].to(torch.int64), minlength=len(sites))
    first_hist_items: List[Tuple[str, int, float]] = []
    for i, c in enumerate(hist.tolist()):
        if c <= 0:
            continue
        first_hist_items.append((sites[i], int(c), float(c) / float(max(1, explode_n))))
    first_hist_items.sort(key=lambda x: x[1], reverse=True)

    # Top offenders by abs ratio / t value
    offenders_abs: List[Dict[str, Any]] = []
    offenders_t: List[Dict[str, Any]] = []

    def _topk_indices(mask: torch.Tensor, score: torch.Tensor, k: int) -> List[int]:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return []
        sc = score[idx]
        kk = min(int(k), int(idx.numel()))
        _topv, topi = torch.topk(sc, k=kk, largest=True, sorted=True)
        sel = idx[topi]
        return [int(x) for x in sel.tolist()]

    top_abs_idx = _topk_indices(abs_cpu, max_abs_ratio_cpu, max_print)
    top_t_idx = _topk_indices(t_cpu, max_t_val_cpu, max_print)

    def _site_name_from_idx(si: int) -> str:
        if si < 0 or si >= len(sites):
            return ""
        return sites[int(si)]

    for i in top_abs_idx:
        sidx = int(max_abs_site_cpu[i].item())
        sname = _site_name_from_idx(sidx)
        abs_ref = 0.0
        if sname and sname in ref_bounds:
            ra, rb = ref_bounds[sname]
            abs_ref = max(abs(float(ra)), abs(float(rb)), 1e-6)
        offenders_abs.append({
            "sample_i": i,
            "orig_i": _orig_index(i),
            "label": int(labels_cpu[i].item()),
            "path": _maybe_path(i),
            "first_expl_site": _site_name_from_idx(int(first_cpu[i].item())),
            "max_abs_site": sname,
            "max_abs_ratio": float(max_abs_ratio_cpu[i].item()),
            "max_abs": float(max_abs_val_cpu[i].item()),
            "abs_ref": float(abs_ref),
            "max_t_site": _site_name_from_idx(int(max_t_site_cpu[i].item())),
            "max_t": float(max_t_val_cpu[i].item()),
            "abs_hard": bool(abs_cpu[i].item()),
            "t_hard": bool(t_cpu[i].item()),
            "nonfinite": bool(nonfin_cpu[i].item()),
            "oor_any": bool(oor_cpu[i].item()),
        })

    for i in top_t_idx:
        offenders_t.append({
            "sample_i": i,
            "orig_i": _orig_index(i),
            "label": int(labels_cpu[i].item()),
            "path": _maybe_path(i),
            "first_expl_site": _site_name_from_idx(int(first_cpu[i].item())),
            "max_t_site": _site_name_from_idx(int(max_t_site_cpu[i].item())),
            "max_t": float(max_t_val_cpu[i].item()),
            "max_abs_site": _site_name_from_idx(int(max_abs_site_cpu[i].item())),
            "max_abs_ratio": float(max_abs_ratio_cpu[i].item()),
            "max_abs": float(max_abs_val_cpu[i].item()),
            "abs_hard": bool(abs_cpu[i].item()),
            "t_hard": bool(t_cpu[i].item()),
            "nonfinite": bool(nonfin_cpu[i].item()),
            "oor_any": bool(oor_cpu[i].item()),
        })

    # Print report
    print("\n" + "=" * 118)
    print(f"SAMPLE GUARD REPORT: {tag}")
    print("=" * 118)
    print(f"[sample-guard] N={n}  explode={explode_n} ({explode_rate * 100:.3f}%)  "
          f"abs_hard={abs_n}  t_hard={t_n}  nonfinite={nonfin_n}  "
          f"oor_any={oor_n} ({oor_rate * 100:.3f}%)  "
          f"(oor_any counted={count_oor_any}, included_in_explode={include_oor_in_explode})")

    if explode_n == 0:
        print("[sample-guard] No exploding samples under hard criteria.")
    else:
        print("[sample-guard] First exploding site histogram (top-10 by count):")
        for s, c, frac in first_hist_items[:10]:
            print(f"  - {s:<32s}  count={c:<5d}  frac_of_expl={frac * 100:.2f}%")

        # Per-site explode counts
        per_site_sorted = sorted(
            ((s, d["explode"], d["abs_hard"], d["t_hard"], d["nonfinite"]) for s, d in site_counts.items()),
            key=lambda x: x[1], reverse=True)
        print("[sample-guard] Sites with most exploding images (top-10):")
        for s, cexp, cab, ct, cnf in per_site_sorted[:10]:
            if cexp <= 0:
                continue
            print(f"  - {s:<32s}  explode_imgs={cexp:<5d}  abs_hard={cab:<5d}  t_hard={ct:<5d}  nonfinite={cnf:<5d}")

        print("[sample-guard] Top offending samples by max_abs_ratio (among abs_hard samples):")
        for j, rec in enumerate(offenders_abs[:max_print], 1):
            path_note = f" | path={rec['path']}" if rec.get("path") else ""
            print(f"  #{j:02d} i={rec['sample_i']:<5d} orig={rec['orig_i']:<6d} y={rec['label']:<3d} "
                  f"first={rec['first_expl_site']:<24s} abs_site={rec['max_abs_site']:<26s} "
                  f"abs_ratio={rec['max_abs_ratio']:.2f} abs={rec['max_abs']:.3e} "
                  f"t_max={rec['max_t']:.2f} t_site={rec['max_t_site']:<26s}{path_note}")

        print("[sample-guard] Top offending samples by max_t (among t_hard samples):")
        for j, rec in enumerate(offenders_t[:max_print], 1):
            path_note = f" | path={rec['path']}" if rec.get("path") else ""
            print(f"  #{j:02d} i={rec['sample_i']:<5d} orig={rec['orig_i']:<6d} y={rec['label']:<3d} "
                  f"first={rec['first_expl_site']:<24s} t_site={rec['max_t_site']:<26s} "
                  f"t_max={rec['max_t']:.2f} abs_ratio={rec['max_abs_ratio']:.2f} abs={rec['max_abs']:.3e}{path_note}")

    # Save CSVs
    if save_csv_flag and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(tag))
        # per-sample CSV (only for samples that explode OR have abs/t hard OR nonfinite OR oor_any)
        mask_any = expl_cpu | abs_cpu | t_cpu | nonfin_cpu | (oor_cpu if count_oor_any else torch.zeros_like(oor_cpu))
        rows = []
        idx_any = torch.nonzero(mask_any, as_tuple=False).flatten().tolist()
        for i in idx_any:
            i = int(i)
            rows.append({
                "sample_i": i,
                "orig_i": _orig_index(i),
                "label": int(labels_cpu[i].item()),
                "path": _maybe_path(i),
                "explode": bool(expl_cpu[i].item()),
                "abs_hard": bool(abs_cpu[i].item()),
                "t_hard": bool(t_cpu[i].item()),
                "nonfinite": bool(nonfin_cpu[i].item()),
                "oor_any": bool(oor_cpu[i].item()),
                "first_expl_site": _site_name_from_idx(int(first_cpu[i].item())),
                "max_abs_ratio": float(max_abs_ratio_cpu[i].item()),
                "max_abs_site": _site_name_from_idx(int(max_abs_site_cpu[i].item())),
                "max_abs": float(max_abs_val_cpu[i].item()),
                "max_t_site": _site_name_from_idx(int(max_t_site_cpu[i].item())),
                "max_t": float(max_t_val_cpu[i].item()),
            })
        save_csv(rows, out_dir / f"sample_guard_{safe_tag}_per_sample.csv")

        # per-site CSV
        rows2 = []
        for s in sites:
            d = site_counts[s]
            rows2.append({
                "site": s,
                "explode_imgs": int(d["explode"]),
                "abs_hard_imgs": int(d["abs_hard"]),
                "t_hard_imgs": int(d["t_hard"]),
                "nonfinite_imgs": int(d["nonfinite"]),
                "oor_any_imgs": int(d["oor_any"]),
            })
        save_csv(rows2, out_dir / f"sample_guard_{safe_tag}_per_site.csv")

        print(f"[sample-guard] CSVs saved: {out_dir / f'sample_guard_{safe_tag}_per_sample.csv'}")
        print(f"[sample-guard]             {out_dir / f'sample_guard_{safe_tag}_per_site.csv'}")

    return {
        "n": n,
        "explode_n": explode_n,
        "explode_rate": explode_rate,
        "abs_hard_n": abs_n,
        "t_hard_n": t_n,
        "nonfinite_n": nonfin_n,
        "oor_any_n": oor_n,
        "first_site_hist": first_hist_items,
        "offenders_abs": offenders_abs,
        "offenders_t": offenders_t,
        "site_counts": site_counts,
    }


# =============================================================================
# Baseline logits
# =============================================================================

@torch.no_grad()
def compute_baseline_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, float]:
    outs: List[torch.Tensor] = []
    correct_t = torch.zeros((), device=device, dtype=torch.int64)
    n_total = 0
    micro_bs = int(BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU)
    for xb_big, yb_big in loader:
        big_bs = int(yb_big.numel())
        for j in range(0, big_bs, micro_bs):
            xb = xb_big[j:j + micro_bs].to(device, non_blocking=True)
            yb = yb_big[j:j + micro_bs].to(device, non_blocking=True)
        logits = model(xb).detach().float()
        outs.append(logits.cpu())
        pred = logits.argmax(dim=1)
        correct_t += (pred == yb).sum()
        n_total += int(yb.numel())
    out = torch.cat(outs, dim=0)
    if device.type == "cuda":
        try:
            out = out.contiguous().pin_memory()
        except Exception:
            pass
    correct = int(correct_t.item())
    return out, float(correct / max(1, n_total))


# =============================================================================
# DP (knapsack) + error table
# =============================================================================


# =============================================================================
# Test eval (report-only; NEVER used for schedule decisions)
# =============================================================================

@torch.no_grad()
def _eval_on_test(
    *,
    tag: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    schedule: Dict[str, int],
    base_logits: torch.Tensor,
    base_acc: float,
) -> EvalResult:
    schedule_ref.update(schedule)
    ev = eval_model(model, loader, device, sites, used_bounds_ref, baseline_logits_cpu=base_logits, collect_stats=False)
    print(
        f"[test-{tag}] acc={_fmt_pct(ev.acc)} | "
        f"delta_vs_base={_fmt_pp((ev.acc - float(base_acc)) * 100.0)} | "
        f"logit_mse={ev.logit_mse:.6e}"
    )
    return ev


def eval_on_test_fast(
    *,
    stage: str,
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    schedule: Dict[str, int],
    base_logits: Optional[torch.Tensor],
    base_acc: Optional[float],
) -> Optional[EvalResult]:
    if (not ENABLE_STAGE_TEST_EVAL) or loader is None or base_logits is None or base_acc is None:
        return None
    print(f"[eval_on_test_fast] stage={stage}")
    return _eval_on_test(
        tag=f"fast:{stage}",
        model=model,
        loader=loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=schedule,
        base_logits=base_logits,
        base_acc=float(base_acc),
    )


def eval_on_test_full(
    *,
    stage: str,
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    schedule: Dict[str, int],
    base_logits: Optional[torch.Tensor],
    base_acc: Optional[float],
) -> Optional[EvalResult]:
    if (not ENABLE_STAGE_TEST_EVAL) or loader is None or base_logits is None or base_acc is None:
        return None
    print(f"[eval_on_test_full] stage={stage}")
    return _eval_on_test(
        tag=f"full:{stage}",
        model=model,
        loader=loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=schedule,
        base_logits=base_logits,
        base_acc=float(base_acc),
    )


def dp_allocate_schedule(sites: List[str], candidates: List[int], per_site_error: Dict[str, Dict[int, float]],
                         budget: int) -> Dict[str, int]:
    INF = 1e30
    cand = sorted(set(int(x) for x in candidates))
    L = len(sites)
    B = int(budget)

    dp = [[INF] * (B + 1) for _ in range(L + 1)]
    choice: List[List[Optional[int]]] = [[None] * (B + 1) for _ in range(L + 1)]
    prevb: List[List[Optional[int]]] = [[None] * (B + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0

    for i, s in enumerate(sites, start=1):
        for b in range(B + 1):
            if dp[i - 1][b] >= INF:
                continue
            for d in cand:
                c = depth_cost_from_ranges(d)
                nb = b + c
                if nb > B:
                    continue
                e = per_site_error.get(s, {}).get(d, INF)
                val = dp[i - 1][b] + e
                if val < dp[i][nb]:
                    dp[i][nb] = val
                    choice[i][nb] = d
                    prevb[i][nb] = b

    best_b = min(range(B + 1), key=lambda b: dp[L][b])
    if dp[L][best_b] >= INF:
        raise RuntimeError("DP failed: no feasible schedule.")

    sched: Dict[str, int] = {}
    b = best_b
    for i in range(L, 0, -1):
        s = sites[i - 1]
        d = choice[i][b]
        pb = prevb[i][b]
        if d is None or pb is None:
            raise RuntimeError("DP backtrack failed.")
        sched[s] = int(d)
        b = int(pb)

    # 保存 DP schedule 到文件
    repo_root = find_repo_root(Path(__file__).parent)
    out_dir = (repo_root / OUTPUT_DIR_REL / Path(f"sweep_schedules/budget_{budget}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dp_schedule_file = out_dir / "dp_schedule.csv"
    save_csv([{"site": s, "degree": int(d)} for s, d in sched.items()], dp_schedule_file)
    print(f"[dp] DP schedule saved to {dp_schedule_file}")

    return sched


def compute_site_importance_rank(per_site_error: Dict[str, Dict[int, float]], sites: List[str]) -> List[str]:
    min_deg = min(CANDIDATE_DEGREES)
    rank = []
    for s in sites:
        e_min = per_site_error.get(s, {}).get(min_deg, None)
        if e_min is None:
            vals = list(per_site_error.get(s, {}).values())
            e_min = max(vals) if vals else 0.0
        e_best = min(per_site_error.get(s, {}).values()) if per_site_error.get(s) else e_min
        sens = float(e_min - e_best)
        rank.append((sens, s))
    rank.sort(key=lambda x: x[0], reverse=True)
    return [s for _sens, s in rank]


# =============================================================================
# Proposal pools & moves
# =============================================================================

def compute_risk_rank_from_full(ev_full: EvalResult, ref_bounds: Dict[str, Tuple[float, float]], sites: List[str]) -> \
List[str]:
    if not ev_full.stats:
        return sites
    scored = []
    for s in sites:
        st = ev_full.stats.get(s)
        if st is None:
            continue
        a_ref, b_ref = ref_bounds.get(s, (-1.0, 1.0))
        abs_ref = max(abs(a_ref), abs(b_ref), 1e-6)
        abs_ratio = st.abs_max / abs_ref
        score = max(abs_ratio / ABS_MULT_SOFT, st.t_max / T_MAX_SOFT, st.oor_rate / max(OOR_SOFT, 1e-12))
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _sc, s in scored]


def proposal_pool(sites: List[str], importance_rank: List[str], risk_rank: List[str]) -> List[str]:
    pool = []
    seen = set()
    for s in (importance_rank[:PROPOSAL_TOPM_IMPORTANCE] + risk_rank[:PROPOSAL_TOPM_RISK]):
        if s in seen:
            continue
        if s in sites:
            pool.append(s)
            seen.add(s)
        if len(pool) >= PROPOSAL_POOL_MAX:
            break
    return pool if pool else sites


def random_cost_neutral_single_move(rng: np.random.Generator, site_pool: List[str], schedule: Dict[str, int],
                                    cost_buckets: Dict[int, List[int]]) -> Optional[Dict[str, int]]:
    s = rng.choice(site_pool)
    cur = schedule[s]
    cc = depth_cost_from_ranges(cur)
    options = [d for d in cost_buckets.get(cc, []) if d != cur]
    if not options:
        return None
    nd = int(rng.choice(options))
    return {s: nd}


def random_cost_neutral_swap_move(rng: np.random.Generator, site_pool: List[str], sites: List[str],
                                  schedule: Dict[str, int], cost_buckets: Dict[int, List[int]]) -> Optional[
    Dict[str, int]]:
    if len(sites) < 2:
        return None
    s1 = rng.choice(site_pool)
    s2 = rng.choice(site_pool if rng.random() < 0.7 else sites)
    if s2 == s1:
        return None

    d1, d2 = schedule[s1], schedule[s2]
    c1, c2 = depth_cost_from_ranges(d1), depth_cost_from_ranges(d2)

    all_degs = [d for lst in cost_buckets.values() for d in lst]
    nd1 = int(rng.choice(all_degs))
    if nd1 == d1:
        return None
    nc1 = depth_cost_from_ranges(nd1)
    delta = nc1 - c1
    target_c2 = c2 - delta
    if target_c2 not in cost_buckets:
        return None
    options2 = [d for d in cost_buckets[target_c2] if d != d2]
    if not options2:
        return None
    nd2 = int(rng.choice(options2))
    return {s1: nd1, s2: nd2}


def random_cost_neutral_tri_move(rng: np.random.Generator, site_pool: List[str], sites: List[str],
                                 schedule: Dict[str, int], cost_buckets: Dict[int, List[int]], max_tries: int = 20) -> \
Optional[Dict[str, int]]:
    if not ENABLE_TRI_MOVES or len(sites) < 3:
        return None
    all_degs = [d for lst in cost_buckets.values() for d in lst]
    for _ in range(max_tries):
        s1 = rng.choice(site_pool)
        s2 = rng.choice(site_pool)
        s3 = rng.choice(site_pool if rng.random() < 0.6 else sites)
        if len({s1, s2, s3}) != 3:
            continue
        d1o, d2o, d3o = schedule[s1], schedule[s2], schedule[s3]
        c1o, c2o, c3o = depth_cost_from_ranges(d1o), depth_cost_from_ranges(d2o), depth_cost_from_ranges(d3o)

        nd1 = int(rng.choice(all_degs))
        nd2 = int(rng.choice(all_degs))
        c1n, c2n = depth_cost_from_ranges(nd1), depth_cost_from_ranges(nd2)
        if nd1 == d1o and nd2 == d2o:
            continue
        delta = (c1n - c1o) + (c2n - c2o)
        target_c3 = c3o - delta
        if target_c3 not in cost_buckets:
            continue
        options3 = [d for d in cost_buckets[target_c3] if d != d3o]
        if not options3:
            continue
        nd3 = int(rng.choice(options3))
        return {s1: nd1, s2: nd2, s3: nd3}
    return None


def should_run_full_guard(accepts_so_far: int, rel_improv: float) -> bool:
    if FULL_GUARD_EVERY_K_ACCEPTS > 0 and (accepts_so_far % FULL_GUARD_EVERY_K_ACCEPTS == 0):
        return True
    if rel_improv >= FULL_GUARD_TRIGGER_REL_IMPROV:
        return True
    return False


# =============================================================================
# Pareto helpers (VAL selection with acc guidance)
# =============================================================================

def _dominates_obj_acc(
    a_obj: float, a_acc: float,
    b_obj: float, b_acc: float,
    *,
    obj_eps_rel: float,
    acc_eps: float,
) -> bool:
    """
    True if A dominates B for objectives:
      - minimize obj
      - maximize acc
    with epsilons.
    """
    # A no-worse than B?
    obj_ok = a_obj <= b_obj * (1.0 + obj_eps_rel) + 1e-18
    acc_ok = a_acc >= b_acc - acc_eps
    if not (obj_ok and acc_ok):
        return False

    # at least one strict improvement (beyond eps)
    obj_strict = a_obj < b_obj * (1.0 - 1e-12)
    acc_strict = a_acc > b_acc + 1e-12
    return bool(obj_strict or acc_strict)


def pareto_front_indices_obj_acc(
    objs: List[float],
    accs: List[float],
    *,
    obj_eps_rel: float = 0.0,
    acc_eps: float = 0.0,
) -> List[int]:
    n = len(objs)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if _dominates_obj_acc(objs[j], accs[j], objs[i], accs[i], obj_eps_rel=obj_eps_rel, acc_eps=acc_eps):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def pareto_candidate_order(
    candidates: List[Dict[str, Any]],
    *,
    cur_obj: float,
    cur_acc: float,
    rng: np.random.Generator,
    max_try: int,
) -> List[Dict[str, Any]]:
    """
    Return candidates ordered for attempts. Uses Pareto-front of (obj↓, acc↑) vs current schedule.
    Each candidate dict must contain: {"obj": float, "acc": float, ...}
    """
    if (not ENABLE_PARETO_SELECTION) or (not candidates):
        # fallback: best obj
        return sorted(candidates, key=lambda c: float(c["obj"]))[:max_try]

    tmp = [{"obj": float(cur_obj), "acc": float(cur_acc), "__is_cur__": True}] + candidates
    objs = [float(x["obj"]) for x in tmp]
    accs = [float(x["acc"]) for x in tmp]

    front_idx = pareto_front_indices_obj_acc(objs, accs, obj_eps_rel=float(PARETO_OBJ_EPS_REL),
                                             acc_eps=float(PARETO_ACC_EPS))
    front = [tmp[i] for i in front_idx if not tmp[i].get("__is_cur__", False)]

    if not front:
        return []

    if rng.random() < float(PARETO_UNIFORM_PICK_PROB):
        rng.shuffle(front)
        return front[:max_try]

    if str(PARETO_PICK_MODE).lower() == "score":
        alpha = float(PARETO_SCORE_ALPHA)

        # higher score is better
        def _score(c):
            obj = max(float(c["obj"]), 1e-30)
            return float(c["acc"]) - alpha * math.log(obj)

        front.sort(key=_score, reverse=True)
        return front[:max_try]

    # default: prefer higher acc, then lower obj
    front.sort(key=lambda c: (-float(c["acc"]), float(c["obj"])))
    return front[:max_try]


def candidate_admissible(
    *,
    cur_obj: float,
    cur_acc: float,
    cand_obj: float,
    cand_acc: float,
) -> bool:
    """Decide whether a candidate is worth considering.

    Default: require objective improvement.
    If ENABLE_ACC_OBJ_TRADEOFF is True, also allow acc-driven moves when:
      - acc improves by at least ACC_GAIN_MIN_PP
      - obj does not worsen too much (<= OBJ_WORSEN_MAX_RATIO * cur_obj)
    """
    if cand_obj <= cur_obj - 1e-12:
        return True
    if not ENABLE_ACC_OBJ_TRADEOFF:
        return False
    min_gain = float(ACC_GAIN_MIN_PP) / 100.0
    if cand_acc >= cur_acc + min_gain:
        if cand_obj <= cur_obj * float(OBJ_WORSEN_MAX_RATIO) + 1e-18:
            return True
    return False


# =============================================================================
# Joint search (fast->val->(gated) full)
# =============================================================================


def joint_search(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    cost_buckets: Dict[int, List[int]],
    budget: int,
    fast_loader: DataLoader,
    fast_base_logits: torch.Tensor,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    dp_val_acc: float,
    full_calib_loader: DataLoader,
    max_steps: int,
    rng: np.random.Generator,
    start_schedule: Dict[str, int],
    importance_rank: List[str],
) -> Tuple[Dict[str, int], EvalResult, EvalResult, Dict[str, int]]:
    """
    Joint search with:
      FAST (MSE-only) pre-screen
      VAL eval (obj + acc)
      gated FULL guard (strict + optional relaxed sample-level acceptance)
      Pareto selection on (val_obj ↓, val_acc ↑) with uniform randomness guidance.
    """
    schedule = dict(start_schedule)
    schedule_ref.update(schedule)

    # initial FULL stats (for risk proposal pool)
    ev_full_cur = eval_model(model, full_calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                             collect_stats=True)
    ev_full_init = ev_full_cur  # for report-only reuse (DP-start schedule)
    full_schedule_cur = dict(schedule)  # schedule at which ev_full_cur was computed
    risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
    pool = proposal_pool(sites, importance_rank, risk_rank)

    # current VAL metrics
    ev_val = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                        collect_stats=True)
    pen_val = penalize_and_guard(ev_val, ref_bounds)
    cur_obj = float(pen_val.obj)
    cur_acc = float(ev_val.acc)

    print(
        f"[joint] init: cost={schedule_cost(schedule)}/{budget} | "
        f"val_obj={cur_obj:.6e} val_acc={_fmt_pct(cur_acc)} | "
        f"full_abs={ev_full_cur.abs_x_global:.2f} full_t={ev_full_cur.t_global:.2f} full_oor={ev_full_cur.worst_oor * 100:.4f}%"
    )

    accepts = 0
    for step in range(1, max_steps + 1):
        # periodically refresh pool using current FULL stats
        if ENABLE_SITE_PRIORITY_PROPOSAL and (accepts > 0) and (FULL_GUARD_EVERY_K_ACCEPTS > 0) and (
            accepts % FULL_GUARD_EVERY_K_ACCEPTS == 0):
            risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
            pool = proposal_pool(sites, importance_rank, risk_rank)

        # candidate moves
        cand_moves: List[Dict[str, int]] = []
        for _ in range(JOINT_MOVE_TRIALS_PER_STEP):
            use_pool = ENABLE_SITE_PRIORITY_PROPOSAL and (rng.random() < PROPOSAL_POOL_PROB)
            site_pool = pool if use_pool else sites
            r = rng.random()
            if r < 0.30:
                mv = random_cost_neutral_single_move(rng, site_pool, schedule, cost_buckets)
            elif r < 0.78:
                mv = random_cost_neutral_swap_move(rng, site_pool, sites, schedule, cost_buckets)
            else:
                mv = random_cost_neutral_tri_move(rng, site_pool, sites, schedule,
                                                  cost_buckets) if ENABLE_TRI_MOVES else None
            if mv is not None:
                cand_moves.append(mv)

        # dedup
        uniq, seen = [], set()
        for mv in cand_moves:
            key = tuple(sorted(mv.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(mv)
        cand_moves = uniq
        if not cand_moves:
            print(f"[joint] stop@step{step:03d}: no candidates generated.")
            break

        # FAST (MSE-only) rank
        scored_fast: List[Tuple[float, Dict[str, int]]] = []
        for mv in cand_moves:
            trial = apply_move(schedule, mv)
            if schedule_cost(trial) > budget:
                continue
            schedule_ref.update(trial)
            mse = eval_mse_cached("fast", trial, sites, model, fast_loader, device, used_bounds_ref, fast_base_logits)
            scored_fast.append((mse, mv))
        if not scored_fast:
            print(f"[joint] stop@step{step:03d}: no feasible candidates.")
            break
        scored_fast.sort(key=lambda x: x[0])
        top_fast = [mv for _mse, mv in scored_fast[:max(JOINT_TOPK_VAL * 4, JOINT_TOPK_VAL)]]

        # VAL eval (obj + acc)
        val_cands: List[Dict[str, Any]] = []
        for mv in top_fast[:JOINT_TOPK_VAL]:
            trial = apply_move(schedule, mv)
            schedule_ref.update(trial)
            evv = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                             collect_stats=True)
            penv = penalize_and_guard(evv, ref_bounds)
            if not penv.ok:
                continue
            if ENABLE_ACC_GUARD and (evv.acc < (dp_val_acc - ACC_GUARD_DROP_PP / 100.0)):
                continue
            if not candidate_admissible(cur_obj=cur_obj, cur_acc=cur_acc, cand_obj=float(penv.obj),
                                        cand_acc=float(evv.acc)):
                continue
            val_cands.append({"obj": float(penv.obj), "acc": float(evv.acc), "mv": mv})

        if not val_cands:
            print(f"[joint] stop@step{step:03d}: no VAL-feasible candidates.")
            break

        # Pareto order (vs current)
        ordered = pareto_candidate_order(val_cands, cur_obj=cur_obj, cur_acc=cur_acc, rng=rng,
                                         max_try=max(int(PARETO_MAX_CANDS_TRY), int(FULL_GUARD_TOP_CANDS_PER_STEP)))

        if not ordered:
            print(f"[joint] stop@step{step:03d}: no Pareto-admissible candidates.")
            break

        accepted = False
        for cand in ordered:
            mv = cand["mv"]
            objv = float(cand["obj"])
            accv = float(cand["acc"])

            trial = apply_move(schedule, mv)

            # FULL-guard gating is keyed to objective-improvement only; acc-improving-only moves still get periodic full-checks.
            rel_improv_obj = max(0.0, (cur_obj - objv) / max(cur_obj, 1e-12))
            run_full = should_run_full_guard(accepts + 1, rel_improv_obj)

            sg = None
            if run_full:
                schedule_ref.update(trial)
                ev_full_c = eval_model(model, full_calib_loader, device, sites, used_bounds_ref,
                                       baseline_logits_cpu=None, collect_stats=True)
                pen_full_c = penalize_and_guard(ev_full_c, ref_bounds)
                ok_full, sg = full_guard_check(
                    pen_full=pen_full_c,
                    model=model,
                    loader=full_calib_loader,
                    device=device,
                    sites=sites,
                    used_bounds=used_bounds_ref,
                    ref_bounds=ref_bounds,
                )
                if not ok_full:
                    continue
                ev_full_cur = ev_full_c
                full_schedule_cur = dict(trial)

            schedule = trial
            schedule_ref.update(schedule)
            cur_obj = objv
            cur_acc = accv
            accepts += 1

            mv_str = ", ".join([f"{k}->{v}" for k, v in mv.items()])
            tag = "full" if run_full else "nofull"
            sg_str = ""
            if sg is not None:
                sg_str = f" | sg(explode={sg['explode_rate'] * 100:.3f}%, oor_any={sg['oor_any_rate'] * 100:.2f}%)"
            print(
                f"[joint] step{step:03d}: accept({tag}) {{{mv_str}}} | "
                f"val_obj={cur_obj:.6e} val_acc={_fmt_pct(cur_acc)} | rel_obj_improv={rel_improv_obj * 100:.2f}%{sg_str}"
            )
            accepted = True
            break

        if not accepted:
            print(f"[joint] stop@step{step:03d}: Pareto candidates failed FULL guard (when triggered).")
            break

    schedule_ref.update(schedule)
    return schedule, ev_full_cur, ev_full_init, full_schedule_cur


def cd_site_subset(sites: List[str], importance_rank: List[str], risk_rank: List[str]) -> List[str]:
    if not ENABLE_CD_SITE_SUBSET:
        return sites
    pool = []
    seen = set()
    for s in (importance_rank[:CD_TOPM_IMPORTANCE] + risk_rank[:CD_TOPM_RISK]):
        if s in seen:
            continue
        if s in sites:
            pool.append(s)
            seen.add(s)
        if len(pool) >= CD_SITE_SUBSET_MAX:
            break
    return pool if pool else sites


def global_coordinate_descent(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    cost_buckets: Dict[int, List[int]],
    budget: int,
    fast_loader: DataLoader,
    fast_base_logits: torch.Tensor,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    dp_val_acc: float,
    full_calib_loader: DataLoader,
    rng: np.random.Generator,
    importance_rank: List[str],
    ev_full_cur: EvalResult,
) -> Tuple[Dict[str, int], EvalResult]:
    schedule = dict(schedule_ref)
    accepts_total = 0

    risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
    subset = cd_site_subset(sites, importance_rank, risk_rank)

    # current val
    schedule_ref.update(schedule)
    ev_cur = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                        collect_stats=True)
    pen_cur = penalize_and_guard(ev_cur, ref_bounds)
    cur_obj = float(pen_cur.obj)
    cur_acc = float(ev_cur.acc)

    for pass_i in range(1, CD_MAX_PASSES + 1):
        accepts_pass = 0
        improved_any = False

        subset_shuf = list(subset)
        rng.shuffle(subset_shuf)

        for s in subset_shuf:
            cur_d = schedule[s]
            cur_c = depth_cost_from_ranges(cur_d)
            cand_same = [d for d in cost_buckets.get(cur_c, []) if d != cur_d]
            if not cand_same:
                continue

            # FAST pre-screen
            fast_scored: List[Tuple[float, int]] = []
            for nd in cand_same:
                trial = dict(schedule)
                trial[s] = int(nd)
                if schedule_cost(trial) > budget:
                    continue
                schedule_ref.update(trial)
                mse = eval_mse_cached("fast", trial, sites, model, fast_loader, device, used_bounds_ref,
                                      fast_base_logits)
                fast_scored.append((mse, int(nd)))
            if not fast_scored:
                continue
            fast_scored.sort(key=lambda x: x[0])
            top_nds = [nd for _m, nd in fast_scored[:max(1, CD_FAST_TOPK_VAL)]]

            # VAL metrics for candidates
            val_cands: List[Dict[str, Any]] = []
            for nd in top_nds:
                trial = dict(schedule)
                trial[s] = int(nd)
                schedule_ref.update(trial)
                evv = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                                 collect_stats=True)
                penv = penalize_and_guard(evv, ref_bounds)
                if not penv.ok:
                    continue
                if ENABLE_ACC_GUARD and (evv.acc < (dp_val_acc - ACC_GUARD_DROP_PP / 100.0)):
                    continue
                if not candidate_admissible(cur_obj=cur_obj, cur_acc=cur_acc, cand_obj=float(penv.obj),
                                            cand_acc=float(evv.acc)):
                    continue
                val_cands.append({"obj": float(penv.obj), "acc": float(evv.acc), "nd": int(nd)})

            if not val_cands:
                continue

            ordered = pareto_candidate_order(val_cands, cur_obj=cur_obj, cur_acc=cur_acc, rng=rng,
                                             max_try=max(int(PARETO_MAX_CANDS_TRY), 3))
            if not ordered:
                continue

            accepted_site = False
            for cand in ordered:
                nd = int(cand["nd"])
                objv = float(cand["obj"])
                accv = float(cand["acc"])

                trial = dict(schedule)
                trial[s] = int(nd)

                rel_improv_obj = max(0.0, (cur_obj - objv) / max(cur_obj, 1e-12))
                run_full = should_run_full_guard(accepts_total + 1, rel_improv_obj)

                sg = None
                if run_full:
                    schedule_ref.update(trial)
                    ev_full_c = eval_model(model, full_calib_loader, device, sites, used_bounds_ref,
                                           baseline_logits_cpu=None, collect_stats=True)
                    pen_full_c = penalize_and_guard(ev_full_c, ref_bounds)
                    ok_full, sg = full_guard_check(
                        pen_full=pen_full_c,
                        model=model,
                        loader=full_calib_loader,
                        device=device,
                        sites=sites,
                        used_bounds=used_bounds_ref,
                        ref_bounds=ref_bounds,
                    )
                    if not ok_full:
                        schedule_ref.update(schedule)
                        continue

                    ev_full_cur = ev_full_c
                    risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
                    subset = cd_site_subset(sites, importance_rank, risk_rank)

                schedule = trial
                schedule_ref.update(schedule)
                cur_obj = objv
                cur_acc = accv
                accepts_total += 1
                accepts_pass += 1
                improved_any = True
                accepted_site = True

                tag = "full" if run_full else "nofull"
                sg_str = ""
                if sg is not None:
                    sg_str = f" | sg(explode={sg['explode_rate'] * 100:.3f}%, oor_any={sg['oor_any_rate'] * 100:.2f}%)"
                print(
                    f"[global-cd] pass{pass_i} accept({tag}): {s}->{nd} | val_obj={cur_obj:.6e} acc={_fmt_pct(cur_acc)}{sg_str}")
                break

            if accepted_site and (accepts_pass >= CD_MAX_ACCEPTS_PER_PASS):
                break

        if not improved_any:
            break

    schedule_ref.update(schedule)
    return schedule, ev_full_cur


def global_beam_refinement(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    cost_buckets: Dict[int, List[int]],
    budget: int,
    fast_loader: DataLoader,
    fast_base_logits: torch.Tensor,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    dp_val_acc: float,
    full_calib_loader: DataLoader,
    rng: np.random.Generator,
    importance_rank: List[str],
    ev_full_cur: EvalResult,
    start_schedule: Dict[str, int],
    best_obj_ref: float,
) -> Tuple[Dict[str, int], float, EvalResult]:
    """
    Beam refinement using Pareto selection on (val_obj ↓, val_acc ↑).
    """
    # beam elements: (obj, schedule, acc)
    schedule_ref.update(start_schedule)
    ev_start = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                          collect_stats=True)
    pen_start = penalize_and_guard(ev_start, ref_bounds)
    start_obj = float(pen_start.obj)
    beam: List[Tuple[float, Dict[str, int], float]] = [(start_obj, dict(start_schedule), float(ev_start.acc))]

    best_schedule = dict(start_schedule)
    best_obj = float(best_obj_ref)
    best_acc = float(ev_start.acc)

    risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
    pool = proposal_pool(sites, importance_rank, risk_rank)

    for step in range(1, GLOBAL_BEAM_STEPS + 1):
        candidates_fast: List[Tuple[float, Dict[str, int]]] = []
        for _obj_p, sched_p, _acc_p in beam:
            # update pool from latest full stats
            risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
            pool = proposal_pool(sites, importance_rank, risk_rank)

            for _ in range(GLOBAL_BEAM_RANDOM_MOVES_PER_PARENT):
                use_pool = ENABLE_SITE_PRIORITY_PROPOSAL and (rng.random() < PROPOSAL_POOL_PROB)
                site_pool = pool if use_pool else sites
                if rng.random() < 0.70:
                    mv = random_cost_neutral_swap_move(rng, site_pool, sites, sched_p, cost_buckets)
                else:
                    mv = random_cost_neutral_tri_move(rng, site_pool, sites, sched_p,
                                                      cost_buckets) if ENABLE_TRI_MOVES else None
                if mv is None:
                    continue
                trial = apply_move(sched_p, mv)
                if schedule_cost(trial) > budget:
                    continue
                schedule_ref.update(trial)
                mse = eval_mse_cached("fast", trial, sites, model, fast_loader, device, used_bounds_ref,
                                      fast_base_logits)
                candidates_fast.append((mse, trial))

        if not candidates_fast:
            break
        candidates_fast.sort(key=lambda x: x[0])
        candidates_fast = candidates_fast[:GLOBAL_BEAM_TOPK_FAST]

        # VAL eval all top-K
        val_cands: List[Dict[str, Any]] = []
        for _mse, trial in candidates_fast[:GLOBAL_BEAM_TOPK_VAL]:
            schedule_ref.update(trial)
            evv = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                             collect_stats=True)
            penv = penalize_and_guard(evv, ref_bounds)
            if not penv.ok:
                continue
            if ENABLE_ACC_GUARD and (evv.acc < (dp_val_acc - ACC_GUARD_DROP_PP / 100.0)):
                continue
            if not candidate_admissible(cur_obj=best_obj, cur_acc=best_acc, cand_obj=float(penv.obj),
                                        cand_acc=float(evv.acc)):
                continue
            val_cands.append({"obj": float(penv.obj), "acc": float(evv.acc), "sched": dict(trial)})

        if not val_cands:
            break

        # Pareto front vs current best (so we don't fill beam with dominated points)
        tmp = [{"obj": float(best_obj), "acc": float(best_acc), "__is_best__": True}] + val_cands
        objs = [float(x["obj"]) for x in tmp]
        accs = [float(x["acc"]) for x in tmp]
        front_idx = pareto_front_indices_obj_acc(objs, accs, obj_eps_rel=float(PARETO_OBJ_EPS_REL),
                                                 acc_eps=float(PARETO_ACC_EPS))
        front = [tmp[i] for i in front_idx if not tmp[i].get("__is_best__", False)]
        if not front:
            break

        if rng.random() < float(PARETO_UNIFORM_PICK_PROB):
            rng.shuffle(front)
        else:
            front.sort(key=lambda c: (-float(c["acc"]), float(c["obj"])))

        # build next beam with gated FULL checks
        next_beam: List[Tuple[float, Dict[str, int], float]] = []
        for i, cand in enumerate(front[:GLOBAL_BEAM_WIDTH], start=1):
            trial = dict(cand["sched"])
            objv = float(cand["obj"])
            accv = float(cand["acc"])

            run_full = (i <= max(1, BEAM_FULL_GUARD_TOP))
            if run_full:
                schedule_ref.update(trial)
                ev_full = eval_model(model, full_calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                     collect_stats=True)
                pen_full = penalize_and_guard(ev_full, ref_bounds)
                ok_full, _sg = full_guard_check(
                    pen_full=pen_full,
                    model=model,
                    loader=full_calib_loader,
                    device=device,
                    sites=sites,
                    used_bounds=used_bounds_ref,
                    ref_bounds=ref_bounds,
                )
                if not ok_full:
                    continue
                ev_full_cur = ev_full

            next_beam.append((objv, trial, accv))

        if not next_beam:
            break

        # keep beam sorted for stable logs (best acc first, then best obj)
        next_beam.sort(key=lambda x: (-float(x[2]), float(x[0])))
        beam = next_beam

        # update best_schedule from Pareto among {best + beam}
        cand_all = [{"obj": float(best_obj), "acc": float(best_acc), "sched": dict(best_schedule), "__is_cur__": True}]
        cand_all += [{"obj": float(o), "acc": float(a), "sched": dict(s)} for (o, s, a) in beam]
        objs2 = [float(x["obj"]) for x in cand_all]
        accs2 = [float(x["acc"]) for x in cand_all]
        front2_idx = pareto_front_indices_obj_acc(objs2, accs2, obj_eps_rel=float(PARETO_OBJ_EPS_REL),
                                                  acc_eps=float(PARETO_ACC_EPS))
        front2 = [cand_all[i] for i in front2_idx]
        # pick a representative from Pareto set for "best"
        front2.sort(key=lambda c: (-float(c["acc"]), float(c["obj"])))
        picked = front2[0]
        best_schedule = dict(picked["sched"])
        best_obj = float(picked["obj"])
        best_acc = float(picked["acc"])

        print(
            f"[global-beam] step{step:02d}: beam_size={len(beam)} | best(val_obj={best_obj:.6e}, acc={_fmt_pct(best_acc)})")

    schedule_ref.update(best_schedule)
    return best_schedule, best_obj, ev_full_cur


def global_refinement(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    cost_buckets: Dict[int, List[int]],
    budget: int,
    fast_loader: DataLoader,
    fast_base_logits: torch.Tensor,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    dp_val_acc: float,
    full_calib_loader: DataLoader,
    rng: np.random.Generator,
    importance_rank: List[str],
    ev_full_cur: EvalResult,
) -> Tuple[Dict[str, int], EvalResult, float, float]:
    # start from current schedule_ref
    best_schedule = dict(schedule_ref)
    schedule_ref.update(best_schedule)
    ev_best = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                         collect_stats=True)
    pen_best = penalize_and_guard(ev_best, ref_bounds)
    best_obj = float(pen_best.obj)
    best_acc = float(ev_best.acc)
    print(f"[global] start: val_obj={best_obj:.6e} acc={_fmt_pct(best_acc)}")

    for r in range(GLOBAL_REFINEMENT_RESTARTS):
        # kick schedule by random cost-neutral single moves on proposal pool
        risk_rank = compute_risk_rank_from_full(ev_full_cur, ref_bounds, sites)
        pool = proposal_pool(sites, importance_rank, risk_rank)

        kicked = dict(best_schedule)
        ok = False
        for _try in range(GLOBAL_REFINEMENT_KICK_TRIES):
            kicked = dict(best_schedule)
            pick_sites = rng.choice(pool, size=min(GLOBAL_REFINEMENT_KICK_SITES, len(pool)), replace=False).tolist()
            for s in pick_sites:
                cd = kicked[s]
                cc = depth_cost_from_ranges(cd)
                opts = [d for d in cost_buckets[cc] if d != cd]
                if opts:
                    kicked[s] = int(rng.choice(opts))
            schedule_ref.update(kicked)
            # cheap check: val guard only
            evk = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                             collect_stats=True)
            penk = penalize_and_guard(evk, ref_bounds)
            if penk.ok and (not ENABLE_ACC_GUARD or evk.acc >= (dp_val_acc - ACC_GUARD_DROP_PP / 100.0)):
                ok = True
                break
        if not ok:
            continue

        print(f"[global] restart{r:02d}: init val_obj={penk.obj:.6e} acc={_fmt_pct(evk.acc)}")

        schedule_ref.update(kicked)
        # CD refinement
        cd_sched, ev_full_cur = global_coordinate_descent(
            model, device, sites, schedule_ref, used_bounds_ref, ref_bounds, cost_buckets, budget,
            fast_loader, fast_base_logits, val_loader, val_base_logits, dp_val_acc, full_calib_loader,
            rng, importance_rank, ev_full_cur
        )

        schedule_ref.update(cd_sched)
        ev_cd = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                           collect_stats=True)
        pen_cd = penalize_and_guard(ev_cd, ref_bounds)

        if pen_cd.ok and pen_cd.obj < best_obj:
            best_obj = float(pen_cd.obj)
            best_schedule = dict(cd_sched)
            best_acc = float(ev_cd.acc)
            print(f"[global] NEW BEST after CD: val_obj={best_obj:.6e} acc={_fmt_pct(best_acc)}")

        # Beam refinement
        if RUN_GLOBAL_BEAM:
            beam_best, beam_obj, ev_full_cur = global_beam_refinement(
                model, device, sites, schedule_ref, used_bounds_ref, ref_bounds, cost_buckets, budget,
                fast_loader, fast_base_logits, val_loader, val_base_logits, dp_val_acc, full_calib_loader,
                rng, importance_rank, ev_full_cur, cd_sched, best_obj
            )
            schedule_ref.update(beam_best)
            if beam_obj < best_obj:
                best_obj = float(beam_obj)
                best_schedule = dict(beam_best)
                # recompute acc for print
                ev_tmp = eval_model(model, val_loader, device, sites, used_bounds_ref,
                                    baseline_logits_cpu=val_base_logits, collect_stats=False)
                best_acc = float(ev_tmp.acc)

    schedule_ref.update(best_schedule)
    return best_schedule, ev_full_cur, best_obj, best_acc


# =============================================================================
# Local window refinement (same as before, moderate)
# =============================================================================

def degree_neighbors(current: int, candidates: List[int], radius: int) -> List[int]:
    cands = sorted(set(int(x) for x in candidates))
    if current not in cands:
        idx = int(np.argmin([abs(x - current) for x in cands]))
        current = cands[idx]
    idx = cands.index(current)
    lo = max(0, idx - radius)
    hi = min(len(cands), idx + radius + 1)
    return cands[lo:hi]


def pick_risky_sites_from_stats(ev_full: EvalResult, ref_bounds: Dict[str, Tuple[float, float]], sites: List[str],
                                topk: int) -> List[str]:
    scored = []
    for s in sites:
        st = ev_full.stats.get(s)
        if st is None:
            continue
        a_ref, b_ref = ref_bounds.get(s, (-1.0, 1.0))
        abs_ref = max(abs(a_ref), abs(b_ref), 1e-6)
        abs_ratio = st.abs_max / abs_ref
        score = max(abs_ratio / ABS_MULT_SOFT, st.t_max / T_MAX_SOFT, st.oor_rate / max(OOR_SOFT, 1e-12))
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _sc, s in scored[:topk]]


def make_windows_around_sites(sites_order: List[str], centers: List[str], window_size: int) -> List[List[str]]:
    n = len(sites_order)
    w = int(window_size)
    half = w // 2
    idx_map = {s: i for i, s in enumerate(sites_order)}
    windows, seen = [], set()
    for c in centers:
        if c not in idx_map:
            continue
        i = idx_map[c]
        lo = max(0, i - half)
        hi = min(n, lo + w)
        lo = max(0, hi - w)
        win = sites_order[lo:hi]
        key = tuple(win)
        if key in seen:
            continue
        seen.add(key)
        windows.append(win)
    return windows


def local_window_refinement(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    ref_bounds: Dict[str, Tuple[float, float]],
    per_site_error: Dict[str, Dict[int, float]],
    budget: int,
    fast_loader: DataLoader,
    fast_base_logits: torch.Tensor,
    val_loader: DataLoader,
    val_base_logits: torch.Tensor,
    dp_val_acc: float,
    full_calib_loader: DataLoader,
    rng: np.random.Generator,
    ev_full_cur: EvalResult,
) -> Tuple[Dict[str, int], EvalResult]:
    if not ENABLE_LOCAL_WINDOW_SEARCH:
        return dict(schedule_ref), ev_full_cur

    cur_schedule = dict(schedule_ref)
    schedule_ref.update(cur_schedule)

    ev_cur_val = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                            collect_stats=True)
    pen_cur_val = penalize_and_guard(ev_cur_val, ref_bounds)
    cur_obj, cur_acc = float(pen_cur_val.obj), float(ev_cur_val.acc)

    risky_sites = pick_risky_sites_from_stats(ev_full_cur, ref_bounds, sites, topk=LOCAL_WINDOW_TOPK_SITES)
    windows = make_windows_around_sites(sites, risky_sites, LOCAL_WINDOW_SIZE)
    if not windows:
        return cur_schedule, ev_full_cur

    print(f"[local-window] risky_sites: {', '.join(risky_sites)}")
    print(f"[local-window] windows: {len(windows)} (size={LOCAL_WINDOW_SIZE})")

    cand_sorted = sorted(set(int(x) for x in CANDIDATE_DEGREES))
    for wi, win in enumerate(windows, start=1):
        target_cost = sum(depth_cost_from_ranges(cur_schedule[s]) for s in win)

        # options per site (neighbors)
        options: Dict[str, List[int]] = {}
        for s in win:
            cur_d = cur_schedule[s]
            opts = degree_neighbors(cur_d, cand_sorted, radius=LOCAL_DEG_NEIGHBOR_RADIUS)
            if cur_d not in opts:
                opts = sorted(set(opts + [cur_d]))
            options[s] = opts

        combos: List[Tuple[float, Dict[str, int]]] = []
        win_list = list(win)

        def dfs(i: int, cost_so_far: int, assign: Dict[str, int]) -> None:
            if i == len(win_list):
                if cost_so_far == target_cost:
                    proxy = 0.0
                    for ss, dd in assign.items():
                        proxy += float(per_site_error.get(ss, {}).get(dd, 0.0))
                    combos.append((proxy, dict(assign)))
                return
            s = win_list[i]
            for d in options[s]:
                c = depth_cost_from_ranges(d)
                new_cost = cost_so_far + c
                if new_cost > target_cost:
                    continue
                # bound remaining
                rem = win_list[i + 1:]
                min_rem = 0
                max_rem = 0
                for rr in rem:
                    costs = [depth_cost_from_ranges(dd) for dd in options[rr]]
                    min_rem += min(costs)
                    max_rem += max(costs)
                if not (new_cost + min_rem <= target_cost <= new_cost + max_rem):
                    continue
                assign[s] = int(d)
                dfs(i + 1, new_cost, assign)
                assign.pop(s, None)

        dfs(0, 0, {})

        if not combos:
            continue
        combos.sort(key=lambda x: x[0])
        combos = combos[:LOCAL_PROXY_TOPK]

        # FAST on combos
        fast_scored: List[Tuple[float, Dict[str, int]]] = []
        for _proxy, asg in combos:
            trial = dict(cur_schedule)
            trial.update(asg)
            if schedule_cost(trial) > budget:
                continue
            schedule_ref.update(trial)
            mse = eval_mse_cached("fast", trial, sites, model, fast_loader, device, used_bounds_ref, fast_base_logits)
            fast_scored.append((mse, asg))
        if not fast_scored:
            continue
        fast_scored.sort(key=lambda x: x[0])
        fast_scored = fast_scored[:LOCAL_FAST_TOPK]

        # VAL (obj + acc)
        val_cands: List[Dict[str, Any]] = []
        for _mse, asg in fast_scored[: max(1, LOCAL_VAL_TOPK * 2)]:
            trial = dict(cur_schedule)
            trial.update(asg)
            schedule_ref.update(trial)
            evv = eval_model(model, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=val_base_logits,
                             collect_stats=True)
            penv = penalize_and_guard(evv, ref_bounds)
            if not penv.ok:
                continue
            if ENABLE_ACC_GUARD and (evv.acc < (dp_val_acc - ACC_GUARD_DROP_PP / 100.0)):
                continue
            if not candidate_admissible(cur_obj=cur_obj, cur_acc=cur_acc, cand_obj=float(penv.obj),
                                        cand_acc=float(evv.acc)):
                continue
            val_cands.append({"obj": float(penv.obj), "acc": float(evv.acc), "asg": dict(asg)})

        if not val_cands:
            continue

        ordered = pareto_candidate_order(val_cands, cur_obj=cur_obj, cur_acc=cur_acc, rng=rng,
                                         max_try=max(int(PARETO_MAX_CANDS_TRY), int(LOCAL_FULL_TOPK)))
        if not ordered:
            continue

        accepted = False
        for cand in ordered:
            asg = dict(cand["asg"])
            objv = float(cand["obj"])
            accv = float(cand["acc"])

            trial = dict(cur_schedule)
            trial.update(asg)
            schedule_ref.update(trial)

            ev_full_c = eval_model(model, full_calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                   collect_stats=True)
            pen_full_c = penalize_and_guard(ev_full_c, ref_bounds)
            ok_full, sg = full_guard_check(
                pen_full=pen_full_c,
                model=model,
                loader=full_calib_loader,
                device=device,
                sites=sites,
                used_bounds=used_bounds_ref,
                ref_bounds=ref_bounds,
            )
            if not ok_full:
                continue

            cur_schedule = trial
            cur_obj = objv
            cur_acc = accv
            ev_full_cur = ev_full_c
            schedule_ref.update(cur_schedule)

            sg_str = ""
            if sg is not None:
                sg_str = f" | sg(explode={sg['explode_rate'] * 100:.3f}%, oor_any={sg['oor_any_rate'] * 100:.2f}%)"
            print(f"[local-window] win{wi:02d}: ACCEPT val_obj={cur_obj:.6e} acc={_fmt_pct(cur_acc)}{sg_str}")
            accepted = True
            break

        if not accepted:
            print(f"[local-window] win{wi:02d}: no FULL-feasible Pareto candidate.")

    schedule_ref.update(cur_schedule)
    return cur_schedule, ev_full_cur


def _sched_with_exact_sites(base: Dict[str, int], exact_sites: Sequence[str]) -> Dict[str, int]:
    out = dict(base)
    for s in exact_sites:
        out[s] = 0  # SiteActivation.forward(): deg<=0 => torch.relu(x)
    return out


@torch.no_grad()
def _eval_schedule_val_mse_acc(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    schedule: Dict[str, int],
    val_loader: DataLoader,
    base_val_logits: torch.Tensor,
) -> Tuple[float, float]:
    schedule_ref.update(schedule)
    ev = eval_model(
        model,
        val_loader,
        device,
        sites,
        used_bounds_ref,
        baseline_logits_cpu=base_val_logits,
        collect_stats=False,
    )
    return float(ev.logit_mse), float(ev.acc)


def exact_relu_diag_1to3_sites(
    model: nn.Module,
    device: torch.device,
    sites: List[str],
    schedule_ref: Dict[str, int],
    used_bounds_ref: Dict[str, Tuple[float, float]],
    base_schedule: Dict[str, int],
    val_loader: DataLoader,
    base_val_logits: torch.Tensor,
    out_dir: Path,
    print_top: int = 15,
    topk_single: int = 12,
) -> Dict[str, Any]:
    """
    Runs:
      - exhaustive single-site exact ReLU override for every site
      - exhaustive pairs/triples among top-K single-site improvements

    Selection is on VAL only.
    Returns a dict containing best1/best2/best3 info and full tables.
    """
    out: Dict[str, Any] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    base_mse, base_acc = _eval_schedule_val_mse_acc(
        model, device, sites, schedule_ref, used_bounds_ref, base_schedule, val_loader, base_val_logits
    )

    # 1) singles
    singles: List[Dict[str, Any]] = []
    for s in sites:
        sched = _sched_with_exact_sites(base_schedule, [s])
        mse, acc = _eval_schedule_val_mse_acc(
            model, device, sites, schedule_ref, used_bounds_ref, sched, val_loader, base_val_logits
        )
        singles.append({
            "site": s,
            "exact_k": 1,
            "sites": s,
            "mse": mse,
            "acc": acc,
            "delta_acc_pp": (acc - base_acc) * 100.0,
            "delta_mse": mse - base_mse,
        })

    singles.sort(key=lambda r: (-(r["delta_acc_pp"]), r["mse"]))
    out["singles"] = singles

    print("\n" + "=" * 110)
    print("DIAG: Single-site exact ReLU overrides on VAL (selection metric = Δacc)")
    print("=" * 110)
    print(f"[diag] base schedule on VAL: acc={_fmt_pct(base_acc)} mse={base_mse:.6e}")
    for i, r in enumerate(singles[:max(1, print_top)], start=1):
        print(
            f"#{i:02d}  {r['site']:<30s}  Δacc={r['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(r['acc'])}  mse={r['mse']:.6e}")

    best1 = singles[0]
    out["best1"] = best1

    top_sites = [r["site"] for r in singles[:max(1, min(topk_single, len(singles)))]]
    out["top_sites_for_combo"] = top_sites

    # 2) pairs
    pairs: List[Dict[str, Any]] = []
    if len(top_sites) >= 2:
        for a, b in itertools.combinations(top_sites, 2):
            sched = _sched_with_exact_sites(base_schedule, [a, b])
            mse, acc = _eval_schedule_val_mse_acc(
                model, device, sites, schedule_ref, used_bounds_ref, sched, val_loader, base_val_logits
            )
            pairs.append({
                "site": "",
                "exact_k": 2,
                "sites": f"{a},{b}",
                "mse": mse,
                "acc": acc,
                "delta_acc_pp": (acc - base_acc) * 100.0,
                "delta_mse": mse - base_mse,
            })
        pairs.sort(key=lambda r: (-(r["delta_acc_pp"]), r["mse"]))
    out["pairs"] = pairs
    best2 = pairs[0] if pairs else None
    out["best2"] = best2

    print("\n" + "=" * 110)
    print(f"DIAG: Pair exact ReLU overrides on VAL (enumerated among top-{len(top_sites)} single sites)")
    print("=" * 110)
    if not pairs:
        print("[diag] skipped (not enough candidate sites).")
    else:
        for i, r in enumerate(pairs[:max(1, print_top)], start=1):
            print(
                f"#{i:02d}  {r['sites']:<60s}  Δacc={r['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(r['acc'])}  mse={r['mse']:.6e}")

    # 3) triples
    triples: List[Dict[str, Any]] = []
    if len(top_sites) >= 3:
        for a, b, c in itertools.combinations(top_sites, 3):
            sched = _sched_with_exact_sites(base_schedule, [a, b, c])
            mse, acc = _eval_schedule_val_mse_acc(
                model, device, sites, schedule_ref, used_bounds_ref, sched, val_loader, base_val_logits
            )
            triples.append({
                "site": "",
                "exact_k": 3,
                "sites": f"{a},{b},{c}",
                "mse": mse,
                "acc": acc,
                "delta_acc_pp": (acc - base_acc) * 100.0,
                "delta_mse": mse - base_mse,
            })
        triples.sort(key=lambda r: (-(r["delta_acc_pp"]), r["mse"]))
    out["triples"] = triples
    best3 = triples[0] if triples else None
    out["best3"] = best3

    print("\n" + "=" * 110)
    print(f"DIAG: Triple exact ReLU overrides on VAL (enumerated among top-{len(top_sites)} single sites)")
    print("=" * 110)
    if not triples:
        print("[diag] skipped (not enough candidate sites).")
    else:
        for i, r in enumerate(triples[:max(1, print_top)], start=1):
            print(
                f"#{i:02d}  {r['sites']:<60s}  Δacc={r['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(r['acc'])}  mse={r['mse']:.6e}")

    # save csv
    save_csv(singles, out_dir / "diag_exact_relu_singles_val.csv")
    if pairs:
        save_csv(pairs, out_dir / "diag_exact_relu_pairs_val.csv")
    if triples:
        save_csv(triples, out_dir / "diag_exact_relu_triples_val.csv")

    # summary
    print("\n" + "=" * 110)
    print("DIAG SUMMARY (VAL-only selection)")
    print("=" * 110)
    print(f"[diag] BASE  : acc={_fmt_pct(base_acc)} mse={base_mse:.6e}")
    if best1:
        print(
            f"[diag] BEST1 : sites={best1['sites']}  Δacc={best1['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(best1['acc'])}  mse={best1['mse']:.6e}")
    if best2:
        print(
            f"[diag] BEST2 : sites={best2['sites']}  Δacc={best2['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(best2['acc'])}  mse={best2['mse']:.6e}")
    if best3:
        print(
            f"[diag] BEST3 : sites={best3['sites']}  Δacc={best3['delta_acc_pp']:+.3f} pp  acc={_fmt_pct(best3['acc'])}  mse={best3['mse']:.6e}")
    print("[diag] CSVs saved under:", str(out_dir))

    return out


# =============================================================================
# Main pipeline
# =============================================================================


def render_final_schedule_bounds_table(
    calib_report_rows: List[Dict[str, Any]],
    *,
    title: str = "FINAL schedule: per-site activation INPUT bounds on CALIB (ref bounds + used bounds + observed min/max + t/oor)",
) -> str:
    """Render the FINAL schedule bounds table as a string.

    Expected keys per row (flexible):
      - site, deg (or degree)
      - ref_a, ref_b
      - used_a, used_b
      - min_obs (or min), max_obs (or max)
      - abs_max (or abs)
      - t_max (or tmax)
      - oor_pct (or oor_percent or oor(%))
    """
    lines: List[str] = []
    lines.append("=" * 140)
    lines.append(title)
    lines.append("=" * 140)
    lines.append(
        f"{'site':<28} {'deg':>5s}  {'ref_a':>9s} {'ref_b':>9s} {'used_a':>9s} {'used_b':>9s}"
        f" {'min_obs':>10s} {'max_obs':>10s} {'abs_max':>10s} {'t_max':>7s} {'oor(%)':>8s}"
    )
    for r in calib_report_rows:
        site = str(r.get("site", ""))
        deg = int(r.get("deg", r.get("degree", 0)) or 0)
        ref_a = float(r.get("ref_a", float("nan")))
        ref_b = float(r.get("ref_b", float("nan")))
        used_a = float(r.get("used_a", float("nan")))
        used_b = float(r.get("used_b", float("nan")))

        min_obs = float(r.get("min_obs", r.get("min", float("nan"))))
        max_obs = float(r.get("max_obs", r.get("max", float("nan"))))
        abs_max = float(r.get("abs_max", r.get("abs", float("nan"))))
        t_max = float(r.get("t_max", r.get("tmax", float("nan"))))

        oor = float(r.get("oor_pct", r.get("oor_percent", r.get("oor(%)", float("nan")))))

        lines.append(
            f"{site:<28} {deg:>5d}  {ref_a:>9.3f} {ref_b:>9.3f} {used_a:>9.3f} {used_b:>9.3f}"
            f" {min_obs:>10.3f} {max_obs:>10.3f} {abs_max:>10.3f} {t_max:>7.3f} {oor:>8.3f}"
        )
    return "\n".join(lines)


def print_final_schedule_bounds_table(
    calib_report_rows: List[Dict[str, Any]],
    *,
    title: str = "FINAL schedule: per-site activation INPUT bounds on CALIB (ref bounds + used bounds + observed min/max + t/oor)",
) -> None:
    """Pretty-print the FINAL schedule bounds table."""
    print(render_final_schedule_bounds_table(calib_report_rows, title=title))


def save_final_schedule_bounds_table(
    calib_report_rows: List[Dict[str, Any]],
    *,
    txt_path: Path,
    csv_path: Optional[Path] = None,
    title: str = "FINAL schedule: per-site activation INPUT bounds on CALIB (ref bounds + used bounds + observed min/max + t/oor)",
) -> None:
    """Save the FINAL schedule bounds table (txt + optional csv)."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(render_final_schedule_bounds_table(calib_report_rows, title=title) + "\n", encoding="utf-8")
    if csv_path is not None:
        save_csv(calib_report_rows, csv_path)

def main() -> None:
    set_seeds(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)

    device = pick_device()
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    repo_root = find_repo_root(Path(__file__).parent)
    weights_path = (repo_root / WEIGHTS_REL).resolve()
    out_dir = (repo_root / OUTPUT_DIR_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device} | batch_size={batch_size}")
    # print(f"[version] {VERSION_TAG}")
    print(f"[repo] {repo_root}")
    print(f"[weights] {weights_path}")
    print(f"[dataset] {DATASET_NAME}")
    print(f"[model] {MODEL_NAME}")
    print(f"[no-clip] USE_CLIP_X={USE_CLIP_X}, CLAMP_T_TO_UNIT={CLAMP_T_TO_UNIT}")
    print(f"[budget] TOTAL_BUDGET={TOTAL_BUDGET} (cost table ranges={DEPTH_COST_RANGES})")
    print(
        f"[speed] gated_full: top={FULL_GUARD_TOP_CANDS_PER_STEP}, everyK={FULL_GUARD_EVERY_K_ACCEPTS}, rel={FULL_GUARD_TRIGGER_REL_IMPROV}; CD_subset={ENABLE_CD_SITE_SUBSET}, CD_topk_val={CD_FAST_TOPK_VAL}, beam_full_top={BEAM_FULL_GUARD_TOP}")
    print(
        f"[search] tri_moves={ENABLE_TRI_MOVES}, local_window={ENABLE_LOCAL_WINDOW_SEARCH}, outer_recalib={ENABLE_OUTER_RECALIB}")
    print(
        f"[subsets] calib={CALIB_N} sweep={DP_SWEEP_N} search={JOINT_SEARCH_N} val={JOINT_VAL_N} fast={JOINT_FAST_N} test={('all' if TEST_MAX_N is None else TEST_MAX_N)}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    ds_train = build_dataset(DATASET_NAME, "train", repo_root)
    ds_test = build_dataset(DATASET_NAME, "test", repo_root)

    calib_idx = sample_indices(len(ds_train), CALIB_N, GLOBAL_SEED)
    n_cal = len(calib_idx)
    n_search = min(JOINT_SEARCH_N, n_cal)
    n_val = min(JOINT_VAL_N, max(0, n_cal - n_search))
    search_idx = calib_idx[:n_search]
    val_idx = calib_idx[n_search:n_search + n_val]
    sweep_idx = calib_idx[:min(DP_SWEEP_N, n_cal)]
    fast_idx = search_idx[:min(JOINT_FAST_N, len(search_idx))]

    test_idx = None
    if TEST_MAX_N is not None:
        test_idx = sample_indices(len(ds_test), TEST_MAX_N, GLOBAL_SEED + 1)

    # -------------------------------------------------------------------------
    # Auto-tune DataLoader batch size (loader throughput) WITHOUT changing results.
    # We keep compute micro-batches fixed (batch_size) and split fetched batches inside eval loops.
    # For QUANTILE bounds collection, keep calib_loader batch_size unchanged to preserve sampling behavior.
    # -------------------------------------------------------------------------
    micro_batch = int(batch_size)
    tuned_loader_bs = micro_batch
    if AUTO_TUNE_LOADER_BATCH:
        tuned_loader_bs = _probe_loader_batch_size(
            ds_train,
            indices=sweep_idx,  # representative subset
            micro_batch=micro_batch,
            device=device,
            start_mult=int(AUTO_LOADER_START_MULT),
            max_mult=int(AUTO_LOADER_MAX_MULT),
        )
    # Calib loader: preserve quantile sampling behavior by keeping the original micro-batch.
    calib_loader_bs = micro_batch if (BOUND_MODE.lower() == "quantile") else int(tuned_loader_bs)
    eval_loader_bs = int(tuned_loader_bs)
    print(f"[auto-batch] micro_batch={micro_batch} | loader_batch={eval_loader_bs} | calib_loader_batch={calib_loader_bs}")


    calib_loader = build_loader(ds_train, calib_idx, calib_loader_bs, device)
    sweep_loader = build_loader(ds_train, sweep_idx, eval_loader_bs, device)
    search_loader = build_loader(ds_train, search_idx, eval_loader_bs, device)  # not used much in v2
    val_loader = build_loader(ds_train, val_idx, eval_loader_bs, device)
    fast_loader = build_loader(ds_train, fast_idx, eval_loader_bs, device)
    test_loader = build_loader(ds_test, test_idx, eval_loader_bs, device)

    # Test-fast subset (report-only). NOTE: this is still TEST data; do NOT use it for config/schedule decisions.
    test_fast_idx = None
    test_fast_loader = None
    if (TEST_FAST_N is not None) and (int(TEST_FAST_N) > 0):
        if test_idx is None:
            test_fast_idx = list(range(min(int(TEST_FAST_N), len(ds_test))))
        else:
            test_fast_idx = test_idx[: min(int(TEST_FAST_N), len(test_idx))]
    test_fast_loader = build_loader(ds_test, test_fast_idx, eval_loader_bs, device)

    print(
        f"[subsets-actual] calib={len(calib_idx)} sweep={len(sweep_idx)} search={len(search_idx)} val={len(val_idx)} fast={len(fast_idx)} test={len(test_idx) if test_idx else len(ds_test)}")

    # model
    sd = load_checkpoint(weights_path)
    model_base = build_model(MODEL_NAME, sd, num_classes_for_dataset(DATASET_NAME)).eval()

    schedule_ref: Dict[str, int] = {}
    used_bounds_ref: Dict[str, Tuple[float, float]] = {}

    model_fx, _ = instrument_relu_sites_fx(model_base, CANDIDATE_DEGREES, schedule_ref, used_bounds_ref)
    model_fx = model_fx.to(device).eval()
    sites = list_site_modules(model_fx)
    print(f"[sites] {len(sites)} activation sites instrumented via FX.")

    # baseline schedule exact
    set_schedule(schedule_ref, sites, 0)

    # baseline bounds on calib
    print(f"[bounds] collecting baseline bounds on CALIB (mode={BOUND_MODE}, margin={RANGE_MARGIN_FRAC}) ...")
    ref_bounds = collect_bounds_from_loader(model_fx, calib_loader, device, sites, mode=BOUND_MODE,
                                            margin_frac=RANGE_MARGIN_FRAC, q_low=Q_LOW, q_high=Q_HIGH)
    sym_bounds = make_symmetric_bounds(ref_bounds)
    used_bounds_ref.clear()
    used_bounds_ref.update(ref_bounds)
    rebuild_all_site_coeffs(model_fx)

    # baseline logits on subsets
    print("[baseline] computing baseline logits on SWEEP/FAST/VAL (exact ReLU) ...")
    set_schedule(schedule_ref, sites, 0)
    base_sweep_logits, _ = compute_baseline_logits(model_fx, sweep_loader, device)
    base_fast_logits, _ = compute_baseline_logits(model_fx, fast_loader, device)
    base_val_logits, base_val_acc_exact = compute_baseline_logits(model_fx, val_loader, device)
    print(f"[baseline] VAL acc (exact ReLU) = {_fmt_pct(base_val_acc_exact)}")

    # Speed: keep baseline logits on the same device as eval to avoid per-batch CPU→GPU copies.
    # This does NOT change any numeric results (same values, just relocated once).
    if base_sweep_logits.device != device:
        base_sweep_logits = base_sweep_logits.to(device, non_blocking=True)
    if base_fast_logits.device != device:
        base_fast_logits = base_fast_logits.to(device, non_blocking=True)
    if base_val_logits.device != device:
        base_val_logits = base_val_logits.to(device, non_blocking=True)
    base_sweep_logits = base_sweep_logits.contiguous()
    base_fast_logits = base_fast_logits.contiguous()
    base_val_logits = base_val_logits.contiguous()


    # baseline logits on TEST (report only; do NOT use for schedule decisions)
    base_test_logits = None
    base_test_acc = None
    base_test_fast_logits = None
    base_test_fast_acc = None
    if ENABLE_STAGE_TEST_EVAL:
        print("[baseline] computing baseline logits on TEST (exact ReLU) ...")
        set_schedule(schedule_ref, sites, 0)
        base_test_logits, base_test_acc = compute_baseline_logits(model_fx, test_loader, device)
        if test_fast_loader is not None:
            base_test_fast_logits, base_test_fast_acc = compute_baseline_logits(model_fx, test_fast_loader, device)
        print(f"[baseline] TEST acc (exact ReLU) = {_fmt_pct(float(base_test_acc))}")


    # DP one-layer error table (proxy)  -- reuse if possible
    table_best_path = out_dir / DP_ONE_LAYER_ERROR_TABLE_BEST_NAME
    table_mixed_path = out_dir / DP_ONE_LAYER_ERROR_TABLE_MIXED_NAME

    per_site_error: Optional[Dict[str, Dict[int, float]]] = None
    per_site_best_scale: Optional[Dict[str, Dict[int, str]]] = None

    if REUSE_DP_ONE_LAYER_ERROR_TABLE:
        per_site_error, per_site_best_scale = load_dp_one_layer_error_table_mixed_csv(
            table_best_path, table_mixed_path, sites, CANDIDATE_DEGREES
        )
        if per_site_error is not None and per_site_best_scale is not None:
            print(f"[dp] reuse one-layer error table: {table_mixed_path if table_mixed_path.exists() else table_best_path}")

    if per_site_error is None or per_site_best_scale is None:
        # DP one-layer error table (proxy)
        print("[dp] building per-site one-layer error table on SWEEP subset ...")
        per_site_error: Dict[str, Dict[int, float]] = {s: {} for s in sites}
        per_site_best_scale: Dict[str, Dict[int, str]] = {s: {} for s in sites}

        # Fast lookup: name -> SiteActivation module (for per-site coeff rebuilds)
        site_module_map: Dict[str, SiteActivation] = {nm: m for nm, m in model_fx.named_modules() if
                                                      isinstance(m, SiteActivation)}

        dp_rows_best: List[Dict[str, Any]] = []
        dp_rows_full: List[Dict[str, Any]] = []

        cand_degs = list(CANDIDATE_DEGREES)

        for si, s in enumerate(sites, start=1):
            m_site = site_module_map.get(s, None)

            # --- MINMAX: rebuild once, eval all degrees ---
            used_bounds_ref[s] = ref_bounds.get(s, (-1.0, 1.0))
            if m_site is not None:
                m_site.rebuild_coeffs()

            mse_min_by_deg: Dict[int, float] = {}
            set_schedule(schedule_ref, sites, 0)
            for di, d in enumerate(cand_degs, start=1):
                schedule_ref[s] = int(d)
                ev_min = eval_model(
                    model_fx,
                    sweep_loader,
                    device,
                    sites,
                    used_bounds_ref,
                    baseline_logits_cpu=base_sweep_logits, collect_stats=False
                )
                mse_min_by_deg[int(d)] = float(ev_min.logit_mse)

                if (di % 10 == 0) or (di == len(cand_degs)):
                    print(f"[dp] site {si:02d}/{len(sites)} {s} | minmax done {di:02d}/{len(cand_degs)}")

            # --- SYMMETRIC: rebuild once, eval all degrees ---
            used_bounds_ref[s] = sym_bounds.get(s, ref_bounds.get(s, (-1.0, 1.0)))
            if m_site is not None:
                m_site.rebuild_coeffs()

            mse_sym_by_deg: Dict[int, float] = {}
            set_schedule(schedule_ref, sites, 0)
            for di, d in enumerate(cand_degs, start=1):
                schedule_ref[s] = int(d)
                ev_sym = eval_model(
                    model_fx,
                    sweep_loader,
                    device,
                    sites,
                    used_bounds_ref,
                    baseline_logits_cpu=base_sweep_logits, collect_stats=False
                )
                mse_sym_by_deg[int(d)] = float(ev_sym.logit_mse)

                if (di % 10 == 0) or (di == len(cand_degs)):
                    print(f"[dp] site {si:02d}/{len(sites)} {s} | sym done {di:02d}/{len(cand_degs)}")

            # pick best per degree
            for d in cand_degs:
                d = int(d)
                mse_min = float(mse_min_by_deg[d])
                mse_sym = float(mse_sym_by_deg[d])

                best_scale = "minmax"
                best_mse = mse_min
                if ENABLE_MIXED_SCALING_IN_DP and (mse_sym < mse_min):
                    best_scale = "symmetric"
                    best_mse = mse_sym

                per_site_error[s][d] = best_mse
                per_site_best_scale[s][d] = best_scale

                dp_rows_best.append({"site": s, "degree": d, "mse": best_mse})
                dp_rows_full.append({
                    "site": s, "degree": d,
                    "mse_minmax": mse_min, "mse_symmetric": mse_sym,
                    "mse_best": best_mse, "best_scale": best_scale,
                })

            # restore this site's bounds to minmax for subsequent phases
            used_bounds_ref[s] = ref_bounds.get(s, (-1.0, 1.0))
            if m_site is not None:
                m_site.rebuild_coeffs()

        # Keep the original CSV name for compatibility (best-MSE table).
        save_csv(dp_rows_best, out_dir / "dp_one_layer_error_table_no_clip.csv")
        # Also save a verbose table with both scaling modes.
        save_csv(dp_rows_full, out_dir / "dp_one_layer_error_table_no_clip_mixed_scaling.csv")

        print(f"[csv] saved one-layer error table to: {out_dir / 'dp_one_layer_error_table_no_clip.csv'}")

    assert per_site_error is not None
    assert per_site_best_scale is not None

    importance_rank = compute_site_importance_rank(per_site_error, sites)


    # -------------------------------------------------------------------------
    # Budget sweep: compare DP schedule vs best UNIFORM schedule under each budget.
    # Record (if available):
    #   - val_obj (penalized), val_acc
    #   - test_acc, test_logit_mse (report-only; requires ENABLE_STAGE_TEST_EVAL)
    # Save JSON to out_dir / BUDGET_SWEEP_JSON_NAME.
    # -------------------------------------------------------------------------
    if ENABLE_BUDGET_SWEEP:
        import json as _json

        results: List[Dict[str, Any]] = []
        budgets = list(range(int(BUDGET_SWEEP_START), int(BUDGET_SWEEP_END) + 1))

        # scaling mode chosen per (site,deg) in DP proxy table
        scale_mode_ref: Dict[str, str] = {s: "minmax" for s in sites}

        for budget in budgets:
            print("\n" + "=" * 120)
            print(f"[sweep] TOTAL_BUDGET={budget}")
            print("=" * 120)

            # --------------------------
            # DP schedule
            # --------------------------
            dp_schedule = dp_allocate_schedule(sites, CANDIDATE_DEGREES, per_site_error, budget)
            dp_cost = schedule_cost(dp_schedule)
            print(f"[dp] schedule cost={dp_cost}/{budget}")

            apply_schedule_and_scaling(
                model_fx=model_fx,
                sites=sites,
                schedule_ref=schedule_ref,
                used_bounds_ref=used_bounds_ref,
                schedule=dp_schedule,
                ref_bounds=ref_bounds,
                sym_bounds=sym_bounds,
                per_site_best_scale=per_site_best_scale,
                scale_mode_ref=scale_mode_ref,
            )

            ev_dp_val = eval_model(
                model_fx, val_loader, device, sites, used_bounds_ref,
                baseline_logits_cpu=base_val_logits, collect_stats=True
            )
            pen_dp_val = penalize_and_guard(ev_dp_val, ref_bounds)
            dp_val_obj = float(pen_dp_val.obj)
            dp_val_acc = float(ev_dp_val.acc)
            print(f"[dp] VAL: obj={dp_val_obj:.6e} acc={_fmt_pct(dp_val_acc)}")

            dp_test_acc = None
            dp_test_logit_mse = None
            if ENABLE_STAGE_TEST_EVAL and (base_test_logits is not None) and (base_test_acc is not None):
                ev_dp_test = eval_model(
                    model_fx, test_loader, device, sites, used_bounds_ref,
                    baseline_logits_cpu=base_test_logits, collect_stats=False
                )
                dp_test_acc = float(ev_dp_test.acc)
                dp_test_logit_mse = float(ev_dp_test.logit_mse)
                print(f"[test-full:dp] acc={_fmt_pct(dp_test_acc)} | logit_mse={dp_test_logit_mse:.6e}")


            calib_report_rows: List[Dict[str, Any]] = []
            for s in sites:
                deg = int(dp_schedule.get(s, 0))

                if s in ref_bounds:
                    ref_a, ref_b = ref_bounds[s]
                else:
                    ref_a, ref_b = (float("nan"), float("nan"))

                if s in used_bounds_ref:
                    used_a, used_b = used_bounds_ref[s]
                else:
                    used_a, used_b = (float("nan"), float("nan"))

                st = None
                try:
                    st = ev_dp_test.stats.get(s) if (ev_dp_test is not None and ev_dp_test.stats is not None) else None
                except Exception:
                    st = None

                abs_max = float(st.abs_max) if st is not None else float("nan")
                t_max = float(st.t_max) if st is not None else float("nan")
                oor_pct = float(st.oor_rate * 100.0) if st is not None else float("nan")

                calib_report_rows.append(
                    {
                        "site": s,
                        "deg": deg,
                        "ref_a": float(ref_a),
                        "ref_b": float(ref_b),
                        "used_a": float(used_a),
                        "used_b": float(used_b),
                        "abs_max": abs_max,
                        "t_max": t_max,
                        "oor_pct": oor_pct,
                    }
                )

            # print_final_schedule_bounds_table(calib_report_rows)
            repo_root = find_repo_root(Path(__file__).parent)
            out_dir = (repo_root / OUTPUT_DIR_REL / Path(f"sweep_schedules/budget_{budget}")).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            save_csv(calib_report_rows, out_dir / 'final_schedule_with_scaling_and_bounds.csv')

            # # --------------------------
            # # UNIFORM schedules (deg choice)
            # #   candidates are filtered by budget in uniform_candidate_list()
            # #   best selection: (val_obj asc) then (val_acc desc) then (deg asc)
            # # --------------------------
            # uniform_all: List[Dict[str, Any]] = []
            # best_uniform: Optional[Dict[str, Any]] = None
            #
            # uni_deg_list = uniform_candidate_list(sites, CANDIDATE_DEGREES, budget) if ENABLE_UNIFORM_CANDIDATES else []
            # if not uni_deg_list:
            #     print("[uniform] no feasible uniform degrees under this budget.")
            #
            # for ud in uni_deg_list:
            #     ud = int(ud)
            #     uni_sched = build_uniform_schedule(sites, ud)
            #     uni_cost = schedule_cost(uni_sched)
            #     if uni_cost > budget:
            #         continue  # safety
            #
            #     apply_schedule_and_scaling(
            #         model_fx=model_fx,
            #         sites=sites,
            #         schedule_ref=schedule_ref,
            #         used_bounds_ref=used_bounds_ref,
            #         schedule=uni_sched,
            #         ref_bounds=ref_bounds,
            #         sym_bounds=sym_bounds,
            #         per_site_best_scale=per_site_best_scale,
            #         scale_mode_ref=scale_mode_ref,
            #     )
            #
            #     ev_u_val = eval_model(
            #         model_fx, val_loader, device, sites, used_bounds_ref,
            #         baseline_logits_cpu=base_val_logits, collect_stats=True
            #     )
            #     pen_u_val = penalize_and_guard(ev_u_val, ref_bounds)
            #     u_val_obj = float(pen_u_val.obj)
            #     u_val_acc = float(ev_u_val.acc)
            #
            #     u_test_acc = None
            #     u_test_logit_mse = None
            #     if ENABLE_STAGE_TEST_EVAL and (base_test_logits is not None) and (base_test_acc is not None):
            #         ev_u_test_full = eval_on_test_full(
            #             stage="uniform",
            #             model=model_fx,
            #             loader=test_loader,
            #             device=device,
            #             sites=sites,
            #             schedule_ref=schedule_ref,
            #             used_bounds_ref=used_bounds_ref,
            #             schedule=uni_sched,
            #             base_logits=base_test_logits,
            #             base_acc=base_test_acc,
            #         )
            #         if ev_u_test_full is not None:
            #             u_test_acc = float(ev_u_test_full.acc)
            #             u_test_logit_mse = float(ev_u_test_full.logit_mse)
            #
            #     rec = {
            #         "deg": int(ud),
            #         "cost": int(uni_cost),
            #         "val_obj": u_val_obj,
            #         "val_acc": u_val_acc,
            #         "test_acc": u_test_acc,
            #         "test_logit_mse": u_test_logit_mse,
            #     }
            #     if SAVE_ALL_UNIFORM_PER_BUDGET:
            #         uniform_all.append(dict(rec))
            #
            #     if best_uniform is None:
            #         best_uniform = dict(rec)
            #     else:
            #         better_obj = rec["val_obj"] < best_uniform["val_obj"] - 1e-18
            #         tie_obj = abs(rec["val_obj"] - best_uniform["val_obj"]) <= 1e-18
            #         better_acc = rec["val_acc"] > best_uniform["val_acc"] + 1e-18
            #         tie_acc = abs(rec["val_acc"] - best_uniform["val_acc"]) <= 1e-18
            #         better_deg = rec["deg"] < best_uniform["deg"]
            #         if better_obj or (tie_obj and (better_acc or (tie_acc and better_deg))):
            #             best_uniform = dict(rec)
            #
            # if best_uniform is not None:
            #     print(f"[uniform] BEST(deg={best_uniform['deg']}) VAL: obj={best_uniform['val_obj']:.6e} acc={_fmt_pct(best_uniform['val_acc'])}")

            results.append({
                "total_budget": int(budget),
                "dp": {
                    "cost": int(dp_cost),
                    "val_obj": dp_val_obj,
                    "val_acc": dp_val_acc,
                    "test_acc": dp_test_acc,
                    "test_logit_mse": dp_test_logit_mse,
                },
                # "uniform_best": best_uniform,
                # "uniform_all": (uniform_all if SAVE_ALL_UNIFORM_PER_BUDGET else None),
            })

        out_json = out_dir / BUDGET_SWEEP_JSON_NAME
        payload = {
            "meta": {
                "dataset": str(DATASET_NAME),
                "model": str(MODEL_NAME),
                "seed": int(GLOBAL_SEED),
                "candidate_degrees": [int(x) for x in CANDIDATE_DEGREES],
                "uniform_candidate_degs": [int(x) for x in UNIFORM_CANDIDATE_DEGS],
                "budget_range": [int(BUDGET_SWEEP_START), int(BUDGET_SWEEP_END)],
                "reuse_dp_one_layer_error_table": bool(REUSE_DP_ONE_LAYER_ERROR_TABLE),
                "dp_one_layer_error_table_best": str(out_dir / DP_ONE_LAYER_ERROR_TABLE_BEST_NAME),
                "dp_one_layer_error_table_mixed": str(out_dir / DP_ONE_LAYER_ERROR_TABLE_MIXED_NAME),
            },
            "rows": results,
        }
        with out_json.open("w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[json] saved budget sweep DP-vs-UNIFORM table to: {out_json}")
        return

    # return
    # DP schedule
    dp_schedule = dp_allocate_schedule(sites, CANDIDATE_DEGREES, per_site_error, TOTAL_BUDGET)
    print(f"[dp] schedule cost={schedule_cost(dp_schedule)}/{TOTAL_BUDGET}")

    cost_buckets = build_cost_buckets(CANDIDATE_DEGREES)

    schedule_ref.update(dp_schedule)

    # Apply mixed scaling choice selected in DP proxy table (per-site).
    scale_mode_ref: Dict[str, str] = {s: 'minmax' for s in sites}
    if ENABLE_MIXED_SCALING_IN_DP:
        n_sym = 0
        for s in sites:
            d = int(dp_schedule.get(s, 0))
            mode = str(per_site_best_scale.get(s, {}).get(d, 'minmax'))
            scale_mode_ref[s] = mode
            if mode == 'symmetric':
                used_bounds_ref[s] = sym_bounds.get(s, ref_bounds.get(s, (-1.0, 1.0)))
                n_sym += 1
            else:
                used_bounds_ref[s] = ref_bounds.get(s, (-1.0, 1.0))
        rebuild_all_site_coeffs(model_fx)
        print(f'[dp] mixed scaling enabled: symmetric_sites={n_sym}/{len(sites)}')
    else:
        used_bounds_ref.clear()
        used_bounds_ref.update(ref_bounds)
        rebuild_all_site_coeffs(model_fx)
    ev_dp_val = eval_model(model_fx, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=base_val_logits,
                           collect_stats=True)
    pen_dp_val = penalize_and_guard(ev_dp_val, ref_bounds)
    dp_val_acc = float(ev_dp_val.acc)
    print(f"[dp] VAL: obj={pen_dp_val.obj:.6e} acc={_fmt_pct(dp_val_acc)}")

    # Decide START schedule for joint/global: if DP violates FULL guard, prefer a feasible uniform schedule.
    start_name = "DP"
    start_schedule = dict(dp_schedule)

    # FULL guard check for DP on full calibration set
    ev_dp_full = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                            collect_stats=True)
    pen_dp_full = penalize_and_guard(ev_dp_full, ref_bounds)
    ok_dp_full, sg_dp_full = full_guard_check(
        pen_full=pen_dp_full,
        model=model_fx,
        loader=calib_loader,
        device=device,
        sites=sites,
        used_bounds=used_bounds_ref,
        ref_bounds=ref_bounds,
    )
    sg_str = ""
    if sg_dp_full is not None:
        sg_str = f" | sg(explode={sg_dp_full['explode_rate'] * 100:.3f}%, oor_any={sg_dp_full['oor_any_rate'] * 100:.2f}%)"
    print(
        f"[dp] FULL guard: strict_ok={pen_dp_full.ok} relaxed_ok={ok_dp_full} | "
        f"abs_x={ev_dp_full.abs_x_global:.2f} t={ev_dp_full.t_global:.2f} "
        f"worst_oor={ev_dp_full.worst_oor * 100:.4f}%{sg_str}"
    )

    # [optional] sample-level explosion diagnostic on CALIB for DP schedule
    if ENABLE_SAMPLE_GUARD_REPORT and SAMPLE_GUARD_REPORT_AT_DP_FULL:
        if (not SAMPLE_GUARD_ONLY_ON_FAIL) or (not ok_dp_full):
            print("[dp] running SAMPLE-GUARD per-image diagnostic on CALIB (DP schedule) ...")
            _ = sample_guard_explosion_report(
                model_fx, calib_loader, device, sites, used_bounds_ref, ref_bounds,
                tag="dp_full",
                out_dir=out_dir,
                max_print=SAMPLE_GUARD_MAX_OFFENDERS_PRINT,
                save_csv_flag=SAMPLE_GUARD_SAVE_CSV,
                count_oor_any=SAMPLE_GUARD_COUNT_OOR_ANY,
                include_oor_in_explode=SAMPLE_GUARD_INCLUDE_OOR_IN_EXPLODE,
            )

    # Decide START schedule for joint/global: if DP violates (relaxed) FULL guard, prefer a feasible uniform schedule.
    if not ok_dp_full:
        best_uni = None  # (val_obj, val_acc, deg, schedule, ev_full)
        for ud in uniform_candidate_list(sites, CANDIDATE_DEGREES, TOTAL_BUDGET):
            uni = build_uniform_schedule(sites, ud)
            schedule_ref.update(uni)
            ev_full_u = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                   collect_stats=True)
            pen_full_u = penalize_and_guard(ev_full_u, ref_bounds)
            ok_full_u, _sg_u = full_guard_check(
                pen_full=pen_full_u,
                model=model_fx,
                loader=calib_loader,
                device=device,
                sites=sites,
                used_bounds=used_bounds_ref,
                ref_bounds=ref_bounds,
            )
            if not ok_full_u:
                continue

            # If feasible, rank by VAL penalized objective
            ev_val_u = eval_model(model_fx, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=base_val_logits,
                                  collect_stats=True)
            pen_val_u = penalize_and_guard(ev_val_u, ref_bounds)
            print(f"[start] UNIFORM(deg={ud}) FULL ok. VAL: obj={pen_val_u.obj:.6e} acc={_fmt_pct(ev_val_u.acc)}")
            cand = (float(pen_val_u.obj), float(ev_val_u.acc), int(ud), dict(uni), ev_full_u)
            if (best_uni is None) or (cand[0] < best_uni[0]):
                best_uni = cand

        if best_uni is not None:
            _obj, _acc, ud, uni_sched, ev_full_u = best_uni
            start_name = f"UNIFORM(deg={ud})"
            start_schedule = dict(uni_sched)
            print(f"[start] choose {start_name} as start (DP infeasible under relaxed FULL guard).")

    # Report-only TEST eval (fast + full) after DP is ready.
    # IMPORTANT: this information must NOT be used for schedule/config iteration.
    eval_on_test_fast(
        stage="after_dp",
        model=model_fx,
        loader=test_fast_loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=dp_schedule,
        base_logits=base_test_fast_logits,
        base_acc=base_test_fast_acc,
    )
    eval_on_test_full(
        stage="after_dp",
        model=model_fx,
        loader=test_loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=dp_schedule,
        base_logits=base_test_logits,
        base_acc=base_test_acc,
    )
    # joint
    print("[joint] running joint search (swap/tri) with GATED FULL guard ...")
    joint_schedule, ev_full_cur = joint_search(
        model_fx, device, sites, schedule_ref, used_bounds_ref, ref_bounds,
        cost_buckets, TOTAL_BUDGET,
        fast_loader, base_fast_logits,
        val_loader, base_val_logits,
        dp_val_acc,
        calib_loader,
        JOINT_MAX_STEPS,
        rng,
        start_schedule,
        importance_rank
    )

    refined_schedule = dict(joint_schedule)

    # Report-only TEST eval after JOINT search (do NOT use for decisions)
    eval_on_test_fast(
        stage="after_joint",
        model=model_fx,
        loader=test_fast_loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=refined_schedule,
        base_logits=base_test_fast_logits,
        base_acc=base_test_fast_acc,
    )
    eval_on_test_full(
        stage="after_joint",
        model=model_fx,
        loader=test_loader,
        device=device,
        sites=sites,
        schedule_ref=schedule_ref,
        used_bounds_ref=used_bounds_ref,
        schedule=refined_schedule,
        base_logits=base_test_logits,
        base_acc=base_test_acc,
    )

    # global refinement (multi-start): start from DP / JOINT / UNIFORM to avoid being stuck in a single basin.
    best_obj = float("inf")
    best_acc = 0.0
    global_start_cands: List[Dict[str, Any]] = []

    if RUN_GLOBAL_REFINEMENT:
        print("[global] running global refinement (kick + fast-CD + (optional) beam) ...")

        start_list: List[Tuple[str, Dict[str, int]]] = []
        if ENABLE_MULTI_START_GLOBAL:
            if MULTI_START_INCLUDE_DP:
                start_list.append(("GLOBAL_FROM_DP", dict(dp_schedule)))
            if MULTI_START_INCLUDE_JOINT:
                start_list.append(("GLOBAL_FROM_JOINT", dict(joint_schedule)))
            if MULTI_START_INCLUDE_UNIFORM:
                for ud in uniform_candidate_list(sites, CANDIDATE_DEGREES, TOTAL_BUDGET):
                    start_list.append((f"GLOBAL_FROM_UNIFORM(deg={ud})", build_uniform_schedule(sites, ud)))
        else:
            start_list.append(("GLOBAL_FROM_CUR", dict(refined_schedule)))

        branch_results: List[Dict[str, Any]] = []
        for st_name, st_sched in start_list:
            schedule_ref.update(st_sched)
            ev_full_start = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                       collect_stats=True)

            sched_out, ev_full_out, _obj_out, _acc_out = global_refinement(
                model_fx, device, sites, schedule_ref, used_bounds_ref, ref_bounds,
                cost_buckets, TOTAL_BUDGET,
                fast_loader, base_fast_logits,
                val_loader, base_val_logits,
                dp_val_acc,
                calib_loader,
                rng,
                importance_rank,
                ev_full_start
            )

            # FULL feasibility check (strict + relaxed)
            schedule_ref.update(sched_out)
            ev_full_chk = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                     collect_stats=True)
            pen_full_chk = penalize_and_guard(ev_full_chk, ref_bounds)
            ok_full_chk, sg_chk = full_guard_check(
                pen_full=pen_full_chk,
                model=model_fx,
                loader=calib_loader,
                device=device,
                sites=sites,
                used_bounds=used_bounds_ref,
                ref_bounds=ref_bounds,
            )

            # VAL metrics for Pareto selection among starts
            ev_val_chk = eval_model(model_fx, val_loader, device, sites, used_bounds_ref,
                                    baseline_logits_cpu=base_val_logits, collect_stats=True)
            pen_val_chk = penalize_and_guard(ev_val_chk, ref_bounds)

            branch_results.append({
                "name": st_name,
                "schedule": dict(sched_out),
                "val_obj": float(pen_val_chk.obj),
                "val_acc": float(ev_val_chk.acc),
                "full_ok": bool(ok_full_chk),
                "full_strict_ok": bool(pen_full_chk.ok),
                "full_ev": ev_full_chk,
                "sg": sg_chk,
            })

        feasible_br = [b for b in branch_results if b["full_ok"]]
        if feasible_br:
            objs = [float(b["val_obj"]) for b in feasible_br]
            accs = [float(b["val_acc"]) for b in feasible_br]
            front_idx = pareto_front_indices_obj_acc(objs, accs, obj_eps_rel=float(PARETO_OBJ_EPS_REL),
                                                     acc_eps=float(PARETO_ACC_EPS))
            pareto_set = [feasible_br[i] for i in front_idx]
            if (len(pareto_set) > 1) and (rng.random() < float(PARETO_UNIFORM_PICK_PROB)):
                chosen_br = pareto_set[int(rng.integers(0, len(pareto_set)))]
            else:
                pareto_set.sort(key=lambda b: (-float(b["val_acc"]), float(b["val_obj"])))
                chosen_br = pareto_set[0]

            refined_schedule = dict(chosen_br["schedule"])
            ev_full_cur = chosen_br["full_ev"]
            best_obj = float(chosen_br["val_obj"])
            best_acc = float(chosen_br["val_acc"])
            print(
                f"[global-multistart] chose {chosen_br['name']} | val_obj={best_obj:.6e} acc={_fmt_pct(best_acc)} | full_strict_ok={chosen_br['full_strict_ok']}")
        else:
            print(
                "[global-multistart] WARNING: no FULL-feasible schedule among multi-start results; keep current schedule.")
            schedule_ref.update(refined_schedule)
            ev_full_cur = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                     collect_stats=True)
            ev_tmp = eval_model(model_fx, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=base_val_logits,
                                collect_stats=True)
            pen_tmp = penalize_and_guard(ev_tmp, ref_bounds)
            best_obj = float(pen_tmp.obj)
            best_acc = float(ev_tmp.acc)

        global_start_cands = branch_results

        # Report-only TEST eval after GLOBAL refinement (do NOT use for decisions)
        eval_on_test_fast(
            stage="after_global",
            model=model_fx,
            loader=test_fast_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=refined_schedule,
            base_logits=base_test_fast_logits,
            base_acc=base_test_fast_acc,
        )
        eval_on_test_full(
            stage="after_global",
            model=model_fx,
            loader=test_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=refined_schedule,
            base_logits=base_test_logits,
            base_acc=base_test_acc,
        )
        # local window
        if ENABLE_LOCAL_WINDOW_SEARCH:
            print("[local-window] running local window exact/beam refinement ...")
            schedule_ref.update(refined_schedule)
            refined_schedule, ev_full_cur = local_window_refinement(
                model_fx, device, sites, schedule_ref, used_bounds_ref, ref_bounds,
                per_site_error, TOTAL_BUDGET,
                fast_loader, base_fast_logits,
                val_loader, base_val_logits,
                dp_val_acc,
                calib_loader,
                rng,
                ev_full_cur
            )

        # Report-only TEST eval after LOCAL-window refinement (do NOT use for decisions)
        eval_on_test_fast(
            stage="after_local",
            model=model_fx,
            loader=test_fast_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=refined_schedule,
            base_logits=base_test_fast_logits,
            base_acc=base_test_fast_acc,
        )
        eval_on_test_full(
            stage="after_local",
            model=model_fx,
            loader=test_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=refined_schedule,
            base_logits=base_test_logits,
            base_acc=base_test_acc,
        )
        # outer loop: optional re-calibration of USED bounds under the *current* schedule (CALIB), then refine again.
        if ENABLE_OUTER_RECALIB and OUTER_RECALIB_ITERS > 0:
            for it in range(1, OUTER_RECALIB_ITERS + 1):
                print(
                    f"[outer] iter {it}/{OUTER_RECALIB_ITERS}: collect bounds on CALIB under current schedule, then refine ...")

                # Re-collect bounds under current schedule (CALIB)
                schedule_ref.update(refined_schedule)
                used_bounds_new = collect_bounds_from_loader(
                    model_fx,
                    calib_loader,
                    device,
                    sites,
                    mode=BOUNDS_MODE,
                    margin_frac=BOUNDS_MARGIN_FRAC,
                    q_low=BOUNDS_Q_LOW,
                    q_high=BOUNDS_Q_HIGH,
                )
                used_bounds_ref.clear()
                used_bounds_ref.update(used_bounds_new)

                # Rebuild coeff caches (safe even if coeffs are bounds-independent)
                rebuild_all_site_coeffs(model_fx)

                # Refresh FULL stats for risk proposal
                ev_full_cur = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                         collect_stats=True)

                # Re-run refinement under updated bounds
                if RUN_GLOBAL_REFINEMENT:
                    refined_schedule, ev_full_cur, best_obj, best_acc = global_refinement(
                        model_fx,
                        device,
                        sites,
                        schedule_ref,
                        used_bounds_ref,
                        ref_bounds,
                        cost_buckets,
                        TOTAL_BUDGET,
                        fast_loader,
                        base_fast_logits,
                        val_loader,
                        base_val_logits,
                        dp_val_acc,
                        calib_loader,
                        rng,
                        importance_rank,
                        ev_full_cur,
                    )
                    schedule_ref.update(refined_schedule)

                if ENABLE_LOCAL_WINDOW_SEARCH:
                    print("[outer] local window refinement ...")
                    refined_schedule, ev_full_cur = local_window_refinement(
                        model_fx,
                        device,
                        sites,
                        schedule_ref,
                        used_bounds_ref,
                        ref_bounds,
                        per_site_error,
                        TOTAL_BUDGET,
                        fast_loader,
                        base_fast_logits,
                        val_loader,
                        base_val_logits,
                        dp_val_acc,
                        calib_loader,
                        rng,
                        ev_full_cur,
                    )
                    schedule_ref.update(refined_schedule)

                # Report-only TEST eval after OUTER iteration (do NOT use for schedule/config decisions)
                eval_on_test_fast(
                    stage=f"after_outer{it}",
                    model=model_fx,
                    loader=test_fast_loader,
                    device=device,
                    sites=sites,
                    schedule_ref=schedule_ref,
                    used_bounds_ref=used_bounds_ref,
                    schedule=refined_schedule,
                    base_logits=base_test_fast_logits,
                    base_acc=base_test_fast_acc,
                )
                eval_on_test_full(
                    stage=f"after_outer{it}",
                    model=model_fx,
                    loader=test_loader,
                    device=device,
                    sites=sites,
                    schedule_ref=schedule_ref,
                    used_bounds_ref=used_bounds_ref,
                    schedule=refined_schedule,
                    base_logits=base_test_logits,
                    base_acc=base_test_acc,
                )
        # Candidate pool: DP / JOINT / REF (+ optional UNIFORM)
        candidates: List[Tuple[str, Dict[str, int]]] = [
            ("DP", dict(dp_schedule)),
            ("JOINT", dict(joint_schedule)),
            ("REF", dict(refined_schedule)),
        ]
        # add multi-start global candidates (if any)
        if 'global_start_cands' in locals():
            for _b in global_start_cands:
                candidates.append((_b["name"], dict(_b["schedule"])))
        if ENABLE_UNIFORM_CANDIDATES:
            for ud in uniform_candidate_list(sites, CANDIDATE_DEGREES, TOTAL_BUDGET):
                candidates.append((f"UNIFORM(deg={ud})", build_uniform_schedule(sites, ud)))

        # Dedup by schedule content
        seen_sched = set()
        uniq_cands: List[Tuple[str, Dict[str, int]]] = []
        for nm, sc in candidates:
            key = tuple(sorted(sc.items()))
            if key in seen_sched:
                continue
            seen_sched.add(key)
            uniq_cands.append((nm, sc))
        candidates = uniq_cands

        cand_info: List[Dict[str, Any]] = []
        feasible: List[Dict[str, Any]] = []

        for name, sched in candidates:
            schedule_ref.update(sched)

            # VAL metrics (used for Pareto selection; never uses TEST)
            evv = eval_model(model_fx, val_loader, device, sites, used_bounds_ref, baseline_logits_cpu=base_val_logits,
                             collect_stats=True)
            penv = penalize_and_guard(evv, ref_bounds)

            # FULL guard on CALIB
            ev_full = eval_model(model_fx, calib_loader, device, sites, used_bounds_ref, baseline_logits_cpu=None,
                                 collect_stats=True)
            pen_full = penalize_and_guard(ev_full, ref_bounds)

            ok_full, sg = full_guard_check(
                pen_full=pen_full,
                model=model_fx,
                loader=calib_loader,
                device=device,
                sites=sites,
                used_bounds=used_bounds_ref,
                ref_bounds=ref_bounds,
            )
            full_pen = float(pen_full.obj)

            sg_str = ""
            if sg is not None:
                sg_str = f" | sg(explode={sg['explode_rate'] * 100:.3f}%, oor_any={sg['oor_any_rate'] * 100:.2f}%)"

            print(
                f"[cand] {name:<16s}: "
                f"val_obj={penv.obj:.6e} val_acc={_fmt_pct(evv.acc)} | "
                f"full_strict_ok={pen_full.ok} full_relaxed_ok={ok_full} full_pen={full_pen:.3e} "
                f"abs_x={ev_full.abs_x_global:.2f} t={ev_full.t_global:.2f} worst_oor={ev_full.worst_oor * 100:.4f}%{sg_str}"
            )

            info = {
                "name": name,
                "val_obj": float(penv.obj),
                "val_acc": float(evv.acc),
                "schedule": dict(sched),
                "full": ev_full,
                "full_strict_ok": bool(pen_full.ok),
                "full_relaxed_ok": bool(ok_full),
                "full_pen": float(full_pen),
                "sg": sg,
            }
            cand_info.append(info)
            if ok_full:
                feasible.append(info)

        # Final pick: Pareto selection on VAL among FULL-feasible candidates
        if not feasible:
            # No FULL-feasible candidate found. Pick the least violating one (min full_pen).
            cand_info.sort(key=lambda x: float(x["full_pen"]))
            chosen = cand_info[0]
            final_name = chosen["name"]
            final_schedule = dict(chosen["schedule"])
            final_full = chosen["full"]
            print(
                f"[final] WARNING: no FULL-feasible schedule. Fallback={final_name} with full_pen={chosen['full_pen']:.3e}.")
        else:
            objs = [float(c["val_obj"]) for c in feasible]
            accs = [float(c["val_acc"]) for c in feasible]
            front_idx = pareto_front_indices_obj_acc(objs, accs, obj_eps_rel=float(PARETO_OBJ_EPS_REL),
                                                     acc_eps=float(PARETO_ACC_EPS))
            pareto_set = [feasible[i] for i in front_idx]

            if (len(pareto_set) > 1) and (rng.random() < float(FINAL_RANDOM_PICK_PROB)):
                chosen = pareto_set[int(rng.integers(0, len(pareto_set)))]
            else:
                acc_best = max(float(c["val_acc"]) for c in pareto_set) if pareto_set else 0.0
                tol = float(FINAL_ACC_TOL_PP) / 100.0
                near_best = [c for c in pareto_set if float(c["val_acc"]) >= acc_best - tol]
                if not near_best:
                    near_best = pareto_set

                if FINAL_PREFER_STRICT_WITHIN_TOL:
                    strict_nb = [c for c in near_best if bool(c.get("full_strict_ok", False))]
                    if strict_nb:
                        near_best = strict_nb


                def _sort_key_final(c: Dict[str, Any]) -> Tuple[float, ...]:
                    full_ev = c["full"]
                    if FINAL_TIEBREAK_BY_STABILITY:
                        return (
                            float(c["val_obj"]),
                            float(full_ev.abs_x_global),
                            float(full_ev.t_global),
                            float(full_ev.worst_oor),
                            -float(c["val_acc"]),
                        )
                    else:
                        return (float(c["val_obj"]), -float(c["val_acc"]))


                near_best.sort(key=_sort_key_final)
                chosen = near_best[0]

            final_name = chosen["name"]
            final_schedule = dict(chosen["schedule"])
            final_full = chosen["full"]

            print(
                f"[final] Selected {final_name} from Pareto(VAL): "
                f"val_obj={chosen['val_obj']:.6e} val_acc={_fmt_pct(chosen['val_acc'])} | "
                f"full_strict_ok={chosen['full_strict_ok']} full_pen={chosen['full_pen']:.3e}"
            )
        schedule_ref.update(final_schedule)
        save_csv([{"site": s, "degree": int(final_schedule[s]), "cost": depth_cost_from_ranges(final_schedule[s])} for s in
                  sites], out_dir / "final_schedule_no_clip.csv")
        print(f"[final] choose {final_name} schedule. saved to {out_dir / 'final_schedule_no_clip.csv'}")

        # Extra artifact: store the per-site scaling mode + used bounds (helps reproducibility when mixed scaling is enabled).
        try:
            rows_scal: List[Dict[str, Any]] = []
            for s in sites:
                ra, rb = ref_bounds.get(s, (float('nan'), float('nan')))
                ua, ub = used_bounds_ref.get(s, (float('nan'), float('nan')))
                rows_scal.append({
                    'site': s,
                    'degree': int(final_schedule.get(s, 0)),
                    'scale': str(scale_mode_ref.get(s, 'minmax')),
                    'ref_a': float(ra), 'ref_b': float(rb),
                    'used_a': float(ua), 'used_b': float(ub),
                })
            save_csv(rows_scal, out_dir / 'final_schedule_with_scaling_and_bounds.csv')
        except Exception as _e:
            print(f'[warn] failed to save final_schedule_with_scaling_and_bounds.csv: {_e}')

        # -------------------------------------------------------------------------
        # Print FINAL schedule degrees (for copy/paste / deployment)
        # -------------------------------------------------------------------------
        print("\n" + "=" * 118)
        print(
            f"FINAL schedule degrees (deployment view): {final_name} | cost={schedule_cost(final_schedule)}/{TOTAL_BUDGET}")
        print("=" * 118)
        print(f"{'site':<32s} {'deg':>4s} {'cost':>4s}")
        for s in sites:
            d = int(final_schedule[s])
            c = int(depth_cost_from_ranges(d))
            print(f"{s:<32s} {d:>4d} {c:>4d}")
        print("\n[final-schedule-dict] = {")
        for s in sites:
            print(f"  '{s}': {int(final_schedule[s])},")
        print("}")

        # -------------------------------------------------------------------------
        # FINAL schedule: per-site activation INPUT bounds on CALIB
        #   (ref bounds + used bounds + observed abs/t/oor)
        #
        # We reuse `final_full` (already computed on FULL calib loader) so this does
        # not affect the search and avoids extra passes.
        # -------------------------------------------------------------------------
        try:
            calib_report_rows: List[Dict[str, Any]] = []
            for s in sites:
                deg = int(final_schedule.get(s, 0))

                if s in ref_bounds:
                    ref_a, ref_b = ref_bounds[s]
                else:
                    ref_a, ref_b = (float("nan"), float("nan"))

                if s in used_bounds_ref:
                    used_a, used_b = used_bounds_ref[s]
                else:
                    used_a, used_b = (float("nan"), float("nan"))

                st = None
                try:
                    st = final_full.stats.get(s) if (final_full is not None and final_full.stats is not None) else None
                except Exception:
                    st = None

                abs_max = float(st.abs_max) if st is not None else float("nan")
                t_max = float(st.t_max) if st is not None else float("nan")
                oor_pct = float(st.oor_rate * 100.0) if st is not None else float("nan")

                calib_report_rows.append(
                    {
                        "site": s,
                        "deg": deg,
                        "ref_a": float(ref_a),
                        "ref_b": float(ref_b),
                        "used_a": float(used_a),
                        "used_b": float(used_b),
                        "abs_max": abs_max,
                        "t_max": t_max,
                        "oor_pct": oor_pct,
                    }
                )

            print_final_schedule_bounds_table(calib_report_rows)
        except Exception as _e:
            print(f"[warn] failed to print final bounds table: {_e}")

            # [optional] sample-level explosion diagnostic on CALIB for FINAL schedule
        if ENABLE_SAMPLE_GUARD_REPORT and SAMPLE_GUARD_REPORT_AT_FINAL_FULL:
            strict_ok = bool(chosen.get("full_strict_ok", True))
            relaxed_ok = bool(chosen.get("full_relaxed_ok", True))
            if (not SAMPLE_GUARD_ONLY_ON_FAIL) or (not strict_ok) or (not relaxed_ok):
                print("[final] running SAMPLE-GUARD per-image diagnostic on CALIB (final schedule) ...")
                _ = sample_guard_explosion_report(
                    model_fx, calib_loader, device, sites, used_bounds_ref, ref_bounds,
                    tag="final_full",
                    out_dir=out_dir,
                    max_print=SAMPLE_GUARD_MAX_OFFENDERS_PRINT,
                    save_csv_flag=SAMPLE_GUARD_SAVE_CSV,
                    count_oor_any=SAMPLE_GUARD_COUNT_OOR_ANY,
                    include_oor_in_explode=SAMPLE_GUARD_INCLUDE_OOR_IN_EXPLODE,
                )

        # Report-only TEST eval for the finally selected schedule
        eval_on_test_fast(
            stage="final_selected",
            model=model_fx,
            loader=test_fast_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=final_schedule,
            base_logits=base_test_fast_logits,
            base_acc=base_test_fast_acc,
        )
        eval_on_test_full(
            stage="final_selected",
            model=model_fx,
            loader=test_loader,
            device=device,
            sites=sites,
            schedule_ref=schedule_ref,
            used_bounds_ref=used_bounds_ref,
            schedule=final_schedule,
            base_logits=base_test_logits,
            base_acc=base_test_acc,
        )

        print(f"[done] outputs in: {out_dir}")

if __name__ == "__main__":
    main()
