#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S5 — convergence of confined Z(L) and DeltaF(L)
for the final Domb–Joyce partition-function estimator.

Revision target
---------------
This version is aligned with the adopted manuscript workflow and Reviewer 1/2
methodology:

1. The canonical unconfined R_g values are read from Rg_MASTER_FINAL.csv.
   No independent R_g simulation is performed here.
2. Z(L) is estimated by independent block simulations using the same
   lattice Domb–Joyce ensemble used by the force calculations.
3. Uncertainty is block-based. Low-ESS/weight-degeneracy diagnostics are
   retained in the output and are never used to delete individual points.
4. DeltaF(L) is defined relative to an explicitly stated large-but-finite
   reference width L_ref for each w. It is therefore a convergence/reference
   quantity, not an unconfined-limit claim.
5. All convergence points are retained. No smoothing, fitting, sign
   selection, or manual outlier removal is performed.
6. The exact lattice convention is recorded: even integer wall separation L,
   absorbing walls at x=0 and x=L, accessible layers 1,...,L-1, and the
   tether at x=L/2.
7. The provenance report records the exact master-R_g rows, master-file hash,
   simulation settings, block diagnostics, and pairwise convergence tests.

Estimator
---------
Each uniformly generated lattice walk has proposal probability
6^{-(N-1)}. For a Domb–Joyce overlap count U,

    W = 6^(N-1) exp(-w U),

so the sample mean of W estimates the confined partition sum Z(L) up to the
same common normalization used consistently across widths. We report

    ln Z(L) = ln[ mean(W) ]

using independent blocks. Within each block the effective sample size (ESS)
is computed from the normalized importance weights.

Free-energy difference
----------------------

    DeltaF(L)/(k_B T) = ln Z(L_ref) - ln Z(L).

L_ref is deliberately finite and is labelled as such in the figure/report.

Typical usage on Kaggle
-----------------------
python FigS5_Z_DeltaF_Convergence_MASTER_REVISED_FINAL.py \
  --rg-master /kaggle/input/datasets/soumyajyotikabi/n200-rg/Rg_MASTER_FINAL.csv \
  --production --nproc 4

Smoke test
----------
python FigS5_Z_DeltaF_Convergence_MASTER_REVISED_FINAL.py \
  --rg-master /kaggle/input/datasets/soumyajyotikabi/n200-rg/Rg_MASTER_FINAL.csv \
  --smoke --nproc 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_NAME = "FigS5_Z_DeltaF_Convergence_MASTER_REVISED_FINAL.py"
SCRIPT_VERSION = "1.0"

# -----------------------------------------------------------------------------
# Canonical settings
# -----------------------------------------------------------------------------
N_DEFAULT = 200
W_DEFAULT = [0.0, 0.3, 0.5]
ATTEMPTS_DEFAULT = [6000, 15000, 30000, 60000]
N_BLOCKS_DEFAULT = 30

SMOKE_ATTEMPTS = [600, 1200, 2400]
SMOKE_BLOCKS = 6
SMOKE_W = [0.0, 0.5]
SMOKE_L = [12, 16, 20]

L_MIN_QUANTITATIVE = 12
CONVERGENCE_TARGET_L_OVER_RG = 2.5
L_REF_FACTOR = 5.5

# Weight-quality criteria. These are diagnostics, not point-selection rules.
NEFF_THRESHOLD = 20.0
GOOD_BLOCK_FRACTION = 0.80
MAX_WEIGHT_FRACTION_CAUTION = 0.20
MIN_VALID_BLOCKS = 3

DEFAULT_OUT_PREFIX = "FigS5_Z_DeltaF_Convergence_MASTER_REVISED_FINAL"

STEPS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int32,
)

# Publication-ready style used by the main-figure workflow.
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "axes.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def float_equal(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def bootstrap_ci(values, n_boot=5000, seed=123456):
    """Percentile bootstrap CI for a mean; used only as a diagnostic report."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = np.mean(x[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# -----------------------------------------------------------------------------
# Canonical master-Rg loading
# -----------------------------------------------------------------------------
def load_master_rg(master_csv: str | Path, N: int, w_values: list[float]):
    """Load exactly one canonical DJ row per requested (N, w)."""
    master_csv = Path(master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"Master Rg CSV not found: {master_csv}")

    with master_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    required = {"model", "N", "w", "Rg", "Rg_err", "quality_flag"}
    if not rows:
        raise ValueError("Master Rg CSV is empty.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"Master Rg CSV missing required columns: {sorted(missing)}")

    selected = {}
    source_rows = {}
    for target_w in w_values:
        candidates = []
        for row in rows:
            model = str(row.get("model", "")).strip()
            try:
                row_N = int(float(row.get("N", "nan")))
                row_w = float(row.get("w", "nan"))
            except (TypeError, ValueError):
                continue
            if model == "DJ" and row_N == int(N) and float_equal(row_w, target_w):
                candidates.append(row)

        if len(candidates) == 0:
            raise ValueError(
                f"No matching master Rg row found for model=DJ, N={N}, w={target_w:g}."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple matching master Rg rows found for model=DJ, N={N}, w={target_w:g}."
            )

        row = candidates[0]
        Rg = finite_float(row.get("Rg"))
        Rg_err = finite_float(row.get("Rg_err"))
        if not (np.isfinite(Rg) and Rg > 0):
            raise ValueError(f"Invalid master Rg for w={target_w:g}: {Rg}")
        if not (np.isfinite(Rg_err) and Rg_err >= 0):
            raise ValueError(f"Invalid master Rg_err for w={target_w:g}: {Rg_err}")

        selected[target_w] = {"Rg": Rg, "Rg_err": Rg_err}
        source_rows[target_w] = dict(row)

    return selected, source_rows, sha256_file(master_csv)


# -----------------------------------------------------------------------------
# Domb–Joyce chain and confined partition function
# -----------------------------------------------------------------------------
def overlap_count(positions: np.ndarray) -> int:
    """Nonbonded same-site overlap count U for a single chain."""
    occupied = {}
    for i, pos in enumerate(positions):
        key = (int(pos[0]), int(pos[1]), int(pos[2]))
        occupied.setdefault(key, []).append(i)

    count = 0
    for indices in occupied.values():
        m = len(indices)
        if m < 2:
            continue
        for ii in range(m):
            i = indices[ii]
            for jj in range(ii + 1, m):
                j = indices[jj]
                if abs(i - j) > 1:
                    count += 1
    return int(count)


def generate_walk(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one uniformly proposed cubic-lattice walk with N monomers."""
    steps_idx = rng.integers(0, len(STEPS), size=N - 1)
    pos = np.zeros((N, 3), dtype=np.int32)
    if N > 1:
        pos[1:] = np.cumsum(STEPS[steps_idx], axis=0, dtype=np.int32)
    return pos


def one_log_weight(N: int, w: float, L: int, rng: np.random.Generator) -> tuple[float, bool, int]:
    """Return log importance weight, survival flag, and overlap count."""
    pos = generate_walk(N, rng)
    # Exact lattice convention used in the force workflow:
    # absorbing walls at x=0,L; accessible layers are 1,...,L-1;
    # tether is at x=L/2 for even L.
    x = pos[:, 0] + L // 2
    survives = bool(np.all((x >= 1) & (x <= L - 1)))
    if not survives:
        return -np.inf, False, 0

    U = overlap_count(pos)
    logW = (N - 1) * math.log(6.0) - float(w) * float(U)
    return float(logW), True, int(U)


def summarize_block(log_weights: np.ndarray):
    """Summarize one block of fixed attempted-chain count."""
    log_weights = np.asarray(log_weights, dtype=float)
    n = int(log_weights.size)
    finite = np.isfinite(log_weights)
    if n == 0 or not np.any(finite):
        return {
            "logZ": -np.inf,
            "neff": 0.0,
            "max_weight_fraction": 1.0,
            "finite": 0,
            "survival_fraction": 0.0,
        }

    lw = log_weights[finite]
    m = float(np.max(lw))
    scaled = np.exp(lw - m)
    sumw = float(np.sum(scaled))
    sumw2 = float(np.sum(scaled * scaled))
    # Denominator is the TOTAL number of attempted chains, including zero
    # statistical-weight/dead trajectories, as required by the importance
    # sampling estimator.
    logZ = m + math.log(sumw) - math.log(n)
    neff = (sumw * sumw / sumw2) if sumw2 > 0 else 0.0
    max_frac = float(np.max(scaled / sumw)) if sumw > 0 else 1.0

    return {
        "logZ": float(logZ),
        "neff": float(min(neff, n)),
        "max_weight_fraction": max_frac,
        "finite": int(np.sum(finite)),
        "survival_fraction": float(np.sum(finite) / n),
    }


def quality_from_blocks(block_records):
    valid = [r for r in block_records if np.isfinite(r["logZ"])]
    if len(valid) < MIN_VALID_BLOCKS:
        return "UNRESOLVED"

    good_neff = sum(r["neff"] >= NEFF_THRESHOLD for r in valid)
    frac_good = good_neff / len(valid)
    max_frac = max(r["max_weight_fraction"] for r in valid)

    if frac_good >= GOOD_BLOCK_FRACTION and max_frac < MAX_WEIGHT_FRACTION_CAUTION:
        return "OK"
    return "CAUTION"


def block_worker(N, w, L, attempts_per_block, seed):
    rng = np.random.default_rng(seed)
    log_weights = np.full(attempts_per_block, -np.inf, dtype=float)
    for i in range(attempts_per_block):
        log_weights[i], _, _ = one_log_weight(N, w, L, rng)
    return summarize_block(log_weights)


def sample_logZ(
    N: int,
    w: float,
    L: int,
    n_attempts: int,
    n_blocks: int,
    base_seed: int,
    nproc: int = 1,
):
    """Estimate ln Z(L) with independent fixed-size blocks."""
    if L <= 1 or L % 2 != 0:
        raise ValueError(f"L must be an even wall separation >1; received L={L}.")
    if n_attempts < n_blocks:
        raise ValueError("n_attempts must be >= n_blocks.")

    adjusted = n_attempts - (n_attempts % n_blocks)
    if adjusted != n_attempts:
        print(
            f"  [INFO] Adjusting attempts {n_attempts} -> {adjusted} "
            f"for divisibility by {n_blocks}."
        )
    n_attempts = adjusted
    attempts_per_block = n_attempts // n_blocks

    # A deterministic seed is assigned to each physical block by the tuple
    # (base_seed, w, L, n_attempts, block_index), making blocks independent
    # and reproducible irrespective of execution order.
    seeds = []
    for b in range(n_blocks):
        w_code = int(round(float(w) * 10000))
        ss = np.random.SeedSequence(
            [int(base_seed), int(N), w_code, int(L), int(n_attempts), int(b)]
        )
        seeds.append(int(ss.generate_state(1, dtype=np.uint64)[0]))

    t0 = time.time()
    records = []
    if nproc > 1:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = [
                ex.submit(block_worker, N, w, L, attempts_per_block, seed)
                for seed in seeds
            ]
            for fut in as_completed(futures):
                records.append(fut.result())
    else:
        for seed in seeds:
            records.append(block_worker(N, w, L, attempts_per_block, seed))

    # Execution order can differ in parallel mode; all downstream summaries
    # use order-independent statistics.
    logz_vals = np.array([r["logZ"] for r in records], dtype=float)
    valid = logz_vals[np.isfinite(logz_vals)]
    logz_mean = float(np.mean(valid)) if valid.size else np.nan
    logz_sem = (
        float(np.std(valid, ddof=1) / np.sqrt(valid.size)) if valid.size >= 2 else np.nan
    )
    ci_low, ci_high = bootstrap_ci(valid, seed=base_seed + 991)

    quality = quality_from_blocks(records)
    neff = np.array([r["neff"] for r in records], dtype=float)
    maxfrac = np.array([r["max_weight_fraction"] for r in records], dtype=float)
    finite = np.array([r["finite"] for r in records], dtype=float)
    surv = np.array([r["survival_fraction"] for r in records], dtype=float)

    return {
        "N": int(N),
        "w": float(w),
        "L": int(L),
        "n_attempts": int(n_attempts),
        "n_blocks": int(n_blocks),
        "attempts_per_block": int(attempts_per_block),
        "block_logZ": [float(x) for x in logz_vals],
        "block_neff": [float(x) for x in neff],
        "block_max_weight_fraction": [float(x) for x in maxfrac],
        "block_finite": [int(x) for x in finite],
        "block_survival_fraction": [float(x) for x in surv],
        "logZ_mean": logz_mean,
        "logZ_sem": logz_sem,
        "logZ_bootstrap_CI95_low": ci_low,
        "logZ_bootstrap_CI95_high": ci_high,
        "neff_mean": float(np.mean(neff)) if neff.size else 0.0,
        "neff_median": float(np.median(neff)) if neff.size else 0.0,
        "neff_min": float(np.min(neff)) if neff.size else 0.0,
        "max_weight_fraction": float(np.max(maxfrac)) if maxfrac.size else np.nan,
        "mean_survival_fraction": float(np.mean(surv)) if surv.size else 0.0,
        "valid_blocks": int(valid.size),
        "valid_block_fraction": float(valid.size / n_blocks),
        "quality": quality,
        "elapsed_sec": float(time.time() - t0),
    }


# -----------------------------------------------------------------------------
# L-grid and free-energy calculations
# -----------------------------------------------------------------------------
def choose_L_values(Rg: float, w: float, smoke: bool = False) -> list[int]:
    if smoke:
        return [L for L in SMOKE_L if L % 2 == 0]

    # Preserve the established convergence grid used in the original S5 audit
    # while making the Rg dependence explicit and reproducible.
    if float_equal(w, 0.0):
        targets = np.array([10.0 / Rg, 14.0 / Rg, 22.0 / Rg])
        L_vals = [10, 14, 22]
        _ = targets  # retained only to document the Rg-relative grid
    else:
        targets = np.array([0.8, 1.5, 2.5, 4.0], dtype=float)
        L_vals = []
        for x in targets:
            L_float = x * Rg
            L_int = int(2 * round(L_float / 2.0))
            if L_int >= L_MIN_QUANTITATIVE:
                L_vals.append(L_int)

    return sorted(set(int(L) for L in L_vals if L > 1 and L % 2 == 0))


def choose_reference_L(Rg: float, data_L_values: list[int]) -> int:
    L_ref = max(30, int(2 * math.ceil((L_REF_FACTOR * Rg) / 2.0)))
    if L_ref <= max(data_L_values):
        L_ref = max(data_L_values) + 2
    if L_ref % 2:
        L_ref += 1
    return int(L_ref)


def make_deltaF(z_data, z_ref):
    """Independent-reference error propagation for DeltaF/(kBT)."""
    a = float(z_data["logZ_mean"])
    b = float(z_ref["logZ_mean"])
    sa = float(z_data["logZ_sem"])
    sb = float(z_ref["logZ_sem"])
    dF = b - a
    err = math.sqrt(sa * sa + sb * sb) if np.isfinite(sa + sb) else np.nan
    return float(dF), float(err)


def zscore_difference(x1, s1, x2, s2):
    den = math.sqrt(float(s1) ** 2 + float(s2) ** 2)
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(abs(float(x2) - float(x1)) / den)


def choose_convergence_L(rows, target_x=CONVERGENCE_TARGET_L_OVER_RG):
    """Choose the data point closest to target L/Rg for panel (c,d)."""
    candidates = [r for r in rows if r["row_role"] == "DATA"]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r["L_over_Rg"] - target_x))["L"]


# -----------------------------------------------------------------------------
# Study orchestration
# -----------------------------------------------------------------------------
def run_study(N, master_rg, w_values, attempts, n_blocks, seed, nproc, smoke=False):
    rows = []
    raw = {}

    for iw, w in enumerate(w_values):
        Rg = master_rg[w]["Rg"]
        Rg_err = master_rg[w]["Rg_err"]
        L_values = choose_L_values(Rg, w, smoke=smoke)
        if not L_values:
            raise ValueError(f"No L values generated for w={w:g}, Rg={Rg}")
        L_ref = choose_reference_L(Rg, L_values)

        print(
            f"\nw={w:g}: Rg={Rg:.9f} +/- {Rg_err:.9f}; "
            f"L_values={L_values}; L_ref={L_ref}; "
            f"L_ref/Rg={L_ref/Rg:.3f}"
        )

        for n in attempts:
            print(f"  Convergence level n={n:,}")

            # Independent reference run for the same convergence level.
            ref_key = (float(w), int(L_ref), int(n))
            zref = raw.get(ref_key)
            if zref is None:
                zref = sample_logZ(
                    N, w, L_ref, n, n_blocks,
                    base_seed=seed + 100000 * (iw + 1),
                    nproc=nproc,
                )
                raw[ref_key] = zref

            rows.append(
                {
                    "row_role": "REFERENCE",
                    "model": "DJ",
                    "N": int(N),
                    "w": float(w),
                    "L": int(L_ref),
                    "L_ref": int(L_ref),
                    "L_over_Rg": float(L_ref / Rg),
                    "n_attempts": int(n),
                    "logZ": zref["logZ_mean"],
                    "logZ_err": zref["logZ_sem"],
                    "logZ_bootstrap_CI95_low": zref["logZ_bootstrap_CI95_low"],
                    "logZ_bootstrap_CI95_high": zref["logZ_bootstrap_CI95_high"],
                    "DeltaF_over_kBT": 0.0,
                    "DeltaF_err_over_kBT": 0.0,
                    "Neff_mean": zref["neff_mean"],
                    "Neff_median": zref["neff_median"],
                    "Neff_min": zref["neff_min"],
                    "max_weight_fraction": zref["max_weight_fraction"],
                    "mean_survival_fraction": zref["mean_survival_fraction"],
                    "valid_blocks": zref["valid_blocks"],
                    "valid_block_fraction": zref["valid_block_fraction"],
                    "Z_quality": zref["quality"],
                    "DeltaF_quality": "REFERENCE",
                    "Rg_master": Rg,
                    "Rg_master_err": Rg_err,
                }
            )

            for L in L_values:
                key = (float(w), int(L), int(n))
                z = raw.get(key)
                if z is None:
                    z = sample_logZ(
                        N, w, L, n, n_blocks,
                        base_seed=seed + 100000 * (iw + 1) + 1000 * (L // 2),
                        nproc=nproc,
                    )
                    raw[key] = z

                dF, dFerr = make_deltaF(z, zref)
                if z["quality"] == "UNRESOLVED" or zref["quality"] == "UNRESOLVED":
                    df_quality = "UNRESOLVED"
                elif z["quality"] == "OK" and zref["quality"] == "OK":
                    df_quality = "OK"
                else:
                    df_quality = "CAUTION"

                rows.append(
                    {
                        "row_role": "DATA",
                        "model": "DJ",
                        "N": int(N),
                        "w": float(w),
                        "L": int(L),
                        "L_ref": int(L_ref),
                        "L_over_Rg": float(L / Rg),
                        "n_attempts": int(n),
                        "logZ": z["logZ_mean"],
                        "logZ_err": z["logZ_sem"],
                        "logZ_bootstrap_CI95_low": z["logZ_bootstrap_CI95_low"],
                        "logZ_bootstrap_CI95_high": z["logZ_bootstrap_CI95_high"],
                        "DeltaF_over_kBT": dF,
                        "DeltaF_err_over_kBT": dFerr,
                        "Neff_mean": z["neff_mean"],
                        "Neff_median": z["neff_median"],
                        "Neff_min": z["neff_min"],
                        "max_weight_fraction": z["max_weight_fraction"],
                        "mean_survival_fraction": z["mean_survival_fraction"],
                        "valid_blocks": z["valid_blocks"],
                        "valid_block_fraction": z["valid_block_fraction"],
                        "Z_quality": z["quality"],
                        "DeltaF_quality": df_quality,
                        "Rg_master": Rg,
                        "Rg_master_err": Rg_err,
                    }
                )

    return rows, raw


def pairwise_convergence(rows, master_rg):
    out = []
    for w in sorted({float(r["w"]) for r in rows}):
        for L in sorted({int(r["L"]) for r in rows if r["w"] == w and r["row_role"] == "DATA"}):
            sub = sorted(
                [
                    r
                    for r in rows
                    if r["w"] == w and r["L"] == L and r["row_role"] == "DATA"
                ],
                key=lambda r: r["n_attempts"],
            )
            for a, b in zip(sub[:-1], sub[1:]):
                zlog = zscore_difference(a["logZ"], a["logZ_err"], b["logZ"], b["logZ_err"])
                zdf = zscore_difference(
                    a["DeltaF_over_kBT"],
                    a["DeltaF_err_over_kBT"],
                    b["DeltaF_over_kBT"],
                    b["DeltaF_err_over_kBT"],
                )
                rel_logz = 100.0 * abs(b["logZ"] - a["logZ"]) / max(abs(b["logZ"]), 1e-12)
                rel_df = 100.0 * abs(b["DeltaF_over_kBT"] - a["DeltaF_over_kBT"]) / max(
                    abs(b["DeltaF_over_kBT"]), 1e-8
                )
                out.append(
                    {
                        "N": int(a["N"]),
                        "w": float(w),
                        "L": int(L),
                        "L_over_Rg": float(L / master_rg[w]["Rg"]),
                        "n1": int(a["n_attempts"]),
                        "n2": int(b["n_attempts"]),
                        "z_logZ": zlog,
                        "z_DeltaF": zdf,
                        "relative_logZ_change_pct": rel_logz,
                        "relative_DeltaF_change_pct": rel_df,
                        "Neff_min_n1": float(a["Neff_min"]),
                        "Neff_min_n2": float(b["Neff_min"]),
                        "Z_quality_n1": a["Z_quality"],
                        "Z_quality_n2": b["Z_quality"],
                        "DeltaF_quality_n1": a["DeltaF_quality"],
                        "DeltaF_quality_n2": b["DeltaF_quality"],
                    }
                )
    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, out_prefix, master_rg, target_x=CONVERGENCE_TARGET_L_OVER_RG):
    wvals = sorted({float(r["w"]) for r in rows})
    nvals = sorted({int(r["n_attempts"]) for r in rows})
    colors = {0.0: "#440154", 0.3: "#21918c", 0.5: "#cc4778"}
    markers = {6000: "o", 15000: "s", 30000: "^", 60000: "D", 600: "o", 1200: "s", 2400: "^"}

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    fig.subplots_adjust(wspace=0.28, hspace=0.30)

    # (a) ln Z(L): all rows, reference shown explicitly.
    ax = axes[0, 0]
    for w in wvals:
        for n in nvals:
            sub = [r for r in rows if r["w"] == w and r["n_attempts"] == n]
            sub.sort(key=lambda r: r["L"])
            if not sub:
                continue
            label = f"$w={w:g}$, {n//1000}k" if n >= 1000 else f"$w={w:g}$, {n}"
            style = "o-" if n == max(nvals) else markers.get(n, "o")
            if n == max(nvals):
                ax.errorbar(
                    [r["L"] for r in sub],
                    [r["logZ"] for r in sub],
                    yerr=[r["logZ_err"] for r in sub],
                    fmt="o-",
                    ms=5.5,
                    lw=1.4,
                    capsize=2.5,
                    color=colors.get(w),
                    label=label,
                )
            else:
                ax.errorbar(
                    [r["L"] for r in sub],
                    [r["logZ"] for r in sub],
                    yerr=[r["logZ_err"] for r in sub],
                    fmt=markers.get(n, "o"),
                    ms=5.2,
                    ls="none",
                    alpha=0.72,
                    capsize=2.0,
                    color=colors.get(w),
                    label=label,
                )
    ax.set_xlabel(r"Slit width $L/a$")
    ax.set_ylabel(r"$\ln Z(L)$")
    ax.set_title("(a) Partition-function convergence", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2, loc="best")

    # (b) DeltaF/(kBT), excluding finite-width reference rows.
    ax = axes[0, 1]
    for w in wvals:
        for n in nvals:
            sub = [
                r
                for r in rows
                if r["w"] == w and r["n_attempts"] == n and r["row_role"] == "DATA"
            ]
            sub.sort(key=lambda r: r["L"])
            if not sub:
                continue
            if n == max(nvals):
                ax.errorbar(
                    [r["L"] for r in sub],
                    [r["DeltaF_over_kBT"] for r in sub],
                    yerr=[r["DeltaF_err_over_kBT"] for r in sub],
                    fmt="o-",
                    ms=5.5,
                    lw=1.4,
                    capsize=2.5,
                    color=colors.get(w),
                    label=fr"$w={w:g}$",
                )
            else:
                ax.errorbar(
                    [r["L"] for r in sub],
                    [r["DeltaF_over_kBT"] for r in sub],
                    yerr=[r["DeltaF_err_over_kBT"] for r in sub],
                    fmt=markers.get(n, "o"),
                    ms=5.2,
                    ls="none",
                    alpha=0.72,
                    capsize=2.0,
                    color=colors.get(w),
                )
    ax.axhline(0.0, lw=0.9, alpha=0.5)
    ax.set_xlabel(r"Slit width $L/a$")
    ax.set_ylabel(r"$\Delta F(L)/(k_B T)$")
    ax.set_title("(b) Free-energy difference", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, loc="best")

    # (c) ln Z convergence at a comparable Rg-scaled point.
    ax = axes[1, 0]
    chosen_labels = []
    for w in wvals:
        sub_all = [r for r in rows if r["w"] == w and r["row_role"] == "DATA"]
        if not sub_all:
            continue
        L_pick = choose_convergence_L(sub_all, target_x=target_x)
        sub = sorted([r for r in sub_all if r["L"] == L_pick], key=lambda r: r["n_attempts"])
        chosen_labels.append((w, L_pick, sub[0]["L_over_Rg"]))
        ax.errorbar(
            [r["n_attempts"] for r in sub],
            [r["logZ"] for r in sub],
            yerr=[r["logZ_err"] for r in sub],
            fmt="o-",
            ms=5.5,
            lw=1.4,
            capsize=2.5,
            color=colors.get(w),
            label=fr"$w={w:g}$, $L/R_g={sub[0]['L_over_Rg']:.2f}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total attempted chains")
    ax.set_ylabel(r"$\ln Z(L)$")
    ax.set_title(fr"(c) $\ln Z$ convergence near $L/R_g={target_x:g}$", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.5, loc="best")

    # (d) DeltaF convergence at the same representative point.
    ax = axes[1, 1]
    for w, L_pick, _ in chosen_labels:
        sub_all = [
            r
            for r in rows
            if r["w"] == w and r["L"] == L_pick and r["row_role"] == "DATA"
        ]
        sub = sorted(sub_all, key=lambda r: r["n_attempts"])
        ax.errorbar(
            [r["n_attempts"] for r in sub],
            [r["DeltaF_over_kBT"] for r in sub],
            yerr=[r["DeltaF_err_over_kBT"] for r in sub],
            fmt="o-",
            ms=5.5,
            lw=1.4,
            capsize=2.5,
            color=colors.get(w),
            label=fr"$w={w:g}$, $L={L_pick}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total attempted chains")
    ax.set_ylabel(r"$\Delta F(L)/(k_B T)$")
    ax.set_title("(d) Free-energy convergence", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.5, loc="best")

    fig.suptitle(
        "Supplementary Fig. S5 — convergence of $Z(L)$ and $\\Delta F(L)$",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Supplementary Fig. S5: master-Rg-aware Z and DeltaF convergence audit."
    )
    parser.add_argument("--rg-master", required=True, help="Path to Rg_MASTER_FINAL.csv")
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--w-values", type=float, nargs="+", default=W_DEFAULT)
    parser.add_argument("--attempts", type=int, nargs="+", default=ATTEMPTS_DEFAULT)
    parser.add_argument("--blocks", type=int, default=N_BLOCKS_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    if args.N < 2:
        raise ValueError("N must be >= 2.")
    if args.blocks < 3:
        raise ValueError("At least three independent blocks are required.")
    if args.nproc < 1:
        raise ValueError("--nproc must be >= 1.")
    if any(n < args.blocks for n in args.attempts):
        raise ValueError("Every attempts value must be >= number of blocks.")
    if args.smoke and args.production:
        raise ValueError("Use either --smoke or --production, not both.")

    if args.smoke:
        w_values = list(SMOKE_W)
        attempts = list(SMOKE_ATTEMPTS)
        n_blocks = SMOKE_BLOCKS
    else:
        w_values = list(args.w_values)
        attempts = list(args.attempts)
        n_blocks = int(args.blocks)

    # Normalize / validate the requested w values.
    if any(w < 0 for w in w_values):
        raise ValueError("Domb–Joyce strength w must be non-negative.")
    w_values = sorted(w_values)

    master, master_rows, master_hash = load_master_rg(args.rg_master, args.N, w_values)

    print("=" * 88)
    print("SUPPLEMENTARY FIG. S5 — CANONICAL MASTER-Rg Z/DeltaF CONVERGENCE")
    print("=" * 88)
    print(f"Script            : {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"Master CSV        : {args.rg_master}")
    print(f"Master SHA256     : {master_hash}")
    print(f"N                 : {args.N}")
    print(f"DJ w values       : {w_values}")
    print(f"Attempts           : {attempts}")
    print(f"Blocks             : {n_blocks}")
    print(f"Parallel processes : {args.nproc}")
    print("Monte Carlo Rg     : OFF (canonical master values only)")
    print("Smoothing/fitting  : OFF")
    print("Sign selection     : OFF / not applicable")
    print("Lattice             : even integer L; absorbing walls x=0,L; accessible 1...L-1;")
    print("                    tether fixed at x=L/2")
    print("DeltaF reference    : finite L_ref per w; not an unconfined-limit claim")
    print("=" * 88)

    for w in w_values:
        row = master_rows[w]
        print(
            f"w={w:g}: master Rg={float(row['Rg']):.9f} +/- "
            f"{float(row['Rg_err']):.9f}; quality={row.get('quality_flag','')}; "
            f"min_ESS={row.get('min_ESS','nan')}"
        )

    rows, _raw = run_study(
        args.N,
        master,
        w_values,
        attempts,
        n_blocks,
        args.seed,
        args.nproc,
        smoke=args.smoke,
    )
    conv = pairwise_convergence(rows, master)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True) if out_prefix.parent != Path(".") else None

    points_csv = f"{out_prefix}_points.csv"
    convergence_csv = f"{out_prefix}_pairwise_convergence.csv"
    report_json = f"{out_prefix}_report.json"

    write_csv(rows, points_csv)
    if conv:
        write_csv(conv, convergence_csv)
    else:
        write_csv([{"note": "No pairwise convergence rows available."}], convergence_csv)

    plot_results(rows, str(out_prefix), master, target_x=CONVERGENCE_TARGET_L_OVER_RG)

    report = {
        "metadata": {
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "N": int(args.N),
            "w_values": [float(w) for w in w_values],
            "attempts": [int(n) for n in attempts],
            "n_blocks": int(n_blocks),
            "nproc": int(args.nproc),
            "seed": int(args.seed),
            "master_csv": str(Path(args.rg_master).resolve()),
            "master_sha256": master_hash,
            "master_rows": {str(w): master_rows[w] for w in w_values},
            "Rg_source": "Rg_MASTER_FINAL.csv; no Rg simulation in S5",
            "estimator": "uniform cubic-lattice walk proposal with importance weight 6^(N-1) exp(-w U)",
            "zero_weight_dead_chains_included": True,
            "zero_weight_denominator": "all attempted chains",
            "block_uncertainty": "SEM across independent block lnZ estimates",
            "bootstrap_diagnostic": "95% percentile CI of block mean",
            "deltaF_definition": "DeltaF/(kBT) = ln Z(L_ref) - ln Z(L)",
            "reference_note": "L_ref is finite and explicitly not an unconfined-limit normalization",
            "lattice_geometry": {
                "wall_positions": [0, "L"],
                "accessible_layers": "1,...,L-1",
                "tether": "x=L/2",
                "wall_separation": "even integer L",
                "boundary_type": "absorbing / survival condition",
            },
            "quality_thresholds": {
                "NEFF_threshold": NEFF_THRESHOLD,
                "good_block_fraction": GOOD_BLOCK_FRACTION,
                "max_weight_fraction_caution": MAX_WEIGHT_FRACTION_CAUTION,
                "minimum_valid_blocks": MIN_VALID_BLOCKS,
            },
            "smoothing": False,
            "fitting": False,
            "manual_point_deletion": False,
        },
        "outputs": {
            "points_csv": points_csv,
            "pairwise_convergence_csv": convergence_csv,
            "figure_png": f"{out_prefix}.png",
            "figure_pdf": f"{out_prefix}.pdf",
        },
        "pairwise_convergence": conv,
    }

    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    print("\n[OK] Wrote:")
    print(f"  {points_csv}")
    print(f"  {convergence_csv}")
    print(f"  {out_prefix}.png")
    print(f"  {out_prefix}.pdf")
    print(f"  {report_json}")


if __name__ == "__main__":
    main()
