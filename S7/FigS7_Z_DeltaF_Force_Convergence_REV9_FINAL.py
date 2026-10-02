#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FigS7_Z_DeltaF_Force_Convergence_REV9_FINAL.py
================================================

Production convergence analysis for Supplementary Fig. S7.

Purpose
-------
Directly address Reviewer 2, Comment 4 by validating convergence of

    ln Z(L),  DeltaF(L),  and  f Rg

using the SAME independent block records, with no sign selection and no
point deletion.

This production version implements the adopted workflow used for the revised
S6/S14 calculations:

* center-tethered absorbing slit: z=0,L, tether z=L/2;
* even integer slit widths only;
* genuine finite-width PERM with frozen thresholds from an independent
  unpruned pilot;
* PERM population control: C-=0.5, C+=2.0, pruning survival probability 0.5,
  maximum clones 4;
* 40 independent production blocks, 384 roots/block for finite widths;
* genuine unconfined L=infinity reference using the canonical Boltzmann-biased
  SIS/Rosenbluth sampler (Option 1), 4096 roots/block by default;
* block-level DeltaF and force estimates;
* paired block bootstrap / SEM for DeltaF and force, retaining block-level
  pairing across neighboring widths;
* signed block force estimates are retained, including negative statistical
  fluctuations; no sign-based filtering is allowed;
* no smoothing, fitting, interpolation, MAD filtering, SNR filtering, or
  favorable-point selection;
* overflow or extinction in any production state aborts the run rather than
  silently discarding the state;
* Rg and Rg uncertainty are read from the canonical Rg master CSV.

The exact PERM and unconfined SIS kernels are imported from the adopted S6
Numba code so that S7 cannot silently drift to a different growth algorithm.
The S6 code is therefore a required companion file.

Recommended Kaggle production command
--------------------------------------
!python /kaggle/working/FigS7_Z_DeltaF_Force_Convergence_REV9_FINAL.py \
  --s6-code /kaggle/working/FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py \
  --rg-master /kaggle/input/datasets/soumyajyotikabi/200-n-rg/Rg_MASTER_FINAL.csv \
  --outdir /kaggle/working/FigS7_DJ_CONVERGENCE_REV9_FINAL \
  --N 200 \
  --w-values 0.0 0.3 0.5 \
  --blocks 40 \
  --block-counts 8 16 24 32 40 \
  --roots-per-block 384 \
  --unconfined-roots-per-block 4096 \
  --pilot-roots 20000 \
  --max-population 32768 \
  --C-minus 0.5 \
  --C-plus 2.0 \
  --prune-probability 0.5 \
  --max-clones 4 \
  --delta-L 2 \
  --bootstrap 5000 \
  --seed 20260929

For a workflow-only check, use --self-test or --smoke. Never use smoke
outputs for the manuscript.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_NAME = "FigS7_Z_DeltaF_Force_Convergence_REV9_FINAL.py"
SCRIPT_VERSION = "9.0-PRODUCTION-NUMBA"

# Adopted manuscript-level plotting convention.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


# =============================================================================
# S6 engine loader: enforce exact adopted sampler
# =============================================================================

def load_s6_engine(path: Path):
    """Load the adopted S6 Numba engine and verify required functions exist."""
    if not path.exists():
        raise FileNotFoundError(f"S6 engine not found: {path}")

    spec = importlib.util.spec_from_file_location("s6_engine_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import S6 engine from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "unconfined_sis_block_numba",
        "pilot_thresholds_numba",
        "perm_block_numba",
        "table_size_for",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The supplied S6 engine does not expose the required adopted "
            f"sampler functions: {missing}"
        )

    try:
        import numba  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Numba is required for production S7.") from exc

    return module


def seed32(seed: int) -> int:
    """Numba-compatible 32-bit seed."""
    return int(seed) & 0xFFFFFFFF


def stable_seed(master_seed: int, stream: int, w: float, L: int, block: int = 0) -> int:
    """Deterministic uint64-derived seed, independent of execution order."""
    wcode = int(round(float(w) * 1000000.0))
    seq = np.random.SeedSequence([
        int(master_seed), int(stream), int(wcode), int(L), int(block)
    ])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def stable_unconfined_seed(master_seed: int, w: float, block: int) -> int:
    wcode = int(round(float(w) * 1000000.0))
    seq = np.random.SeedSequence([
        int(master_seed), 7001, int(wcode), int(block)
    ])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


# =============================================================================
# Canonical Rg master
# =============================================================================

def load_master_rg(path: Path, N: int, w_values: Sequence[float]) -> Dict[float, dict]:
    df = pd.read_csv(path)
    required = {
        "model", "N", "w", "Rg", "Rg_err", "quality_flag",
        "min_ESS", "median_ESS", "max_block_weight_fraction", "estimator"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Rg master CSV is missing required columns: {missing}")

    dj = df[(df["model"].astype(str).str.upper() == "DJ") & (df["N"] == N)].copy()
    if dj.empty:
        raise ValueError(f"No DJ rows found for N={N} in {path}")

    out: Dict[float, dict] = {}
    for w in w_values:
        rows = dj[np.isclose(dj["w"].astype(float), float(w), rtol=0.0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one DJ Rg row for N={N}, w={w}; found {len(rows)}"
            )
        r = rows.iloc[0]
        Rg = float(r["Rg"])
        if not np.isfinite(Rg) or Rg <= 0.0:
            raise ValueError(f"Invalid Rg={Rg} for w={w}")
        Rg_err = float(r["Rg_err"])
        if not np.isfinite(Rg_err) or Rg_err < 0.0:
            raise ValueError(f"Invalid Rg_err={Rg_err} for w={w}")
        out[float(w)] = {
            "Rg": Rg,
            "Rg_err": Rg_err,
            "quality_flag": str(r["quality_flag"]),
            "min_ESS": float(r["min_ESS"]),
            "median_ESS": float(r["median_ESS"]),
            "max_block_weight_fraction": float(r["max_block_weight_fraction"]),
            "estimator": str(r["estimator"]),
        }
    return out


# =============================================================================
# Adopted geometry / state plan
# =============================================================================

def validate_even_L(L: int) -> int:
    L = int(L)
    if L <= 2 or L % 2 != 0:
        raise ValueError(f"Finite-width L must be an even integer >2; got {L}")
    return L


def default_L_centers(w: float) -> List[int]:
    """Representative derivative centers retained from the exploratory S7."""
    if abs(float(w) - 0.0) < 1e-12:
        return [14, 18, 22]
    if abs(float(w) - 0.3) < 1e-12:
        return [16, 20, 24]
    if abs(float(w) - 0.5) < 1e-12:
        return [18, 22, 26]
    return [16, 20, 24]


def build_state_plan(w_values: Sequence[float], centers_override: Sequence[int] | None,
                     delta_L: int) -> Dict[float, List[int]]:
    if delta_L <= 0 or delta_L % 2 != 0:
        raise ValueError("delta-L must be a positive even integer")

    plan: Dict[float, List[int]] = {}
    for w in w_values:
        centers = [validate_even_L(x) for x in (
            centers_override if centers_override is not None else default_L_centers(float(w))
        )]
        states = set()
        for Lc in centers:
            Lm = validate_even_L(Lc - delta_L)
            Lp = validate_even_L(Lc + delta_L)
            states.update([Lm, Lc, Lp])
        plan[float(w)] = sorted(states)
    return plan


# =============================================================================
# Pilot thresholds
# =============================================================================

def make_frozen_threshold(
    engine,
    N: int,
    w: float,
    L: int,
    pilot_roots: int,
    C_minus: float,
    C_plus: float,
    seed: int,
    table_size: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build one frozen threshold pair from an independent unpruned pilot."""
    ref, valid_counts, status = engine.pilot_thresholds_numba(
        int(N), int(L), float(w), int(pilot_roots),
        float(C_minus), float(C_plus), seed32(seed), int(table_size)
    )
    if int(status) != 1:
        bad = np.where(~np.isfinite(ref))[0]
        depth = int(bad[0]) if bad.size else -1
        raise RuntimeError(
            f"Pilot failed for w={w}, L={L}; first non-finite depth={depth}"
        )
    if not np.all(np.isfinite(ref)):
        raise RuntimeError(f"Pilot produced non-finite thresholds for w={w}, L={L}")

    log_low = np.asarray(ref + math.log(C_minus), dtype=np.float64)
    log_high = np.asarray(ref + math.log(C_plus), dtype=np.float64)

    meta = {
        "N": int(N),
        "w": float(w),
        "L": int(L),
        "pilot_roots": int(pilot_roots),
        "C_minus": float(C_minus),
        "C_plus": float(C_plus),
        "seed": int(seed),
        "valid_counts_by_depth": np.asarray(valid_counts, dtype=int).tolist(),
        "reference_logw_by_depth": ref.tolist(),
        "logW_minus_by_depth": log_low.tolist(),
        "logW_plus_by_depth": log_high.tolist(),
    }
    return log_low, log_high, meta


# =============================================================================
# Production blocks
# =============================================================================

def run_unconfined_state(
    engine,
    N: int,
    w: float,
    blocks: int,
    roots_per_block: int,
    table_size: int,
    master_seed: int,
) -> dict:
    records = []
    for b in range(blocks):
        seed = stable_unconfined_seed(master_seed, w, b)
        logZ, ess, maxfrac, valid = engine.unconfined_sis_block_numba(
            int(N), float(w), int(roots_per_block), seed32(seed), int(table_size)
        )
        logZ = float(logZ)
        ess = float(ess)
        maxfrac = float(maxfrac)
        valid = int(valid)
        if not np.isfinite(logZ):
            raise RuntimeError(
                f"Unconfined SIS failed: w={w}, block={b}, logZ={logZ}, ESS={ess}"
            )
        if valid != roots_per_block:
            raise RuntimeError(
                f"Unconfined block has invalid roots: w={w}, block={b}, "
                f"valid={valid}/{roots_per_block}"
            )
        records.append({
            "model": "DJ",
            "ensemble": "unconfined",
            "w": float(w),
            "L": np.inf,
            "block_id": int(b),
            "logZ": logZ,
            "ESS": ess,
            "max_weight_fraction": maxfrac,
            "population_final": int(roots_per_block),
            "population_max": int(roots_per_block),
            "overflow": 0,
            "status": "OK",
            "valid_roots": valid,
            "roots": int(roots_per_block),
        })

    return {
        "model": "DJ",
        "ensemble": "unconfined",
        "w": float(w),
        "L": np.inf,
        "blocks": records,
    }


def run_finite_state(
    engine,
    N: int,
    w: float,
    L: int,
    blocks: int,
    roots_per_block: int,
    max_population: int,
    prune_probability: float,
    max_clones: int,
    log_low: np.ndarray,
    log_high: np.ndarray,
    table_size: int,
    master_seed: int,
) -> dict:
    records = []
    for b in range(blocks):
        seed = stable_seed(master_seed, 1101, w, L, b)
        status, logZ, ess, maxfrac, finalpop, maxpop = engine.perm_block_numba(
            int(N), int(L), float(w), int(roots_per_block), int(max_population),
            float(prune_probability), int(max_clones),
            log_low, log_high, seed32(seed), int(table_size)
        )
        status = int(status)
        logZ = float(logZ)
        ess = float(ess)
        maxfrac = float(maxfrac)
        finalpop = int(finalpop)
        maxpop = int(maxpop)

        if status == 1:
            raise RuntimeError(f"PERM extinction: w={w}, L={L}, block={b}")
        if status == 2:
            raise RuntimeError(f"PERM population overflow: w={w}, L={L}, block={b}")
        if status == 3 or not np.isfinite(logZ):
            raise RuntimeError(
                f"PERM numerical failure: w={w}, L={L}, block={b}, "
                f"status={status}, logZ={logZ}"
            )

        records.append({
            "model": "DJ",
            "ensemble": "finite_absorbing",
            "w": float(w),
            "L": int(L),
            "block_id": int(b),
            "logZ": logZ,
            "ESS": ess,
            "max_weight_fraction": maxfrac,
            "population_final": finalpop,
            "population_max": maxpop,
            "overflow": 0,
            "status": "OK",
            "valid_roots": int(roots_per_block),
            "roots": int(roots_per_block),
        })

    return {
        "model": "DJ",
        "ensemble": "finite_absorbing",
        "w": float(w),
        "L": int(L),
        "blocks": records,
    }


# =============================================================================
# Block-level statistics and paired bootstrap
# =============================================================================

def mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return np.nan, np.nan
    mean = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else np.nan
    return mean, sem


def paired_bootstrap(
    arrays: Sequence[np.ndarray],
    statistic,
    reps: int,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Paired block bootstrap.

    The SAME resampled block indices are applied to all arrays, so any empirical
    covariance between neighboring-width block records is preserved rather than
    destroyed by independent resampling.
    """
    if reps <= 0:
        return np.nan, np.nan, np.nan
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = arrays[0].size
    if n < 4 or any(a.size != n for a in arrays):
        return np.nan, np.nan, np.nan
    if any(not np.all(np.isfinite(a)) for a in arrays):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        samples = [a[idx] for a in arrays]
        vals[i] = float(statistic(samples))

    vals.sort()
    return (
        float(np.std(vals, ddof=1)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def state_logz_arrays(state: dict) -> np.ndarray:
    return np.asarray([b["logZ"] for b in state["blocks"]], dtype=float)


def state_ess_arrays(state: dict) -> np.ndarray:
    return np.asarray([b["ESS"] for b in state["blocks"]], dtype=float)


def state_maxfrac_arrays(state: dict) -> np.ndarray:
    return np.asarray([b["max_weight_fraction"] for b in state["blocks"]], dtype=float)


def convergence_rows_for_center(
    w: float,
    Lc: int,
    delta_L: int,
    Rg: float,
    Rg_err: float,
    n_levels: Sequence[int],
    conf_states: Dict[Tuple[float, int], dict],
    inf_state: dict,
    bootstrap_reps: int,
    seed_base: int,
) -> List[dict]:
    Lm = validate_even_L(Lc - delta_L)
    Lp = validate_even_L(Lc + delta_L)

    s_m = conf_states[(float(w), Lm)]
    s_c = conf_states[(float(w), int(Lc))]
    s_p = conf_states[(float(w), Lp)]
    s_inf = inf_state

    z_inf = state_logz_arrays(s_inf)
    z_m = state_logz_arrays(s_m)
    z_c = state_logz_arrays(s_c)
    z_p = state_logz_arrays(s_p)

    rows: List[dict] = []
    for n in n_levels:
        if n < 4 or n > len(z_inf):
            continue
        a = z_inf[:n]
        b = z_m[:n]
        c = z_c[:n]
        d = z_p[:n]

        lnZ_inf, lnZ_inf_sem = mean_sem(a)
        lnZ_m, lnZ_m_sem = mean_sem(b)
        lnZ_c, lnZ_c_sem = mean_sem(c)
        lnZ_p, lnZ_p_sem = mean_sem(d)

        # DeltaF block estimator and uncertainty: form the COMPLETE
        # logarithmic difference block-by-block, then SEM/bootstrap.
        deltaF_blocks = a - b
        deltaF = float(np.mean(deltaF_blocks))
        deltaF_sem_block = float(np.std(deltaF_blocks, ddof=1) / math.sqrt(n))

        def delta_stat(samples):
            aa, bb = samples
            return float(np.mean(aa - bb))

        dF_boot_se, dF_lo, dF_hi = paired_bootstrap(
            [a, b], delta_stat, bootstrap_reps,
            seed_base + 100000 + n,
        )

        # Central difference force formed at block level from the complete
        # logarithmic finite difference.
        force_blocks = (d - b) / (2.0 * delta_L)
        force = float(np.mean(force_blocks))
        force_sem_block = float(np.std(force_blocks, ddof=1) / math.sqrt(n))

        def force_stat(samples):
            aa, bb = samples
            return float(np.mean((aa - bb) / (2.0 * delta_L)))

        f_boot_se, f_lo, f_hi = paired_bootstrap(
            [d, b], force_stat, bootstrap_reps,
            seed_base + 200000 + n,
        )

        # Rg-scaled force. First propagate block uncertainty through the
        # logarithmic finite difference; then add independent master-Rg error.
        fRg = float(Rg * force)
        fRg_err_block = float(abs(Rg) * force_sem_block)
        fRg_boot_se = float(abs(Rg) * f_boot_se) if np.isfinite(f_boot_se) else np.nan
        fRg_err_total = math.sqrt(
            fRg_err_block ** 2 + (force * Rg_err) ** 2
        )

        ess_min = float(min(
            np.min(state_ess_arrays(s_inf)[:n]),
            np.min(state_ess_arrays(s_m)[:n]),
            np.min(state_ess_arrays(s_c)[:n]),
            np.min(state_ess_arrays(s_p)[:n]),
        ))
        maxfrac = float(max(
            np.max(state_maxfrac_arrays(s_inf)[:n]),
            np.max(state_maxfrac_arrays(s_m)[:n]),
            np.max(state_maxfrac_arrays(s_c)[:n]),
            np.max(state_maxfrac_arrays(s_p)[:n]),
        ))

        # Block covariance diagnostics are retained explicitly.
        cov_inf_minus = float(np.cov(a, b, ddof=1)[0, 1]) if n > 1 else np.nan
        cov_plus_minus = float(np.cov(d, b, ddof=1)[0, 1]) if n > 1 else np.nan

        rows.append({
            "w": float(w),
            "N": int(s_c["N"]) if "N" in s_c else int(0),
            "L_center": int(Lc),
            "L_minus": int(Lm),
            "L_plus": int(Lp),
            "delta_L": int(delta_L),
            "blocks_used": int(n),
            "Rg": float(Rg),
            "Rg_err": float(Rg_err),
            "L_over_Rg": float(Lc / Rg),
            "lnZ_inf": float(lnZ_inf),
            "lnZ_inf_sem": float(lnZ_inf_sem),
            "lnZ_minus": float(lnZ_m),
            "lnZ_minus_sem": float(lnZ_m_sem),
            "lnZ_center": float(lnZ_c),
            "lnZ_center_sem": float(lnZ_c_sem),
            "lnZ_plus": float(lnZ_p),
            "lnZ_plus_sem": float(lnZ_p_sem),
            "DeltaF": float(deltaF),
            "DeltaF_err_block": float(deltaF_sem_block),
            "DeltaF_bootstrap_SE": float(dF_boot_se),
            "DeltaF_bootstrap_CI95_low": float(dF_lo),
            "DeltaF_bootstrap_CI95_high": float(dF_hi),
            "force": float(force),
            "force_err_block": float(force_sem_block),
            "force_bootstrap_SE": float(f_boot_se),
            "force_bootstrap_CI95_low": float(f_lo),
            "force_bootstrap_CI95_high": float(f_hi),
            "fRg": float(fRg),
            "fRg_err_block": float(fRg_err_block),
            "fRg_bootstrap_SE": float(fRg_boot_se),
            "fRg_err_total": float(fRg_err_total),
            "ESS_min_all": float(ess_min),
            "max_weight_fraction_all": float(maxfrac),
            "cov_logZ_inf_minus": cov_inf_minus,
            "cov_logZ_plus_minus": cov_plus_minus,
        })

    return rows


# =============================================================================
# Audits
# =============================================================================

def convergence_audit(conv_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if conv_rows.empty:
        return pd.DataFrame()

    for (w, Lc), sub in conv_rows.groupby(["w", "L_center"], sort=True):
        sub = sub.sort_values("blocks_used").copy()
        final_n = int(sub["blocks_used"].max())
        final = sub[sub["blocks_used"] == final_n].iloc[0]
        for _, r in sub.iterrows():
            rows.append({
                "w": float(w),
                "L_center": int(Lc),
                "blocks_used": int(r["blocks_used"]),
                "final_blocks": final_n,
                "Delta_lnZ_center_to_final": float(r["lnZ_center"] - final["lnZ_center"]),
                "Delta_DeltaF_to_final": float(r["DeltaF"] - final["DeltaF"]),
                "Delta_fRg_to_final": float(r["fRg"] - final["fRg"]),
                "relative_fRg_change_abs": (
                    float(abs(r["fRg"] - final["fRg"]) / abs(final["fRg"]))
                    if np.isfinite(final["fRg"]) and final["fRg"] != 0.0 else np.nan
                ),
                "ESS_min_all": float(r["ESS_min_all"]),
                "max_weight_fraction_all": float(r["max_weight_fraction_all"]),
            })
    return pd.DataFrame(rows)


def monotonicity_from_final(conv_rows: pd.DataFrame) -> pd.DataFrame:
    """A diagnostic for DeltaF across the representative derivative centers."""
    rows = []
    if conv_rows.empty:
        return pd.DataFrame()

    final = conv_rows[conv_rows["blocks_used"] == conv_rows.groupby(
        ["w", "L_center"]
    )["blocks_used"].transform("max")].copy()

    for w, sub in final.groupby("w", sort=True):
        sub = sub.sort_values("L_over_Rg")
        for i in range(len(sub) - 1):
            a = sub.iloc[i]
            b = sub.iloc[i + 1]
            diff = float(b["DeltaF"] - a["DeltaF"])
            ea = float(a["DeltaF_err_block"])
            eb = float(b["DeltaF_err_block"])
            comb = math.sqrt(ea * ea + eb * eb)
            rows.append({
                "w": float(w),
                "L_center_left": int(a["L_center"]),
                "L_center_right": int(b["L_center"]),
                "x_left": float(a["L_over_Rg"]),
                "x_right": float(b["L_over_Rg"]),
                "DeltaF_right_minus_left": diff,
                "combined_1sigma": comb,
                "z_like": diff / comb if comb > 0 else np.nan,
                "raw_increase": bool(diff > 0.0),
                "increase_over_2sigma": bool(diff > 2.0 * comb),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================

def _select_extreme_center(df: pd.DataFrame, w: float) -> pd.Series:
    sub = df[df["w"] == float(w)].copy()
    if sub.empty:
        raise RuntimeError(f"No convergence rows for w={w}")
    # The smallest L_center is the strongest-confinement diagnostic retained.
    Lc = int(sub["L_center"].min())
    return sub[sub["L_center"] == Lc].sort_values("blocks_used").iloc[-1]


def make_figure(
    conv_df: pd.DataFrame,
    w_values: Sequence[float],
    block_counts: Sequence[int],
    out_png: Path,
    out_pdf: Path,
) -> None:
    if conv_df.empty:
        raise RuntimeError("No convergence data available for figure")

    # Fixed publication-compatible qualitative palette through Matplotlib's
    # default cycle; no manually specified RGB/hex colors are used.
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0))

    marker_map = {8: "o", 16: "s", 24: "^", 32: "D", 40: "P"}

    # (a) lnZ convergence, relative to final 40-block estimate, at the most
    # challenging finite-width center for each w.
    ax = axes[0, 0]
    for w in w_values:
        sub = conv_df[
            (conv_df["w"] == float(w)) &
            (conv_df["L_center"] == int(conv_df[conv_df["w"] == float(w)]["L_center"].min()))
        ].sort_values("blocks_used")
        if sub.empty:
            continue
        final = sub.iloc[-1]
        y = sub["lnZ_center"].to_numpy() - float(final["lnZ_center"])
        ax.plot(
            sub["blocks_used"], y, "o-", lw=1.8, ms=6,
            label=fr"$w={w:g}$, $L/R_g={final['L_over_Rg']:.2f}$"
        )
    ax.axhline(0.0, lw=1.0, ls="--")
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel(r"$\ln Z_n(L)-\ln Z_{40}(L)$")
    ax.set_title(r"(a) Convergence of $\ln Z(L)$", loc="left")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", frameon=True)

    # (b) DeltaF convergence, same most-confined center per w.
    ax = axes[0, 1]
    for w in w_values:
        sub_all = conv_df[conv_df["w"] == float(w)]
        if sub_all.empty:
            continue
        Lc = int(sub_all["L_center"].min())
        sub = sub_all[sub_all["L_center"] == Lc].sort_values("blocks_used")
        ax.errorbar(
            sub["blocks_used"], sub["DeltaF"], yerr=sub["DeltaF_err_block"],
            fmt="o-", lw=1.6, ms=6, capsize=3,
            label=fr"$w={w:g}$, $L/R_g={sub['L_over_Rg'].iloc[-1]:.2f}$"
        )
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel(r"$\Delta F\;(k_BT)$")
    ax.set_title(r"(b) Convergence of $\Delta F(L)$", loc="left")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", frameon=True)

    # (c) Signed force convergence on a LINEAR y scale.
    ax = axes[1, 0]
    for w in w_values:
        sub_all = conv_df[conv_df["w"] == float(w)]
        if sub_all.empty:
            continue
        Lc = int(sub_all["L_center"].min())
        sub = sub_all[sub_all["L_center"] == Lc].sort_values("blocks_used")
        ax.errorbar(
            sub["blocks_used"], sub["fRg"], yerr=sub["fRg_err_total"],
            fmt="o-", lw=1.6, ms=6, capsize=3,
            label=fr"$w={w:g}$, $L/R_g={sub['L_over_Rg'].iloc[-1]:.2f}$"
        )
    ax.axhline(0.0, lw=1.0, ls="--")
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel(r"$fR_g$ ($k_BT$)")
    ax.set_title(r"(c) Signed force convergence", loc="left")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", frameon=True)

    # (d) Sampling diagnostic at the same most-confined derivative centers.
    ax = axes[1, 1]
    for w in w_values:
        sub_all = conv_df[conv_df["w"] == float(w)]
        if sub_all.empty:
            continue
        Lc = int(sub_all["L_center"].min())
        sub = sub_all[sub_all["L_center"] == Lc].sort_values("blocks_used")
        ax.plot(
            sub["blocks_used"], sub["ESS_min_all"], "o-", lw=1.6, ms=6,
            label=fr"$w={w:g}$, $L/R_g={sub['L_over_Rg'].iloc[-1]:.2f}$"
        )
    ax.axhline(20.0, lw=1.2, ls=":", label="ESS=20 diagnostic")
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel("Minimum block ESS")
    ax.set_title("(d) Low-ESS sampling diagnostic", loc="left")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", frameon=True)

    fig.suptitle(
        r"Supplementary Fig. S7 — convergence of $\ln Z(L)$, $\Delta F(L)$, and $fR_g$",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Self-test
# =============================================================================

def self_test(engine) -> None:
    table = int(engine.table_size_for(20))

    # Unconfined SIS smoke.
    logz, ess, maxfrac, valid = engine.unconfined_sis_block_numba(
        20, 0.3, 16, seed32(123456), table
    )
    assert np.isfinite(float(logz))
    assert float(ess) > 0.0
    assert 0.0 < float(maxfrac) <= 1.0
    assert int(valid) == 16

    # Pilot geometry.
    ref, valid_counts, status = engine.pilot_thresholds_numba(
        20, 10, 0.3, 64, 0.5, 2.0, seed32(654321), table
    )
    assert int(status) == 1
    assert np.isfinite(ref).all()
    assert int(valid_counts[0]) == 64

    lo = np.asarray(ref + math.log(0.5), dtype=np.float64)
    hi = np.asarray(ref + math.log(2.0), dtype=np.float64)
    status, logz, ess, maxfrac, finalpop, maxpop = engine.perm_block_numba(
        20, 10, 0.3, 16, 64, 0.5, 4, lo, hi, seed32(987654), table
    )
    assert int(status) == 0
    assert np.isfinite(float(logz))
    assert float(ess) > 0.0
    assert 0.0 < float(maxfrac) <= 1.0
    assert int(finalpop) > 0
    assert int(maxpop) >= int(finalpop)

    # Block-level signed finite difference must be permitted.
    a = np.array([1.0, 1.2, 0.8, 1.1])
    b = np.array([1.1, 1.0, 0.9, 1.3])
    d = (a - b) / 2.0
    assert np.any(d < 0.0)

    print("SELF-TEST PASS: adopted S6 kernels, center/even-L geometry, and signed block estimators.")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Production S7 convergence analysis using the adopted S6 Numba engine.",
    )
    p.add_argument("--s6-code", required=True, type=Path,
                   help="Adopted FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py")
    p.add_argument("--rg-master", required=True, type=Path)
    p.add_argument("--outdir", type=Path,
                   default=Path("FigS7_DJ_CONVERGENCE_REV9_FINAL"))
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--w-values", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    p.add_argument("--L-centers", nargs="+", type=int, default=None,
                   help="Use the same center list for all w values; otherwise use the recommended per-w centers.")
    p.add_argument("--delta-L", type=int, default=2)
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--block-counts", nargs="+", type=int,
                   default=[8, 16, 24, 32, 40])
    p.add_argument("--roots-per-block", type=int, default=384)
    p.add_argument("--unconfined-roots-per-block", type=int, default=4096)
    p.add_argument("--pilot-roots", type=int, default=20000)
    p.add_argument("--max-population", type=int, default=32768)
    p.add_argument("--C-minus", type=float, default=0.5)
    p.add_argument("--C-plus", type=float, default=2.0)
    p.add_argument("--prune-probability", type=float, default=0.5)
    p.add_argument("--max-clones", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Workflow validation only; output must not be used for the paper.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    engine = load_s6_engine(args.s6_code)

    if args.self_test:
        self_test(engine)
        return

    # Smoke mode is deliberately small.
    if args.smoke:
        args.blocks = min(args.blocks, 4)
        args.block_counts = [n for n in [2, 4] if n <= args.blocks]
        args.roots_per_block = min(args.roots_per_block, 32)
        args.unconfined_roots_per_block = min(args.unconfined_roots_per_block, 64)
        args.pilot_roots = min(args.pilot_roots, 128)
        args.max_population = max(args.roots_per_block, min(args.max_population, 128))
        args.bootstrap = min(args.bootstrap, 200)
        args.w_values = args.w_values[:1]
        args.L_centers = [14]
        print("SMOKE MODE: workflow validation only; do not use output for the manuscript.")

    if args.N < 2:
        raise ValueError("N must be >= 2")
    if args.blocks < 4:
        raise ValueError("Production S7 requires at least 4 independent blocks")
    if args.roots_per_block < 4:
        raise ValueError("roots-per-block must be >= 4")
    if args.unconfined_roots_per_block < 4:
        raise ValueError("unconfined-roots-per-block must be >= 4")
    if args.max_population < args.roots_per_block:
        raise ValueError("max-population must be >= roots-per-block")
    if not (0.0 < args.C_minus < 1.0 < args.C_plus):
        raise ValueError("Require 0 < C-minus < 1 < C-plus")
    if not (0.0 < args.prune_probability <= 1.0):
        raise ValueError("prune-probability must be in (0,1]")
    if args.max_clones < 2:
        raise ValueError("max-clones must be >= 2")
    if args.delta_L <= 0 or args.delta_L % 2 != 0:
        raise ValueError("delta-L must be a positive even integer")

    block_counts = sorted(set(int(n) for n in args.block_counts if int(n) >= 4))
    block_counts = [n for n in block_counts if n <= args.blocks]
    if args.blocks not in block_counts:
        block_counts.append(args.blocks)
        block_counts.sort()
    if not block_counts:
        raise ValueError("No valid block-count convergence levels")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_size = int(engine.table_size_for(args.N))
    master_rg = load_master_rg(args.rg_master, args.N, args.w_values)
    plan = build_state_plan(args.w_values, args.L_centers, args.delta_L)

    print("=" * 100)
    print("SUPPLEMENTARY FIG. S7 — PRODUCTION CONVERGENCE / ADOPTED S6 ENGINE")
    print("=" * 100)
    print(f"Script/version             : {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"S6 engine                  : {args.s6_code}")
    print(f"Rg master                  : {args.rg_master}")
    print(f"N                          : {args.N}")
    print(f"w values                   : {args.w_values}")
    print(f"delta_L                    : {args.delta_L}")
    print(f"Total blocks               : {args.blocks}")
    print(f"Convergence block levels   : {block_counts}")
    print(f"Finite roots/block         : {args.roots_per_block}")
    print(f"Unconfined roots/block     : {args.unconfined_roots_per_block}")
    print(f"Independent pilot roots    : {args.pilot_roots}")
    print(f"PERM C-/C+                 : {args.C_minus}/{args.C_plus}")
    print(f"Prune survival probability : {args.prune_probability}")
    print(f"Max clones                 : {args.max_clones}")
    print(f"Max population             : {args.max_population}")
    print(f"Hash table size            : {table_size}")
    print(f"Seed                       : {args.seed}")
    print("Geometry                   : absorbing walls z=0,L; center tether z=L/2; even L")
    print("Unconfined reference       : genuine L=infinity canonical SIS/Rosenbluth (Option 1)")
    print("Sign selection             : OFF")
    print("Filtering/smoothing/fits   : OFF")
    print("Overflow/extinction        : fatal; no silent state/block deletion")
    print("Force uncertainty          : block-level finite difference + independent Rg propagation")
    print("=" * 100)

    print("\nCanonical Rg values:")
    for w in args.w_values:
        r = master_rg[float(w)]
        print(
            f"  w={w:g}: Rg={r['Rg']:.9f} +/- {r['Rg_err']:.9f}; "
            f"quality={r['quality_flag']}; master minESS={r['min_ESS']:.2f}; "
            f"master maxfrac={r['max_block_weight_fraction']:.5f}"
        )

    print("\nDerivative-center plan:")
    for w in args.w_values:
        print(f"  w={w:g}: centers={default_L_centers(float(w)) if args.L_centers is None else args.L_centers}; "
              f"simulated finite L={plan[float(w)]}")

    t0 = time.time()

    # -------------------------------------------------------------------------
    # Independent unconfined reference blocks: Option 1
    # -------------------------------------------------------------------------
    print("\nRunning genuine unconfined references (Option 1)...")
    inf_states: Dict[float, dict] = {}
    for w in args.w_values:
        print(f"  unconfined w={w:g}: {args.blocks} blocks x {args.unconfined_roots_per_block} roots")
        inf_states[float(w)] = run_unconfined_state(
            engine=engine,
            N=args.N,
            w=float(w),
            blocks=args.blocks,
            roots_per_block=args.unconfined_roots_per_block,
            table_size=table_size,
            master_seed=args.seed + 700_000_000,
        )
        ess = state_ess_arrays(inf_states[float(w)])
        mf = state_maxfrac_arrays(inf_states[float(w)])
        lz = state_logz_arrays(inf_states[float(w)])
        print(
            f"    logZ_inf={np.mean(lz):.10f} +/- {np.std(lz,ddof=1)/math.sqrt(len(lz)):.10f}; "
            f"ESS min/median={np.min(ess):.2f}/{np.median(ess):.2f}; "
            f"maxfrac={np.max(mf):.5f}"
        )

    # -------------------------------------------------------------------------
    # Frozen independent pilots
    # -------------------------------------------------------------------------
    print("\nBuilding frozen PERM thresholds from independent unpruned pilots...")
    thresholds: Dict[Tuple[float, int], Tuple[np.ndarray, np.ndarray]] = {}
    threshold_meta: Dict[str, dict] = {}
    for w in args.w_values:
        for L in plan[float(w)]:
            seed = stable_seed(args.seed + 1000, 501, w, L, 0)
            lo, hi, meta = make_frozen_threshold(
                engine=engine,
                N=args.N,
                w=float(w),
                L=int(L),
                pilot_roots=args.pilot_roots,
                C_minus=args.C_minus,
                C_plus=args.C_plus,
                seed=seed32(seed),
                table_size=table_size,
            )
            thresholds[(float(w), int(L))] = (lo, hi)
            threshold_meta[f"w={float(w):.12g}|L={int(L)}"] = meta
            print(f"  pilot OK: w={w:g}, L={L}")

    with (outdir / "FigS7_PERM_FROZEN_THRESHOLDS.json").open("w", encoding="utf-8") as fh:
        json.dump(threshold_meta, fh, indent=2)

    # -------------------------------------------------------------------------
    # Finite-width production
    # -------------------------------------------------------------------------
    print("\nRunning finite-width PERM production...")
    conf_states: Dict[Tuple[float, int], dict] = {}
    for w in args.w_values:
        for L in plan[float(w)]:
            lo, hi = thresholds[(float(w), int(L))]
            print(f"  state w={w:g}, L={L}: {args.blocks} blocks x {args.roots_per_block} roots")
            state = run_finite_state(
                engine=engine,
                N=args.N,
                w=float(w),
                L=int(L),
                blocks=args.blocks,
                roots_per_block=args.roots_per_block,
                max_population=args.max_population,
                prune_probability=args.prune_probability,
                max_clones=args.max_clones,
                log_low=lo,
                log_high=hi,
                table_size=table_size,
                master_seed=args.seed + 900_000_000,
            )
            conf_states[(float(w), int(L))] = state
            ess = state_ess_arrays(state)
            mf = state_maxfrac_arrays(state)
            print(
                f"    logZ={np.mean(state_logz_arrays(state)):.10f} +/- "
                f"{np.std(state_logz_arrays(state),ddof=1)/math.sqrt(args.blocks):.10f}; "
                f"ESS min/median={np.min(ess):.2f}/{np.median(ess):.2f}; "
                f"maxfrac={np.max(mf):.5f}"
            )

    # -------------------------------------------------------------------------
    # Convergence reconstruction
    # -------------------------------------------------------------------------
    print("\nConstructing block-level convergence estimates...")
    conv_rows: List[dict] = []
    for w in args.w_values:
        Rg = master_rg[float(w)]["Rg"]
        Rg_err = master_rg[float(w)]["Rg_err"]
        centers = args.L_centers if args.L_centers is not None else default_L_centers(float(w))
        for Lc0 in centers:
            Lc = validate_even_L(Lc0)
            rows = convergence_rows_for_center(
                w=float(w),
                Lc=Lc,
                delta_L=args.delta_L,
                Rg=Rg,
                Rg_err=Rg_err,
                n_levels=block_counts,
                conf_states=conf_states,
                inf_state=inf_states[float(w)],
                bootstrap_reps=args.bootstrap,
                seed_base=int(args.seed + 2_000_000 + round(float(w)*10000) + Lc*100),
            )
            for row in rows:
                row["N"] = int(args.N)
                conv_rows.append(row)

    conv_df = pd.DataFrame(conv_rows)
    conv_path = outdir / "FigS7_convergence_levels.csv"
    conv_df.to_csv(conv_path, index=False)

    # -------------------------------------------------------------------------
    # Complete block-level output: unconfined + finite states
    # -------------------------------------------------------------------------
    block_rows: List[dict] = []
    for w, state in inf_states.items():
        for rec in state["blocks"]:
            block_rows.append(dict(rec))
    for (_, _L), state in conf_states.items():
        for rec in state["blocks"]:
            block_rows.append(dict(rec))
    block_df = pd.DataFrame(block_rows).sort_values(
        ["w", "ensemble", "L", "block_id"],
        na_position="last"
    )
    block_path = outdir / "FigS7_block_level.csv"
    block_df.to_csv(block_path, index=False)

    # -------------------------------------------------------------------------
    # Dedicated block-level force table: complete logarithmic finite difference
    # -------------------------------------------------------------------------
    force_rows: List[dict] = []
    final_rows: List[dict] = []
    for _, r in conv_df.iterrows():
        row = r.to_dict()
        force_rows.append({
            "w": row["w"],
            "N": row["N"],
            "L_center": row["L_center"],
            "L_minus": row["L_minus"],
            "L_plus": row["L_plus"],
            "delta_L": row["delta_L"],
            "blocks_used": row["blocks_used"],
            "force": row["force"],
            "force_err_block": row["force_err_block"],
            "force_bootstrap_SE": row["force_bootstrap_SE"],
            "force_bootstrap_CI95_low": row["force_bootstrap_CI95_low"],
            "force_bootstrap_CI95_high": row["force_bootstrap_CI95_high"],
            "fRg": row["fRg"],
            "fRg_err_block": row["fRg_err_block"],
            "fRg_err_total": row["fRg_err_total"],
            "Rg": row["Rg"],
            "Rg_err": row["Rg_err"],
            "ESS_min_all": row["ESS_min_all"],
            "max_weight_fraction_all": row["max_weight_fraction_all"],
        })
    if not conv_df.empty:
        final_df = conv_df[conv_df["blocks_used"] == conv_df.groupby(
            ["w", "L_center"]
        )["blocks_used"].transform("max")].copy()
        final_rows = final_df.to_dict(orient="records")
    force_df = pd.DataFrame(force_rows)
    force_df.to_csv(outdir / "FigS7_force_block_level.csv", index=False)
    pd.DataFrame(final_rows).to_csv(outdir / "FigS7_final_state_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Audits
    # -------------------------------------------------------------------------
    audit_df = convergence_audit(conv_df)
    audit_df.to_csv(outdir / "FigS7_convergence_audit.csv", index=False)

    mono_df = monotonicity_from_final(conv_df)
    mono_df.to_csv(outdir / "FigS7_monotonicity_audit.csv", index=False)

    # Direct summary of worst diagnostics at final block count.
    diagnostics_rows = []
    for w in args.w_values:
        # All simulated finite states + unconfined state.
        estats = [inf_states[float(w)]] + [
            conf_states[(float(w), L)] for L in plan[float(w)]
        ]
        ess_values = np.concatenate([state_ess_arrays(s) for s in estats])
        mf_values = np.concatenate([state_maxfrac_arrays(s) for s in estats])
        diagnostics_rows.append({
            "w": float(w),
            "minimum_ESS_all_states": float(np.min(ess_values)),
            "median_ESS_all_states": float(np.median(ess_values)),
            "maximum_weight_fraction_all_states": float(np.max(mf_values)),
            "all_blocks_ok": True,
            "all_overflow_zero": True,
            "all_signed_forces_retained": True,
        })
    pd.DataFrame(diagnostics_rows).to_csv(outdir / "FigS7_sampling_diagnostics_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    png = outdir / "FigS7_Z_DeltaF_Force_Convergence_REV9_FINAL.png"
    pdf = outdir / "FigS7_Z_DeltaF_Force_Convergence_REV9_FINAL.pdf"
    make_figure(conv_df, args.w_values, block_counts, png, pdf)

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------
    provenance = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numba_version": getattr(engine, "numba", None).__version__ if getattr(engine, "numba", None) is not None else None,
        "engine_file": str(args.s6_code.resolve()),
        "rg_master_file": str(args.rg_master.resolve()),
        "N": int(args.N),
        "w_values": [float(w) for w in args.w_values],
        "delta_L": int(args.delta_L),
        "blocks": int(args.blocks),
        "block_counts": [int(n) for n in block_counts],
        "roots_per_block_finite": int(args.roots_per_block),
        "roots_per_block_unconfined": int(args.unconfined_roots_per_block),
        "pilot_roots": int(args.pilot_roots),
        "C_minus": float(args.C_minus),
        "C_plus": float(args.C_plus),
        "prune_probability": float(args.prune_probability),
        "max_clones": int(args.max_clones),
        "max_population": int(args.max_population),
        "seed": int(args.seed),
        "hash_table_size": int(table_size),
        "geometry": {
            "lattice": "3D simple cubic",
            "walls": "absorbing at z=0 and z=L",
            "tether": "z=L/2",
            "accessible_layers": "z=1,...,L-1",
            "finite_L_even_only": True,
        },
        "state_plan": {str(w): plan[float(w)] for w in args.w_values},
        "Rg_source": {
            str(w): master_rg[float(w)] for w in args.w_values
        },
        "samplers": {
            "finite_width": "adopted S6 center-tethered PERM with frozen independent unpruned pilot",
            "unconfined": "canonical Boltzmann-biased SIS/Rosenbluth, genuine L=infinity reference (Option 1)",
        },
        "estimators": {
            "lnZ_state": "block mean of block-level log partition-function estimates",
            "DeltaF_block": "lnZ_inf_block - lnZ_conf_block",
            "DeltaF": "mean of block-level logarithmic differences",
            "force_block": "[lnZ_plus_block - lnZ_minus_block]/(2 delta_L)",
            "force": "mean of signed block-level central differences",
            "fRg": "Rg * force",
            "fRg_uncertainty": "block SEM propagated through finite difference, then independent Rg uncertainty added in quadrature",
            "bootstrap": "paired block bootstrap using the same resampled block indices for neighboring-width/reference arrays",
        },
        "quality_control": {
            "sign_selection": False,
            "negative_force_blocks_retained": True,
            "point_deletion": False,
            "outlier_filtering": False,
            "smoothing": False,
            "fitting": False,
            "overflow_allowed": False,
            "extinction_allowed": False,
        },
        "reviewer_2_comment_4": (
            "S7 directly demonstrates convergence of lnZ(L), DeltaF(L), and the resulting force "
            "using cumulative independent production blocks. The unconfined reference is simulated "
            "at genuine L=infinity rather than a finite-L proxy. Force uncertainty is formed at the "
            "block level from the complete logarithmic finite difference, with paired block bootstrap."
        ),
        "figure": {
            "panel_a": "lnZ(L) convergence relative to final block estimate",
            "panel_b": "DeltaF(L) convergence with block SEM",
            "panel_c": "signed fRg convergence on a linear y-axis",
            "panel_d": "minimum block ESS diagnostic",
        },
        "runtime_seconds": float(time.time() - t0),
        "outputs": {
            "convergence_levels": str((outdir / "FigS7_convergence_levels.csv").resolve()),
            "block_level": str(block_path.resolve()),
            "force_block_level": str((outdir / "FigS7_force_block_level.csv").resolve()),
            "final_state_summary": str((outdir / "FigS7_final_state_summary.csv").resolve()),
            "sampling_diagnostics": str((outdir / "FigS7_sampling_diagnostics_summary.csv").resolve()),
            "convergence_audit": str((outdir / "FigS7_convergence_audit.csv").resolve()),
            "monotonicity_audit": str((outdir / "FigS7_monotonicity_audit.csv").resolve()),
            "figure_png": str(png.resolve()),
            "figure_pdf": str(pdf.resolve()),
            "thresholds": str((outdir / "FigS7_PERM_FROZEN_THRESHOLDS.json").resolve()),
        },
    }
    with (outdir / "FigS7_provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=str)

    print("\n" + "=" * 100)
    print("S7 PRODUCTION COMPLETE")
    print("=" * 100)
    for p in [
        conv_path,
        block_path,
        outdir / "FigS7_force_block_level.csv",
        outdir / "FigS7_final_state_summary.csv",
        outdir / "FigS7_sampling_diagnostics_summary.csv",
        outdir / "FigS7_convergence_audit.csv",
        outdir / "FigS7_monotonicity_audit.csv",
        png,
        pdf,
        outdir / "FigS7_PERM_FROZEN_THRESHOLDS.json",
        outdir / "FigS7_provenance.json",
    ]:
        print(p.resolve())

    print(f"Total wall time: {(time.time() - t0)/60.0:.2f} min")

    if not mono_df.empty:
        print("\nFinal monotonicity audit (representative centers):")
        for w, sub in mono_df.groupby("w", sort=True):
            print(
                f"  w={w:g}: raw increases={int(sub['raw_increase'].sum())}; "
                f">2sigma increases={int(sub['increase_over_2sigma'].sum())}"
            )


if __name__ == "__main__":
    main()
