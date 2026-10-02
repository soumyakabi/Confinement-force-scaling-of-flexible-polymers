#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S9 — Gaussian-chain confinement under two distinct
one-coordinate boundary ensembles.

REV4: direct-survival Gaussian benchmark
-----------------------------------------
This revision replaces the confined Rosenbluth estimator used in the earlier
S9 implementation by a direct uniform six-direction random-walk survival
estimator for the absorbing ensemble.

Why:
    For an ideal Gaussian lattice walk, the absorbing partition function is
        Z(L) = 6^(N-1) S(L),
    where S(L) is the probability that an unconstrained six-direction walk,
    tethered at z=L/2, remains in 1 <= z <= L-1 at every step.
    This bounded Bernoulli/survival estimator avoids the strong Rosenbluth
    weight degeneracy observed in the earlier S9 benchmark.

Boundary ensembles
-------------------
Absorbing:
    Survival/first-passage ensemble. Walls at z=0,L, even L, tether z=L/2.
    A trajectory is counted only if every monomer remains in z=1,...,L-1.
    The unconfined reference is exactly Z_free = 6^(N-1).

Reflecting:
    Discrete fold-back / mirror stochastic rule, NOT generic reflecting
    boundaries. Every attempted lattice step is accepted after fold-back,
    so there are exactly 6 stochastic choices at every step:
        Z_ref = 6^(N-1), DeltaF_ref = 0, f_ref = 0,
    exactly for the ideal lattice walk.

Mechanical derivative
---------------------
One-coordinate slit force conjugate to the wall separation L:
    f(L) = kBT [ln Z(L+dL) - ln Z(L-dL)] / (2 dL)
with dL = 2 lattice units by default. Both walls are displaced
symmetrically and the tether is re-centered at z=(L/2) in each derivative
state. This is not a single-wall force.

Signed estimates
----------------
All signed block force estimates are retained. No sign filtering, point
deletion, smoothing, fitting, or favorable-point selection is performed.

Common random numbers (CRN)
---------------------------
Within each independent block, the same unconstrained random-walk direction
sequence is replayed at all widths. Since the tether is always re-centered at
L/2, a trajectory represented by its relative z-displacement q_i survives
at width L exactly when
    |q_i| <= L/2 - 1
for every monomer. Thus neighboring widths use the same underlying walks and
the force difference retains the block-level covariance.

Free-energy estimator
---------------------
Each block provides a survival fraction p_b and therefore
    logZ_b(L) = log[6^(N-1)] + log(p_b).
The final estimator is
    logZ(L) = log(mean_b exp(logZ_b)),
equivalent to using all trials because blocks have equal size.
Thus
    DeltaF(L) = logZ_free - logZ(L).
For uncertainty, the complete nonlinear free-energy estimator is evaluated
inside a block bootstrap:
    DeltaF* = logZ_free - log(mean_b* exp(logZ_b)).
The force uncertainty is computed from the signed paired block force values.

Exact lattice benchmark
-----------------------
The absorbing lattice partition function is computed independently by exact
integer transfer recursion. Exact free energy and the exact same central
difference (dL=2) force are reported alongside the Monte Carlo estimates.

Outputs
-------
Fig_S9_Gaussian_BC_REV4.png
Fig_S9_Gaussian_BC_REV4.pdf
S9_points.csv
S9_block_level.csv
S9_force_block_level.csv
S9_convergence.csv
S9_exact_audit.csv
S9_provenance.json

No external Rg file is required: for the lattice Gaussian convention,
    Rg = sqrt(N/6), with a=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import numba
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    numba = None
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco


SCRIPT_NAME = "S9_Gaussian_BC_REV4_DIRECT_SURVIVAL_FINAL.py"
SCRIPT_VERSION = "4.0-DIRECT-SURVIVAL-CRN"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.3,
    "savefig.dpi": 600,
})


# =============================================================================
# Low-level random-walk kernels
# =============================================================================

@njit(cache=True)
def _generate_relative_z_minmax(N: int, n_trials: int, seed: int):
    """
    Generate unconstrained six-direction walks and return, for each trial,
    the minimum and maximum cumulative z displacement relative to the tether.

    The same relative trajectories can then be tested against many even slit
    widths without resampling.
    """
    np.random.seed(seed)

    qmin = np.zeros(n_trials, dtype=np.int32)
    qmax = np.zeros(n_trials, dtype=np.int32)

    for t in range(n_trials):
        q = 0
        qlo = 0
        qhi = 0

        for _ in range(N - 1):
            u = np.random.random()
            idx = int(u * 6.0)
            if idx >= 6:
                idx = 5

            if idx == 4:
                q -= 1
            elif idx == 5:
                q += 1

            if q < qlo:
                qlo = q
            if q > qhi:
                qhi = q

        qmin[t] = qlo
        qmax[t] = qhi

    return qmin, qmax


def _log_survival_fraction(qmin: np.ndarray, qmax: np.ndarray,
                           L: int) -> Tuple[float, int]:
    """Return log survival fraction and number of surviving walks."""
    half = L // 2
    bound = half - 1
    survive = (qmin >= -bound) & (qmax <= bound)
    n_survive = int(np.count_nonzero(survive))
    if n_survive == 0:
        return -math.inf, 0
    return math.log(n_survive / qmin.size), n_survive


def sample_blocks(
    N: int,
    widths: Sequence[int],
    trials_per_block: int,
    blocks: int,
    seed: int,
) -> Dict[str, object]:
    """
    Generate independent random-walk blocks. Within each block, the same
    unconstrained walks are reused at every width (CRN).
    """
    width_list = list(sorted(set(int(L) for L in widths)))
    logz_block = {L: np.empty(blocks, dtype=float) for L in width_list}
    surv_block = {L: np.empty(blocks, dtype=float) for L in width_list}
    n_survive_block = {L: np.empty(blocks, dtype=np.int64) for L in width_list}

    logz_free = (N - 1) * math.log(6.0)

    for b in range(blocks):
        block_seed = int(
            np.random.SeedSequence([int(seed), 730001, int(b)])
            .generate_state(1, dtype=np.uint64)[0]
            & np.uint64(0xFFFFFFFF)
        )

        qmin, qmax = _generate_relative_z_minmax(
            int(N), int(trials_per_block), block_seed
        )

        for L in width_list:
            logp, ns = _log_survival_fraction(qmin, qmax, L)
            surv_block[L][b] = math.exp(logp) if np.isfinite(logp) else 0.0
            n_survive_block[L][b] = ns
            logz_block[L][b] = (
                logz_free + logp if np.isfinite(logp) else -math.inf
            )

    return {
        "logz_block": logz_block,
        "surv_block": surv_block,
        "n_survive_block": n_survive_block,
        "trials_per_block": trials_per_block,
        "blocks": blocks,
        "logz_free": logz_free,
    }


# =============================================================================
# Stable estimators / bootstrap
# =============================================================================

def logmeanexp(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -math.inf
    m = float(np.max(x))
    return float(m + math.log(float(np.mean(np.exp(x - m)))))


def safe_sem(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def bootstrap_deltaF_from_logz(
    logz_blocks: Sequence[float],
    logz_free: float,
    reps: int,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Nonlinear bootstrap for DeltaF = logZ_free - log(mean Z_blocks).
    Returns SE, 95% CI low, 95% CI high.
    """
    x = np.asarray(logz_blocks, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 or reps <= 0:
        return math.nan, math.nan, math.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    out = np.empty(reps, dtype=float)

    for i in range(reps):
        out[i] = logz_free - logmeanexp(x[idx[i]])

    return (
        float(np.std(out, ddof=1)),
        float(np.percentile(out, 2.5)),
        float(np.percentile(out, 97.5)),
    )


def bootstrap_force_mean(
    force_blocks: Sequence[float],
    reps: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Bootstrap the signed paired-block force mean."""
    x = np.asarray(force_blocks, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 or reps <= 0:
        return math.nan, math.nan, math.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    means = x[idx].mean(axis=1)

    return (
        float(np.std(means, ddof=1)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


# =============================================================================
# Exact lattice benchmark
# =============================================================================

def exact_absorbing_logz(N: int, L: int) -> float:
    """Exact absorbing cubic-lattice partition function."""
    if L < 4 or L % 2 != 0:
        raise ValueError("Exact recursion requires even L >= 4.")

    v = [0] * (L + 1)
    v[L // 2] = 1

    for _ in range(N - 1):
        nv = [0] * (L + 1)
        for z in range(1, L):
            c = v[z]
            if c == 0:
                continue
            nv[z] += 4 * c
            if z - 1 >= 1:
                nv[z - 1] += c
            if z + 1 <= L - 1:
                nv[z + 1] += c
        v = nv

    Z = sum(v[1:L])
    if Z <= 0:
        return -math.inf
    return math.log(Z)


# =============================================================================
# Point and convergence assembly
# =============================================================================

def assemble_points(
    N: int,
    centers: Sequence[int],
    delta_L: int,
    results: Dict[str, object],
    exact_logs: Dict[int, float],
    exact_force: Dict[int, float],
    bootstrap_reps: int,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    logz_block = results["logz_block"]
    surv_block = results["surv_block"]
    n_survive_block = results["n_survive_block"]
    blocks = int(results["blocks"])
    trials = int(results["trials_per_block"])
    logz_free = float(results["logz_free"])
    Rg = math.sqrt(N / 6.0)

    point_rows = []
    force_rows = []
    exact_rows = []

    for L in centers:
        Lm, Lp = L - delta_L, L + delta_L

        lc = np.asarray(logz_block[L], dtype=float)
        lm = np.asarray(logz_block[Lm], dtype=float)
        lp = np.asarray(logz_block[Lp], dtype=float)

        # Final nonlinear free-energy estimator from the block partition
        # functions; equivalent to pooling all equal-sized trials.
        logz_hat = logmeanexp(lc)
        DeltaF = logz_free - logz_hat

        f_blocks = (lp - lm) / (2.0 * delta_L)
        f_mean = float(np.mean(f_blocks))
        f_sem = safe_sem(f_blocks)

        f_boot_se, f_lo, f_hi = bootstrap_force_mean(
            f_blocks,
            bootstrap_reps,
            seed + 500000 + 17 * L,
        )

        df_boot_se, df_lo, df_hi = bootstrap_deltaF_from_logz(
            lc,
            logz_free,
            bootstrap_reps,
            seed + 700000 + 19 * L,
        )

        # Independent-width reference for the CRN variance-reduction audit.
        lm_var = float(np.var(lm, ddof=1)) if blocks > 1 else math.nan
        lp_var = float(np.var(lp, ddof=1)) if blocks > 1 else math.nan
        f_sem_if_ind = (
            math.sqrt((lm_var + lp_var) / blocks) / (2.0 * delta_L)
            if np.isfinite(lm_var) and np.isfinite(lp_var)
            else math.nan
        )

        exact_f = exact_force[L]
        exact_F = logz_free - exact_logs[L]
        rel_f_err = (
            100.0 * abs(f_mean - exact_f) / abs(exact_f)
            if abs(exact_f) > 0
            else math.nan
        )

        # Sampling diagnostics for all three widths entering the finite difference.
        surv_min = min(
            float(np.min(surv_block[Lm])),
            float(np.min(surv_block[L])),
            float(np.min(surv_block[Lp])),
        )
        min_survivors = min(
            int(np.min(n_survive_block[Lm])),
            int(np.min(n_survive_block[L])),
            int(np.min(n_survive_block[Lp])),
        )

        # For a bounded survival estimator, a simple block ESS diagnostic is
        # based on Bernoulli survival probabilities:
        #     ESS = B * p(1-p) / Var(p_b)
        # with p_b treated as block-level survival estimates.
        p_all = np.concatenate([
            surv_block[Lm], surv_block[L], surv_block[Lp]
        ])
        pbar = float(np.mean(p_all))
        pvar = float(np.var(p_all, ddof=1)) if p_all.size > 1 else math.nan
        ess_survival = (
            float(p_all.size * pbar * (1.0 - pbar) / pvar)
            if pvar > 0 and 0 < pbar < 1
            else math.inf
        )

        # A more direct within-block Bernoulli effective count is simply the
        # smallest number of surviving trajectories among the three widths.
        quality = "OK"
        if min_survivors < 50:
            quality = "CHECK_survivors"

        point_rows.append({
            "boundary": "absorbing (survival/first-passage)",
            "N": N,
            "L": int(L),
            "L_over_Rg": float(L / Rg),
            "delta_L": int(delta_L),
            "logZ_free": logz_free,
            "logZ_center": float(logz_hat),
            "logZ_center_SE_block": safe_sem(lc),
            "DeltaF_kBT": float(DeltaF),
            "DeltaF_bootstrap_SE": float(df_boot_se),
            "DeltaF_bootstrap_CI95_low": float(df_lo),
            "DeltaF_bootstrap_CI95_high": float(df_hi),
            "force_kBT": float(f_mean),
            "force_SE_block": float(f_sem),
            "force_bootstrap_SE": float(f_boot_se),
            "force_bootstrap_CI95_low": float(f_lo),
            "force_bootstrap_CI95_high": float(f_hi),
            "fRg": float(f_mean * Rg),
            "fRg_err": float(f_sem * Rg),
            "fRg_bootstrap_SE": float(f_boot_se * Rg),
            "fRg_bootstrap_CI95_low": float(f_lo * Rg),
            "fRg_bootstrap_CI95_high": float(f_hi * Rg),
            "fRg_SEM_if_independent": float(
                f_sem_if_ind * Rg if np.isfinite(f_sem_if_ind) else math.nan
            ),
            "CRN_variance_reduction_factor": float(
                f_sem_if_ind / f_sem if f_sem > 0 and np.isfinite(f_sem_if_ind)
                else math.nan
            ),
            "exact_logZ": float(exact_logs[L]),
            "exact_DeltaF_kBT": float(exact_F),
            "exact_fRg": float(exact_f * Rg),
            "relative_force_error_pct": float(rel_f_err),
            "min_survivors_across_derivative_states": int(min_survivors),
            "min_block_survival_fraction": float(surv_min),
            "diagnostic_survival_ESS": float(ess_survival),
            "quality_flag": quality,
            "signed_estimates_retained": True,
            "sign_selection": False,
            "common_random_numbers_across_widths": True,
            "n_blocks": blocks,
            "trials_per_block": trials,
        })

        for b in range(blocks):
            force_rows.append({
                "L_center": int(L),
                "L_minus": int(Lm),
                "L_plus": int(Lp),
                "block": int(b + 1),
                "logZ_minus": float(lm[b]),
                "logZ_plus": float(lp[b]),
                "survival_minus": float(surv_block[Lm][b]),
                "survival_plus": float(surv_block[Lp][b]),
                "force_block_kBT": float(f_blocks[b]),
                "force_block_fRg": float(f_blocks[b] * Rg),
                "signed": True,
            })

        exact_rows.append({
            "L": int(L),
            "L_over_Rg": float(L / Rg),
            "exact_logZ": float(exact_logs[L]),
            "exact_DeltaF_kBT": float(exact_F),
            "exact_force_kBT": float(exact_f),
            "exact_fRg": float(exact_f * Rg),
        })

    # Reflecting exact rows
    for L in centers:
        point_rows.append({
            "boundary": "reflecting (fold-back stochastic rule)",
            "N": N,
            "L": int(L),
            "L_over_Rg": float(L / Rg),
            "delta_L": int(delta_L),
            "logZ_free": logz_free,
            "logZ_center": logz_free,
            "logZ_center_SE_block": 0.0,
            "DeltaF_kBT": 0.0,
            "DeltaF_bootstrap_SE": 0.0,
            "DeltaF_bootstrap_CI95_low": 0.0,
            "DeltaF_bootstrap_CI95_high": 0.0,
            "force_kBT": 0.0,
            "force_SE_block": 0.0,
            "force_bootstrap_SE": 0.0,
            "force_bootstrap_CI95_low": 0.0,
            "force_bootstrap_CI95_high": 0.0,
            "fRg": 0.0,
            "fRg_err": 0.0,
            "fRg_bootstrap_SE": 0.0,
            "fRg_bootstrap_CI95_low": 0.0,
            "fRg_bootstrap_CI95_high": 0.0,
            "fRg_SEM_if_independent": 0.0,
            "CRN_variance_reduction_factor": math.nan,
            "exact_logZ": logz_free,
            "exact_DeltaF_kBT": 0.0,
            "exact_fRg": 0.0,
            "relative_force_error_pct": 0.0,
            "min_survivors_across_derivative_states": -1,
            "min_block_survival_fraction": math.nan,
            "diagnostic_survival_ESS": math.nan,
            "quality_flag": "EXACT_DISCRETE_RESULT",
            "signed_estimates_retained": True,
            "sign_selection": False,
            "common_random_numbers_across_widths": True,
            "n_blocks": blocks,
            "trials_per_block": trials,
            "ensemble_definition":
                "Discrete fold-back/mirror rule; each attempted step is "
                "reflected at the boundary; exactly six stochastic choices "
                "per step; Z_ref=6^(N-1), DeltaF=0, f=0 exactly.",
        })

    return point_rows, force_rows, exact_rows


def convergence_rows(
    N: int,
    centers: Sequence[int],
    delta_L: int,
    results: Dict[str, object],
    levels: Sequence[int],
) -> List[dict]:
    logz_block = results["logz_block"]
    logz_free = float(results["logz_free"])
    Rg = math.sqrt(N / 6.0)
    out = []

    for L in centers:
        Lm, Lp = L - delta_L, L + delta_L
        for n in levels:
            if n > results["blocks"]:
                continue

            lc = np.asarray(logz_block[L][:n], dtype=float)
            lm = np.asarray(logz_block[Lm][:n], dtype=float)
            lp = np.asarray(logz_block[Lp][:n], dtype=float)

            logz_n = logmeanexp(lc)
            F_n = logz_free - logz_n
            fb = (lp - lm) / (2.0 * delta_L)
            f_n = float(np.mean(fb))
            f_se = safe_sem(fb)

            out.append({
                "L_center": int(L),
                "L_over_Rg": float(L / Rg),
                "blocks_used": int(n),
                "logZ_center": float(logz_n),
                "DeltaF_kBT": float(F_n),
                "force_kBT": float(f_n),
                "force_err_kBT": float(f_se),
                "fRg": float(f_n * Rg),
                "fRg_err": float(f_se * Rg),
            })

    return out


# =============================================================================
# Plot
# =============================================================================

def make_figure(
    point_rows: List[dict],
    conv_rows: List[dict],
    exact_rows: List[dict],
    N: int,
    Rg: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    abs_rows = [
        r for r in point_rows
        if r["boundary"].startswith("absorbing")
    ]
    abs_rows.sort(key=lambda r: r["L_over_Rg"])
    exact_rows = sorted(exact_rows, key=lambda r: r["L_over_Rg"])

    x = np.array([r["L_over_Rg"] for r in abs_rows], dtype=float)
    F = np.array([r["DeltaF_kBT"] for r in abs_rows], dtype=float)
    Ferr = np.array([r["DeltaF_bootstrap_SE"] for r in abs_rows], dtype=float)
    f = np.array([r["fRg"] for r in abs_rows], dtype=float)
    ferr = np.array([r["fRg_err"] for r in abs_rows], dtype=float)

    xe = np.array([r["L_over_Rg"] for r in exact_rows], dtype=float)
    Fe = np.array([r["exact_DeltaF_kBT"] for r in exact_rows], dtype=float)
    fe = np.array([r["exact_fRg"] for r in exact_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.8))

    # (a) Free energy: MC vs exact
    ax = axes[0, 0]
    ax.errorbar(
        x, F, yerr=Ferr, fmt="o-", ms=6.5, lw=1.8, capsize=3,
        label="Absorbing MC (survival)"
    )
    ax.plot(
        xe, Fe, "k--", lw=1.8,
        label="Absorbing exact lattice"
    )
    ax.plot(
        x, np.zeros_like(x), "s:", ms=5.5, lw=1.5,
        label="Reflecting (fold-back; exact)"
    )
    ax.set_xlabel(r"$L/R_g$")
    ax.set_ylabel(r"$\Delta F\;(k_BT)$")
    ax.set_title("(a) Confinement free energy", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    # (b) Force: MC vs exact
    ax = axes[0, 1]
    ax.errorbar(
        x, f, yerr=ferr, fmt="o-", ms=6.5, lw=1.8, capsize=3,
        label="Absorbing MC (CRN paired)"
    )
    ax.plot(
        xe, fe, "k--", lw=1.8,
        label="Absorbing exact lattice"
    )
    ax.plot(
        x, np.zeros_like(x), "s:", ms=5.5, lw=1.5,
        label="Reflecting (fold-back; exact)"
    )
    ax.axhline(0.0, ls=":", lw=0.9)
    ax.set_xlabel(r"$L/R_g$")
    ax.set_ylabel(r"$fR_g\;(k_BT)$")
    ax.set_title("(b) Signed entropic force", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    # (c) Cumulative force convergence at L=10
    Lmin = min(r["L_center"] for r in conv_rows)
    sub = [
        r for r in conv_rows
        if r["L_center"] == Lmin
    ]
    sub.sort(key=lambda r: r["blocks_used"])
    ax = axes[1, 0]
    xs = np.array([r["blocks_used"] for r in sub], dtype=int)
    ys = np.array([r["fRg"] for r in sub], dtype=float)
    ye = np.array([r["fRg_err"] for r in sub], dtype=float)
    exact_force = next(
        r["exact_fRg"] for r in exact_rows if int(round(
            r["L_over_Rg"] * Rg
        )) == Lmin
    )
    ax.errorbar(
        xs, ys, yerr=ye, fmt="o-", ms=6, lw=1.7, capsize=3,
        label=fr"MC, $L={Lmin}$"
    )
    ax.axhline(
        exact_force, ls="--", lw=1.7,
        label="Exact lattice"
    )
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel(r"$fR_g\;(k_BT)$")
    ax.set_title("(c) Force convergence at strongest confinement", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    # (d) CRN variance reduction
    ax = axes[1, 1]
    ratios = np.array([
        r["CRN_variance_reduction_factor"]
        for r in abs_rows
        if np.isfinite(r["CRN_variance_reduction_factor"])
    ], dtype=float)
    xr = np.arange(ratios.size)
    ax.plot(xr, ratios, "o-", ms=5.5, lw=1.6, label="Independent-width SEM / CRN SEM")
    ax.axhline(1.0, ls="--", lw=1.2, label="No variance reduction")
    ax.set_xticks(xr)
    ax.set_xticklabels(
        [f"{r['L_over_Rg']:.2f}" for r in abs_rows
         if np.isfinite(r["CRN_variance_reduction_factor"])],
        rotation=45, ha="right", fontsize=9
    )
    ax.set_xlabel(r"$L/R_g$")
    ax.set_ylabel("SEM ratio")
    ax.set_title("(d) CRN variance reduction", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    fig.suptitle(
        rf"Supplementary Fig. S9 — Gaussian chain in a slit "
        rf"($N={N}$, $R_g=\sqrt{{N/6}}={Rg:.4f}$)",
        fontsize=15.5, fontweight="bold", y=0.995
    )

    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.105,
        top=0.91, wspace=0.27, hspace=0.32
    )

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Self-test
# =============================================================================

def run_self_tests() -> None:
    # Exact reflecting normalization.
    N = 20
    assert math.isclose(
        (N - 1) * math.log(6.0),
        math.log(6.0 ** (N - 1)),
        rel_tol=1e-13, abs_tol=1e-13,
    )

    # For one-width direct survival at sufficiently wide slit, all paths
    # need not survive, so only check that the kernel executes and returns
    # sensible extrema.
    qmin, qmax = _generate_relative_z_minmax(20, 128, 12345)
    assert qmin.shape == (128,)
    assert qmax.shape == (128,)
    assert np.all(qmin <= 0)
    assert np.all(qmax >= 0)

    # Exact recursion must never exceed the free-space partition count.
    L = 20
    exact = exact_absorbing_logz(N, L)
    logz_free = (N - 1) * math.log(6.0)
    assert np.isfinite(exact)
    assert exact <= logz_free + 1e-12

    print("=" * 82)
    print("S9 REV4 DIRECT-SURVIVAL SELF-TEST")
    print("=" * 82)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("PASS: direct relative-z kernel")
    print("PASS: exact absorbing lattice recursion")
    print("PASS: reflecting exact normalization")
    print("All self-tests passed.")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--trials-per-block", type=int, default=4000)
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--delta-L", type=int, default=2)
    p.add_argument(
        "--centers", nargs="+", type=int,
        default=[10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
    )
    p.add_argument(
        "--block-counts", nargs="+", type=int,
        default=[8, 16, 24, 32, 40],
    )
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument(
        "--outdir", type=Path,
        default=Path("S9_REV4_FINAL"),
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_tests()
        return

    if args.smoke:
        args.N = 60
        args.trials_per_block = 1000
        args.blocks = 8
        args.centers = [6, 8, 10, 12]
        args.block_counts = [2, 4, 8]
        args.bootstrap_reps = 300

    if args.N < 2:
        raise ValueError("N must be >= 2.")
    if args.blocks < 4:
        raise ValueError("Use at least 4 independent blocks.")
    if args.trials_per_block < 100:
        raise ValueError("Use at least 100 trials per block.")
    if args.delta_L <= 0 or args.delta_L % 2 != 0:
        raise ValueError("delta_L must be a positive even integer.")

    centers = sorted(set(int(L) for L in args.centers))
    if not centers:
        raise ValueError("No center widths supplied.")

    for L in centers:
        if L < 4 or L % 2 != 0:
            raise ValueError(f"Center L={L} must be even and >=4.")
        if L - args.delta_L < 4:
            raise ValueError(
                f"Center L={L} too small for delta_L={args.delta_L}; "
                "use a larger center width."
            )

    levels = sorted(set(int(n) for n in args.block_counts if 1 <= n <= args.blocks))
    if not levels:
        raise ValueError("No valid block counts supplied.")

    Rg = math.sqrt(args.N / 6.0)

    all_widths = sorted({
        L for L in centers
    } | {
        L - args.delta_L for L in centers
    } | {
        L + args.delta_L for L in centers
    })

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"{SCRIPT_NAME}  |  version {SCRIPT_VERSION}")
    print("=" * 90)
    print(f"Numba available      : {NUMBA_AVAILABLE}")
    print(f"N                    : {args.N}")
    print(f"Rg                   : {Rg:.9f}")
    print(f"trials/block         : {args.trials_per_block}")
    print(f"blocks               : {args.blocks}")
    print(f"total trials/width   : {args.trials_per_block * args.blocks}")
    print(f"delta L              : {args.delta_L}")
    print(f"center widths        : {centers}")
    print(f"all required widths  : {all_widths}")
    print(f"block checkpoints    : {levels}")
    print("absorbing estimator  : direct six-direction survival probability")
    print("reflecting ensemble  : exact fold-back/mirror result")
    print("CRN                  : ON")
    print("sign filtering       : OFF")
    print("=" * 90)

    t0 = time.time()

    results = sample_blocks(
        N=args.N,
        widths=all_widths,
        trials_per_block=args.trials_per_block,
        blocks=args.blocks,
        seed=args.seed,
    )

    exact_logs = {
        L: exact_absorbing_logz(args.N, L)
        for L in all_widths
    }

    exact_force = {
        L: (
            exact_logs[L + args.delta_L] -
            exact_logs[L - args.delta_L]
        ) / (2.0 * args.delta_L)
        for L in centers
    }

    point_rows, force_rows, exact_rows = assemble_points(
        N=args.N,
        centers=centers,
        delta_L=args.delta_L,
        results=results,
        exact_logs=exact_logs,
        exact_force=exact_force,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    conv_rows = convergence_rows(
        N=args.N,
        centers=centers,
        delta_L=args.delta_L,
        results=results,
        levels=levels,
    )

    # -------------------------------------------------------------------------
    # Save block-level survival / logZ table
    # -------------------------------------------------------------------------
    block_rows = []
    for L in all_widths:
        for b in range(args.blocks):
            block_rows.append({
                "L": int(L),
                "L_over_Rg": float(L / Rg),
                "block": int(b + 1),
                "trials_in_block": int(args.trials_per_block),
                "survivors": int(results["n_survive_block"][L][b]),
                "survival_fraction": float(results["surv_block"][L][b]),
                "logZ_block": float(results["logz_block"][L][b]),
                "exact_logZ": float(exact_logs[L]),
                "logZ_abs_error": float(
                    abs(results["logz_block"][L][b] - exact_logs[L])
                ),
            })

    # -------------------------------------------------------------------------
    # Save all outputs
    # -------------------------------------------------------------------------
    def write_csv(path: Path, rows: List[dict]) -> None:
        """Write heterogeneous row dictionaries safely.

        Some output tables intentionally contain a small amount of extra
        metadata on only a subset of rows (e.g. the exact reflecting-row
        ensemble definition).  Build the header from the union of all keys
        and write missing entries as empty cells rather than aborting.
        """
        if not rows:
            path.write_text("no_rows\n", encoding="utf-8")
            return
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=fieldnames, extrasaction="ignore", restval=""
            )
            writer.writeheader()
            writer.writerows(rows)

    points_csv = outdir / "S9_points.csv"
    block_csv = outdir / "S9_block_level.csv"
    force_csv = outdir / "S9_force_block_level.csv"
    conv_csv = outdir / "S9_convergence.csv"
    exact_csv = outdir / "S9_exact_audit.csv"

    write_csv(points_csv, point_rows)
    write_csv(block_csv, block_rows)
    write_csv(force_csv, force_rows)
    write_csv(conv_csv, conv_rows)
    write_csv(exact_csv, exact_rows)

    # Figure
    png = outdir / "Fig_S9_Gaussian_BC_REV4.png"
    pdf = outdir / "Fig_S9_Gaussian_BC_REV4.pdf"
    make_figure(
        point_rows=point_rows,
        conv_rows=conv_rows,
        exact_rows=exact_rows,
        N=args.N,
        Rg=Rg,
        out_png=png,
        out_pdf=pdf,
    )

    # -------------------------------------------------------------------------
    # Exact audit summary
    # -------------------------------------------------------------------------
    mc_logz = {
        L: logmeanexp(results["logz_block"][L])
        for L in all_widths
    }
    max_logz_err = max(
        abs(mc_logz[L] - exact_logs[L])
        for L in all_widths
    )

    absorbing_points = [
        r for r in point_rows
        if r["boundary"].startswith("absorbing")
    ]

    max_rel_force_error = max(
        r["relative_force_error_pct"]
        for r in absorbing_points
        if np.isfinite(r["relative_force_error_pct"])
    )

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------
    provenance = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "numba": getattr(numba, "__version__", None),
        "N": args.N,
        "Rg": Rg,
        "Rg_definition": "sqrt(N/6), a=1",
        "trials_per_block": args.trials_per_block,
        "blocks": args.blocks,
        "total_trials_per_width": args.trials_per_block * args.blocks,
        "delta_L": args.delta_L,
        "center_widths": centers,
        "all_widths": all_widths,
        "block_checkpoints": levels,
        "geometry": {
            "confinement": "one-coordinate slit",
            "coordinate": "z",
            "walls": "z=0 and z=L",
            "tether": "z=L/2; even L",
            "accessible_layers_absorbing": "z=1,...,L-1",
            "mechanical_derivative":
                "[ln Z(L+dL)-ln Z(L-dL)]/(2 dL), "
                "both walls displaced symmetrically; tether re-centered at L/2",
        },
        "ensembles": {
            "absorbing":
                "survival/first-passage ensemble of ordinary six-direction "
                "unconstrained lattice walks",
            "reflecting":
                "discrete fold-back/mirror stochastic rule, not generic "
                "reflecting walls; exact Z=6^(N-1), DeltaF=0, f=0",
        },
        "absorbing_estimator": {
            "Z_L":
                "6^(N-1) times the empirical survival probability",
            "logZ_block":
                "log(6^(N-1)) + log(survivors/trials_per_block)",
            "final_logZ":
                "log(mean_b exp(logZ_block_b)); equal-sized blocks",
            "DeltaF":
                "logZ_free - final_logZ",
            "DeltaF_uncertainty":
                "nonlinear block bootstrap of the complete DeltaF estimator",
        },
        "force_estimator": {
            "block_force":
                "[lnZ_b(L+dL)-lnZ_b(L-dL)]/(2 dL)",
            "point_force":
                "mean of signed paired block-force estimates",
            "force_uncertainty":
                "SEM of signed paired block-force estimates; bootstrap audit also saved",
            "sign_selection": False,
        },
        "common_random_numbers": {
            "enabled": True,
            "description":
                "Same unconstrained relative z trajectories are replayed "
                "at all widths within each independent block; tether is "
                "re-centered at L/2 for each width.",
        },
        "exact_audit": {
            "method": "exact integer transfer recursion",
            "max_abs_logZ_error": float(max_logz_err),
            "max_relative_force_error_pct": float(max_rel_force_error),
        },
        "outputs": {
            "figure_png": str(png.resolve()),
            "figure_pdf": str(pdf.resolve()),
            "points_csv": str(points_csv.resolve()),
            "block_csv": str(block_csv.resolve()),
            "force_csv": str(force_csv.resolve()),
            "convergence_csv": str(conv_csv.resolve()),
            "exact_audit_csv": str(exact_csv.resolve()),
        },
        "no_data_selection": {
            "negative_force_removed": False,
            "point_removed_for_bad_sign": False,
            "outlier_filter": False,
            "smoothing": False,
            "fitting": False,
        },
        "runtime_seconds": float(time.time() - t0),
    }

    prov_path = outdir / "S9_provenance.json"
    prov_path.write_text(
        json.dumps(provenance, indent=2, default=float),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("S9 REV4 SUMMARY")
    print("=" * 90)

    for r in absorbing_points:
        print(
            f"L={r['L']:3d}  L/Rg={r['L_over_Rg']:.4f}  "
            f"DeltaF={r['DeltaF_kBT']:.6f}  "
            f"fRg={r['fRg']:.6f} +/- {r['fRg_err']:.6f}  "
            f"exact fRg={r['exact_fRg']:.6f}  "
            f"rel.err={r['relative_force_error_pct']:.3f}%  "
            f"min survivors={r['min_survivors_across_derivative_states']}  "
            f"CRN={r['CRN_variance_reduction_factor']:.2f}x  "
            f"[{r['quality_flag']}]"
        )

    print(f"\nMax |logZ_MC - logZ_exact| = {max_logz_err:.6e}")
    print(f"Max relative force error   = {max_rel_force_error:.4f}%")
    print("Reflecting fold-back result: DeltaF=0, f=0 exactly.")
    print(f"Runtime: {(time.time()-t0)/60.0:.2f} min")

    print("\nFiles:")
    for p in [
        png, pdf, points_csv, block_csv,
        force_csv, conv_csv, exact_csv, prov_path
    ]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
