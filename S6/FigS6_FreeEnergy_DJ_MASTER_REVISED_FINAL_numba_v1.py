#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S6 — Domb–Joyce confinement free energy crossover
NUMBA / CENTER-TETHERED / EVEN-L production workflow

Scientific workflow
-------------------
1. R_g(N,w) is read only from the canonical Rg_MASTER_FINAL.csv.
2. The confined ensemble is the same center-tethered absorbing-slit ensemble
   used in the revised S14/PERM workflow:
       walls at z=0 and z=L,
       tether at z=L/2,
       0 < z < L for subsequent monomers,
       L is required to be an even integer.
3. Finite-width Z(L) is estimated with genuine PERM population control.
4. PERM thresholds are frozen from independent, unpruned pilots.
5. The unconfined reference Z(infinity) is estimated separately with the
   canonical unconfined Boltzmann-biased SIS/Rosenbluth estimator used by
   Rg_MASTER_FINAL; no finite-width surrogate and no PERM are used for L=inf.
6. No sign filtering, point deletion, smoothing, interpolation, or outlier
   filtering is applied to the final Delta F data.
7. Uncertainties use independent block bootstrap of
       Delta F = log(mean Z_inf) - log(mean Z_L).
8. Reviewer diagnostics separately audit
       (f) w-dependence for L/Rg < 3 and >= 3,
       (g) monotonicity of Delta F with increasing L/Rg.

Important geometry choice
-------------------------
All finite widths are physical, common even integer L values shared by every w.
The plotted x-axis is the realized L/R_g(w).  The code records the exact realized
L/R_g value; there is no interpolation onto a common reduced-width grid.

Recommended production command
------------------------------
python FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py \
  --rg-master /kaggle/input/datasets/soumyajyotikabi/200-n-rg/Rg_MASTER_FINAL.csv \
  --outdir /kaggle/working/FigS6_DJ_MASTER_FINAL \
  --N 200 \
  --w-values 0.2 0.5 1.0 \
  --L-values 10 12 14 18 20 24 26 30 34 \
  --blocks 20 \
  --roots-per-block 256 \
  --unconfined-roots-per-block 2048 \
  --pilot-roots 3000 \
  --max-population 4096 \
  --C-minus 0.5 \
  --C-plus 2.0 \
  --prune-probability 0.5 \
  --max-clones 4 \
  --bootstrap 5000 \
  --seed 20260928

Validation-only plan check:
  add --dry-run

Smoke test:
  add --smoke

Requires:
  numpy, pandas, matplotlib, numba
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import numba
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Numba is required for this script. Install/enable numba before running."
    ) from exc


SCRIPT_NAME = Path(__file__).name
SCRIPT_VERSION = "1.0-NUMBA-CENTER-EVEN-L"

STEPS_X = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
STEPS_Y = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
STEPS_Z = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.5,
})


# =============================================================================
# Stable scalar helpers
# =============================================================================

def logmeanexp(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -np.inf
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def bootstrap_log_ratio(logz_conf: Sequence[float],
                        logz_inf: Sequence[float],
                        n_boot: int,
                        seed: int) -> Tuple[float, float, float]:
    """Bootstrap DeltaF = log(mean Z_inf) - log(mean Z_L)."""
    a = np.asarray(logz_conf, dtype=float)
    b = np.asarray(logz_inf, dtype=float)
    if a.size < 4 or b.size < 4:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    na, nb = a.size, b.size
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        vals[i] = logmeanexp(b[ib]) - logmeanexp(a[ia])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.std(vals, ddof=1)), float(lo), float(hi)


def monotonicity_diagnostic(x: np.ndarray,
                            y: np.ndarray,
                            yerr: np.ndarray) -> Dict[str, object]:
    order = np.argsort(x)
    xs = np.asarray(x)[order]
    ys = np.asarray(y)[order]
    es = np.asarray(yerr)[order]
    diffs = np.diff(ys)
    diff_err = np.sqrt(es[1:] ** 2 + es[:-1] ** 2)
    raw = np.where(diffs > 0)[0]
    significant = np.where(diffs > 2.0 * diff_err)[0]
    rows = []
    for j, d in enumerate(diffs):
        rows.append({
            "x_left": float(xs[j]),
            "x_right": float(xs[j + 1]),
            "delta_F_right_minus_left": float(d),
            "combined_1sigma": float(diff_err[j]),
            "z_like": float(d / diff_err[j]) if diff_err[j] > 0 else np.nan,
            "raw_increase": bool(d > 0),
            "increase_over_2sigma": bool(d > 2.0 * diff_err[j]),
        })
    return {
        "n_raw_violations": int(raw.size),
        "n_gt_2sigma_violations": int(significant.size),
        "pairs": rows,
    }


def pairwise_region_diagnostic(results_by_w: Dict[float, Dict[str, np.ndarray]],
                                w1: float,
                                w2: float,
                                x_cut: float = 3.0,
                                tolerance: float = 0.18):
    """Descriptive nearest-neighbour comparison; never filters the data."""
    a = results_by_w[w1]
    b = results_by_w[w2]
    rows = []
    for i, xa in enumerate(a["x"]):
        j = int(np.argmin(np.abs(b["x"] - xa)))
        xb = b["x"][j]
        if abs(float(xa - xb)) <= tolerance:
            ea = a["F_err"][i]
            eb = b["F_err"][j]
            diff = a["F"][i] - b["F"][j]
            comb = math.sqrt(ea * ea + eb * eb)
            rows.append({
                "w1": w1,
                "w2": w2,
                "x1": float(xa),
                "x2": float(xb),
                "x_mean": float(0.5 * (xa + xb)),
                "region": "strong" if 0.5 * (xa + xb) < x_cut else "weak",
                "F1_minus_F2": float(diff),
                "combined_1sigma": float(comb),
                "compatible_within_1sigma": bool(abs(diff) <= comb),
            })
    return rows


def summarize_pairwise(rows):
    out = {}
    for region in ("strong", "weak"):
        sub = [r for r in rows if r["region"] == region]
        if not sub:
            out[region] = {"n": 0}
            continue
        compatible = sum(bool(r["compatible_within_1sigma"]) for r in sub)
        out[region] = {
            "n": len(sub),
            "n_compatible_1sigma": compatible,
            "fraction_compatible_1sigma": compatible / len(sub),
            "mean_difference": float(np.mean([r["F1_minus_F2"] for r in sub])),
        }
    return out


# =============================================================================
# Hash-table occupancy kernels — same approach as the revised S14 PERM code
# =============================================================================

@njit(cache=True)
def coord_key(x, y, z):
    bias = np.int64(1_000_000)
    mask = np.uint64((1 << 21) - 1)
    ux = np.uint64(np.int64(x) + bias) & mask
    uy = np.uint64(np.int64(y) + bias) & mask
    uz = np.uint64(np.int64(z) + bias) & mask
    return ((ux << np.uint64(42)) |
            (uy << np.uint64(21)) |
            uz)


@njit(cache=True)
def hidx(key, mask):
    return int((key * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(mask))


@njit(cache=True)
def occ(keys, counts, used, table_size, x, y, z):
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
    for _ in range(1024):
        if used[idx] == 0:
            return 0
        if keys[idx] == key:
            return int(counts[idx])
        idx += np.int64(1)
        if idx >= table_size:
            idx = np.int64(0)
    return -1


@njit(cache=True)
def ins(keys, counts, used, table_size, x, y, z):
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
    for _ in range(1024):
        if used[idx] == 0:
            used[idx] = 1
            keys[idx] = key
            counts[idx] = 1
            return True
        if keys[idx] == key:
            counts[idx] += 1
            return True
        idx += np.int64(1)
        if idx >= table_size:
            idx = np.int64(0)
    return False


@njit(cache=True)
def logsumexp6(v0, v1, v2, v3, v4, v5):
    m = v0
    if v1 > m:
        m = v1
    if v2 > m:
        m = v2
    if v3 > m:
        m = v3
    if v4 > m:
        m = v4
    if v5 > m:
        m = v5
    if not np.isfinite(m):
        return -np.inf
    s = (math.exp(v0 - m) if np.isfinite(v0) else 0.0)
    s += (math.exp(v1 - m) if np.isfinite(v1) else 0.0)
    s += (math.exp(v2 - m) if np.isfinite(v2) else 0.0)
    s += (math.exp(v3 - m) if np.isfinite(v3) else 0.0)
    s += (math.exp(v4 - m) if np.isfinite(v4) else 0.0)
    s += (math.exp(v5 - m) if np.isfinite(v5) else 0.0)
    if s <= 0.0:
        return -np.inf
    return m + math.log(s)


@njit(cache=True)
def weighted_choice6(v0, v1, v2, v3, v4, v5, log_norm):
    """Sample among six candidates whose log relative weights are v_i."""
    u = np.random.random()
    total = 0.0
    vals = np.empty(6, dtype=np.float64)
    vals[0] = v0
    vals[1] = v1
    vals[2] = v2
    vals[3] = v3
    vals[4] = v4
    vals[5] = v5
    for i in range(6):
        if np.isfinite(vals[i]):
            total += math.exp(vals[i] - log_norm)
    target = u * total
    acc = 0.0
    for i in range(6):
        if np.isfinite(vals[i]):
            acc += math.exp(vals[i] - log_norm)
            if target <= acc:
                return i
    # Floating-point fallthrough: choose the last finite candidate.
    for i in range(5, -1, -1):
        if np.isfinite(vals[i]):
            return i
    return -1


# =============================================================================
# Canonical unconfined SIS / Rosenbluth block
# =============================================================================

@njit(cache=True)
def unconfined_sis_block_numba(N, w, roots, seed, table_size):
    """Canonical unconfined DJ SIS/Rosenbluth block."""
    np.random.seed(seed)

    logw_values = np.full(roots, -np.inf, dtype=np.float64)
    keys = np.zeros(table_size, dtype=np.uint64)
    counts = np.zeros(table_size, dtype=np.int16)
    used = np.zeros(table_size, dtype=np.uint8)

    for r in range(roots):
        used[:] = 0
        counts[:] = 0
        x = 0
        y = 0
        z = 0
        ok = ins(keys, counts, used, table_size, x, y, z)
        if not ok:
            continue
        lw = 0.0

        for _ in range(1, N):
            trial = np.full(6, -np.inf, dtype=np.float64)
            for d in range(6):
                xx = x + STEPS_X[d]
                yy = y + STEPS_Y[d]
                zz = z + STEPS_Z[d]
                ov = occ(keys, counts, used, table_size, xx, yy, zz)
                if ov < 0:
                    ok = False
                    break
                trial[d] = -w * ov
            if not ok:
                break

            lnorm = logsumexp6(trial[0], trial[1], trial[2],
                               trial[3], trial[4], trial[5])
            if not np.isfinite(lnorm):
                ok = False
                break
            idx = weighted_choice6(trial[0], trial[1], trial[2],
                                   trial[3], trial[4], trial[5], lnorm)
            if idx < 0:
                ok = False
                break
            lw += lnorm
            x += STEPS_X[idx]
            y += STEPS_Y[idx]
            z += STEPS_Z[idx]
            if not ins(keys, counts, used, table_size, x, y, z):
                ok = False
                break

        if ok:
            logw_values[r] = lw

    finite = np.isfinite(logw_values)
    vals = logw_values[finite]
    nvalid = vals.size
    if nvalid == 0:
        return -np.inf, 0.0, 1.0, 0

    m = np.max(vals)
    ww = np.exp(vals - m)
    s1 = np.sum(ww)
    s2 = np.sum(ww * ww)
    if s1 <= 0.0 or not np.isfinite(s1):
        return -np.inf, 0.0, 1.0, nvalid

    logZ = m + math.log(s1 / roots)
    ess = s1 * s1 / s2 if s2 > 0.0 else 0.0
    maxfrac = np.max(ww) / s1
    return logZ, ess, maxfrac, nvalid


# =============================================================================
# Unpruned pilot for finite-width PERM thresholds
# =============================================================================

@njit(cache=True)
def pilot_thresholds_numba(N, L, w, pilot_roots, c_minus, c_plus,
                           seed, table_size):
    """Unpruned pilot; threshold reference is log(mean original-root weight)."""
    np.random.seed(seed)

    keys = np.zeros((pilot_roots, table_size), dtype=np.uint64)
    counts = np.zeros((pilot_roots, table_size), dtype=np.int16)
    used = np.zeros((pilot_roots, table_size), dtype=np.uint8)
    x = np.zeros(pilot_roots, dtype=np.int64)
    y = np.zeros(pilot_roots, dtype=np.int64)
    z = np.zeros(pilot_roots, dtype=np.int64)
    alive = np.ones(pilot_roots, dtype=np.uint8)
    logw = np.zeros(pilot_roots, dtype=np.float64)

    z0 = L // 2
    for r in range(pilot_roots):
        z[r] = z0
        ins(keys[r], counts[r], used[r], table_size, 0, 0, z0)

    ref = np.full(N, np.nan, dtype=np.float64)
    ref[0] = 0.0
    valid_counts = np.zeros(N, dtype=np.int64)
    valid_counts[0] = pilot_roots

    for depth in range(1, N):
        # Grow every surviving pilot root by one monomer.
        for r in range(pilot_roots):
            if alive[r] == 0:
                continue

            v0 = -np.inf
            v1 = -np.inf
            v2 = -np.inf
            v3 = -np.inf
            v4 = -np.inf
            v5 = -np.inf
            vals = np.empty(6, dtype=np.float64)
            vals[:] = -np.inf

            for d in range(6):
                xx = x[r] + STEPS_X[d]
                yy = y[r] + STEPS_Y[d]
                zz = z[r] + STEPS_Z[d]
                if zz <= 0 or zz >= L:
                    continue
                ov = occ(keys[r], counts[r], used[r], table_size,
                         xx, yy, zz)
                if ov < 0:
                    continue
                vals[d] = -w * ov

            v0, v1, v2, v3, v4, v5 = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
            lnorm = logsumexp6(v0, v1, v2, v3, v4, v5)
            if not np.isfinite(lnorm):
                alive[r] = 0
                continue

            idx = weighted_choice6(v0, v1, v2, v3, v4, v5, lnorm)
            if idx < 0:
                alive[r] = 0
                continue

            logw[r] += lnorm
            x[r] += STEPS_X[idx]
            y[r] += STEPS_Y[idx]
            z[r] += STEPS_Z[idx]
            if not ins(keys[r], counts[r], used[r], table_size,
                       x[r], y[r], z[r]):
                alive[r] = 0
                continue

        # Compute log(mean W * I_alive) over ORIGINAL pilot roots.
        maxlw = -np.inf
        n_alive = 0
        for r in range(pilot_roots):
            if alive[r] != 0:
                n_alive += 1
                if logw[r] > maxlw:
                    maxlw = logw[r]
        valid_counts[depth] = n_alive
        if n_alive == 0:
            return ref, valid_counts, 0

        sum_scaled = 0.0
        for r in range(pilot_roots):
            if alive[r] != 0:
                sum_scaled += math.exp(logw[r] - maxlw)
        ref[depth] = maxlw + math.log(sum_scaled / pilot_roots)

    logm = ref + math.log(c_minus)
    logp = ref + math.log(c_plus)
    return ref, valid_counts, 1


# =============================================================================
# Genuine finite-width PERM block
# =============================================================================

@njit(cache=True)
def perm_block_numba(N, L, w, roots, max_population,
                     prune_survival_probability, max_clones,
                     log_minus, log_plus, seed, table_size):
    """One independent finite-width PERM block.

    Geometry: center tether z=L/2, absorbing walls at z=0,L.
    Returns status 0=OK, 1=extinction, 2=overflow, 3=numeric failure.
    """
    np.random.seed(seed)

    coords_a = np.zeros((max_population, N, 3), dtype=np.int32)
    coords_b = np.zeros((max_population, N, 3), dtype=np.int32)
    keys_a = np.zeros((max_population, table_size), dtype=np.uint64)
    keys_b = np.zeros((max_population, table_size), dtype=np.uint64)
    counts_a = np.zeros((max_population, table_size), dtype=np.int16)
    counts_b = np.zeros((max_population, table_size), dtype=np.int16)
    used_a = np.zeros((max_population, table_size), dtype=np.uint8)
    used_b = np.zeros((max_population, table_size), dtype=np.uint8)
    logw_a = np.full(max_population, -np.inf, dtype=np.float64)
    logw_b = np.full(max_population, -np.inf, dtype=np.float64)

    z0 = L // 2
    pop = roots
    for r in range(roots):
        coords_a[r, 0, 0] = 0
        coords_a[r, 0, 1] = 0
        coords_a[r, 0, 2] = z0
        logw_a[r] = 0.0
        if not ins(keys_a[r], counts_a[r], used_a[r], table_size,
                   0, 0, z0):
            return 3, -np.inf, 0.0, 1.0, 0, roots

    max_pop_seen = roots

    for depth in range(1, N):
        grown = 0

        # Grow every existing walker by one step.
        # The parent remains in buffer A until its child state is copied to B.
        for parent in range(pop):
            if not np.isfinite(logw_a[parent]):
                continue

            x = coords_a[parent, depth - 1, 0]
            y = coords_a[parent, depth - 1, 1]
            z = coords_a[parent, depth - 1, 2]

            vals = np.full(6, -np.inf, dtype=np.float64)
            for d in range(6):
                xx = x + STEPS_X[d]
                yy = y + STEPS_Y[d]
                zz = z + STEPS_Z[d]
                if zz <= 0 or zz >= L:
                    continue
                ov = occ(keys_a[parent], counts_a[parent], used_a[parent],
                         table_size, xx, yy, zz)
                if ov < 0:
                    vals[d] = -np.inf
                else:
                    vals[d] = -w * ov

            lnorm = logsumexp6(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
            if not np.isfinite(lnorm):
                continue
            idx = weighted_choice6(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], lnorm)
            if idx < 0:
                continue

            nx = x + STEPS_X[idx]
            ny = y + STEPS_Y[idx]
            nz = z + STEPS_Z[idx]
            child_logw = logw_a[parent] + lnorm

            if grown >= max_population:
                return 2, -np.inf, 0.0, 1.0, grown, max_pop_seen
            child = grown
            grown += 1

            # Copy parent trajectory and occupancy state, then insert the new site.
            for q in range(depth):
                coords_b[child, q, 0] = coords_a[parent, q, 0]
                coords_b[child, q, 1] = coords_a[parent, q, 1]
                coords_b[child, q, 2] = coords_a[parent, q, 2]
            coords_b[child, depth, 0] = nx
            coords_b[child, depth, 1] = ny
            coords_b[child, depth, 2] = nz

            for q in range(table_size):
                keys_b[child, q] = keys_a[parent, q]
                counts_b[child, q] = counts_a[parent, q]
                used_b[child, q] = used_a[parent, q]

            if not ins(keys_b[child], counts_b[child], used_b[child], table_size,
                       nx, ny, nz):
                logw_b[child] = -np.inf
                continue
            logw_b[child] = child_logw

        if grown == 0:
            return 1, -np.inf, 0.0, 1.0, 0, max_pop_seen

        # Population control at this depth, using frozen pilot thresholds.
        next_pop = 0
        lm = log_minus[depth]
        lp = log_plus[depth]
        for child in range(grown):
            lw = logw_b[child]
            if not np.isfinite(lw):
                continue

            if lw > lp:
                # Deterministic enrichment by cloning identical child states.
                ratio = math.exp(min(lw - lp, 20.0))
                nk = int(math.ceil(ratio))
                if nk < 2:
                    nk = 2
                if nk > max_clones:
                    nk = max_clones
                split_logw = lw - math.log(nk)
                for c in range(nk):
                    if next_pop >= max_population:
                        return 2, -np.inf, 0.0, 1.0, next_pop, max_pop_seen
                    dest = next_pop
                    next_pop += 1
                    for q in range(depth + 1):
                        coords_a[dest, q, 0] = coords_b[child, q, 0]
                        coords_a[dest, q, 1] = coords_b[child, q, 1]
                        coords_a[dest, q, 2] = coords_b[child, q, 2]
                    for q in range(table_size):
                        keys_a[dest, q] = keys_b[child, q]
                        counts_a[dest, q] = counts_b[child, q]
                        used_a[dest, q] = used_b[child, q]
                    logw_a[dest] = split_logw

            elif lw < lm:
                # Same convention as revised S14: prune-survival probability p;
                # if retained, compensate W -> W/p.
                if np.random.random() < prune_survival_probability:
                    if next_pop >= max_population:
                        return 2, -np.inf, 0.0, 1.0, next_pop, max_pop_seen
                    dest = next_pop
                    next_pop += 1
                    for q in range(depth + 1):
                        coords_a[dest, q, 0] = coords_b[child, q, 0]
                        coords_a[dest, q, 1] = coords_b[child, q, 1]
                        coords_a[dest, q, 2] = coords_b[child, q, 2]
                    for q in range(table_size):
                        keys_a[dest, q] = keys_b[child, q]
                        counts_a[dest, q] = counts_b[child, q]
                        used_a[dest, q] = used_b[child, q]
                    logw_a[dest] = lw - math.log(prune_survival_probability)

            else:
                if next_pop >= max_population:
                    return 2, -np.inf, 0.0, 1.0, next_pop, max_pop_seen
                dest = next_pop
                next_pop += 1
                for q in range(depth + 1):
                    coords_a[dest, q, 0] = coords_b[child, q, 0]
                    coords_a[dest, q, 1] = coords_b[child, q, 1]
                    coords_a[dest, q, 2] = coords_b[child, q, 2]
                for q in range(table_size):
                    keys_a[dest, q] = keys_b[child, q]
                    counts_a[dest, q] = counts_b[child, q]
                    used_a[dest, q] = used_b[child, q]
                logw_a[dest] = lw

        # Clear the output log-weight buffer used next round.
        for i in range(max_population):
            logw_b[i] = -np.inf

        if next_pop == 0:
            return 1, -np.inf, 0.0, 1.0, 0, max_pop_seen

        pop = next_pop
        if pop > max_pop_seen:
            max_pop_seen = pop

    # Final partition-function estimate from population weights.
    m = -np.inf
    for i in range(pop):
        if np.isfinite(logw_a[i]) and logw_a[i] > m:
            m = logw_a[i]
    if not np.isfinite(m):
        return 3, -np.inf, 0.0, 1.0, pop, max_pop_seen

    s1 = 0.0
    s2 = 0.0
    max_scaled = 0.0
    for i in range(pop):
        if np.isfinite(logw_a[i]):
            ww = math.exp(logw_a[i] - m)
            s1 += ww
            s2 += ww * ww
            if ww > max_scaled:
                max_scaled = ww
    if s1 <= 0.0 or not np.isfinite(s1):
        return 3, -np.inf, 0.0, 1.0, pop, max_pop_seen

    logZ = m + math.log(s1 / roots)
    ess = (s1 * s1) / s2 if s2 > 0.0 else 0.0
    maxfrac = max_scaled / s1
    return 0, logZ, ess, maxfrac, pop, max_pop_seen


# =============================================================================
# Python wrappers around Numba kernels
# =============================================================================

def table_size_for(N: int) -> int:
    size = 1
    while size < 3 * (N + 64):
        size *= 2
    return size


def load_master_rg(path: str, N: int, w_values: Sequence[float]):
    df = pd.read_csv(path)
    required = {
        "model", "N", "w", "Rg", "Rg_err", "quality_flag",
        "min_ESS", "median_ESS", "max_block_weight_fraction", "estimator"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master Rg CSV is missing required columns: {missing}")
    dj = df[(df["model"].astype(str).str.upper() == "DJ") & (df["N"] == N)].copy()
    if dj.empty:
        raise ValueError(f"No DJ rows found for N={N}: {path}")

    out = {}
    for w in w_values:
        rows = dj[np.isclose(dj["w"].astype(float), float(w), rtol=0.0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one DJ row for N={N}, w={w}; found {len(rows)}")
        r = rows.iloc[0].to_dict()
        Rg = float(r["Rg"])
        if not np.isfinite(Rg) or Rg <= 0:
            raise ValueError(f"Invalid Rg for w={w}: {Rg}")
        out[float(w)] = {
            "Rg": Rg,
            "Rg_err": float(r["Rg_err"]),
            "quality_flag": str(r["quality_flag"]),
            "min_ESS": float(r["min_ESS"]),
            "median_ESS": float(r["median_ESS"]),
            "max_block_weight_fraction": float(r["max_block_weight_fraction"]),
            "estimator": str(r["estimator"]),
            "source_row": r,
        }
    return out


def build_frozen_thresholds(args, table_size):
    thresholds = {}
    state_index = 0
    for w in args.w_values:
        for L in args.L_values:
            seed = args.seed + 17_000_003 + 1_000_003 * state_index
            ref, valid_counts, ok = pilot_thresholds_numba(
                args.N, int(L), float(w), args.pilot_roots,
                args.C_minus, args.C_plus, seed, table_size
            )
            if ok != 1:
                bad = np.where(valid_counts == 0)[0]
                depth = int(bad[0]) if bad.size else -1
                raise RuntimeError(
                    f"PERM pilot failed: N={args.N}, w={w}, L={L}, depth={depth}"
                )
            thresholds[f"w={float(w):.12g}|L={int(L)}"] = {
                "N": args.N,
                "w": float(w),
                "L": int(L),
                "pilot_roots": args.pilot_roots,
                "reference_logw_by_depth": ref.tolist(),
                "logW_minus_by_depth": (ref + math.log(args.C_minus)).tolist(),
                "logW_plus_by_depth": (ref + math.log(args.C_plus)).tolist(),
                "valid_counts_by_depth": valid_counts.tolist(),
                "C_minus": float(args.C_minus),
                "C_plus": float(args.C_plus),
                "seed": int(seed),
            }
            state_index += 1
    return thresholds


def run_unconfined_blocks(args, w: float, table_size: int, nblocks: int, roots: int):
    rows = []
    logs = []
    for b in range(nblocks):
        seed = args.seed + 300_000_007 + int(round(w * 1000)) * 1009 + 10_000_019 * b
        logZ, ess, maxfrac, valid = unconfined_sis_block_numba(
            args.N, float(w), int(roots), int(seed), table_size
        )
        if not np.isfinite(logZ):
            raise RuntimeError(f"Unconfined SIS failed for w={w}, block={b}")
        logs.append(float(logZ))
        rows.append({
            "w": float(w),
            "block": b,
            "logZ": float(logZ),
            "ESS": float(ess),
            "max_weight_fraction": float(maxfrac),
            "valid_chains": int(valid),
            "roots": int(roots),
            "sampler": "canonical_unconfined_SIS",
        })
    return logs, rows


def run_finite_state(args, w: float, L: int, thresholds, table_size: int):
    sid = f"w={float(w):.12g}|L={int(L)}"
    th = thresholds[sid]
    lm = np.asarray(th["logW_minus_by_depth"], dtype=np.float64)
    lp = np.asarray(th["logW_plus_by_depth"], dtype=np.float64)

    logs = []
    records = []
    for b in range(args.blocks):
        seed = args.seed + 101_000_003 * (1 + b + int(round(w * 10)) * 100 + int(L))
        status, logZ, ess, maxfrac, finalpop, maxpop = perm_block_numba(
            args.N,
            int(L),
            float(w),
            int(args.roots_per_block),
            int(args.max_population),
            float(args.prune_probability),
            int(args.max_clones),
            lm,
            lp,
            int(seed),
            table_size,
        )
        if status == 1:
            raise RuntimeError(f"PERM extinction: N={args.N}, w={w}, L={L}, block={b}")
        if status == 2:
            raise RuntimeError(f"PERM population overflow: N={args.N}, w={w}, L={L}, block={b}")
        if status == 3:
            raise RuntimeError(f"PERM numerical failure: N={args.N}, w={w}, L={L}, block={b}")
        logs.append(float(logZ))
        records.append({
            "w": float(w),
            "L": int(L),
            "block": b,
            "logZ": float(logZ),
            "ESS": float(ess),
            "max_weight_fraction": float(maxfrac),
            "population_final": int(finalpop),
            "population_max_seen": int(maxpop),
            "roots": int(args.roots_per_block),
            "status": "OK",
            "sampler": "NUMBA_PERM",
            "tether_z": int(L // 2),
        })
    return logs, records


def block_convergence(logs):
    out = {}
    for n in (12, 16, len(logs)):
        if n <= len(logs):
            out[str(n)] = logmeanexp(logs[:n])
    return out


def make_plot(data, master_rg, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    colors = plt.cm.plasma(np.linspace(0.18, 0.88, len(data)))
    for color, (w, d) in zip(colors, sorted(data.items())):
        order = np.argsort(d["x"])
        ax.errorbar(
            d["x"][order], d["F"][order], yerr=d["F_err"][order],
            fmt="o-", ms=6.5, lw=1.6, capsize=3,
            color=color,
            label=rf"$w={w:g}$" + (" (Rg caution)" if str(master_rg[w]["quality_flag"]).upper() != "OK" else ""),
        )
    ax.axvline(3.0, color="0.45", ls="--", lw=1.0, alpha=0.8)
    y0, y1 = ax.get_ylim()
    ax.text(3.03, y1 - 0.04 * (y1 - y0), r"$L/R_g=3$", fontsize=11, va="top", ha="left")
    ax.set_xlabel(r"Scaled slit width $L/R_g(w)$")
    ax.set_ylabel(r"Confinement free energy $\Delta F\;(k_BT)$")
    ax.set_title(r"Domb--Joyce confinement free energy crossover ($N=200$)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run_selftest():
    print("Numba version:", numba.__version__)
    # Small internal kernel tests for geometry and unconfined SIS.
    N = 20
    L = 12
    assert L % 2 == 0
    table = table_size_for(N)
    logZ, ess, maxf, valid = unconfined_sis_block_numba(N, 0.2, 64, 1234, table)
    assert np.isfinite(logZ) and ess > 0 and valid == 64
    ref, counts, ok = pilot_thresholds_numba(N, L, 0.2, 64, 0.5, 2.0, 5678, table)
    assert ok == 1 and np.all(np.isfinite(ref)) and counts[-1] > 0
    status, logZp, essp, maxfp, fp, mp = perm_block_numba(
        N, L, 0.2, 32, 128, 0.5, 4,
        ref + math.log(0.5), ref + math.log(2.0), 91011, table
    )
    assert status == 0 and np.isfinite(logZp) and essp > 0 and fp > 0 and mp <= 128
    print("SELFTEST PASS: unconfined SIS, center-tether pilot, and finite-width PERM.")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--rg-master", required=True)
    p.add_argument("--outdir", default="FigS6_DJ_MASTER_FINAL")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--w-values", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    p.add_argument("--L-values", nargs="+", type=int, default=[10, 12, 14, 18, 20, 24, 26, 30, 34])
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--roots-per-block", type=int, default=256)
    p.add_argument("--unconfined-roots-per-block", type=int, default=2048)
    p.add_argument("--pilot-roots", type=int, default=3000)
    p.add_argument("--max-population", type=int, default=4096)
    p.add_argument("--C-minus", type=float, default=0.5)
    p.add_argument("--C-plus", type=float, default=2.0)
    p.add_argument("--prune-probability", type=float, default=0.5,
                   help="Survival probability in the low-weight Russian-roulette step; same convention as revised S14.")
    p.add_argument("--max-clones", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Reduced N/blocks/roots/L settings for workflow validation only.")
    args = p.parse_args()

    if args.self_test:
        run_selftest()
        return

    if args.N < 2:
        raise ValueError("N must be >= 2")
    if any(int(L) <= 2 or int(L) % 2 != 0 for L in args.L_values):
        raise ValueError(
            f"Every finite slit width L must be a positive even integer for center tethering; got {args.L_values}"
        )
    if args.blocks < 4:
        raise ValueError("Use at least 4 independent blocks")
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

    if args.smoke:
        args.N = min(args.N, 40)
        args.blocks = min(args.blocks, 4)
        args.roots_per_block = min(args.roots_per_block, 32)
        args.unconfined_roots_per_block = min(args.unconfined_roots_per_block, 64)
        args.pilot_roots = min(args.pilot_roots, 128)
        args.max_population = min(args.max_population, 128)
        args.bootstrap = min(args.bootstrap, 500)
        args.L_values = [int(L) for L in args.L_values[:3]]
        print("SMOKE MODE: workflow validation only; do not use output for the manuscript.")

    os.makedirs(args.outdir, exist_ok=True)
    master_rg = load_master_rg(args.rg_master, args.N, args.w_values)
    table_size = table_size_for(args.N)

    print("=" * 98)
    print("SUPPLEMENTARY FIG. S6 — NUMBA / CENTER-TETHERED / EVEN-L / MASTER-Rg")
    print("=" * 98)
    print(f"Script/version        : {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"Numba                 : {numba.__version__}")
    print(f"Master CSV            : {args.rg_master}")
    print(f"N                     : {args.N}")
    print(f"DJ w values           : {args.w_values}")
    print(f"Physical even L values: {args.L_values}")
    print(f"Blocks                : {args.blocks}")
    print(f"Roots/block (finite)  : {args.roots_per_block}")
    print(f"Roots/block (inf)     : {args.unconfined_roots_per_block}")
    print(f"Pilot roots/state     : {args.pilot_roots}")
    print(f"PERM C-/C+            : {args.C_minus}/{args.C_plus}")
    print(f"Max population        : {args.max_population}")
    print(f"Hash table size       : {table_size}")
    print("Geometry              : absorbing walls z=0,L; center tether z=L/2; even L")
    print("Unconfined reference  : canonical SIS/Rosenbluth, L=None")
    print("Filtering/smoothing   : OFF")
    print("=" * 98)

    for w in args.w_values:
        r = master_rg[float(w)]
        print(
            f"master Rg(w={w:g}) = {r['Rg']:.9f} +/- {r['Rg_err']:.9f}; "
            f"quality={r['quality_flag']}; minESS={r['min_ESS']:.2f}"
        )

    if args.dry_run:
        print("\nDRY RUN: master dataset validated; no Monte Carlo performed.")
        for w in args.w_values:
            Rg = master_rg[float(w)]["Rg"]
            vals = [L / Rg for L in args.L_values]
            print(f"  w={w:g}: realized L/Rg = {[round(v, 6) for v in vals]}")
        return

    t0 = time.time()

    # ---------------------------------------------------------------------
    # Frozen PERM thresholds for finite-width states only.
    # ---------------------------------------------------------------------
    print("\nBuilding independent frozen PERM thresholds for finite-width states...")
    thresholds = build_frozen_thresholds(args, table_size)
    thr_path = os.path.join(args.outdir, "FigS6_PERM_FROZEN_THRESHOLDS.json")
    with open(thr_path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)

    # ---------------------------------------------------------------------
    # Unconfined reference blocks.
    # ---------------------------------------------------------------------
    all_state_rows = []
    all_block_rows = []
    conf_logs = {}
    inf_logs = {}
    inf_rows = {}

    print("\nRunning genuine unconfined SIS references...")
    for w in args.w_values:
        logs, rows = run_unconfined_blocks(
            args, float(w), table_size, args.blocks, args.unconfined_roots_per_block
        )
        inf_logs[float(w)] = logs
        inf_rows[float(w)] = rows
        zmean = logmeanexp(logs)
        print(
            f"  w={w:g}: logZ_inf={zmean:.8f} +/- "
            f"{np.std(logs, ddof=1) / math.sqrt(len(logs)):.8f}; "
            f"minESS={min(r['ESS'] for r in rows):.2f}"
        )
        for r in rows:
            all_block_rows.append(dict(r, L="inf", tether_z=0))

    # ---------------------------------------------------------------------
    # Finite-width PERM states.
    # ---------------------------------------------------------------------
    production = {}
    print("\nRunning center-tethered finite-width PERM states...")
    for w in args.w_values:
        for L in args.L_values:
            print(f"  state w={w:g}, L={L}, tether z={L//2}")
            logs, rows = run_finite_state(args, float(w), int(L), thresholds, table_size)
            key = (float(w), int(L))
            production[key] = logs
            for r in rows:
                all_block_rows.append(r)
            zmean = logmeanexp(logs)
            print(
                f"      logZ={zmean:.8f}; minESS={min(r['ESS'] for r in rows):.2f}; "
                f"maxWeightFrac={max(r['max_weight_fraction'] for r in rows):.4f}; "
                f"maxPop={max(r['population_max_seen'] for r in rows)}"
            )

    # ---------------------------------------------------------------------
    # Build Delta F data.
    # ---------------------------------------------------------------------
    data = {}
    point_rows = []
    mono_rows = []
    pair_rows = []
    pair_summaries = {}
    convergence_rows = []

    for w in args.w_values:
        w = float(w)
        logz_inf = logmeanexp(inf_logs[w])
        x_list, F_list, E_list = [], [], []
        for L in args.L_values:
            L = int(L)
            logsL = production[(w, L)]
            logz_L = logmeanexp(logsL)
            DeltaF = logz_inf - logz_L
            boot_seed = args.seed + 700_000_007 + int(round(w * 1000)) * 1009 + L
            F_se, F_lo, F_hi = bootstrap_log_ratio(logsL, inf_logs[w], args.bootstrap, boot_seed)
            Rg = master_rg[w]["Rg"]
            Rg_err = master_rg[w]["Rg_err"]
            x = L / Rg
            xerr = L * Rg_err / (Rg * Rg)

            conv_inf = block_convergence(inf_logs[w])
            conv_L = block_convergence(logsL)
            conv = {
                str(n): conv_inf[str(n)] - conv_L[str(n)]
                for n in (12, 16, args.blocks)
                if str(n) in conv_inf and str(n) in conv_L
            }

            point_rows.append({
                "model": "DJ",
                "N": args.N,
                "w": w,
                "L_realized": L,
                "tether_z": L // 2,
                "Rg_master": Rg,
                "Rg_master_err": Rg_err,
                "Rg_master_quality": master_rg[w]["quality_flag"],
                "L_over_Rg": x,
                "L_over_Rg_err": xerr,
                "logZ_L": logz_L,
                "logZ_inf": logz_inf,
                "DeltaF": DeltaF,
                "DeltaF_bootstrap_SE": F_se,
                "DeltaF_bootstrap_CI95_low": F_lo,
                "DeltaF_bootstrap_CI95_high": F_hi,
                "DeltaF_12": conv.get("12", np.nan),
                "DeltaF_16": conv.get("16", np.nan),
                f"DeltaF_{args.blocks}": conv.get(str(args.blocks), np.nan),
                "status": "OK",
                "geometry": "center-tethered absorbing slit, even L",
            })
            x_list.append(x)
            F_list.append(DeltaF)
            E_list.append(F_se)

        data[w] = {
            "x": np.asarray(x_list, dtype=float),
            "F": np.asarray(F_list, dtype=float),
            "F_err": np.asarray(E_list, dtype=float),
            "Rg": master_rg[w]["Rg"],
            "Rg_err": master_rg[w]["Rg_err"],
        }

        md = monotonicity_diagnostic(data[w]["x"], data[w]["F"], data[w]["F_err"])
        for pair in md["pairs"]:
            mono_rows.append({"w": w, **pair})
        print(
            f"Monotonicity audit w={w:g}: raw increases={md['n_raw_violations']}; "
            f">2sigma increases={md['n_gt_2sigma_violations']}"
        )

        for w1, w2 in [(0.2, 0.5), (0.5, 1.0), (0.2, 1.0)]:
            if w1 in data and w2 in data and (w, L) == (w, int(args.L_values[-1])):
                rows = pairwise_region_diagnostic(data, w1, w2, x_cut=3.0, tolerance=0.18)
                pair_rows.extend(rows)
                pair_summaries[f"{w1:g}_vs_{w2:g}"] = summarize_pairwise(rows)

    # ---------------------------------------------------------------------
    # Convergence state audit.
    # ---------------------------------------------------------------------
    for w in args.w_values:
        w = float(w)
        ci = block_convergence(inf_logs[w])
        for L in args.L_values:
            L = int(L)
            cf = block_convergence(production[(w, L)])
            for n in (12, 16, args.blocks):
                k = str(n)
                if k in ci and k in cf:
                    v = ci[k] - cf[k]
                    convergence_rows.append({
                        "w": w,
                        "L": L,
                        "blocks_used": n,
                        "DeltaF_blocks": v,
                    })

    # ---------------------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------------------
    pd.DataFrame(point_rows).to_csv(os.path.join(args.outdir, "FigS6_point_level.csv"), index=False)
    pd.DataFrame(mono_rows).to_csv(os.path.join(args.outdir, "FigS6_monotonicity_audit.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(args.outdir, "FigS6_w_dependence_region_audit.csv"), index=False)
    pd.DataFrame(convergence_rows).to_csv(os.path.join(args.outdir, "FigS6_convergence_audit.csv"), index=False)
    pd.DataFrame(all_block_rows).to_csv(os.path.join(args.outdir, "FigS6_block_level.csv"), index=False)

    state_rows = []
    for w in args.w_values:
        w = float(w)
        inf = inf_rows[w]
        state_rows.append({
            "state_id": f"w={w:g}|L=inf",
            "w": w,
            "L": "inf",
            "tether_z": 0,
            "logZ_mean": logmeanexp(inf_logs[w]),
            "logZ_SE_block": np.std(inf_logs[w], ddof=1) / math.sqrt(len(inf_logs[w])),
            "ESS_min": min(r["ESS"] for r in inf),
            "ESS_median": float(np.median([r["ESS"] for r in inf])),
            "max_weight_fraction_max": max(r["max_weight_fraction"] for r in inf),
            "sampler": "canonical_unconfined_SIS",
        })
        for L in args.L_values:
            logs = production[(w, int(L))]
            br = [r for r in all_block_rows if r.get("w") == w and r.get("L") == int(L)]
            state_rows.append({
                "state_id": f"w={w:g}|L={int(L)}",
                "w": w,
                "L": int(L),
                "tether_z": int(L // 2),
                "logZ_mean": logmeanexp(logs),
                "logZ_SE_block": np.std(logs, ddof=1) / math.sqrt(len(logs)),
                "ESS_min": min(r["ESS"] for r in br),
                "ESS_median": float(np.median([r["ESS"] for r in br])),
                "max_weight_fraction_max": max(r["max_weight_fraction"] for r in br),
                "population_max_seen": max(r["population_max_seen"] for r in br),
                "sampler": "NUMBA_PERM",
            })
    pd.DataFrame(state_rows).to_csv(os.path.join(args.outdir, "FigS6_state_level_production.csv"), index=False)

    provenance = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numba_version": numba.__version__,
        "figure": "Supplementary Fig. S6",
        "N": args.N,
        "w_values": [float(w) for w in args.w_values],
        "physical_even_L_values": [int(L) for L in args.L_values],
        "master_Rg_file": os.path.abspath(args.rg_master),
        "geometry": {
            "walls": "absorbing at z=0 and z=L",
            "allowed_region": "0 < z < L",
            "tether": "z=L/2",
            "L_constraint": "even integer",
            "common_physical_L_grid": True,
        },
        "finite_sampler": "NUMBA PERM with frozen independent unpruned thresholds",
        "unconfined_sampler": "canonical Boltzmann-biased SIS/Rosenbluth, no PERM",
        "roots_per_block_finite": args.roots_per_block,
        "roots_per_block_unconfined": args.unconfined_roots_per_block,
        "pilot_roots": args.pilot_roots,
        "blocks": args.blocks,
        "C_minus": args.C_minus,
        "C_plus": args.C_plus,
        "prune_survival_probability": args.prune_probability,
        "max_clones": args.max_clones,
        "max_population": args.max_population,
        "filtering": False,
        "smoothing": False,
        "sign_selection": False,
        "reviewer_1_comment_f": "w-dependence audited below and above L/Rg=3; raw data retained",
        "reviewer_1_comment_g": "monotonicity audited using signed raw DeltaF and >2sigma pairwise increases",
        "pairwise_region_summary": pair_summaries,
    }
    with open(os.path.join(args.outdir, "FigS6_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=float)

    png = os.path.join(args.outdir, "FigS6_DJ_FreeEnergy_MASTER_REVISED.png")
    pdf = os.path.join(args.outdir, "FigS6_DJ_FreeEnergy_MASTER_REVISED.pdf")
    make_plot(data, master_rg, png, pdf)

    elapsed = (time.time() - t0) / 60.0
    print("\n" + "=" * 98)
    print("S6 COMPLETE")
    print(f"Elapsed minutes: {elapsed:.2f}")
    print("Outputs written to:", os.path.abspath(args.outdir))
    print("Pairwise reviewer-(f) summaries:")
    for k, v in pair_summaries.items():
        print(f"  {k}: {v}")
    print("=" * 98)


if __name__ == "__main__":
    main()
