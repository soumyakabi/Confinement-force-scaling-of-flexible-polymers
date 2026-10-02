#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FigS8_FreeEnergy_Overlap_REV8_FINAL.py

Free-energy and overlap comparison of Gaussian, Domb-Joyce, and strict
self-avoiding walks in a center-tethered absorbing slit.

Workflow identical to S14, Fig7ii, FigS6-numba, FigS7-numba:
  * numba PERM (hash-table, dual buffer, Russian roulette, weight-preserving clones)
  * frozen thresholds from an independent unpruned pilot
  * genuine unconfined reference (L = infinity)
  * same physical L grid for all models
  * center tether at L/2, even L only, absorbing walls at 0 and L
  * Rg from the canonical Rg_MASTER_FINAL.csv
  * cumulative-block convergence at n = 4, 8, 12, 16, 20

Reviewer 1, comment (h):
  "In Fig. S8, the confinement free energy of the Domb-Joyce model appears
   to be larger than those of both the Gaussian and SAW models ... The
   physical reason for this behavior is not clear."

This revision fixes the reference and L-grid ambiguities that produced the
original appearance, and adds a fixed-L interpolation panel so the crossover
Gaussian -> DJ(w) -> SAW is directly visible.

No sign selection. No smoothing. No point deletion.
"""

from __future__ import annotations
import argparse, json, math, os, platform, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    def njit(*a, **k):
        def deco(f): return f
        return deco


def _seed32(x) -> int:
    return int(x) & 0xFFFFFFFF


SCRIPT_NAME = "FigS8_FreeEnergy_Overlap_REV8_FINAL.py"
SCRIPT_VERSION = "8.0-PRODUCTION-NUMBA"
KBT = 1.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13, "axes.labelsize": 16, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.fontsize": 9, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.3, "savefig.dpi": 600,
})


# =============================================================================
# Numba kernels (center tether, even L; DJ and strict SAW share the engine)
# =============================================================================

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _coord_key(x, y, z):
        bias = np.int64(1_000_000)
        mask = np.uint64((1 << 21) - 1)
        ux = np.uint64(np.int64(x) + bias) & mask
        uy = np.uint64(np.int64(y) + bias) & mask
        uz = np.uint64(np.int64(z) + bias) & mask
        return (ux << np.uint64(42)) | (uy << np.uint64(21)) | uz

    @njit(cache=True)
    def _hidx(key, mask):
        return int((key * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(mask))

    @njit(cache=True)
    def _occ(keys, counts, used, table_size, x, y, z):
        key = _coord_key(x, y, z)
        idx = np.int64(_hidx(key, table_size - 1))
        for _ in range(2048):
            if used[idx] == 0:
                return 0
            if keys[idx] == key:
                return int(counts[idx])
            idx += np.int64(1)
            if idx >= table_size:
                idx = np.int64(0)
        return -1

    @njit(cache=True)
    def _ins(keys, counts, used, table_size, x, y, z):
        key = _coord_key(x, y, z)
        idx = np.int64(_hidx(key, table_size - 1))
        for _ in range(2048):
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
    def _logsumexp6(vals):
        m = -np.inf
        for i in range(6):
            v = vals[i]
            if np.isfinite(v) and v > m:
                m = v
        if not np.isfinite(m):
            return -np.inf
        s = 0.0
        for i in range(6):
            v = vals[i]
            if np.isfinite(v):
                s += math.exp(v - m)
        return m + math.log(s) if s > 0.0 else -np.inf

    @njit(cache=True)
    def _pilot_kernel(N, L, w, saw, unconfined, roots, seed):
        """
        Unpruned pilot.
          saw=0 : Domb-Joyce (Boltzmann-weighted moves, weight = logsumexp of trials)
          saw=1 : strict SAW (uniform among unvisited in-slit moves, weight = log count)
          unconfined=1 : no walls at all (genuine L=infinity reference)
        Returns pilot[k] = log(mean_root W_k I_alive), k = 0..N.
        """
        np.random.seed(seed)
        ts = 1
        while ts < 3 * (N + 64):
            ts *= 2

        dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
        dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
        dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

        logsum = np.full(N + 1, -np.inf, dtype=np.float64)
        logsum[0] = 0.0

        z0 = 0 if unconfined == 1 else L // 2

        for r in range(roots):
            keys = np.zeros(ts, dtype=np.uint64)
            counts = np.zeros(ts, dtype=np.int16)
            used = np.zeros(ts, dtype=np.uint8)

            x = 0; y = 0; z = z0
            if not _ins(keys, counts, used, ts, x, y, z):
                continue

            lw = 0.0
            for k in range(1, N):
                a = logsum[k]; b = lw
                if not np.isfinite(a):
                    logsum[k] = b
                elif b > a:
                    logsum[k] = b + math.log1p(math.exp(a - b))
                else:
                    logsum[k] = a + math.log1p(math.exp(b - a))

                trial = np.full(6, -np.inf, dtype=np.float64)
                bad = False
                for d in range(6):
                    xx = x + dx[d]; yy = y + dy[d]; zz = z + dz[d]
                    if unconfined == 0:
                        if zz <= 0 or zz >= L:
                            continue
                    ov = _occ(keys, counts, used, ts, xx, yy, zz)
                    if ov < 0:
                        bad = True; break
                    if saw == 1:
                        if ov == 0:
                            trial[d] = 0.0
                    else:
                        trial[d] = -w * ov
                if bad:
                    break

                nvalid = 0
                for d in range(6):
                    if np.isfinite(trial[d]):
                        nvalid += 1
                if nvalid == 0:
                    break

                if saw == 1:
                    norm = math.log(float(nvalid))
                    u = np.random.random() * nvalid
                    acc = 0.0; chosen = -1
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += 1.0
                            if u <= acc:
                                chosen = d; break
                else:
                    norm = _logsumexp6(trial)
                    if not np.isfinite(norm):
                        break
                    m = -np.inf
                    for d in range(6):
                        if np.isfinite(trial[d]) and trial[d] > m:
                            m = trial[d]
                    total = 0.0
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            total += math.exp(trial[d] - m)
                    u = np.random.random() * total
                    acc = 0.0; chosen = -1
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += math.exp(trial[d] - m)
                            if u <= acc:
                                chosen = d; break
                if chosen < 0:
                    break

                lw += norm
                x += dx[chosen]; y += dy[chosen]; z += dz[chosen]
                if not _ins(keys, counts, used, ts, x, y, z):
                    break

            if np.isfinite(lw):
                a = logsum[N]; b = lw
                if not np.isfinite(a):
                    logsum[N] = b
                elif b > a:
                    logsum[N] = b + math.log1p(math.exp(a - b))
                else:
                    logsum[N] = a + math.log1p(math.exp(b - a))

        lr = math.log(roots)
        for k in range(N + 1):
            if np.isfinite(logsum[k]):
                logsum[k] -= lr
        return logsum

    @njit(cache=True)
    def _perm_block_kernel(N, L, w, saw, unconfined, roots, max_pop, seed,
                           prune_p, max_clones, log_low, log_high):
        """One production block. Returns (logZ, ESS, maxfrac, pop, maxpop, prune, clone, overflow)."""
        np.random.seed(seed)
        ts = 1
        while ts < 3 * (N + 64):
            ts *= 2

        ca = np.zeros((max_pop, N, 3), dtype=np.int32)
        cb = np.zeros((max_pop, N, 3), dtype=np.int32)
        ka = np.zeros((max_pop, ts), dtype=np.uint64)
        kb = np.zeros((max_pop, ts), dtype=np.uint64)
        na = np.zeros((max_pop, ts), dtype=np.int16)
        nb = np.zeros((max_pop, ts), dtype=np.int16)
        ua = np.zeros((max_pop, ts), dtype=np.uint8)
        ub = np.zeros((max_pop, ts), dtype=np.uint8)
        la = np.full(max_pop, -np.inf, dtype=np.float64)
        lb = np.full(max_pop, -np.inf, dtype=np.float64)

        dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
        dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
        dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

        pop = roots
        if pop > max_pop:
            return (-np.inf, 0.0, 1.0, 0, 0, 0, 0, 1)

        z0 = 0 if unconfined == 1 else L // 2
        for r in range(pop):
            ca[r, 0, 0] = 0; ca[r, 0, 1] = 0; ca[r, 0, 2] = z0
            la[r] = 0.0
            if not _ins(ka[r], na[r], ua[r], ts, 0, 0, z0):
                return (-np.inf, 0.0, 1.0, 0, 0, 0, 0, 1)

        pr_deaths = 0; cl_events = 0; ov_events = 0; mx_pop = pop

        for k in range(1, N):
            new_pop = 0
            low_log = log_low[k + 1]
            high_log = log_high[k + 1]

            for parent in range(pop):
                if not np.isfinite(la[parent]):
                    continue
                x = ca[parent, k - 1, 0]
                y = ca[parent, k - 1, 1]
                z = ca[parent, k - 1, 2]

                trial = np.full(6, -np.inf, dtype=np.float64)
                bad = False
                for d in range(6):
                    xx = x + dx[d]; yy = y + dy[d]; zz = z + dz[d]
                    if unconfined == 0:
                        if zz <= 0 or zz >= L:
                            continue
                    ov = _occ(ka[parent], na[parent], ua[parent], ts, xx, yy, zz)
                    if ov < 0:
                        bad = True; break
                    if saw == 1:
                        if ov == 0:
                            trial[d] = 0.0
                    else:
                        trial[d] = -w * ov
                if bad:
                    continue

                nvalid = 0
                for d in range(6):
                    if np.isfinite(trial[d]):
                        nvalid += 1
                if nvalid == 0:
                    continue

                if saw == 1:
                    norm = math.log(float(nvalid))
                    u = np.random.random() * nvalid
                    acc = 0.0; chosen = -1
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += 1.0
                            if u <= acc:
                                chosen = d; break
                else:
                    norm = _logsumexp6(trial)
                    if not np.isfinite(norm):
                        continue
                    m = -np.inf
                    for d in range(6):
                        if np.isfinite(trial[d]) and trial[d] > m:
                            m = trial[d]
                    total = 0.0
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            total += math.exp(trial[d] - m)
                    u = np.random.random() * total
                    acc = 0.0; chosen = -1
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += math.exp(trial[d] - m)
                            if u <= acc:
                                chosen = d; break
                if chosen < 0:
                    continue

                nx = x + dx[chosen]; ny = y + dy[chosen]; nz = z + dz[chosen]
                new_lw = la[parent] + norm

                if new_lw < low_log:
                    if np.random.random() < prune_p:
                        pr_deaths += 1
                        continue
                    new_lw -= math.log(1.0 - prune_p)

                nk = 1
                if new_lw > high_log:
                    ratio = math.exp(min(20.0, new_lw - high_log))
                    nk = int(math.ceil(ratio))
                    if nk < 2: nk = 2
                    if nk > max_clones: nk = max_clones

                if new_pop + nk > max_pop:
                    ov_events += 1
                    return (-np.inf, 0.0, 1.0, new_pop, mx_pop,
                            pr_deaths, cl_events, ov_events)

                child_logw = new_lw - math.log(nk)
                child = new_pop; new_pop += 1
                for q in range(k):
                    cb[child, q, 0] = ca[parent, q, 0]
                    cb[child, q, 1] = ca[parent, q, 1]
                    cb[child, q, 2] = ca[parent, q, 2]
                cb[child, k, 0] = nx; cb[child, k, 1] = ny; cb[child, k, 2] = nz
                for q in range(ts):
                    kb[child, q] = ka[parent, q]
                    nb[child, q] = na[parent, q]
                    ub[child, q] = ua[parent, q]
                if not _ins(kb[child], nb[child], ub[child], ts, nx, ny, nz):
                    new_pop -= 1
                    continue
                lb[child] = child_logw

                for c in range(1, nk):
                    clone = new_pop; new_pop += 1
                    cl_events += 1
                    for q in range(k + 1):
                        cb[clone, q, 0] = cb[child, q, 0]
                        cb[clone, q, 1] = cb[child, q, 1]
                        cb[clone, q, 2] = cb[child, q, 2]
                    for q in range(ts):
                        kb[clone, q] = kb[child, q]
                        nb[clone, q] = nb[child, q]
                        ub[clone, q] = ub[child, q]
                    lb[clone] = child_logw

            tmp = ca; ca = cb; cb = tmp
            tmp = ka; ka = kb; kb = tmp
            tmp = na; na = nb; nb = tmp
            tmp = ua; ua = ub; ub = tmp
            tmp = la; la = lb; lb = tmp

            for i in range(max_pop):
                lb[i] = -np.inf

            pop = new_pop
            if pop > mx_pop:
                mx_pop = pop
            if pop == 0:
                return (-np.inf, 0.0, 1.0, 0, mx_pop,
                        pr_deaths, cl_events, ov_events)

        cnt = 0
        vals = np.empty(pop, dtype=np.float64)
        for i in range(pop):
            if np.isfinite(la[i]):
                vals[cnt] = la[i]; cnt += 1
        vals = vals[:cnt]
        if vals.size == 0:
            return (-np.inf, 0.0, 1.0, 0, mx_pop,
                    pr_deaths, cl_events, ov_events)

        vmax = np.max(vals)
        ww = np.exp(vals - vmax)
        s1 = np.sum(ww); s2 = np.sum(ww * ww)
        if s1 <= 0.0 or s2 <= 0.0:
            return (-np.inf, 0.0, 1.0, vals.size, mx_pop,
                    pr_deaths, cl_events, ov_events)

        logz = vmax + math.log(s1 / roots)
        ess = s1 * s1 / s2
        maxfrac = np.max(ww / s1)
        return (logz, ess, maxfrac, vals.size, mx_pop,
                pr_deaths, cl_events, ov_events)



# REV8 updates:
#   * true L=infinity reference for BOTH DJ and strict SAW via unconfined SIS/Rosenbluth
#   * adopted production PERM: 40 blocks x 384 roots, C-/C+=0.5/2.0, max clones=4
#   * cumulative levels 8,16,24,32,40
#   * block-level DeltaF = lnZ_inf,b - lnZ_L,b; uncertainty from block differences
#   * fixed physical L grid for all models
#   * DJ includes w=0, 0.1, 0.3, 0.5; SAW is the w->infinity endpoint
#   * fixed-L categorical comparison and quantitative pairwise-separation audit
#   * no sign selection, filtering, smoothing, fitting, or point deletion


# =============================================================================
# Revised numerical helpers
# =============================================================================

def mean_sem(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(np.mean(x)) if x.size else np.nan), np.nan
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size))


def bootstrap_mean(values, reps, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or reps <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=(int(reps), x.size))
    means = x[idx].mean(axis=1)
    return (float(np.std(means, ddof=1)),
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def stable_seed(master_seed, stream, model_code, wcode, Lcode, block=0):
    ss = np.random.SeedSequence([
        int(master_seed), int(stream), int(model_code), int(wcode), int(Lcode), int(block)
    ])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def load_master_rg(path: Path, N: int, w_values):
    df = pd.read_csv(path)
    required = {"model", "N", "w", "Rg", "Rg_err", "quality_flag"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Rg master missing required columns: {missing}")

    dj = df[(df["model"].astype(str).str.upper() == "DJ") & (df["N"].astype(int) == int(N))]
    if dj.empty:
        raise ValueError(f"No DJ N={N} rows in {path}")

    out = {}
    for w in w_values:
        rows = dj[np.isclose(dj["w"].astype(float), float(w), rtol=0.0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one DJ row for N={N}, w={w}; found {len(rows)}")
        r = rows.iloc[0]
        out[float(w)] = {
            "Rg": float(r["Rg"]),
            "Rg_err": float(r["Rg_err"]),
            "quality_flag": str(r["quality_flag"]),
            "min_ESS": float(r["min_ESS"]) if "min_ESS" in r else np.nan,
            "median_ESS": float(r["median_ESS"]) if "median_ESS" in r else np.nan,
            "max_block_weight_fraction": float(r["max_block_weight_fraction"]) if "max_block_weight_fraction" in r else np.nan,
        }

    saw = df[(df["model"].astype(str).str.upper() == "SAW") & (df["N"].astype(int) == int(N))]
    if len(saw) != 1:
        raise ValueError(f"Expected exactly one SAW row for N={N}; found {len(saw)}")
    r = saw.iloc[0]
    out["saw"] = {
        "Rg": float(r["Rg"]),
        "Rg_err": float(r["Rg_err"]),
        "quality_flag": str(r["quality_flag"]),
        "min_ESS": float(r["min_ESS"]) if "min_ESS" in r else np.nan,
        "median_ESS": float(r["median_ESS"]) if "median_ESS" in r else np.nan,
        "max_block_weight_fraction": float(r["max_block_weight_fraction"]) if "max_block_weight_fraction" in r else np.nan,
    }
    return out


def gaussian_deltaF(N: int, L: float, a: float = 1.0):
    """Exact continuous Gaussian confinement free energy relative to L=infinity."""
    if not np.isfinite(L) or L > 5000:
        return 0.0
    Rg2 = a * a * N / 6.0
    x0 = 0.5 * L
    s = 0.0
    for m in range(20000):
        n = 2 * m + 1
        k = n * math.pi / L
        amp = (4.0 / (n * math.pi)) * math.sin(k * x0)
        term = amp * math.exp(-k * k * Rg2)
        s += term
        if m > 8 and abs(term) < 1e-15 * max(1.0, abs(s)):
            break
    if s <= 0.0 or not np.isfinite(s):
        return float("nan")
    return float(-math.log(s))


# =============================================================================
# True unconfined SIS kernel (DJ and strict SAW)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _unconfined_sis_block_kernel(N, w, saw, roots, seed):
        """One independent unconfined SIS/Rosenbluth block.

        saw=0: DJ Boltzmann-biased proposal q ~ exp(-w DeltaU)
        saw=1: strict SAW Rosenbluth growth, uniform over allowed moves

        The estimator is the mean of the complete-path Rosenbluth weights,
        with dead chains contributing zero weight.
        """
        np.random.seed(seed)
        ts = 1
        while ts < 3 * (N + 64):
            ts *= 2
        dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
        dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
        dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

        logsum = -np.inf
        sumsq_scaled = 0.0
        max_scaled = 0.0
        valid = 0
        max_logw = -np.inf
        logw_vals = np.full(roots, -np.inf, dtype=np.float64)

        for r in range(roots):
            keys = np.zeros(ts, dtype=np.uint64)
            counts = np.zeros(ts, dtype=np.int16)
            used = np.zeros(ts, dtype=np.uint8)
            x = 0; y = 0; z = 0
            if not _ins(keys, counts, used, ts, x, y, z):
                continue
            lw = 0.0
            alive = True

            for k in range(1, N):
                trial = np.full(6, -np.inf, dtype=np.float64)
                bad = False
                for d in range(6):
                    xx = x + dx[d]; yy = y + dy[d]; zz = z + dz[d]
                    ov = _occ(keys, counts, used, ts, xx, yy, zz)
                    if ov < 0:
                        bad = True; break
                    if saw == 1:
                        if ov == 0:
                            trial[d] = 0.0
                    else:
                        trial[d] = -w * ov
                if bad:
                    alive = False; break

                nvalid = 0
                for d in range(6):
                    if np.isfinite(trial[d]):
                        nvalid += 1
                if nvalid == 0:
                    alive = False; break

                if saw == 1:
                    norm = math.log(float(nvalid))
                    u = np.random.random() * nvalid
                    chosen = -1
                    acc = 0.0
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += 1.0
                            if u <= acc:
                                chosen = d; break
                else:
                    norm = _logsumexp6(trial)
                    if not np.isfinite(norm):
                        alive = False; break
                    m = -np.inf
                    for d in range(6):
                        if np.isfinite(trial[d]) and trial[d] > m:
                            m = trial[d]
                    total = 0.0
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            total += math.exp(trial[d] - m)
                    u = np.random.random() * total
                    chosen = -1
                    acc = 0.0
                    for d in range(6):
                        if np.isfinite(trial[d]):
                            acc += math.exp(trial[d] - m)
                            if u <= acc:
                                chosen = d; break

                if chosen < 0:
                    alive = False; break
                lw += norm
                x += dx[chosen]; y += dy[chosen]; z += dz[chosen]
                if not _ins(keys, counts, used, ts, x, y, z):
                    alive = False; break

            if alive and np.isfinite(lw):
                logw_vals[valid] = lw
                valid += 1
                if lw > max_logw:
                    max_logw = lw

        if valid == 0 or not np.isfinite(max_logw):
            return (-np.inf, 0.0, 1.0, 0)

        s1 = 0.0
        s2 = 0.0
        for i in range(valid):
            q = math.exp(logw_vals[i] - max_logw)
            s1 += q
            s2 += q * q
        logZ = max_logw + math.log(s1 / roots)
        ess = s1 * s1 / s2 if s2 > 0.0 else 0.0
        maxfrac = 1.0 / s1
        return (logZ, ess, maxfrac, valid)


# =============================================================================
# Production wrappers
# =============================================================================

def run_unconfined_state(N, w, saw, blocks, roots_per_block, seed):
    records = []
    for b in range(blocks):
        model_code = 2 if saw else 1
        wcode = int(round(float(w) * 1000000.0))
        s = stable_seed(seed, 8101, model_code, wcode, 0, b)
        logZ, ess, maxfrac, valid = _unconfined_sis_block_kernel(
            int(N), float(w), int(saw), int(roots_per_block), int(_seed32(s))
        )
        logZ = float(logZ); ess = float(ess); maxfrac = float(maxfrac); valid = int(valid)
        if not np.isfinite(logZ):
            raise RuntimeError(f"Unconfined SIS failed: saw={saw}, w={w}, block={b}")
        if valid != roots_per_block:
            # For SAW dead chains are possible; they have zero statistical weight.
            # We therefore require only a finite estimator, not all roots alive.
            if valid <= 0:
                raise RuntimeError(f"Unconfined SIS has zero valid chains: saw={saw}, w={w}, block={b}")
        records.append({
            "model": "SAW" if saw else "DJ",
            "ensemble": "unconfined_SIS",
            "w": float("inf") if saw else float(w),
            "L": np.inf,
            "block_id": int(b),
            "logZ": logZ,
            "ESS": ess,
            "max_weight_fraction": maxfrac,
            "population_final": int(valid),
            "population_max": int(valid),
            "prune_deaths": 0,
            "clone_events": 0,
            "overflow": 0,
            "valid_roots": int(valid),
            "roots": int(roots_per_block),
            "status": "OK",
        })
    return {"model": "SAW" if saw else "DJ", "ensemble": "unconfined_SIS",
            "w": float("inf") if saw else float(w), "L": np.inf,
            "saw": int(saw), "blocks": records}


def make_pilot(N, L, w, saw, pilot_roots, c_minus, c_plus, seed):
    pilot = _pilot_kernel(int(N), int(L), float(w), int(saw), 0, int(pilot_roots), _seed32(seed))
    if not np.all(np.isfinite(pilot)):
        raise RuntimeError(f"Pilot failed for saw={saw}, w={w}, L={L}")
    return (pilot + math.log(c_minus), pilot + math.log(c_plus))


def run_finite_state(N, L, w, saw, blocks, roots_per_block, max_population,
                    prune_p, max_clones, log_low, log_high, seed):
    records = []
    for b in range(blocks):
        model_code = 2 if saw else 1
        wcode = 0 if saw else int(round(float(w) * 1000000.0))
        s = stable_seed(seed, 8201, model_code, wcode, int(L), b)
        logZ, ess, maxfrac, pop, maxpop, pr, cl, ov = _perm_block_kernel(
            int(N), int(L), float(w), int(saw), 0, int(roots_per_block),
            int(max_population), _seed32(s), float(prune_p), int(max_clones),
            log_low, log_high
        )
        vals = (float(logZ), float(ess), float(maxfrac), int(pop), int(maxpop), int(pr), int(cl), int(ov))
        if not np.isfinite(vals[0]) or vals[7] != 0:
            raise RuntimeError(f"Finite PERM failed: saw={saw}, w={w}, L={L}, block={b}, result={vals}")
        records.append({
            "model": "SAW" if saw else "DJ",
            "ensemble": "finite_absorbing_PERM",
            "w": float("inf") if saw else float(w),
            "L": int(L), "block_id": int(b), "logZ": vals[0], "ESS": vals[1],
            "max_weight_fraction": vals[2], "population_final": vals[3],
            "population_max": vals[4], "prune_deaths": vals[5],
            "clone_events": vals[6], "overflow": vals[7],
            "valid_roots": int(roots_per_block), "roots": int(roots_per_block),
            "status": "OK",
        })
    return {"model": "SAW" if saw else "DJ", "ensemble": "finite_absorbing_PERM",
            "w": float("inf") if saw else float(w), "L": int(L), "saw": int(saw),
            "blocks": records}


def state_logz_blocks(state):
    return np.asarray([b["logZ"] for b in state["blocks"]], dtype=float)


def state_ess_blocks(state):
    return np.asarray([b["ESS"] for b in state["blocks"]], dtype=float)


def state_maxfrac_blocks(state):
    return np.asarray([b["max_weight_fraction"] for b in state["blocks"]], dtype=float)


def dF_block_statistics(ref, conf, reps, seed):
    a = state_logz_blocks(ref)
    b = state_logz_blocks(conf)
    if len(a) != len(b):
        raise ValueError("Reference and confined state must have the same block count")
    d = a - b
    mean, sem = mean_sem(d)
    bse, lo, hi = bootstrap_mean(d, reps, seed)
    return mean, sem, bse, lo, hi, d


def cumulative_deltaf(ref, conf, n):
    d = state_logz_blocks(ref)[:n] - state_logz_blocks(conf)[:n]
    return mean_sem(d)


def build_states_plan(w_values, L_values):
    plan = []
    for w in w_values:
        plan.append(("DJ", None, float(w), 0))
    plan.append(("SAW", None, float("inf"), 1))
    for L in L_values:
        if int(L) % 2 != 0 or int(L) <= 0:
            raise ValueError(f"All L values must be positive even integers; got {L}")
        for w in w_values:
            plan.append(("DJ", int(L), float(w), 0))
        plan.append(("SAW", int(L), float("inf"), 1))
    return plan


def run_plan(N, w_values, L_values, blocks, roots_per_block, unconfined_roots,
             pilot_roots, max_population, c_minus, c_plus, prune_p, max_clones, seed):
    states = {}
    thresholds = {}
    plan = build_states_plan(w_values, L_values)
    print(f"Total physical states: {len(plan)}")

    # Frozen independent pilots ONLY for finite-width PERM states.
    print("\nBuilding frozen independent PERM pilots...")
    for kind, L, w, saw in plan:
        if L is None:
            continue
        wkey = 0.0 if saw else float(w)
        tag = ("SAW" if saw else "DJ")
        s = stable_seed(seed, 9101, 2 if saw else 1,
                        0 if saw else int(round(w*1e6)), int(L), 0)
        lo, hi = make_pilot(N, L, wkey, saw, pilot_roots, c_minus, c_plus, s)
        thresholds[(tag, int(L), float(wkey), int(saw))] = (lo, hi)
        print(f"  pilot OK: {tag} L={L} w={'inf' if saw else str(w)}")

    # True unconfined references using canonical SIS/Rosenbluth.
    print("\nRunning true unconfined SIS/Rosenbluth references...")
    for w in w_values:
        st = run_unconfined_state(N, float(w), 0, blocks, unconfined_roots, seed + 500_000_000)
        states[(None, float(w), 0)] = st
        e = state_ess_blocks(st); f = state_maxfrac_blocks(st)
        print(f"  DJ w={w:g}: logZ={np.mean(state_logz_blocks(st)):.9f} +/- {np.std(state_logz_blocks(st),ddof=1)/math.sqrt(blocks):.9f}; minESS={np.min(e):.1f}; maxfrac={np.max(f):.4f}")
    st = run_unconfined_state(N, 0.0, 1, blocks, unconfined_roots, seed + 600_000_000)
    states[(None, float("inf"), 1)] = st
    e = state_ess_blocks(st); f = state_maxfrac_blocks(st)
    print(f"  SAW: logZ={np.mean(state_logz_blocks(st)):.9f} +/- {np.std(state_logz_blocks(st),ddof=1)/math.sqrt(blocks):.9f}; minESS={np.min(e):.1f}; maxfrac={np.max(f):.4f}")

    # Finite-width PERM states.
    print("\nRunning finite-width PERM states...")
    for L in L_values:
        for w in w_values:
            key = ("DJ", int(L), float(w), 0)
            lo, hi = thresholds[key]
            st = run_finite_state(N, int(L), float(w), 0, blocks, roots_per_block,
                                  max_population, prune_p, max_clones, lo, hi,
                                  seed + 700_000_000)
            states[(int(L), float(w), 0)] = st
            e = state_ess_blocks(st); f = state_maxfrac_blocks(st)
            if not np.all(np.isfinite(e)) or np.max(f) > 1.0:
                raise RuntimeError(f"Invalid diagnostics in {key}")
            print(f"  DJ w={w:g} L={L}: minESS={np.min(e):.1f}; maxfrac={np.max(f):.4f}")
        key = ("SAW", int(L), 0.0, 1)
        lo, hi = thresholds[key]
        st = run_finite_state(N, int(L), 0.0, 1, blocks, roots_per_block,
                              max_population, prune_p, max_clones, lo, hi,
                              seed + 800_000_000)
        states[(int(L), float("inf"), 1)] = st
        e = state_ess_blocks(st); f = state_maxfrac_blocks(st)
        print(f"  SAW L={L}: minESS={np.min(e):.1f}; maxfrac={np.max(f):.4f}")

    return states, thresholds


# =============================================================================
# Assembly and audits
# =============================================================================

def assemble_rows(states, rg_master, w_values, L_values, N, block_counts, bootstrap, seed):
    rows = []
    for n in block_counts:
        for w in w_values:
            ref = states[(None, float(w), 0)]
            for L in L_values:
                conf = states[(int(L), float(w), 0)]
                bseed = stable_seed(seed, 9301, 1, int(round(w*1e6)), int(L), n)
                F, sem, bse, lo, hi, dblocks = dF_block_statistics(ref, conf, bootstrap, bseed)
                R = rg_master[float(w)]['Rg']; Re = rg_master[float(w)]['Rg_err']
                x = L/R; xe = L*Re/(R*R)
                rows.append({
                    'model':'DJ','w':float(w),'saw':0,'L':int(L),'blocks_used':int(n),
                    'Rg':R,'Rg_err':Re,'Rg_quality':rg_master[float(w)]['quality_flag'],
                    'L_over_Rg':x,'L_over_Rg_err':xe,
                    'logZ_inf':float(np.mean(state_logz_blocks(ref)[:n])),
                    'logZ_L':float(np.mean(state_logz_blocks(conf)[:n])),
                    'DeltaF':F,'DeltaF_err':sem,'DeltaF_bootstrap_SE':bse,
                    'DeltaF_CI95_low':lo,'DeltaF_CI95_high':hi,
                    'ESS_min_all':float(min(np.min(state_ess_blocks(ref)[:n]),np.min(state_ess_blocks(conf)[:n]))),
                    'max_weight_fraction_max':float(max(np.max(state_maxfrac_blocks(ref)[:n]),np.max(state_maxfrac_blocks(conf)[:n]))),
                })
        ref = states[(None,float('inf'),1)]
        R = rg_master['saw']['Rg']; Re = rg_master['saw']['Rg_err']
        for L in L_values:
            conf = states[(int(L),float('inf'),1)]
            bseed = stable_seed(seed, 9401, 2, 0, int(L), n)
            F, sem, bse, lo, hi, dblocks = dF_block_statistics(ref, conf, bootstrap, bseed)
            rows.append({
                'model':'SAW','w':float('inf'),'saw':1,'L':int(L),'blocks_used':int(n),
                'Rg':R,'Rg_err':Re,'Rg_quality':rg_master['saw']['quality_flag'],
                'L_over_Rg':L/R,'L_over_Rg_err':L*Re/(R*R),
                'logZ_inf':float(np.mean(state_logz_blocks(ref)[:n])),
                'logZ_L':float(np.mean(state_logz_blocks(conf)[:n])),
                'DeltaF':F,'DeltaF_err':sem,'DeltaF_bootstrap_SE':bse,
                'DeltaF_CI95_low':lo,'DeltaF_CI95_high':hi,
                'ESS_min_all':float(min(np.min(state_ess_blocks(ref)[:n]),np.min(state_ess_blocks(conf)[:n]))),
                'max_weight_fraction_max':float(max(np.max(state_maxfrac_blocks(ref)[:n]),np.max(state_maxfrac_blocks(conf)[:n]))),
            })
    return rows


def final_rows(rows):
    return [r for r in rows if r['blocks_used'] == max(r['blocks_used'] for r2 in rows for r in [r2])]


def model_ordering_audit(rows, rg_master, w_values, L_values, N, comparison_L_values):
    final_n = max(r['blocks_used'] for r in rows)
    df = pd.DataFrame(rows)
    records=[]
    # Adjacent descriptive comparisons: Gaussian -> DJ(w=0) -> DJ(0.1) -> DJ(0.3) -> DJ(0.5) -> SAW.
    model_nodes=[('Gaussian',0.0,0), *[(f'DJ_w{w:g}',float(w),0) for w in w_values], ('SAW',float('inf'),1)]
    for L in comparison_L_values:
        vals=[]
        Fg=gaussian_deltaF(N,float(L))
        vals.append(('Gaussian',Fg,0.0))
        for w in w_values:
            q=df[(df.model=='DJ') & (df.w==float(w)) & (df.L==int(L)) & (df.blocks_used==final_n)]
            if q.empty: raise RuntimeError(f'Missing DJ comparison row w={w}, L={L}')
            vals.append((f'DJ(w={w:g})',float(q.DeltaF.iloc[0]),float(q.DeltaF_err.iloc[0])))
        q=df[(df.model=='SAW') & (df.L==int(L)) & (df.blocks_used==final_n)]
        if q.empty: raise RuntimeError(f'Missing SAW comparison row L={L}')
        vals.append(('SAW (w->inf)',float(q.DeltaF.iloc[0]),float(q.DeltaF_err.iloc[0])))
        for i in range(len(vals)-1):
            a,fa,ea=vals[i]; b,fb,eb=vals[i+1]
            diff=fb-fa; comb=math.sqrt(ea*ea+eb*eb)
            records.append({
                'L':int(L),'left':a,'right':b,'left_DeltaF':fa,'right_DeltaF':fb,
                'right_minus_left':diff,'combined_1sigma':comb,
                'z_like':diff/comb if comb>0 else np.nan,
                'compatible_with_equal_at_1sigma':abs(diff)<=comb,
                'direction':'right_higher' if diff>0 else ('right_lower' if diff<0 else 'equal')
            })
    return pd.DataFrame(records)


def sampling_summary(rows):
    df=pd.DataFrame(rows)
    # retain final block level only for sampling table
    n=df.blocks_used.max()
    q=df[df.blocks_used==n]
    return q[['model','w','L','blocks_used','ESS_min_all','max_weight_fraction_max','Rg_quality']].drop_duplicates().sort_values(['model','w','L'])


# =============================================================================
# Figure
# =============================================================================

def make_figure(rows, rg_master, w_values, N, L_values, comparison_L_values,
                convergence_L, out_png, out_pdf):
    df=pd.DataFrame(rows)
    n_final=int(df.blocks_used.max())
    final=df[df.blocks_used==n_final]
    fig,axes=plt.subplots(2,2,figsize=(14,10))

    # (a) scaled free energy
    ax=axes[0,0]
    RgG=math.sqrt(N/6.0)
    xg=np.array([L/RgG for L in np.linspace(min(L_values),max(L_values),300)])
    Lg=xg*RgG
    ax.plot(xg,[gaussian_deltaF(N,float(L)) for L in Lg],'k--',lw=2,label='Gaussian (exact)')
    for w in w_values:
        sub=final[(final.model=='DJ')&(final.w==float(w))].sort_values('L_over_Rg')
        ax.errorbar(sub.L_over_Rg,sub.DeltaF,xerr=sub.L_over_Rg_err,yerr=sub.DeltaF_err,
                    fmt='o-',ms=6,capsize=3,label=fr'DJ $w={w:g}$')
    sub=final[final.model=='SAW'].sort_values('L_over_Rg')
    ax.errorbar(sub.L_over_Rg,sub.DeltaF,xerr=sub.L_over_Rg_err,yerr=sub.DeltaF_err,
                fmt='s-',ms=6,capsize=3,label=r'SAW ($w\to\infty$)')
    ax.set_xlabel(r'Scaled slit width $L/R_g$')
    ax.set_ylabel(r'Confinement free energy $\Delta F\;(k_BT)$')
    ax.set_title('(a) Free-energy scaling across models',loc='left')
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9,loc='best')

    # (b) categorical fixed-L comparison
    ax=axes[0,1]
    labels=['Gaussian']+[fr'DJ $w={w:g}$' for w in w_values]+[r'SAW\n($w\to\infty$)']
    xs=np.arange(len(labels))
    for L in comparison_L_values:
        y=[gaussian_deltaF(N,float(L))]
        e=[0.0]
        for w in w_values:
            q=final[(final.model=='DJ')&(final.w==float(w))&(final.L==int(L))].iloc[0]
            y.append(float(q.DeltaF)); e.append(float(q.DeltaF_err))
        q=final[(final.model=='SAW')&(final.L==int(L))].iloc[0]
        y.append(float(q.DeltaF)); e.append(float(q.DeltaF_err))
        ax.errorbar(xs,y,yerr=e,fmt='o-',ms=5,capsize=3,label=fr'$L={L}$')
    ax.set_xticks(xs,labels,rotation=18,ha='right')
    ax.set_ylabel(r'$\Delta F\;(k_BT)$ at fixed physical $L$')
    ax.set_title('(b) Fixed-physical-width model comparison',loc='left')
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9,loc='best')

    # (c) convergence
    ax=axes[1,0]
    for w in w_values:
        q=df[(df.model=='DJ')&(df.w==float(w))&(df.L==int(convergence_L))].sort_values('blocks_used')
        ax.errorbar(q.blocks_used,q.DeltaF,yerr=q.DeltaF_err,fmt='o-',ms=5,capsize=3,label=fr'DJ $w={w:g}$')
    q=df[(df.model=='SAW')&(df.L==int(convergence_L))].sort_values('blocks_used')
    ax.errorbar(q.blocks_used,q.DeltaF,yerr=q.DeltaF_err,fmt='s-',ms=5,capsize=3,label='SAW')
    ax.set_xlabel('Cumulative blocks')
    ax.set_ylabel(r'$\Delta F\;(k_BT)$')
    ax.set_title(fr'(c) $\Delta F$ convergence at $L={convergence_L}$',loc='left')
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9,loc='best')

    # (d) ESS diagnostic at most confined L
    ax=axes[1,1]
    Lmin=min(L_values)
    for w in w_values:
        q=df[(df.model=='DJ')&(df.w==float(w))&(df.L==int(Lmin))].sort_values('blocks_used')
        ax.plot(q.blocks_used,q.ESS_min_all,'o-',ms=5,label=fr'DJ $w={w:g}$')
    q=df[(df.model=='SAW')&(df.L==int(Lmin))].sort_values('blocks_used')
    ax.plot(q.blocks_used,q.ESS_min_all,'s-',ms=5,label='SAW')
    ax.axhline(20.0,ls=':',lw=1.3,label='ESS=20 diagnostic')
    ax.set_xlabel('Cumulative blocks')
    ax.set_ylabel('Minimum block ESS')
    ax.set_title(fr'(d) Low-ESS diagnostic at $L={Lmin}$',loc='left')
    ax.set_yscale('log')
    ax.grid(alpha=0.22,which='both')
    ax.legend(fontsize=9,loc='best')

    fig.suptitle(r'Supplementary Fig. S8 — free-energy comparison across Gaussian, Domb–Joyce, and SAW models',
                 fontsize=16,fontweight='bold',y=0.995)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_png,dpi=600,bbox_inches='tight')
    fig.savefig(out_pdf,bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Self-test and CLI
# =============================================================================

def self_test():
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is required for S8 REV8.')
    # Unconfined SIS: DJ and SAW small tests
    for saw,w in [(0,0.0),(0,0.3),(1,0.0)]:
        z,ess,mf,valid=_unconfined_sis_block_kernel(20,float(w),saw,64,12345+saw)
        assert np.isfinite(z) and ess>0 and mf>0 and valid>0
    # Exact Gaussian large-width limit and positive confinement cost
    assert abs(gaussian_deltaF(200,1e9))<1e-12
    assert gaussian_deltaF(200,12)>0
    print('PASS: Numba + unconfined SIS DJ/SAW smoke kernels')
    print('PASS: Gaussian exact confinement function')
    print('All S8 REV8 self-tests passed.')


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                               description='S8 REV8 production free-energy comparison.')
    p.add_argument('--rg-master',required=True,type=Path)
    p.add_argument('--outdir',type=Path,default=Path('FigS8_REV8_FINAL'))
    p.add_argument('--N',type=int,default=200)
    p.add_argument('--w-values',nargs='+',type=float,default=[0.0,0.1,0.3,0.5])
    p.add_argument('--L-values',nargs='+',type=int,default=[12,16,20,24,28,32])
    p.add_argument('--comparison-L-values',nargs='+',type=int,default=[12,20,28])
    p.add_argument('--convergence-L',type=int,default=24)
    p.add_argument('--blocks',type=int,default=40)
    p.add_argument('--block-counts',nargs='+',type=int,default=[8,16,24,32,40])
    p.add_argument('--roots-per-block',type=int,default=384)
    p.add_argument('--unconfined-roots-per-block',type=int,default=8192)
    p.add_argument('--pilot-roots',type=int,default=20000)
    p.add_argument('--max-population',type=int,default=32768)
    p.add_argument('--C-minus',type=float,default=0.5)
    p.add_argument('--C-plus',type=float,default=2.0)
    p.add_argument('--prune-probability',type=float,default=0.5)
    p.add_argument('--max-clones',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def main():
    args=parse_args()
    if args.self_test:
        self_test(); return
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is required for production S8 REV8.')
    if args.N<2: raise ValueError('N must be >=2')
    if args.blocks<8: raise ValueError('Use at least 8 production blocks')
    if max(args.block_counts)>args.blocks: raise ValueError('block-count exceeds total blocks')
    if args.blocks not in args.block_counts: args.block_counts=sorted(set(args.block_counts+[args.blocks]))
    if any(int(L)<=0 or int(L)%2 for L in args.L_values): raise ValueError('All L-values must be positive even integers')
    for L in args.comparison_L_values+[args.convergence_L]:
        if L not in args.L_values: raise ValueError(f'L={L} must be in L-values')
    if not (0<args.C_minus<1<args.C_plus): raise ValueError('Require 0<C-minus<1<C-plus')
    if not (0<args.prune_probability<=1): raise ValueError('prune-probability must be in (0,1]')
    if args.max_clones<2: raise ValueError('max-clones must be >=2')
    if args.max_population<args.roots_per_block: raise ValueError('max-population must be >= roots-per-block')

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rg=load_master_rg(args.rg_master,args.N,args.w_values)

    print('='*100)
    print(f'{SCRIPT_NAME} v{SCRIPT_VERSION}')
    print('='*100)
    print(f'Rg master                 : {args.rg_master}')
    print(f'N                         : {args.N}')
    print(f'DJ w values               : {args.w_values}')
    print(f'Physical L values         : {args.L_values}')
    print(f'Comparison L values       : {args.comparison_L_values}')
    print(f'Convergence L             : {args.convergence_L}')
    print(f'Blocks                    : {args.blocks}')
    print(f'Block levels              : {args.block_counts}')
    print(f'Finite roots/block        : {args.roots_per_block}')
    print(f'Unconfined SIS roots/block: {args.unconfined_roots_per_block}')
    print(f'Pilot roots               : {args.pilot_roots}')
    print(f'PERM C-/C+                : {args.C_minus}/{args.C_plus}')
    print(f'Prune probability         : {args.prune_probability}')
    print(f'Max clones                : {args.max_clones}')
    print(f'Max population            : {args.max_population}')
    print('Geometry                  : absorbing walls z=0,L; center tether z=L/2; even L')
    print('Unconfined reference      : genuine L=infinity SIS/Rosenbluth (DJ and SAW)')
    print('DeltaF estimator           : blockwise lnZ_inf,b - lnZ_L,b')
    print('Filtering/sign selection   : OFF')
    print('='*100)

    # Save Rg input audit before any simulation.
    rg_rows=[]
    for w in args.w_values:
        r=rg[float(w)]
        rg_rows.append({'model':'DJ','N':args.N,'w':float(w),**r})
    rg_rows.append({'model':'SAW','N':args.N,'w':float('inf'),**rg['saw']})
    pd.DataFrame(rg_rows).to_csv(outdir/'FigS8_Rg_input_audit.csv',index=False)
    for label,r in [('SAW',rg['saw'])]+[(f'DJ w={w:g}',rg[float(w)]) for w in args.w_values]:
        print(f'{label:12s}: Rg={r["Rg"]:.9f} +/- {r["Rg_err"]:.9f} [{r["quality_flag"]}], master minESS={r["min_ESS"]:.2f}, maxfrac={r["max_block_weight_fraction"]:.5f}')
    if rg['saw']['quality_flag'].upper() != 'OK':
        print('WARNING: SAW Rg master row is flagged CAUTION; value is used unchanged and this is recorded in provenance.')

    if args.dry_run:
        print('DRY RUN: input validation only; no production simulation.')
        return

    states, thresholds=run_plan(args.N,args.w_values,args.L_values,args.blocks,args.roots_per_block,
                                args.unconfined_roots_per_block,args.pilot_roots,args.max_population,
                                args.C_minus,args.C_plus,args.prune_probability,args.max_clones,args.seed)

    rows=assemble_rows(states,rg,args.w_values,args.L_values,args.N,args.block_counts,args.bootstrap,args.seed)
    rdf=pd.DataFrame(rows)
    rdf.to_csv(outdir/'FigS8_convergence_levels.csv',index=False)

    block_rows=[]
    for key,st in states.items():
        for b in st['blocks']:
            block_rows.append({
                'model':st['model'],'ensemble':st['ensemble'],'w':st['w'],'L':('inf' if np.isinf(st['L']) else st['L']),
                'saw':st['saw'],'block_id':b['block_id'],'logZ':b['logZ'],'ESS':b['ESS'],
                'max_weight_fraction':b['max_weight_fraction'],'population_final':b['population_final'],
                'population_max':b['population_max'],'prune_deaths':b['prune_deaths'],
                'clone_events':b['clone_events'],'overflow':b['overflow'],'valid_roots':b['valid_roots'],'roots':b['roots']})
    pd.DataFrame(block_rows).to_csv(outdir/'FigS8_block_level.csv',index=False)

    final= rdf[rdf.blocks_used==args.blocks].copy()
    final.to_csv(outdir/'FigS8_final_state_summary.csv',index=False)

    ordering=model_ordering_audit(rows,rg,args.w_values,args.L_values,args.N,args.comparison_L_values)
    ordering.to_csv(outdir/'FigS8_model_ordering_audit.csv',index=False)

    sampling=sampling_summary(rows)
    sampling.to_csv(outdir/'FigS8_sampling_diagnostics_summary.csv',index=False)

    # Convergence audit: change from each cumulative level to the final level for every state.
    conv=[]
    for _,g in rdf.groupby(['model','w','L']):
        g=g.sort_values('blocks_used')
        Ffinal=float(g[g.blocks_used==args.blocks].DeltaF.iloc[0])
        for _,r in g.iterrows():
            conv.append({'model':r.model,'w':r.w,'L':r.L,'blocks_used':int(r.blocks_used),
                         'DeltaF':float(r.DeltaF),'DeltaF_err':float(r.DeltaF_err),
                         'abs_DeltaF_to_final':abs(float(r.DeltaF)-Ffinal),
                         'relative_DeltaF_to_final':abs(float(r.DeltaF)-Ffinal)/max(abs(Ffinal),1e-12),
                         'ESS_min_all':float(r.ESS_min_all),'max_weight_fraction_max':float(r.max_weight_fraction_max)})
    pd.DataFrame(conv).to_csv(outdir/'FigS8_convergence_audit.csv',index=False)

    # Save the actual frozen threshold arrays: small enough to archive and useful for exact auditability.
    thr_arrays = {}
    thr_meta = []
    for idx, (k, (lo_arr, hi_arr)) in enumerate(thresholds.items()):
        lo_key = f'state_{idx:03d}_lo'
        hi_key = f'state_{idx:03d}_hi'
        thr_arrays[lo_key] = np.asarray(lo_arr, dtype=np.float64)
        thr_arrays[hi_key] = np.asarray(hi_arr, dtype=np.float64)
        thr_meta.append({'index':idx,'key':{'model':k[0],'L':k[1],'w':k[2],'saw':k[3]},'lo_key':lo_key,'hi_key':hi_key})
    np.savez_compressed(outdir/'FigS8_PERM_FROZEN_THRESHOLDS.npz', **thr_arrays)
    with (outdir/'FigS8_PERM_threshold_metadata.json').open('w',encoding='utf-8') as fh:
        json.dump({'pilot_roots':args.pilot_roots,'C_minus':args.C_minus,'C_plus':args.C_plus,
                   'states':thr_meta},fh,indent=2,default=float)

    make_figure(rows,rg,args.w_values,args.N,args.L_values,args.comparison_L_values,args.convergence_L,
                outdir/'FigS8_FreeEnergy_Overlap_REV8.png',outdir/'FigS8_FreeEnergy_Overlap_REV8.pdf')

    prov={
        'script_name':SCRIPT_NAME,'script_version':SCRIPT_VERSION,'timestamp':datetime.now().isoformat(),
        'python':sys.version,'platform':platform.platform(),'numba':getattr(numba,'__version__',None),
        'N':args.N,'w_values':args.w_values,'L_values':args.L_values,'comparison_L_values':args.comparison_L_values,
        'convergence_L':args.convergence_L,'blocks':args.blocks,'block_counts':args.block_counts,
        'roots_per_block':args.roots_per_block,'unconfined_roots_per_block':args.unconfined_roots_per_block,
        'pilot_roots':args.pilot_roots,'C_minus':args.C_minus,'C_plus':args.C_plus,
        'prune_probability':args.prune_probability,'max_clones':args.max_clones,'max_population':args.max_population,
        'seed':args.seed,
        'geometry':{'lattice':'3D simple cubic','tether':'z=L/2','walls':'absorbing at z=0 and z=L',
                    'accessible_layers':'1..L-1','even_L_only':True,'unconfined_reference':'genuine L=infinity SIS/Rosenbluth'},
        'estimators':{'Rg':'from canonical merged master CSV',
                      'DeltaF':'mean over independent blockwise [lnZ_inf,b - lnZ_L,b]',
                      'DeltaF_uncertainty':'SEM of blockwise differences; 5000-replicate bootstrap audit',
                      'unconfined':'sequential importance sampling/Rosenbluth for DJ and strict SAW',
                      'finite_width':'PERM with frozen thresholds from independent unpruned pilot'},
        'data_treatment':{'sign_selection':False,'point_deletion':False,'smoothing':False,'fitting':False,'snr_filter':False},
        'reviewer_1_comment_h':{'purpose':'same-physical-L comparison against Gaussian and SAW endpoints',
                               'fixed_physical_L':args.L_values,'comparison_L':args.comparison_L_values},
        'SAW_Rg_note':{'Rg':rg['saw']['Rg'],'Rg_err':rg['saw']['Rg_err'],'quality_flag':rg['saw']['quality_flag'],
                       'origin':'1,000,000-chain high-statistics strict-SAW Rosenbluth rerun uploaded separately'},
        'outputs':{'figure_png':str(outdir/'FigS8_FreeEnergy_Overlap_REV8.png'),
                   'figure_pdf':str(outdir/'FigS8_FreeEnergy_Overlap_REV8.pdf')}
    }
    (outdir/'FigS8_provenance.json').write_text(json.dumps(prov,indent=2,default=float),encoding='utf-8')

    print('\nS8 REV8 written:')
    for p in [outdir/'FigS8_FreeEnergy_Overlap_REV8.png',outdir/'FigS8_FreeEnergy_Overlap_REV8.pdf',
              outdir/'FigS8_convergence_levels.csv',outdir/'FigS8_block_level.csv',outdir/'FigS8_final_state_summary.csv',
              outdir/'FigS8_model_ordering_audit.csv',outdir/'FigS8_sampling_diagnostics_summary.csv',outdir/'FigS8_convergence_audit.csv',
              outdir/'FigS8_Rg_input_audit.csv',outdir/'FigS8_PERM_FROZEN_THRESHOLDS.npz',outdir/'FigS8_provenance.json']:
        print(' ',p.resolve())


if __name__=='__main__':
    main()
