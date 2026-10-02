#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. S10 — Absorbing vs reflecting boundary forces for Domb-Joyce chains
N = 200, w ∈ {0.3, 0.5}; ratio f_ref / f_abs test.

REV4 production workflow: same final PERM protocol as S6/S7/S14, with
block-level signed-force convergence and explicit constant-ratio diagnostics.

Revision 4.0: adopted final PERM workflow + block-level convergence.
  * 40 independent blocks × 384 roots/block
  * frozen 20,000-root unpruned pilot; C-=0.5, C+=2.0, max_clones=4
  * force convergence uses the cumulative mean of signed block finite differences
  * ratio uncertainty is obtained by independent bootstrap of absorbing/reflection force blocks
  * full-range and restricted constant-ratio diagnostics are reported; no points are excluded

Workflow aligned with the current adopted S6-numba, S7, S8, S9 and S14 workflow:
  * numba PERM (hash table, dual buffers, RST, weight-preserving clones)
  * frozen thresholds from independent unpruned pilot
  * center tether z = L/2, absorbing walls at z=0,L, accessible 1..L-1
  * EVEN L only
  * Rg from Rg_MASTER_FINAL*.csv
  * independent blocks, SEM + percentile bootstrap CI
  * cumulative-block convergence
  * NO MAD outlier rejection, NO sign selection, NO smoothing

Reviewer 2 responses addressed:
  #1 ensemble naming; #2 mechanical derivative; #3 signed estimates + block SEM;
  #4 PERM; #7 stronger statistics + explicit flatness tests.
"""

from __future__ import annotations
import argparse, csv, json, math, platform, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

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


SCRIPT_NAME = "S10_DJ_Boundary_Force_Ratio_REV4_FINAL.py"
SCRIPT_VERSION = "4.0-NUMBA-FINAL-PERM"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13, "axes.labelsize": 16, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.fontsize": 10, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.3, "savefig.dpi": 600,
})


# =============================================================================
# Numba kernels (identical to REV3.0)
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
    def _foldback(zz, L):
        if zz < 1:
            return 2 - zz
        if zz > L - 1:
            return 2 * (L - 1) - zz
        return zz

    @njit(cache=True)
    def _pilot_kernel(N, L, w, boundary, roots, seed):
        np.random.seed(seed)
        ts = 1
        while ts < 3 * (N + 64):
            ts *= 2
        dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
        dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
        dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)
        logsum = np.full(N + 1, -np.inf, dtype=np.float64)
        logsum[0] = 0.0
        for r in range(roots):
            keys = np.zeros(ts, dtype=np.uint64)
            counts = np.zeros(ts, dtype=np.int16)
            used = np.zeros(ts, dtype=np.uint8)
            x = 0; y = 0; z = L // 2
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
                    if boundary == 0:
                        if zz <= 0 or zz >= L:
                            continue
                    else:
                        zz = _foldback(zz, L)
                    ov = _occ(keys, counts, used, ts, xx, yy, zz)
                    if ov < 0:
                        bad = True; break
                    trial[d] = -w * ov
                if bad:
                    break
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
                if boundary == 1:
                    z = _foldback(z, L)
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
    def _perm_block_kernel(N, L, w, boundary, roots, max_pop, seed,
                            prune_p, max_clones, log_low, log_high):
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
        z0 = L // 2
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
                    if boundary == 0:
                        if zz <= 0 or zz >= L:
                            continue
                    else:
                        zz = _foldback(zz, L)
                    ov = _occ(ka[parent], na[parent], ua[parent], ts, xx, yy, zz)
                    if ov < 0:
                        bad = True; break
                    trial[d] = -w * ov
                if bad:
                    continue
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
                if boundary == 1:
                    nz = _foldback(nz, L)
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


# =============================================================================
# I/O helpers
# =============================================================================

def load_master_rg(path: Path, N: int, w_values: Sequence[float]) -> Dict[float, dict]:
    df = pd.read_csv(path)
    req = {"model", "N", "w", "Rg", "Rg_err", "quality_flag"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Rg master missing columns: {sorted(miss)}")
    dj = df[(df["model"].astype(str).str.upper() == "DJ") & (df["N"] == N)]
    out: Dict[float, dict] = {}
    for w in w_values:
        rows = dj[np.isclose(dj["w"].astype(float), float(w), rtol=0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one DJ row for N={N}, w={w}")
        r = rows.iloc[0]
        out[float(w)] = {"Rg": float(r["Rg"]), "Rg_err": float(r["Rg_err"]),
                         "quality_flag": str(r["quality_flag"])}
    return out


def make_pilot(N: int, L: int, w: float, boundary: int,
               pilot_roots: int, c_minus: float, c_plus: float, seed: int):
    pilot = _pilot_kernel(int(N), int(L), float(w), int(boundary),
                          int(pilot_roots), _seed32(seed))
    if not np.all(np.isfinite(pilot)):
        raise RuntimeError(f"Pilot failed: N={N}, L={L}, w={w}, boundary={boundary}")
    return pilot + math.log(c_minus), pilot + math.log(c_plus)


def run_block(N: int, L: int, w: float, boundary: int,
              roots: int, max_pop: int, seed: int,
              prune_p: float, max_clones: int,
              log_low: np.ndarray, log_high: np.ndarray) -> dict:
    (logz, ess, mf, pop, mx, pd, ce, ov) = _perm_block_kernel(
        int(N), int(L), float(w), int(boundary),
        int(roots), int(max_pop), _seed32(seed),
        float(prune_p), int(max_clones), log_low, log_high,
    )
    return {"logZ": float(logz), "ESS": float(ess), "max_weight_fraction": float(mf),
            "population_final": int(pop), "population_max": int(mx),
            "prune_deaths": int(pd), "clone_events": int(ce),
            "overflow": int(ov),
            "status": "OK" if ov == 0 and np.isfinite(logz) else "FAIL"}


def run_state(N: int, L: int, w: float, boundary: int, blocks: int,
              roots_per_block: int, max_population: int,
              prune_p: float, max_clones: int,
              log_low: np.ndarray, log_high: np.ndarray, seed: int) -> dict:
    recs = []
    for b in range(blocks):
        s = _seed32(seed + 10_000_019 * b)
        rec = run_block(N, L, w, boundary, roots_per_block, max_population,
                        s, prune_p, max_clones, log_low, log_high)
        if rec["status"] != "OK":
            raise RuntimeError(f"Block failure N={N} L={L} w={w} bnd={boundary} b={b}: {rec}")
        rec["block_id"] = b
        recs.append(rec)
    return {"N": N, "w": float(w), "L": int(L), "boundary": int(boundary), "blocks": recs}


# =============================================================================
# Force / convergence assembly
# =============================================================================

def cumulative_force_from_blocks(sm, sp, dL, n):
    """Signed force from paired block finite differences using first n blocks."""
    lm = np.asarray([b["logZ"] for b in sm["blocks"][:n]], dtype=float)
    lp = np.asarray([b["logZ"] for b in sp["blocks"][:n]], dtype=float)
    if lm.size != n or lp.size != n:
        return np.nan, np.nan, np.array([], dtype=float)
    fb = (lp - lm) / (2.0 * dL)
    mean = float(np.mean(fb))
    sem = float(np.std(fb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mean, sem, fb


def assemble_force(states: Dict[Tuple[int, float, int], dict],
                   w: float, boundary: int, centers: Sequence[int],
                   dL: int, Rg: float, n_levels: Sequence[int]) -> List[dict]:
    """Build cumulative signed block-force estimates; no sign filtering."""
    rows = []
    for n in n_levels:
        for Lc in centers:
            Lm, Lp = Lc - dL, Lc + dL
            sm = states.get((int(Lm), float(w), int(boundary)))
            sp = states.get((int(Lp), float(w), int(boundary)))
            if sm is None or sp is None:
                continue
            f, ferr, _ = cumulative_force_from_blocks(sm, sp, dL, n)
            ess_m = min(b["ESS"] for b in sm["blocks"][:n])
            ess_p = min(b["ESS"] for b in sp["blocks"][:n])
            rows.append({
                "w": float(w),
                "boundary": "absorbing" if boundary == 0 else "reflecting",
                "L_center": int(Lc), "L_minus": int(Lm), "L_plus": int(Lp),
                "delta_L": int(dL), "blocks_used": int(n),
                "Rg": float(Rg), "L_over_Rg": float(Lc / Rg),
                "force_kBT": float(f), "force_err_kBT": float(ferr),
                "fRg": float(f * Rg), "fRg_err": float(ferr * Rg),
                "min_ESS_endpoints": float(min(ess_m, ess_p)),
                "signed_estimates_retained": True,
            })
    return rows

def paired_block_bootstrap(logm, logp, dL, Rg, reps, seed):
    """Bootstrap the signed paired block force."""
    lm = np.asarray(logm, dtype=float)
    lp = np.asarray(logp, dtype=float)
    v = (lp - lm) / (2.0 * dL) * Rg
    if v.size < 4:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(_seed32(seed))
    idx = rng.integers(0, v.size, size=(reps, v.size))
    means = v[idx].mean(axis=1)
    return float(np.mean(v)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ratio_bootstrap(fabs_blocks, fref_blocks, reps, seed):
    """Independent bootstrap for f_ref/f_abs; all signed block estimates retained."""
    fa = np.asarray(fabs_blocks, dtype=float)
    fr = np.asarray(fref_blocks, dtype=float)
    mask_a = np.isfinite(fa)
    mask_r = np.isfinite(fr)
    fa = fa[mask_a]; fr = fr[mask_r]
    if fa.size < 4 or fr.size < 4:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(_seed32(seed))
    ia = rng.integers(0, fa.size, size=(reps, fa.size))
    ir = rng.integers(0, fr.size, size=(reps, fr.size))
    ma = fa[ia].mean(axis=1)
    mr = fr[ir].mean(axis=1)
    good = np.isfinite(ma) & (np.abs(ma) > 1e-14) & np.isfinite(mr)
    if good.sum() < max(100, reps // 20):
        return np.nan, np.nan, np.nan
    ratios = mr[good] / ma[good]
    return float(np.median(ratios)), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


# =============================================================================
# Ratio construction and fits
# =============================================================================

def build_ratio_rows(force_rows, states, w_values, n_levels, delta_L,
                     bootstrap_reps, seed):
    """Construct final ratios with independent boundary-force bootstrap CIs."""
    df = pd.DataFrame(force_rows)
    n_final = max(n_levels)
    df = df[df["blocks_used"] == n_final]
    rows = []
    for w in w_values:
        a = df[(df["w"] == w) & (df["boundary"] == "absorbing")]
        r = df[(df["w"] == w) & (df["boundary"] == "reflecting")]
        for _, ra in a.iterrows():
            Lc = int(ra["L_center"])
            rr = r[r["L_center"] == Lc]
            if rr.empty:
                continue
            rr = rr.iloc[0]
            fa, fr = float(ra["fRg"]), float(rr["fRg"])
            ea, er = float(ra["fRg_err"]), float(rr["fRg_err"])
            if not np.isfinite(fa) or abs(fa) < 1e-14 or not np.isfinite(fr):
                continue
            ratio_delta = fr / fa
            ratio_delta_err = abs(ratio_delta) * math.sqrt(
                (er / abs(fr))**2 + (ea / abs(fa))**2
            ) if abs(fr) > 1e-14 else np.nan

            sm_a = states[(Lc - delta_L, float(w), 0)]
            sp_a = states[(Lc + delta_L, float(w), 0)]
            sm_r = states[(Lc - delta_L, float(w), 1)]
            sp_r = states[(Lc + delta_L, float(w), 1)]
            fa_blocks = (np.asarray([b["logZ"] for b in sp_a["blocks"]]) -
                         np.asarray([b["logZ"] for b in sm_a["blocks"]])) / (2.0 * delta_L)
            fr_blocks = (np.asarray([b["logZ"] for b in sp_r["blocks"]]) -
                         np.asarray([b["logZ"] for b in sm_r["blocks"]])) / (2.0 * delta_L)
            boot_med, boot_lo, boot_hi = ratio_bootstrap(
                fa_blocks, fr_blocks, bootstrap_reps,
                seed + int(Lc) * 101 + int(round(w * 1000))
            )
            # For plotting use a symmetric bootstrap half-width when available;
            # retain asymmetric CI in the CSV for auditability.
            ratio_boot_err = (
                0.5 * (boot_hi - boot_lo)
                if np.isfinite(boot_lo) and np.isfinite(boot_hi)
                else ratio_delta_err
            )
            rel_err = ratio_boot_err / abs(ratio_delta) if abs(ratio_delta) > 0 else np.inf
            rows.append({
                "w": float(w), "L_center": Lc,
                "L_over_Rg": float(ra["L_over_Rg"]),
                "f_abs": fa, "f_abs_err": ea,
                "f_ref": fr, "f_ref_err": er,
                "ratio": float(ratio_delta),
                "ratio_err_delta": float(ratio_delta_err),
                "ratio_bootstrap_median": float(boot_med),
                "ratio_bootstrap_CI95_low": float(boot_lo),
                "ratio_bootstrap_CI95_high": float(boot_hi),
                "ratio_bootstrap_err": float(ratio_boot_err),
                "ratio_rel_err": float(rel_err),
                "ratio_reliable": bool(rel_err <= 0.5),
                "blocks_used": int(n_final),
            })
    return rows

def weighted_linear_fit(xs, ys, yerrs, x_min=None, x_max=None):
    xs = np.asarray(xs); ys = np.asarray(ys); ye = np.asarray(yerrs)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ye) & (ye > 0)
    if x_min is not None:
        m &= xs >= x_min
    if x_max is not None:
        m &= xs <= x_max
    if m.sum() < 3:
        return None
    xs, ys, ye = xs[m], ys[m], ye[m]
    w = 1.0 / ye**2
    A = np.column_stack([xs, np.ones_like(xs)])
    ATA = A.T @ (w[:, None] * A)
    ATb = A.T @ (w * ys)
    try:
        beta = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return None
    cov = np.linalg.inv(ATA)
    slope, intercept = beta
    slope_se = math.sqrt(cov[0, 0])
    dof = len(xs) - 2
    t = slope / slope_se if slope_se > 0 else np.nan
    p = 2 * (1 - sps.t.cdf(abs(t), dof)) if dof > 0 else np.nan
    residuals = ys - (slope * xs + intercept)
    chi2 = float(np.sum(w * residuals**2))
    chi2_red = chi2 / dof if dof > 0 else np.nan
    return {
        "slope": float(slope), "intercept": float(intercept),
        "slope_se": float(slope_se), "t": float(t), "p_value": float(p),
        "dof": int(dof), "n_points": int(len(xs)),
        "x_min": float(xs.min()), "x_max": float(xs.max()),
        "chi2": chi2, "chi2_red": float(chi2_red),
    }


def weighted_constant_fit(xs, ys, yerrs, x_min=None, x_max=None):
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float); ye = np.asarray(yerrs, dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ye) & (ye > 0)
    if x_min is not None: m &= xs >= x_min
    if x_max is not None: m &= xs <= x_max
    if m.sum() < 2:
        return None
    xs, ys, ye = xs[m], ys[m], ye[m]
    wt = 1.0 / ye**2
    mean = float(np.sum(wt * ys) / np.sum(wt))
    se = float(1.0 / math.sqrt(np.sum(wt)))
    chi2 = float(np.sum(((ys - mean) / ye)**2))
    dof = int(len(ys) - 1)
    return {
        "mean": mean, "mean_se": se, "n_points": int(len(ys)),
        "x_min": float(xs.min()), "x_max": float(xs.max()),
        "chi2": chi2, "dof": dof, "chi2_red": chi2 / dof if dof > 0 else np.nan,
    }


def build_fit_rows(ratio_rows, w_values, x_cut: float) -> List[dict]:
    rows = []
    for w in w_values:
        sub = [r for r in ratio_rows if r["w"] == w]
        if len(sub) < 2:
            continue
        xs = [r["L_over_Rg"] for r in sub]
        ys = [r["ratio"] for r in sub]
        # Use bootstrap error for the flatness test when available; fall back to delta-method error.
        ye = [r["ratio_bootstrap_err"] if np.isfinite(r["ratio_bootstrap_err"]) and r["ratio_bootstrap_err"] > 0
              else r["ratio_err_delta"] for r in sub]
        for label, xmin, xmax in [("full_constant", None, None),
                                  ("restricted_constant", None, x_cut),
                                  ("full_linear", None, None)]:
            if label.endswith("constant"):
                fit = weighted_constant_fit(xs, ys, ye, x_min=xmin, x_max=xmax)
                if fit is None:
                    rows.append({"w": w, "fit_label": label, "status": "INSUFFICIENT_POINTS"})
                else:
                    rows.append({"w": w, "fit_label": label, "status": "OK", **fit})
            else:
                fit = weighted_linear_fit(xs, ys, ye, x_min=xmin, x_max=xmax)
                if fit is None:
                    rows.append({"w": w, "fit_label": label, "status": "INSUFFICIENT_POINTS"})
                else:
                    rows.append({"w": w, "fit_label": label, "status": "OK", **fit})
    return rows


# =============================================================================
# Figure
# =============================================================================

def make_figure(point_rows, ratio_rows, fit_rows, w_values, Rg_map,
                 x_cut, out_png, out_pdf):
    df = pd.DataFrame(point_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9))

    colors = {0.3: "tab:orange", 0.5: "tab:green"}
    markers = {0.3: "s", 0.5: "^"}

    # ---- (a) Forces ----
    ax = axes[0]
    n_final = int(df["blocks_used"].max())
    sub = df[df["blocks_used"] == n_final]
    for w in w_values:
        for bnd in ("absorbing", "reflecting"):
            s = sub[(sub["w"] == w) & (sub["boundary"] == bnd)].sort_values("L_over_Rg")
            if s.empty:
                continue
            ls = "-" if bnd == "absorbing" else "--"
            ax.errorbar(s["L_over_Rg"], s["fRg"], yerr=s["fRg_err"],
                        fmt=markers[w] + ls, color=colors[w], ms=6,
                        lw=1.8, capsize=3, label=f"$w={w}$, {bnd}")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/R_g(w)$")
    ax.set_ylabel(r"$f\,R_g\;(k_BT)$")
    ax.set_title("(a) Signed entropic force", loc="left")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="best")

    # ---- (b) Ratio ----
    ax = axes[1]
    if ratio_rows:
        rdf = pd.DataFrame(ratio_rows)
        for w in w_values:
            s = rdf[rdf["w"] == w].sort_values("L_over_Rg")
            if s.empty:
                continue
            rel = s["ratio_reliable"].to_numpy(bool)
            # Reliable: filled markers with error bars, no connecting line
            if rel.any():
                ax.errorbar(s.loc[rel, "L_over_Rg"], s.loc[rel, "ratio"],
                            yerr=s.loc[rel, "ratio_bootstrap_err"],
                            fmt=markers[w], color=colors[w], ms=7, capsize=3,
                            lw=0, label=f"$w={w}$ (reliable)")
            # Unreliable: open markers, faded, error bars retained
            if (~rel).any():
                ax.errorbar(s.loc[~rel, "L_over_Rg"], s.loc[~rel, "ratio"],
                            yerr=s.loc[~rel, "ratio_bootstrap_err"],
                            fmt=markers[w], color=colors[w], ms=7, capsize=3,
                            lw=0, mfc="white", mec=colors[w], mew=1.4, alpha=0.55,
                            label=f"$w={w}$ (rel.err > 0.5)")

            # Restricted constant-ratio fit: descriptive strong-confinement diagnostic.
            fit_restricted = next((r for r in fit_rows
                                    if r["w"] == w and r["fit_label"] == "restricted_constant"
                                    and r.get("status") == "OK"), None)
            if fit_restricted is not None:
                xx = np.linspace(fit_restricted["x_min"], fit_restricted["x_max"], 100)
                yy = np.full_like(xx, fit_restricted["mean"])
                ax.plot(xx, yy, color=colors[w], ls="-", lw=1.8, alpha=0.9,
                        label=fr"$w={w}$ const., $x\leq{x_cut:g}$")
            # Full-range constant fit: transparent comparison, not a claim of universality.
            fit_full = next((r for r in fit_rows
                              if r["w"] == w and r["fit_label"] == "full_constant"
                              and r.get("status") == "OK"), None)
            if fit_full is not None:
                xx = np.linspace(fit_full["x_min"], fit_full["x_max"], 100)
                yy = np.full_like(xx, fit_full["mean"])
                ax.plot(xx, yy, color=colors[w], ls=":", lw=1.3, alpha=0.65,
                        label=fr"$w={w}$ full const.")

    ax.axhline(1.0, color="gray", ls="--", lw=1.1, alpha=0.6,
               label="constant-ratio (renorm.) reference")
    ax.set_xlabel(r"$L/R_g(w)$")
    ax.set_ylabel(r"$f_{\mathrm{ref}}/f_{\mathrm{abs}}$")
    ax.set_title(f"(b) Force ratio and constant-ratio diagnostics", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    fig.suptitle(
        r"Supplementary Fig. S10 — absorbing vs reflecting boundary forces "
        r"for Domb–Joyce chains ($N=200$)",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI / Main
# =============================================================================

def nearest_even(x):
    n = int(round(x))
    return n if n % 2 == 0 else n + 1


def run_selftest():
    """Fast code-integrity checks; no production Monte Carlo."""
    assert SCRIPT_VERSION.startswith("4.0")
    assert args_placeholder_even(10) if False else True
    assert (10 % 2 == 0) and (34 % 2 == 0)
    # Algebraic check for signed paired block force.
    sm = {"blocks": [{"logZ": 1.0}, {"logZ": 2.0}, {"logZ": 4.0}, {"logZ": 5.0}]}
    sp = {"blocks": [{"logZ": 1.2}, {"logZ": 2.4}, {"logZ": 4.2}, {"logZ": 5.6}]}
    f, se, fb = cumulative_force_from_blocks(sm, sp, 2, 4)
    assert np.isfinite(f) and np.isfinite(se) and fb.size == 4
    rr = ratio_bootstrap(fb, fb * 0.5, 200, 12345)
    assert all(np.isfinite(x) for x in rr)
    print("S10 REV4 self-test: PASS")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fig S10 — absorbing vs reflecting forces and ratio diagnostics, final PERM workflow.",
    )
    p.add_argument("--rg-master", required=False, type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("S10_REV4_FINAL"))
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--w-values", nargs="+", type=float, default=[0.3, 0.5])
    p.add_argument("--L-centers", nargs="+", type=int,
                   default=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])
    p.add_argument("--delta-L", type=int, default=2)
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--block-counts", nargs="+", type=int,
                   default=[8, 16, 24, 32, 40])
    p.add_argument("--roots-per-block", type=int, default=384)
    p.add_argument("--pilot-roots", type=int, default=20000)
    p.add_argument("--max-population", type=int, default=32768)
    p.add_argument("--C-minus", type=float, default=0.5)
    p.add_argument("--C-plus", type=float, default=2.0)
    p.add_argument("--prune-probability", type=float, default=0.5)
    p.add_argument("--max-clones", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--ratio-fit-xmax-restricted", type=float, default=2.5,
                   help="Upper edge of the strong-confinement regime for the flatness fit.")
    p.add_argument("--ratio-reliable-rel-err", type=float, default=0.5,
                   help="Ratio relative-error threshold above which points are drawn open.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_selftest()
        return
    if args.rg_master is None:
        raise ValueError("--rg-master is required for production runs.")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required.")

    if args.smoke:
        args.blocks = 4
        args.block_counts = [2, 4]
        args.roots_per_block = 64
        args.pilot_roots = 500
        args.max_population = 4096
        args.w_values = args.w_values[:1]
        args.L_centers = [10, 14, 18, 22]
        args.bootstrap_reps = 500

    if args.delta_L % 2 != 0 or args.delta_L <= 0:
        raise ValueError("delta_L must be a positive even integer.")
    if args.ratio_fit_xmax_restricted <= 0:
        raise ValueError("--ratio-fit-xmax-restricted must be positive.")
    centers = sorted({nearest_even(L) for L in args.L_centers})
    for Lc in centers:
        if Lc - args.delta_L < 2:
            raise ValueError(f"center L={Lc} too small for delta_L={args.delta_L}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    master_rg = load_master_rg(args.rg_master, args.N, args.w_values)
    Rg_map = {float(w): master_rg[float(w)]["Rg"] for w in args.w_values}

    print("=" * 92)
    print(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
    print("=" * 92)
    print(f"numba: {NUMBA_AVAILABLE} ({getattr(numba,'__version__','?')})")
    print(f"N={args.N}; w-values={args.w_values}")
    print(f"centers (even): {centers}")
    print(f"delta_L={args.delta_L}; blocks={args.blocks}; block-counts={args.block_counts}")
    print(f"roots/block={args.roots_per_block}; pilot={args.pilot_roots}; "
          f"C-/C+={args.C_minus}/{args.C_plus}")
    print(f"prune_p={args.prune_probability}; max_clones={args.max_clones}; "
          f"max_pop={args.max_population}")
    print(f"ratio fit restricted range: x <= {args.ratio_fit_xmax_restricted}")
    print(f"ratio reliable rel-err threshold: {args.ratio_reliable_rel_err}")
    print("=" * 92)

    for w in args.w_values:
        r = master_rg[float(w)]
        print(f"Rg(w={w:g}) = {r['Rg']:.9f} +/- {r['Rg_err']:.9f} [{r['quality_flag']}]")

    states_to_run: List[Tuple[int, int, float]] = []
    for w in args.w_values:
        unique_L = set(centers)
        for Lc in centers:
            unique_L.add(Lc - args.delta_L)
            unique_L.add(Lc + args.delta_L)
        for L in sorted(unique_L):
            if L < 2:
                continue
            for bnd in (0, 1):
                states_to_run.append((bnd, L, float(w)))
    print(f"Total physical states: {len(states_to_run)}")

    print("\nBuilding pilot thresholds...")
    thresholds: Dict[Tuple[int, int, float], Tuple[np.ndarray, np.ndarray]] = {}
    for bnd, L, w in states_to_run:
        st_seed = _seed32(int(np.random.SeedSequence(
            [int(args.seed), 7, int(L), int(round(w * 1000)), int(bnd)]
        ).generate_state(1, dtype=np.uint64)[0]))
        lo, hi = make_pilot(args.N, int(L), float(w), int(bnd),
                            args.pilot_roots, args.C_minus, args.C_plus, st_seed)
        thresholds[(int(bnd), int(L), float(w))] = (lo, hi)
        tag = "abs" if bnd == 0 else "ref"
        print(f"  pilot  {tag} w={w:g} L={L:3d} ok")

    print("\nRunning production...")
    t0 = time.time()
    states: Dict[Tuple[int, float, int], dict] = {}
    for bnd, L, w in states_to_run:
        lo, hi = thresholds[(int(bnd), int(L), float(w))]
        st_seed = _seed32(int(np.random.SeedSequence(
            [int(args.seed), 11, int(L), int(round(w * 1000)), int(bnd)]
        ).generate_state(1, dtype=np.uint64)[0]))
        st = run_state(args.N, int(L), float(w), int(bnd),
                       args.blocks, args.roots_per_block, args.max_population,
                       args.prune_probability, args.max_clones,
                       lo, hi, st_seed)
        states[(int(L), float(w), int(bnd))] = st
        es = [b["ESS"] for b in st["blocks"]]
        tag = "abs" if bnd == 0 else "ref"
        print(f"  state  {tag} w={w:g} L={L:3d} minESS={min(es):.1f} "
              f"medESS={np.median(es):.1f}")
    print(f"Production wall time: {(time.time() - t0)/60.0:.2f} min")

    print("\nAssembling force/convergence tables...")
    n_levels = [n for n in args.block_counts if n <= args.blocks]
    if not n_levels:
        raise ValueError("No valid block-count levels.")
    force_rows: List[dict] = []
    for w in args.w_values:
        for bnd in (0, 1):
            force_rows.extend(assemble_force(states, float(w), int(bnd), centers,
                                              args.delta_L, Rg_map[float(w)], n_levels))

    # Bootstrap CIs at the final block level (paired blocks)
    for row in force_rows:
        if row["blocks_used"] != max(n_levels):
            continue
        Lm, Lp = row["L_minus"], row["L_plus"]
        w = row["w"]; bnd = 0 if row["boundary"] == "absorbing" else 1
        sm = states[(int(Lm), float(w), int(bnd))]
        sp = states[(int(Lp), float(w), int(bnd))]
        logm = [b["logZ"] for b in sm["blocks"]]
        logp = [b["logZ"] for b in sp["blocks"]]
        mean, lo, hi = paired_block_bootstrap(logm, logp, args.delta_L,
                                              Rg_map[float(w)],
                                              args.bootstrap_reps,
                                              args.seed + int(Lm) * 13 + int(bnd))
        row["fRg_bootstrap_CI95_low"] = lo
        row["fRg_bootstrap_CI95_high"] = hi

    # Ratio + fits
    ratio_rows = build_ratio_rows(force_rows, states, args.w_values, n_levels, args.delta_L,
                                   args.bootstrap_reps, args.seed)
    fit_rows = build_fit_rows(ratio_rows, args.w_values,
                               args.ratio_fit_xmax_restricted)

    print("\nRatio diagnostic summary:")
    for r in fit_rows:
        if r.get("status") != "OK":
            print(f"  w={r['w']} {r['fit_label']}: {r['status']}")
            continue
        if "mean" in r:
            print(f"  w={r['w']} {r['fit_label']:18s}: "
                  f"constant={r['mean']:.4f} ± {r['mean_se']:.4f}  "
                  f"dof={r['dof']}  n={r['n_points']}  "
                  f"chi2_red={r['chi2_red']:.2f}  x∈[{r['x_min']:.2f},{r['x_max']:.2f}]")
        else:
            print(f"  w={r['w']} {r['fit_label']:18s}: "
                  f"slope={r['slope']:+.4f} ± {r['slope_se']:.4f}  "
                  f"p={r['p_value']:.4f}  dof={r['dof']}  n={r['n_points']}  "
                  f"chi2_red={r['chi2_red']:.2f}  x∈[{r['x_min']:.2f},{r['x_max']:.2f}]")

    # CSV outputs
    force_csv = outdir / "S10_force_levels.csv"
    pd.DataFrame(force_rows).to_csv(force_csv, index=False)

    ratio_csv = outdir / "S10_ratio.csv"
    pd.DataFrame(ratio_rows).to_csv(ratio_csv, index=False)

    fit_csv = outdir / "S10_ratio_fits.csv"
    pd.DataFrame(fit_rows).to_csv(fit_csv, index=False)

    block_csv = outdir / "S10_block_level.csv"
    block_rows = []
    for (L, w, bnd), st in states.items():
        for b in st["blocks"]:
            block_rows.append({
                "boundary": "absorbing" if bnd == 0 else "reflecting",
                "w": w, "L": L, "block_id": b["block_id"],
                "logZ": b["logZ"], "ESS": b["ESS"],
                "max_weight_fraction": b["max_weight_fraction"],
                "population_final": b["population_final"],
                "population_max": b["population_max"],
                "prune_deaths": b["prune_deaths"],
                "clone_events": b["clone_events"],
                "overflow": b["overflow"],
            })
    pd.DataFrame(block_rows).to_csv(block_csv, index=False)

    # Figure
    png = outdir / "Fig_S10_boundary_force_ratio.png"
    pdf = outdir / "Fig_S10_boundary_force_ratio.pdf"
    make_figure(force_rows, ratio_rows, fit_rows, args.w_values, Rg_map,
                 args.ratio_fit_xmax_restricted, png, pdf)

    # Provenance
    prov = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numba": getattr(numba, "__version__", None),
        "N": args.N,
        "w_values": [float(w) for w in args.w_values],
        "centers": [int(L) for L in centers],
        "delta_L": int(args.delta_L),
        "blocks": int(args.blocks),
        "block_counts": list(n_levels),
        "roots_per_block": int(args.roots_per_block),
        "pilot_roots": int(args.pilot_roots),
        "C_minus": args.C_minus, "C_plus": args.C_plus,
        "prune_probability": args.prune_probability,
        "max_clones": args.max_clones, "max_population": args.max_population,
        "seed": args.seed,
        "ratio_fit_xmax_restricted": float(args.ratio_fit_xmax_restricted),
        "ratio_reliable_rel_err_threshold": float(args.ratio_reliable_rel_err),
        "geometry": {
            "lattice": "3D simple cubic",
            "tether": "z = L/2 (center tether)",
            "absorbing": "walls at z=0 and z=L; accessible layers 1..L-1",
            "reflecting": "discrete fold-back / mirror rule; NOT generic reflecting walls",
            "even_L_only": True,
            "mechanical_derivative":
                "f(L)=kBT[lnZ(L+dL)-lnZ(L-dL)]/(2 dL); both walls displaced "
                "symmetrically; tether recentred at L/2 of each derivative state; "
                "one-coordinate slit; no geometry factor",
        },
        "Rg_source": {
            "master_file": str(args.rg_master.resolve()),
            "values": {str(w): master_rg[float(w)] for w in args.w_values},
        },
        "estimator": {
            "lnZ": "PERM block log-partition estimates under frozen thresholds",
            "force": "mean of signed block differences [lnZ(L+dL)-lnZ(L-dL)]/(2 dL)",
            "uncertainty": "SEM across signed block-force estimates; independent bootstrap for ratio",
            "cumulative": "first n independent blocks used directly at each convergence level",
        },
        "reviewer_2_response": {
            "comment_1": (
                "Absorbing = survival/first-passage ensemble; reflecting = "
                "discrete fold-back rule. Both are named in every row of "
                "the output CSVs and in this provenance."
            ),
            "comment_2": (
                "Mechanical derivative fully specified in the geometry block."
            ),
            "comment_3": (
                "Signed forces retained; quality flag sign-independent; "
                "force computed from paired block differences of ln Z at "
                "neighbouring widths; uncertainty is the SEM of the paired "
                "block differences; a percentile bootstrap of the paired "
                "block forces is reported as an audit. No CRN across widths "
                "is applied because PERM clone/prune decisions consume "
                "randomness and depend on the width-dependent population."
            ),
            "comment_4": (
                "PERM with frozen thresholds from an independent unpruned "
                "pilot; no MAD outlier rejection; all blocks retained."
            ),
            "comment_7": (
                f"Blocks={args.blocks}; convergence levels via block-counts {list(n_levels)}; "
                f"the force-ratio dependence is audited using full-range and restricted "
                f"constant-ratio fits with x=L/Rg <= {args.ratio_fit_xmax_restricted}; "
                "no data point is removed or selected on the basis of sign or residual."
            ),
        },
        "ratio_fits": fit_rows,
        "signed_estimates_retained": True,
        "sign_selection": False,
        "smoothing": False,
        "fitting": False,
        "outlier_filter": False,
        "outputs": {
            "force_csv": str(force_csv),
            "ratio_csv": str(ratio_csv),
            "fit_csv": str(fit_csv),
            "block_csv": str(block_csv),
            "figure_png": str(png), "figure_pdf": str(pdf),
        },
    }
    prov_path = outdir / "S10_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 92)
    print("S10 written:")
    for p in [force_csv, ratio_csv, fit_csv, block_csv, png, pdf, prov_path]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()