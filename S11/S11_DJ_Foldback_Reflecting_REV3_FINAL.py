#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S11 — Fold-back reflecting DJ force, w-crossover
====================================================================

Two panels:
  (a) fRg vs L/Rg(w) for Domb-Joyce chains at N=200,
      w = 0.3 (orange squares) and w = 0.5 (green diamonds),
      under the discrete fold-back / mirror reflecting rule.
      The Gaussian case (w=0) is exactly zero for this rule
      (six allowed moves at every step => Z = 6^(N-1), f = 0)
      and is therefore not plotted.

  (b) Convergence of the paired-block force at two representative
      widths (L=12 strong confinement, L=24 mid confinement)
      for w=0.5, at cumulative block counts 10/20/30/40/50/60.

Workflow aligned with the current adopted S6/S7/S8/S9/S10 production workflow:
  * numba PERM with hash-table walker storage, dual buffers, Russian
    roulette with weight compensation, weight-preserving clones,
    FROZEN thresholds from an independent unpruned pilot.
  * Center tether z=L/2, fold-back walls, accessible layers 1..L-1,
    EVEN L only.
  * Rg from Rg_MASTER_FINAL*.csv.
  * Independent blocks; block-level force SEM; independent-endpoint bootstrap CI.
  * NO MAD rejection, NO sign selection, NO smoothing, NO fitting.
  * Signed forces retained. No force sign selection or post-hoc point filtering.

Reviewer 2 responses:
  #1 ensemble naming: "discrete fold-back reflecting rule" is named in
     every output row, in the figure title, and in the provenance.
     No claim about generic reflecting-wall physics is made.
  #2 mechanical derivative: f(L) = kBT [lnZ(L+dL) - lnZ(L-dL)] / (2 dL);
     both walls displaced symmetrically; tether held at z=L/2 in each
     derivative state; one-coordinate slit; no geometry factor.
  #3 signed estimates retained; block-level SEM and independent-endpoint bootstrap CI.
  #4 PERM with frozen thresholds; convergence panel at 8/16/24/32/40.
  #7 no cross-N claim is made here; N-dependence of the fold-back force
     under the same ensemble is outside the scope of this figure.
"""

from __future__ import annotations
import argparse, csv, json, math, platform, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


SCRIPT_NAME = "S11_DJ_Foldback_Reflecting_REV3_FINAL.py"
SCRIPT_VERSION = "3.0-NUMBA-FINAL-PERM"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13, "axes.labelsize": 16, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.fontsize": 10, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.3, "savefig.dpi": 600,
})


# =============================================================================
# Numba kernels (fold-back reflecting only)
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
    def _pilot_kernel_foldback(N, L, w, roots, seed):
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
                    xx = x + dx[d]; yy = y + dy[d]; zz = _foldback(z + dz[d], L)
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
                x += dx[chosen]; y += dy[chosen]; z = _foldback(z + dz[chosen], L)
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
    def _perm_block_kernel_foldback(N, L, w, roots, max_pop, seed,
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
                    xx = x + dx[d]; yy = y + dy[d]; zz = _foldback(z + dz[d], L)
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
                nx = x + dx[chosen]; ny = y + dy[chosen]; nz = _foldback(z + dz[chosen], L)
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
# Master Rg
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


# =============================================================================
# Pilot + production
# =============================================================================

def make_pilot(N: int, L: int, w: float, pilot_roots: int,
               c_minus: float, c_plus: float, seed: int):
    pilot = _pilot_kernel_foldback(int(N), int(L), float(w),
                                   int(pilot_roots), _seed32(seed))
    if not np.all(np.isfinite(pilot)):
        raise RuntimeError(f"Pilot failed: N={N}, L={L}, w={w}")
    return pilot + math.log(c_minus), pilot + math.log(c_plus)


def run_block(N: int, L: int, w: float, roots: int, max_pop: int, seed: int,
              prune_p: float, max_clones: int,
              log_low: np.ndarray, log_high: np.ndarray) -> dict:
    (logz, ess, mf, pop, mx, pd, ce, ov) = _perm_block_kernel_foldback(
        int(N), int(L), float(w), int(roots), int(max_pop), _seed32(seed),
        float(prune_p), int(max_clones), log_low, log_high,
    )
    return {"logZ": float(logz), "ESS": float(ess), "max_weight_fraction": float(mf),
            "population_final": int(pop), "population_max": int(mx),
            "prune_deaths": int(pd), "clone_events": int(ce), "overflow": int(ov),
            "status": "OK" if ov == 0 and np.isfinite(logz) else "FAIL"}


def run_state(N: int, L: int, w: float, blocks: int,
              roots_per_block: int, max_population: int,
              prune_p: float, max_clones: int,
              log_low: np.ndarray, log_high: np.ndarray, seed: int) -> dict:
    recs = []
    for b in range(blocks):
        s = _seed32(seed + 10_000_019 * b)
        rec = run_block(N, L, w, roots_per_block, max_population,
                        s, prune_p, max_clones, log_low, log_high)
        if rec["status"] != "OK":
            raise RuntimeError(f"Block failure N={N} L={L} w={w} b={b}: {rec}")
        rec["block_id"] = b
        recs.append(rec)
    return {"N": N, "w": float(w), "L": int(L), "blocks": recs}


def cumulative_logz(state: dict, n: int):
    lz = np.array([b["logZ"] for b in state["blocks"][:n]], dtype=float)
    if not np.all(np.isfinite(lz)):
        return np.nan, np.nan
    return float(np.mean(lz)), float(np.std(lz, ddof=1) / math.sqrt(n)) if n > 1 else np.nan


def assemble_force(states: Dict[Tuple[int, float], dict], w: float,
                   centers: Sequence[int], dL: int, Rg: float,
                   n_levels: Sequence[int]) -> List[dict]:
    rows = []
    for n in n_levels:
        for Lc in centers:
            Lm, Lp = Lc - dL, Lc + dL
            sm = states.get((int(Lm), float(w)))
            sp = states.get((int(Lp), float(w)))
            if sm is None or sp is None:
                continue
            lm, sem_m = cumulative_logz(sm, n)
            lp, sem_p = cumulative_logz(sp, n)
            f = (lp - lm) / (2.0 * dL)
            f_sem = math.sqrt(sem_p**2 + sem_m**2) / (2.0 * dL) \
                    if np.isfinite(sem_p) and np.isfinite(sem_m) else float("nan")
            ess_m = min(b["ESS"] for b in sm["blocks"][:n])
            ess_p = min(b["ESS"] for b in sp["blocks"][:n])
            rows.append({
                "w": float(w),
                "boundary": "fold-back reflecting",
                "L_center": int(Lc), "L_minus": int(Lm), "L_plus": int(Lp),
                "delta_L": int(dL), "blocks_used": int(n),
                "Rg": float(Rg), "L_over_Rg": float(Lc / Rg),
                "lnZ_minus": float(lm), "lnZ_plus": float(lp),
                "force_kBT": float(f), "force_err_kBT": float(f_sem),
                "fRg": float(f * Rg), "fRg_err": float(f_sem * Rg),
                "min_ESS_endpoints": float(min(ess_m, ess_p)),
            })
    return rows


def independent_endpoint_bootstrap(logm, logp, dL, Rg, reps, seed):
    """Bootstrap the two independently sampled endpoint ensembles.

    The L-dL and L+dL endpoint simulations use independent random streams.
    We therefore resample the two endpoint block collections independently
    rather than treating equal block indices as correlated pairs.
    """
    logm = np.asarray(logm, dtype=float)
    logp = np.asarray(logp, dtype=float)
    n = min(logm.size, logp.size)
    if n < 2 or not (np.all(np.isfinite(logm[:n])) and np.all(np.isfinite(logp[:n]))):
        return np.nan, np.nan, np.nan
    logm = logm[:n]
    logp = logp[:n]
    rng = np.random.default_rng(_seed32(seed))
    idx_m = rng.integers(0, n, size=(reps, n))
    idx_p = rng.integers(0, n, size=(reps, n))
    means_m = logm[idx_m].mean(axis=1)
    means_p = logp[idx_p].mean(axis=1)
    f_rg = (means_p - means_m) / (2.0 * dL) * Rg
    center = float((logp.mean() - logm.mean()) / (2.0 * dL) * Rg)
    return center, float(np.percentile(f_rg, 2.5)), float(np.percentile(f_rg, 97.5))


# =============================================================================
# Figure (2 panels)
# =============================================================================

def make_figure(point_rows, w_values, Rg_map, out_png, out_pdf):
    df = pd.DataFrame(point_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))

    colors = {0.3: "tab:orange", 0.5: "tab:green"}
    markers = {0.3: "s", 0.5: "D"}

    # ---- (a) Force curves for both w ----
    ax = axes[0]
    n_final = int(df["blocks_used"].max())
    sub = df[df["blocks_used"] == n_final]
    for w in w_values:
        s = sub[sub["w"] == w].sort_values("L_over_Rg")
        if s.empty:
            continue
        if {"fRg_bootstrap_CI95_low", "fRg_bootstrap_CI95_high"}.issubset(s.columns) and \
           np.all(np.isfinite(s["fRg_bootstrap_CI95_low"])) and np.all(np.isfinite(s["fRg_bootstrap_CI95_high"])):
            yerr = np.vstack([s["fRg"] - s["fRg_bootstrap_CI95_low"],
                              s["fRg_bootstrap_CI95_high"] - s["fRg"]])
        else:
            yerr = s["fRg_err"].to_numpy()
        ax.errorbar(s["L_over_Rg"], s["fRg"], yerr=yerr,
                    fmt=markers[w] + "-", ms=7, lw=1.8, capsize=3,
                    color=colors[w],
                    label=fr"$w={w}$, $R_g={Rg_map[w]:.3f}$")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$L/R_g(w)$")
    ax.set_ylabel(r"$f\,R_g\;(k_BT)$")
    ax.set_title(r"(a) Fold-back reflecting force, $N=200$", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    # ---- (b) Convergence ----
    ax = axes[1]
    L_min_center = int(df["L_center"].min())
    L_mid_center = int(sorted(df["L_center"].unique())[len(df["L_center"].unique()) // 2])
    for w, Lc, color, marker in [
            (w_values[0], L_min_center, "tab:red", "o"),
            (w_values[-1], L_min_center, "tab:red", "s"),
            (w_values[0], L_mid_center, "tab:blue", "o"),
            (w_values[-1], L_mid_center, "tab:blue", "s"),
    ]:
        s = df[(df["w"] == w) & (df["L_center"] == Lc)].sort_values("blocks_used")
        if s.empty:
            continue
        ls = "--" if w == w_values[-1] else "-"
        ax.errorbar(s["blocks_used"], s["fRg"], yerr=s["fRg_err"],
                    fmt=marker + ls, ms=6, lw=1.5, capsize=3,
                    color=color,
                    label=fr"$w={w}$, $L={Lc}$ ($L/R_g={s['L_over_Rg'].iloc[0]:.2f}$)")
    ax.set_xlabel("Cumulative blocks")
    ax.set_ylabel(r"$f\,R_g\;(k_BT)$")
    ax.set_title("(b) Block-level force convergence", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="best")

    fig.suptitle(
        r"Supplementary Fig. S11 — fold-back reflecting DJ force "
        r"($N=200$; $w=0.3$, $0.5$; $w=0$ gives $f\equiv 0$ for this rule)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI / Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="S11 fold-back reflecting DJ force, w-crossover (numba PERM).",
    )
    p.add_argument("--rg-master", required=False, type=Path)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path("S11_REV3_FINAL"))
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--w-values", nargs="+", type=float, default=[0.3, 0.5])
    p.add_argument("--L-centers", nargs="+", type=int,
                   default=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
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
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def run_selftest() -> None:
    """Fast internal validation; does not require an Rg master file."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the self-test.")
    L = 6
    assert _foldback(0, L) == 2
    assert _foldback(L, L) == L - 2
    assert _foldback(3, L) == 3
    N = 8
    pilot = _pilot_kernel_foldback(N, L, 0.0, 64, 12345)
    target = (N - 1) * math.log(6.0)
    if not np.isfinite(pilot[N]) or abs(pilot[N] - target) > 1e-12:
        raise AssertionError(f"w=0 fold-back benchmark failed: {pilot[N]} vs {target}")
    lo = pilot + math.log(0.5)
    hi = pilot + math.log(2.0)
    rec = run_block(N, L, 0.0, 32, 512, 54321, 0.5, 4, lo, hi)
    if rec["status"] != "OK" or not np.isfinite(rec["logZ"]):
        raise AssertionError(f"PERM self-test failed: {rec}")
    if abs(rec["logZ"] - target) > 0.25:
        raise AssertionError(f"w=0 PERM benchmark drift too large: {rec['logZ']} vs {target}")
    print("S11 self-test passed: fold-back mapping, exact w=0 benchmark, and PERM kernel.")


def main():
    args = parse_args()
    if args.self_test:
        run_selftest()
        return
    if args.rg_master is None:
        raise ValueError("--rg-master is required for production runs (omit only with --self-test).")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required.")

    if args.smoke:
        args.blocks = 4
        args.block_counts = [2, 4]
        args.roots_per_block = 64
        args.pilot_roots = 500
        args.max_population = 4096
        args.L_centers = [10, 16, 22]
        args.bootstrap_reps = 500

    if args.delta_L % 2 != 0 or args.delta_L <= 0:
        raise ValueError("delta_L must be a positive even integer.")
    centers = sorted({int(L) for L in args.L_centers})
    for Lc in centers:
        if Lc % 2 != 0:
            raise ValueError(f"center L={Lc} must be even (fold-back with L/2 tether)")
        if Lc - args.delta_L < 2:
            raise ValueError(f"center L={Lc} too small for delta_L={args.delta_L}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    master_rg = load_master_rg(args.rg_master, args.N, args.w_values)
    Rg_map = {float(w): master_rg[float(w)]["Rg"] for w in args.w_values}

    print("=" * 92)
    print(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
    print("=" * 92)
    print(f"numba: {NUMBA_AVAILABLE} ({getattr(numba,'__version__','?')})")
    print(f"N={args.N}; w-values={args.w_values}; boundary=fold-back reflecting")
    print(f"centers (even): {centers}")
    print(f"delta_L={args.delta_L}; blocks={args.blocks}; block-counts={args.block_counts}")
    print(f"roots/block={args.roots_per_block}; pilot={args.pilot_roots}; "
          f"C-/C+={args.C_minus}/{args.C_plus}")
    print(f"prune_p={args.prune_probability}; max_clones={args.max_clones}; "
          f"max_pop={args.max_population}")
    print("=" * 92)

    for w in args.w_values:
        r = master_rg[float(w)]
        print(f"Rg(w={w:g}) = {r['Rg']:.9f} +/- {r['Rg_err']:.9f} [{r['quality_flag']}]")

    # Build state list
    states_to_run: List[Tuple[int, float]] = []
    for w in args.w_values:
        unique_L = set(centers)
        for Lc in centers:
            unique_L.add(Lc - args.delta_L)
            unique_L.add(Lc + args.delta_L)
        for L in sorted(unique_L):
            if L < 2 or L % 2 != 0:
                continue
            states_to_run.append((int(L), float(w)))
    print(f"Total physical states: {len(states_to_run)}")

    print("\nBuilding pilot thresholds...")
    thresholds: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}
    for L, w in states_to_run:
        st_seed = _seed32(int(np.random.SeedSequence(
            [int(args.seed), 7, int(L), int(round(w * 1000))]
        ).generate_state(1, dtype=np.uint64)[0]))
        lo, hi = make_pilot(args.N, int(L), float(w),
                            args.pilot_roots, args.C_minus, args.C_plus, st_seed)
        thresholds[(int(L), float(w))] = (lo, hi)
        print(f"  pilot  w={w:g} L={L:3d} ok")

    print("\nRunning production...")
    t0 = time.time()
    states: Dict[Tuple[int, float], dict] = {}
    for L, w in states_to_run:
        lo, hi = thresholds[(int(L), float(w))]
        st_seed = _seed32(int(np.random.SeedSequence(
            [int(args.seed), 11, int(L), int(round(w * 1000))]
        ).generate_state(1, dtype=np.uint64)[0]))
        st = run_state(args.N, int(L), float(w), args.blocks,
                       args.roots_per_block, args.max_population,
                       args.prune_probability, args.max_clones,
                       lo, hi, st_seed)
        states[(int(L), float(w))] = st
        es = [b["ESS"] for b in st["blocks"]]
        print(f"  state  w={w:g} L={L:3d} minESS={min(es):.1f} medESS={np.median(es):.1f}")
    print(f"Production wall time: {(time.time() - t0)/60.0:.2f} min")

    print("\nAssembling force/convergence tables...")
    n_levels = [n for n in args.block_counts if n <= args.blocks]
    if not n_levels:
        raise ValueError("No valid block-count levels.")

    force_rows: List[dict] = []
    for w in args.w_values:
        force_rows.extend(assemble_force(states, float(w), centers,
                                          args.delta_L, Rg_map[float(w)], n_levels))

    # Bootstrap CIs on the final block level
    for row in force_rows:
        if row["blocks_used"] != max(n_levels):
            continue
        Lm, Lp = row["L_minus"], row["L_plus"]
        w = row["w"]
        sm = states[(int(Lm), float(w))]
        sp = states[(int(Lp), float(w))]
        logm = [b["logZ"] for b in sm["blocks"]]
        logp = [b["logZ"] for b in sp["blocks"]]
        _, lo, hi = independent_endpoint_bootstrap(logm, logp, args.delta_L, Rg_map[float(w)],
                                                   args.bootstrap_reps,
                                                   args.seed + int(Lm) * 13 + int(round(w * 100)))
        row["fRg_bootstrap_CI95_low"] = lo
        row["fRg_bootstrap_CI95_high"] = hi
        row["fRg_bootstrap_err_low"] = row["fRg"] - lo if np.isfinite(lo) else np.nan
        row["fRg_bootstrap_err_high"] = hi - row["fRg"] if np.isfinite(hi) else np.nan

    # CSV outputs
    force_csv = outdir / "S11_force_levels.csv"
    pd.DataFrame(force_rows).to_csv(force_csv, index=False)

    block_csv = outdir / "S11_block_level.csv"
    block_rows = []
    for (L, w), st in states.items():
        for b in st["blocks"]:
            block_rows.append({
                "boundary": "fold-back reflecting",
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

    png = outdir / "FigS11_foldback_reflecting.png"
    pdf = outdir / "FigS11_foldback_reflecting.pdf"
    make_figure(force_rows, args.w_values, Rg_map, png, pdf)

    prov = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numba": getattr(numba, "__version__", None),
        "N": args.N,
        "w_values": [float(w) for w in args.w_values],
        "boundary": "fold-back reflecting",
        "centers": [int(L) for L in centers],
        "delta_L": int(args.delta_L),
        "blocks": int(args.blocks), "block_counts": list(n_levels),
        "roots_per_block": int(args.roots_per_block),
        "pilot_roots": int(args.pilot_roots),
        "C_minus": args.C_minus, "C_plus": args.C_plus,
        "prune_probability": args.prune_probability,
        "max_clones": args.max_clones, "max_population": args.max_population,
        "seed": args.seed,
        "geometry": {
            "lattice": "3D simple cubic",
            "tether": "z = L/2 (center tether)",
            "boundary": "discrete fold-back / mirror rule; NOT generic reflecting walls",
            "accessible_layers": "1..L-1",
            "even_L_only": True,
            "mechanical_derivative":
                "f(L)=kBT[lnZ(L+dL)-lnZ(L-dL)]/(2 dL); both walls displaced "
                "symmetrically; tether held at L/2 of each derivative state; "
                "one-coordinate slit; no geometry factor",
        },
        "Rg_source": {
            "master_file": str(args.rg_master.resolve()),
            "values": {str(w): master_rg[float(w)] for w in args.w_values},
        },
        "estimator": {
            "lnZ": "block-averaged log(mean Rosenbluth weight) under PERM with frozen thresholds",
            "force": "blockwise difference of independent endpoint lnZ estimates: (mean lnZ_+ - mean lnZ_-) / (2 dL)",
            "uncertainty": "quadrature SEM from independent endpoint block ensembles; independent-endpoint percentile bootstrap CI",
            "cumulative": "same state-specific seeds; first n production blocks used at each convergence level",
        },
        "gaussian_w0_note": (
            "For the discrete fold-back rule the ideal Gaussian chain has six "
            "allowed moves at every step, hence Z = 6^(N-1), DeltaF = 0, and "
            "f = 0 exactly. w=0 is therefore not plotted."
        ),
        "reviewer_2_response": {
            "comment_1": (
                "The boundary is explicitly the discrete fold-back/mirror rule, "
                "named in every output row, in the figure title, and in this "
                "provenance. No claim about generic reflecting-wall physics is made."
            ),
            "comment_2": (
                "Mechanical derivative fully specified: both walls displaced "
                "symmetrically, tether held at z=L/2 in each derivative state, "
                "one-coordinate slit, no geometry factor."
            ),
            "comment_3": (
                "Signed forces retained; quality flag is sign-independent. The force "
                "uses independent endpoint block ensembles at L-dL and L+dL; the "
                "uncertainty is the block-level SEM from those endpoint estimates, "
                "with an independent-endpoint percentile bootstrap as an audit."
            ),
            "comment_4": (
                "PERM with frozen thresholds from an independent unpruned pilot; "
                "no MAD outlier rejection, no sign filtering, no smoothing, and "
                f"convergence reported at cumulative block counts {list(n_levels)}."
            ),
            "comment_7": (
                "This figure makes a single-ensemble claim at N=200 for two values "
                "of w. No cross-N claim is made; N-dependence of the fold-back "
                "force is outside the scope of this figure."
            ),
        },
        "signed_estimates_retained": True,
        "sign_selection": False,
        "final_plot_uncertainty": "independent-endpoint percentile bootstrap 95% CI; convergence panel uses block-level SEM",
        "smoothing": False,
        "fitting": False,
        "outlier_filter": False,
        "outputs": {
            "force_csv": str(force_csv),
            "block_csv": str(block_csv),
            "figure_png": str(png), "figure_pdf": str(pdf),
        },
    }
    prov_path = outdir / "S11_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 92)
    print("S11 written:")
    for p in [force_csv, block_csv, png, pdf, prov_path]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()