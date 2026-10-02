#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 7(ii) — Domb–Joyce N-dependent confinement-force scaling at fixed w=0.1
=============================================================================

Reviewer-facing production code following the revised manuscript workflow.

PURPOSE
-------
Test whether confinement-force data for Domb–Joyce chains with
N = 100, 150, 200, 250 at fixed excluded-volume strength w=0.1 become
approximately organized by the dimensionless variable L/R_g after scaling the
force by the same unconfined, three-dimensional R_g used for each chain.

MODEL / ENSEMBLE
----------------
Domb–Joyce polymer on a 3D simple-cubic lattice, with:
    - absorbing walls at z=0 and z=L,
    - allowed monomer layers z=1,...,L-1,
    - center tether z=L/2,
    - even integer L only,
    - overlap penalty exp(-w U), here w=0.1,
    - survival / Rosenbluth ensemble: a path that cannot continue contributes
      zero statistical weight to the partition-function estimator.

MECHANICAL DERIVATIVE
---------------------
The force is defined as

    f(L) = d ln Z(L) / dL = -dF/dL,

and evaluated locally on the even lattice by

    f(L) ~= [ ln Z(L+Delta_L) - ln Z(L-Delta_L) ] / (2 Delta_L),

with the default Delta_L=2. Both walls are displaced symmetrically; the tether
therefore remains centered at each derivative state.

SAMPLING / PERM
---------------
A genuine Domb–Joyce PERM estimator is used.

1. An independent UNPRUNED pilot is run once for every unique physical derivative
   state (N, L_state). If the same physical state occurs in two neighboring
   central-width derivatives, the same frozen threshold schedule is reused.
2. Pilot partial-chain log-partition estimates define frozen pruning/enrichment
   thresholds. Production never adapts these thresholds using production data.
3. Production uses Russian-roulette pruning and deterministic enrichment.
   Clones share total parent weight exactly.
4. A population-cap overflow is a HARD INVALIDATION. No production result is
   accepted when overflow occurs; branches are never silently discarded.
5. Force is formed from paired block partition-function estimates. Signed block
   forces are retained. There is NO sign-based deletion or selection.
6. Uncertainty is obtained from the signed block-force distribution, with a
   percentile bootstrap CI as an audit diagnostic.

DIAGNOSTICS REQUIRED BY THE REVIEW WORKFLOW
--------------------------------------------
- convergence of log Z and widening free-energy difference across cumulative
  block counts,
- convergence of the paired force estimate,
- minimum PERM effective sample size and maximum normalized weight fraction,
- overflow count,
- signed force values retained in all raw/block/point files,
- collapse residuals and fit-window sensitivity,
- complete provenance including the canonical R_g source.

IMPORTANT INTERPRETATION
------------------------
The figure is an approximate scaling-collapse test at fixed w=0.1.
It does NOT by itself establish a mathematically exact universal master curve.
Any power-law fit is reported as a finite-range effective exponent.

INPUT
-----
Canonical R_g master CSV produced by Rg_MASTER_FINAL.py. The file must contain
exactly one DJ row at w=0.1 for each requested N.

OUTPUT
------
Fig7ii_DJ_NScaling_PERM_FINAL/
    Fig7ii_DJ_NScaling_PointLevel.csv
    Fig7ii_DJ_NScaling_BlockLevel.csv
    Fig7ii_DJ_NScaling_PilotThresholds.csv
    Fig7ii_DJ_NScaling_Convergence.csv
    Fig7ii_DJ_NScaling_FitDiagnostics.csv
    Fig7ii_DJ_NScaling_CollapseResiduals.csv
    Fig7ii_DJ_NScaling_Provenance.json
    Fig7ii_DJ_NScaling.png
    Fig7ii_DJ_NScaling.pdf

RECOMMENDED TWO-STAGE WORKFLOW (KAGGLE / LINUX)
-----------------------------------------------
1) --plan-only creates and checks the realized even-L physical plan.
2) --pilot-only runs one independent unpruned pilot for each unique (N,L_state)
   and writes frozen thresholds.
3) Full production is run with --pilot-thresholds pointing to that frozen CSV.

Example full production command:
python Fig7ii_DJ_NScaling_PERM_FINAL_v1_2.py \
  --rg-master /kaggle/working/Rg_MASTER_DJ_w0p1/Rg_MASTER_FINAL.csv \
  --pilot-thresholds /kaggle/working/Fig7ii_PILOT/Fig7ii_DJ_NScaling_PilotThresholds.csv \
  --outdir /kaggle/working/Fig7ii_DJ_NScaling_FINAL \
  --N-list 100 150 200 250 \
  --target-ratios 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 \
  --delta-L 2 \
  --blocks 20 \
  --roots-per-block 256 \
  --pilot-roots 2000 \
  --max-population 4096 \
  --C-minus 0.5 \
  --C-plus 2.0 \
  --prune-probability 0.5 \
  --max-clones 4 \
  --workers 2 \
  --seed 20260927

Use --smoke only for code validation; smoke output is never for the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import numba
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This production script requires numba. Install numba before running."
    ) from exc


# =============================================================================
# Scientific constants / defaults
# =============================================================================

SCRIPT_NAME = Path(__file__).name
SCRIPT_VERSION = "1.2-RGPROP-FITFIX"
A = 1.0
KBT = 1.0
W_TARGET = 0.1
NU_SAW = 0.587597
SAW_FORCE_EXPONENT = -(1.0 + 1.0 / NU_SAW)
GAUSSIAN_FORCE_EXPONENT = -3.0

DEFAULT_N_LIST = [100, 150, 200, 250]
DEFAULT_TARGET_RATIOS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                         2.2, 2.4, 2.6, 2.8, 3.0]
DEFAULT_DELTA_L = 2
DEFAULT_BLOCKS = 20
DEFAULT_ROOTS_PER_BLOCK = 256
DEFAULT_PILOT_ROOTS = 2000
DEFAULT_MAX_POPULATION = 4096
DEFAULT_C_MINUS = 0.5
DEFAULT_C_PLUS = 2.0
DEFAULT_PRUNE_P = 0.5
DEFAULT_MAX_CLONES = 4
DEFAULT_BOOTSTRAP_REPS = 4000
DEFAULT_SEED = 20260927

# Quality thresholds are independent of force sign.
ESS_CRITICAL = 2.0
ESS_VERY_LOW = 5.0
ESS_LOW = 10.0
WEIGHT_CRITICAL = 0.80
WEIGHT_HIGH = 0.40

CONVERGENCE_K = (4, 8, 12, 16, 20)
FIT_WINDOWS = ((0.9, 1.6), (0.9, 1.8), (1.0, 1.8))

STEPS = np.asarray(
    [[1, 0, 0], [-1, 0, 0],
     [0, 1, 0], [0, -1, 0],
     [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


# =============================================================================
# Reproducible seeds / hashes
# =============================================================================


def make_seed(base_seed: int, *tokens: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), *[int(t) for t in tokens]])
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Lattice hash helpers
# =============================================================================

@njit(cache=True)
def coord_key(x: int, y: int, z: int) -> np.uint64:
    bias = np.int64(1_000_000)
    mask = np.uint64((1 << 21) - 1)
    ux = np.uint64(np.int64(x) + bias) & mask
    uy = np.uint64(np.int64(y) + bias) & mask
    uz = np.uint64(np.int64(z) + bias) & mask
    return ((ux << np.uint64(42)) |
            (uy << np.uint64(21)) |
            uz)


@njit(cache=True)
def hidx(key: np.uint64, mask: int) -> int:
    return int((key * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(mask))


@njit(cache=True)
def occ(keys, counts, used, table_size: int, x: int, y: int, z: int) -> int:
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
    for _ in range(2048):
        if used[idx] == 0:
            return 0
        if keys[idx] == key:
            return int(counts[idx])
        idx += np.int64(1)
        if idx >= table_size:
            idx = 0
    return -1


@njit(cache=True)
def ins(keys, counts, used, table_size: int, x: int, y: int, z: int) -> bool:
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
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
            idx = 0
    return False


@njit(cache=True)
def logsumexp6(values) -> float:
    m = -np.inf
    for i in range(6):
        v = values[i]
        if np.isfinite(v) and v > m:
            m = v
    if not np.isfinite(m):
        return -np.inf
    s = 0.0
    for i in range(6):
        v = values[i]
        if np.isfinite(v):
            s += math.exp(v - m)
    return m + math.log(s) if s > 0.0 else -np.inf


@njit(cache=True)
def hash_selftest() -> Tuple[bool, bool, int, int]:
    size = 256
    keys = np.zeros(size, dtype=np.uint64)
    counts = np.zeros(size, dtype=np.int16)
    used = np.zeros(size, dtype=np.uint8)
    a = ins(keys, counts, used, size, 1, 2, 3)
    c1 = occ(keys, counts, used, size, 1, 2, 3)
    b = ins(keys, counts, used, size, 1, 2, 3)
    c2 = occ(keys, counts, used, size, 1, 2, 3)
    return a, b, c1, c2


def run_selftest() -> None:
    a, b, c1, c2 = hash_selftest()
    if not (a and b and c1 == 1 and c2 == 2):
        raise RuntimeError(f"Hash-table self-test failed: {a=} {b=} {c1=} {c2=}")
    print("Hash-table self-test: PASS")


# =============================================================================
# Canonical R_g loading
# =============================================================================


def normalize_model(value: object) -> str:
    return str(value).strip().lower().replace("–", "-").replace(" ", "")


def load_master_rg(path: Path, N_list: Sequence[int]) -> Dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Rg master file not found: {path}")

    df = pd.read_csv(path)
    required = {"model", "N", "w", "Rg", "Rg_err", "quality_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rg master missing columns: {sorted(missing)}")

    df = df.copy()
    df["_model"] = df["model"].map(normalize_model)
    df["N_num"] = pd.to_numeric(df["N"], errors="coerce")
    df["w_num"] = pd.to_numeric(df["w"], errors="coerce")

    dj = df[
        (df["_model"] == "dj") &
        np.isclose(df["w_num"], W_TARGET, rtol=0.0, atol=1e-12)
    ].copy()

    if dj.empty:
        raise ValueError("No Domb–Joyce rows at w=0.1 found in the master R_g CSV.")

    out: Dict[int, dict] = {}
    for N in N_list:
        rows = dj[dj["N_num"] == int(N)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one DJ w=0.1 master row for N={N}; found {len(rows)}."
            )
        r = rows.iloc[0]
        Rg = float(r["Rg"])
        Rg_err = float(r["Rg_err"])
        quality = str(r["quality_flag"])
        if not np.isfinite(Rg) or Rg <= 0:
            raise ValueError(f"Invalid R_g for N={N}: {Rg}")
        if not np.isfinite(Rg_err) or Rg_err < 0:
            raise ValueError(f"Invalid R_g error for N={N}: {Rg_err}")
        if quality.upper() not in {"OK", "EXACT"}:
            raise ValueError(
                f"Canonical R_g quality for N={N} is {quality!r}; production requires OK."
            )
        out[int(N)] = {
            "N": int(N),
            "Rg": Rg,
            "Rg_err": Rg_err,
            "quality_flag": quality,
            "model": str(r["model"]),
            "w": float(r["w_num"]),
        }
    return out


# =============================================================================
# Width planning
# =============================================================================


def nearest_even(value: float, minimum: int = 4) -> int:
    width = max(minimum, int(math.floor(value / 2.0 + 0.5) * 2))
    return width


def make_unique_L_plan(
    Rg: float,
    target_ratios: Sequence[float],
    delta_L: int,
) -> List[dict]:
    by_L = {}
    for target in target_ratios:
        if target <= 0:
            raise ValueError(f"Target L/Rg must be positive: {target}")
        L = nearest_even(target * Rg, minimum=max(2 + delta_L, 4))
        if L - delta_L < 2:
            continue
        realized = L / Rg
        item = {
            "target_L_over_Rg": float(target),
            "L": int(L),
            "realized_L_over_Rg": float(realized),
        }
        if L not in by_L:
            by_L[L] = item
        else:
            # Keep the requested target closest to its actual realized ratio.
            old = by_L[L]
            if abs(target - realized) < abs(old["target_L_over_Rg"] - realized):
                by_L[L] = item
    return sorted(by_L.values(), key=lambda d: d["L"])


# =============================================================================
# Domb–Joyce pilot
# =============================================================================

@njit(cache=True)
def pilot_partial(N: int, L: int, w: float, roots: int, seed: int) -> np.ndarray:
    """
    Independent unpruned Domb–Joyce Rosenbluth pilot.

    Returns pilot_logZ[k] = log(mean_root W_k I_alive) for k monomers.
    """
    np.random.seed(seed)

    table_size = 1
    while table_size < 3 * (N + 64):
        table_size *= 2

    dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
    dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
    dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

    logsum = np.full(N + 1, -np.inf, dtype=np.float64)
    logsum[0] = 0.0

    for r in range(roots):
        keys = np.zeros(table_size, dtype=np.uint64)
        counts = np.zeros(table_size, dtype=np.int16)
        used = np.zeros(table_size, dtype=np.uint8)

        x = 0
        y = 0
        z = L // 2
        if not ins(keys, counts, used, table_size, x, y, z):
            continue

        lw = 0.0
        alive = True

        for k in range(1, N):
            # Add current W_k to the pilot mean.
            if alive and np.isfinite(lw):
                old = logsum[k]
                if not np.isfinite(old):
                    logsum[k] = lw
                elif lw > old:
                    logsum[k] = lw + math.log1p(math.exp(old - lw))
                else:
                    logsum[k] = old + math.log1p(math.exp(lw - old))

            trial = np.full(6, -np.inf, dtype=np.float64)
            bad = False
            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]
                if zz <= 0 or zz >= L:
                    continue
                ov = occ(keys, counts, used, table_size, xx, yy, zz)
                if ov < 0:
                    bad = True
                    break
                trial[d] = -w * ov

            if bad:
                alive = False
                break

            norm = logsumexp6(trial)
            if not np.isfinite(norm):
                alive = False
                break

            # Sample direction proportional to exp(-w * overlap).
            m = -np.inf
            for d in range(6):
                if np.isfinite(trial[d]) and trial[d] > m:
                    m = trial[d]
            total = 0.0
            for d in range(6):
                if np.isfinite(trial[d]):
                    total += math.exp(trial[d] - m)

            u = np.random.random() * total
            acc = 0.0
            chosen = -1
            for d in range(6):
                if np.isfinite(trial[d]):
                    acc += math.exp(trial[d] - m)
                    if u <= acc:
                        chosen = d
                        break

            if chosen < 0:
                alive = False
                break

            lw += norm
            x += dx[chosen]
            y += dy[chosen]
            z += dz[chosen]

            if not ins(keys, counts, used, table_size, x, y, z):
                alive = False
                break

        if alive and np.isfinite(lw):
            old = logsum[N]
            if not np.isfinite(old):
                logsum[N] = lw
            elif lw > old:
                logsum[N] = lw + math.log1p(math.exp(old - lw))
            else:
                logsum[N] = old + math.log1p(math.exp(lw - old))

    ln_roots = math.log(roots)
    for k in range(N + 1):
        if np.isfinite(logsum[k]):
            logsum[k] -= ln_roots
    return logsum


def complete_pilot(pilot: np.ndarray) -> np.ndarray:
    out = pilot.copy()
    n = out.size
    valid = np.isfinite(out)
    valid[0] = True
    out[0] = 0.0
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.arange(n, dtype=float) * math.log(2.0)
    if idx.size == 1:
        out[:] = out[0]
        return out
    x = np.arange(n, dtype=float)
    out[~valid] = np.interp(x[~valid], x[idx], out[idx])
    last = idx[-1]
    if last < n - 1:
        take = min(5, idx.size)
        slope = np.polyfit(idx[-take:], out[idx[-take:]], 1)[0]
        for k in range(last + 1, n):
            out[k] = out[last] + slope * (k - last)
    return out


def build_thresholds(
    N: int,
    L: int,
    w: float,
    pilot_roots: int,
    seed: int,
    cminus: float,
    cplus: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pilot_partial(N, L, w, pilot_roots, seed)
    ref = complete_pilot(raw)
    low = ref + math.log(cminus)
    high = ref + math.log(cplus)
    return ref, low, high


# =============================================================================
# Genuine Domb–Joyce PERM production block
# =============================================================================

@njit(cache=True)
def perm_block_dj(
    N: int,
    L: int,
    w: float,
    roots: int,
    max_pop: int,
    seed: int,
    prune_p: float,
    max_clones: int,
    log_low: np.ndarray,
    log_high: np.ndarray,
) -> tuple:
    """
    One independent genuine-PERM production block.

    Returns
    -------
    logZ, ESS, max_weight_fraction, population_ratio, final_population,
    prune_deaths, clone_events, overflow_events.
    """
    np.random.seed(seed)

    table_size = 1
    while table_size < 3 * (N + 64):
        table_size *= 2

    coords_a = np.zeros((max_pop, N, 3), dtype=np.int32)
    coords_b = np.zeros((max_pop, N, 3), dtype=np.int32)

    keys_a = np.zeros((max_pop, table_size), dtype=np.uint64)
    keys_b = np.zeros((max_pop, table_size), dtype=np.uint64)
    counts_a = np.zeros((max_pop, table_size), dtype=np.int16)
    counts_b = np.zeros((max_pop, table_size), dtype=np.int16)
    used_a = np.zeros((max_pop, table_size), dtype=np.uint8)
    used_b = np.zeros((max_pop, table_size), dtype=np.uint8)

    logw_a = np.full(max_pop, -np.inf, dtype=np.float64)
    logw_b = np.full(max_pop, -np.inf, dtype=np.float64)

    dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
    dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
    dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

    pop = roots
    if pop > max_pop:
        return -np.inf, 0.0, 1.0, 0.0, 0, 0, 0, 1

    for r in range(pop):
        coords_a[r, 0, 0] = 0
        coords_a[r, 0, 1] = 0
        coords_a[r, 0, 2] = L // 2
        logw_a[r] = 0.0
        if not ins(keys_a[r], counts_a[r], used_a[r], table_size,
                   0, 0, L // 2):
            return -np.inf, 0.0, 1.0, 0.0, 0, 0, 0, 1

    prune_deaths = 0
    clone_events = 0
    overflow_events = 0

    for k in range(1, N):
        new_pop = 0
        low_log = log_low[k + 1]
        high_log = log_high[k + 1]

        for parent in range(pop):
            if not np.isfinite(logw_a[parent]):
                continue

            x = coords_a[parent, k - 1, 0]
            y = coords_a[parent, k - 1, 1]
            z = coords_a[parent, k - 1, 2]

            trial = np.full(6, -np.inf, dtype=np.float64)
            bad = False
            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]
                if zz <= 0 or zz >= L:
                    continue
                ov = occ(keys_a[parent], counts_a[parent], used_a[parent],
                         table_size, xx, yy, zz)
                if ov < 0:
                    bad = True
                    break
                trial[d] = -w * ov

            if bad:
                continue

            norm = logsumexp6(trial)
            if not np.isfinite(norm):
                continue

            # Sample q_j proportional to exp(-w*overlap_j).
            m = -np.inf
            for d in range(6):
                if np.isfinite(trial[d]) and trial[d] > m:
                    m = trial[d]
            total = 0.0
            for d in range(6):
                if np.isfinite(trial[d]):
                    total += math.exp(trial[d] - m)
            u = np.random.random() * total
            acc = 0.0
            chosen = -1
            for d in range(6):
                if np.isfinite(trial[d]):
                    acc += math.exp(trial[d] - m)
                    if u <= acc:
                        chosen = d
                        break
            if chosen < 0:
                continue

            nx = x + dx[chosen]
            ny = y + dy[chosen]
            nz = z + dz[chosen]
            new_logw = logw_a[parent] + norm

            # Russian roulette pruning.
            if new_logw < low_log:
                if np.random.random() < prune_p:
                    prune_deaths += 1
                    continue
                new_logw -= math.log(1.0 - prune_p)

            # Enrichment factor; total clone weight is exactly preserved.
            # IMPORTANT: PERM clones the SAME newly generated child state.
            # Cloning into distinct next-step directions would change the
            # proposal distribution and bias the estimator because q_j is not
            # generally uniform for the Domb–Joyce model.
            nk = 1
            if new_logw > high_log:
                ratio = math.exp(min(20.0, new_logw - high_log))
                nk = int(math.ceil(ratio))
                if nk < 2:
                    nk = 2
                if nk > max_clones:
                    nk = max_clones

            if new_pop + nk > max_pop:
                # HARD INVALIDATION: no contribution is silently discarded.
                overflow_events += 1
                return (
                    -np.inf, 0.0, 1.0, float(new_pop / roots),
                    new_pop, prune_deaths, clone_events, overflow_events
                )

            child_logw = new_logw - math.log(nk)

            # The already sampled direction is the child state to be enriched.
            cx = nx
            cy = ny
            cz = nz

            # First child.
            child = new_pop
            new_pop += 1
            for q in range(k):
                coords_b[child, q, 0] = coords_a[parent, q, 0]
                coords_b[child, q, 1] = coords_a[parent, q, 1]
                coords_b[child, q, 2] = coords_a[parent, q, 2]
            coords_b[child, k, 0] = cx
            coords_b[child, k, 1] = cy
            coords_b[child, k, 2] = cz

            for q in range(table_size):
                keys_b[child, q] = keys_a[parent, q]
                counts_b[child, q] = counts_a[parent, q]
                used_b[child, q] = used_a[parent, q]

            if not ins(keys_b[child], counts_b[child], used_b[child],
                       table_size, cx, cy, cz):
                overflow_events += 1
                return (
                    -np.inf, 0.0, 1.0, float(new_pop / roots),
                    new_pop, prune_deaths, clone_events, overflow_events
                )
            logw_b[child] = child_logw

            # Additional clones are exact copies of the selected child.
            for c in range(1, nk):
                clone = new_pop
                new_pop += 1
                clone_events += 1
                for q in range(k + 1):
                    coords_b[clone, q, 0] = coords_b[child, q, 0]
                    coords_b[clone, q, 1] = coords_b[child, q, 1]
                    coords_b[clone, q, 2] = coords_b[child, q, 2]
                for q in range(table_size):
                    keys_b[clone, q] = keys_b[child, q]
                    counts_b[clone, q] = counts_b[child, q]
                    used_b[clone, q] = used_b[child, q]
                logw_b[clone] = child_logw

        # Swap population buffers.
        coords_a, coords_b = coords_b, coords_a
        keys_a, keys_b = keys_b, keys_a
        counts_a, counts_b = counts_b, counts_a
        used_a, used_b = used_b, used_a
        logw_a, logw_b = logw_b, logw_a

        for i in range(max_pop):
            logw_b[i] = -np.inf

        pop = new_pop
        if pop == 0:
            return (
                -np.inf, 0.0, 1.0, 0.0,
                0, prune_deaths, clone_events, overflow_events
            )

    vals = logw_a[:pop]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (
            -np.inf, 0.0, 1.0, 0.0,
            0, prune_deaths, clone_events, overflow_events
        )

    vmax = np.max(vals)
    ww = np.exp(vals - vmax)
    s1 = np.sum(ww)
    s2 = np.sum(ww * ww)
    if s1 <= 0.0 or s2 <= 0.0:
        return (
            -np.inf, 0.0, 1.0, float(vals.size / roots),
            vals.size, prune_deaths, clone_events, overflow_events
        )

    logz = vmax + math.log(s1 / roots)
    ess = s1 * s1 / s2
    maxfrac = np.max(ww / s1)
    pop_ratio = vals.size / roots

    return (
        logz, ess, maxfrac, pop_ratio,
        vals.size, prune_deaths, clone_events, overflow_events
    )


# =============================================================================
# Production block wrappers
# =============================================================================

@dataclass
class BlockRecord:
    N: int
    L_center: int
    L_state: int
    side: str
    block_id: int
    Rg: float
    Rg_err: float
    realized_L_over_Rg: float
    logZ: float
    ESS: float
    max_weight_fraction: float
    population_ratio: float
    final_population: int
    prune_deaths: int
    clone_events: int
    overflow_events: int
    roots_per_block: int
    runtime_seconds: float


def run_block_task(task: tuple) -> dict:
    (
        N, L_center, L_state, side, block_id, Rg, Rg_err,
        w, roots, max_pop, prune_p, max_clones,
        log_low, log_high, seed,
    ) = task
    t0 = time.perf_counter()
    out = perm_block_dj(
        N, L_state, w, roots, max_pop, seed,
        prune_p, max_clones, log_low, log_high,
    )
    return asdict(BlockRecord(
        N=N,
        L_center=L_center,
        L_state=L_state,
        side=side,
        block_id=block_id,
        Rg=Rg,
        Rg_err=Rg_err,
        realized_L_over_Rg=L_center / Rg,
        logZ=float(out[0]),
        ESS=float(out[1]),
        max_weight_fraction=float(out[2]),
        population_ratio=float(out[3]),
        final_population=int(out[4]),
        prune_deaths=int(out[5]),
        clone_events=int(out[6]),
        overflow_events=int(out[7]),
        roots_per_block=roots,
        runtime_seconds=time.perf_counter() - t0,
    ))


# =============================================================================
# Statistical helpers
# =============================================================================


def block_sem(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sem = block_sem(x)
    if x.size < 2 or reps <= 0:
        return mean, sem, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    boots = np.mean(x[idx], axis=1)
    return (
        mean,
        sem,
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def quality_flag(min_ess: float, max_weight_fraction: float, overflow: int) -> str:
    # Quality is explicitly independent of force sign.
    if overflow > 0:
        return "INVALID_OVERFLOW"
    if not np.isfinite(min_ess) or not np.isfinite(max_weight_fraction):
        return "INVALID_DIAGNOSTICS"
    if min_ess < ESS_CRITICAL or max_weight_fraction >= WEIGHT_CRITICAL:
        return "CRITICAL_ESS"
    if min_ess < ESS_VERY_LOW:
        return "VERY_LOW_ESS"
    if min_ess < ESS_LOW:
        return "LOW_ESS"
    if max_weight_fraction >= WEIGHT_HIGH:
        return "HIGH_WEIGHT_CONCENTRATION"
    return "OK"


def paired_block_dataframe(side_df: pd.DataFrame, delta_L: int) -> pd.DataFrame:
    """Pair plus/minus block estimates using the requested finite-difference step."""
    if delta_L <= 0:
        raise ValueError("delta_L must be positive.")
    minus = side_df[side_df["side"] == "minus"].copy()
    plus = side_df[side_df["side"] == "plus"].copy()
    minus = minus.sort_values("block_id")
    plus = plus.sort_values("block_id")
    merged = minus.merge(
        plus,
        on="block_id",
        suffixes=("_minus", "_plus"),
        how="inner",
    )
    merged["deltaF_widen_block"] = (
        merged["logZ_plus"] - merged["logZ_minus"]
    )
    merged["force_block"] = (
        merged["deltaF_widen_block"] / (2.0 * float(delta_L))
    )
    return merged


def assemble_point(
    N: int,
    target_ratio: float,
    L_center: int,
    delta_L: int,
    rg: dict,
    block_df: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> Tuple[dict, pd.DataFrame]:
    minus = block_df[block_df["side"] == "minus"].copy()
    plus = block_df[block_df["side"] == "plus"].copy()

    merged = minus.merge(
        plus,
        on="block_id",
        suffixes=("_minus", "_plus"),
        how="inner",
    ).sort_values("block_id")

    # Correct general denominator for arbitrary delta_L.
    merged["deltaF_widen_block"] = merged["logZ_plus"] - merged["logZ_minus"]
    merged["force_block"] = merged["deltaF_widen_block"] / (2.0 * delta_L)
    merged["fRg_block"] = merged["force_block"] * rg["Rg"]

    valid = (
        np.isfinite(merged["logZ_minus"]) &
        np.isfinite(merged["logZ_plus"])
    )
    fvals = merged.loc[valid, "force_block"].to_numpy(float)
    dFvals = merged.loc[valid, "deltaF_widen_block"].to_numpy(float)

    overflow_count = int(
        block_df["overflow_events"].fillna(0).sum()
    )
    min_ess = float(block_df["ESS"].replace([np.inf, -np.inf], np.nan).min()) if len(block_df) else np.nan
    max_weight = float(block_df["max_weight_fraction"].replace([np.inf, -np.inf], np.nan).max()) if len(block_df) else np.nan

    qflag = quality_flag(min_ess, max_weight, overflow_count)

    if not qflag.startswith("OK") and qflag != "OK":
        # Keep signed values for audit, but do not silently publish an invalid point.
        pass

    if fvals.size:
        f_mean, f_sem, f_ci_lo, f_ci_hi = bootstrap_mean_ci(
            fvals, bootstrap_reps, seed
        )
        dF_mean = float(np.mean(dFvals))
        dF_sem = block_sem(dFvals)
    else:
        f_mean = f_sem = f_ci_lo = f_ci_hi = np.nan
        dF_mean = dF_sem = np.nan

    fRg = f_mean * rg["Rg"] if np.isfinite(f_mean) else np.nan
    # Separate force-sampling and R_g contributions.  R_g comes from the
    # independent unconfined master calculation and is therefore propagated
    # independently of the confined-force uncertainty.
    fRg_force_err = f_sem * rg["Rg"] if np.isfinite(f_sem) else np.nan
    fRg_rg_err = abs(f_mean) * rg["Rg_err"] if np.isfinite(f_mean) else np.nan
    if np.isfinite(fRg_force_err) and np.isfinite(fRg_rg_err):
        fRg_sem = float(math.sqrt(fRg_force_err**2 + fRg_rg_err**2))
    else:
        fRg_sem = np.nan
    fRg_ci_lo = f_ci_lo * rg["Rg"] if np.isfinite(f_ci_lo) else np.nan
    fRg_ci_hi = f_ci_hi * rg["Rg"] if np.isfinite(f_ci_hi) else np.nan

    logZ_minus = float(np.mean(minus.loc[np.isfinite(minus["logZ"]), "logZ"])) if np.any(np.isfinite(minus["logZ"])) else np.nan
    logZ_plus = float(np.mean(plus.loc[np.isfinite(plus["logZ"]), "logZ"])) if np.any(np.isfinite(plus["logZ"])) else np.nan
    logZ_minus_err = block_sem(minus.loc[np.isfinite(minus["logZ"]), "logZ"].to_numpy(float))
    logZ_plus_err = block_sem(plus.loc[np.isfinite(plus["logZ"]), "logZ"].to_numpy(float))

    point = {
        "N": N,
        "w": W_TARGET,
        "Rg": rg["Rg"],
        "Rg_err": rg["Rg_err"],
        "Rg_quality": rg["quality_flag"],
        "target_L_over_Rg": float(target_ratio),
        "L": int(L_center),
        "realized_L_over_Rg": float(L_center / rg["Rg"]),
        "L_minus": int(L_center - delta_L),
        "L_plus": int(L_center + delta_L),
        "delta_L": int(delta_L),
        "force": float(f_mean),
        "force_err": float(f_sem),
        "force_bootstrap95_low": float(f_ci_lo),
        "force_bootstrap95_high": float(f_ci_hi),
        "fRg": float(fRg),
        "fRg_err": float(fRg_sem),
        "fRg_force_component_err": float(fRg_force_err),
        "fRg_Rg_component_err": float(fRg_rg_err),
        "fRg_bootstrap95_low": float(fRg_ci_lo),
        "fRg_bootstrap95_high": float(fRg_ci_hi),
        "logZ_minus": float(logZ_minus),
        "logZ_minus_err": float(logZ_minus_err),
        "logZ_plus": float(logZ_plus),
        "logZ_plus_err": float(logZ_plus_err),
        "deltaF_widen": float(dF_mean),
        "deltaF_widen_err": float(dF_sem),
        "valid_force_blocks": int(fvals.size),
        "total_blocks": int(len(merged)),
        "valid_block_fraction": float(fvals.size / len(merged)) if len(merged) else 0.0,
        "min_ESS": min_ess,
        "max_weight_fraction": max_weight,
        "overflow_events": overflow_count,
        "sign_selected": False,
        "excluded": bool(qflag != "OK"),
        "quality_flag": qflag,
    }

    merged["N"] = N
    merged["L_center"] = L_center
    merged["target_L_over_Rg"] = float(target_ratio)
    merged["realized_L_over_Rg"] = float(L_center / rg["Rg"])
    merged["Rg"] = rg["Rg"]
    merged["delta_L"] = delta_L
    return point, merged


# =============================================================================
# Convergence diagnostics
# =============================================================================


def cumulative_convergence(point_block: pd.DataFrame, point: dict) -> pd.DataFrame:
    rows = []
    max_k = int(point["total_blocks"])
    requested = [k for k in CONVERGENCE_K if k <= max_k]

    for k in requested:
        sub = point_block.sort_values("block_id").head(k)
        finite_m = np.isfinite(sub["logZ_minus"])
        finite_p = np.isfinite(sub["logZ_plus"])
        lm = float(np.mean(sub.loc[finite_m, "logZ_minus"])) if finite_m.any() else np.nan
        lp = float(np.mean(sub.loc[finite_p, "logZ_plus"])) if finite_p.any() else np.nan
        dfv = sub["deltaF_widen_block"].to_numpy(float)
        fv = sub["force_block"].to_numpy(float)
        dfv = dfv[np.isfinite(dfv)]
        fv = fv[np.isfinite(fv)]

        rows.append({
            "N": point["N"],
            "L": point["L"],
            "realized_L_over_Rg": point["realized_L_over_Rg"],
            "blocks_used": k,
            "logZ_minus_mean": lm,
            "logZ_plus_mean": lp,
            "deltaF_widen_mean": float(np.mean(dfv)) if dfv.size else np.nan,
            "force_mean": float(np.mean(fv)) if fv.size else np.nan,
            "force_sem": block_sem(fv),
            "overflow_events_cumulative": int(
                sub["overflow_events_minus"].sum() + sub["overflow_events_plus"].sum()
            ),
        })

    return pd.DataFrame(rows)


def add_relative_convergence_metrics(conv: pd.DataFrame) -> pd.DataFrame:
    conv = conv.copy()
    groups = []
    for (N, L), g in conv.groupby(["N", "L"], sort=False):
        g = g.sort_values("blocks_used").copy()
        final = g.iloc[-1]
        g["rel_change_logZ_minus_vs_final"] = g["logZ_minus_mean"] - final["logZ_minus_mean"]
        g["rel_change_logZ_plus_vs_final"] = g["logZ_plus_mean"] - final["logZ_plus_mean"]
        if np.isfinite(final["deltaF_widen_mean"]) and final["deltaF_widen_mean"] != 0:
            g["relative_change_deltaF_vs_final"] = (
                g["deltaF_widen_mean"] / final["deltaF_widen_mean"] - 1.0
            )
        else:
            g["relative_change_deltaF_vs_final"] = np.nan
        if np.isfinite(final["force_mean"]) and final["force_mean"] != 0:
            g["relative_change_force_vs_final"] = (
                g["force_mean"] / final["force_mean"] - 1.0
            )
        else:
            g["relative_change_force_vs_final"] = np.nan
        groups.append(g)
    return pd.concat(groups, ignore_index=True) if groups else conv


# =============================================================================
# Fit diagnostics
# =============================================================================


def weighted_loglog_fit(
    df: pd.DataFrame,
    xmin: float,
    xmax: float,
) -> dict:
    """Weighted log-log diagnostic fit with goodness-of-fit error inflation.

    The formal WLS covariance is retained, but when the residual scatter exceeds
    the stated point uncertainties (reduced chi^2 > 1), the reported parameter
    uncertainties are inflated by sqrt(reduced chi^2).  This avoids presenting an
    artificially precise effective exponent when finite-size/crossover deviations
    dominate the residuals.
    """
    d = df[
        np.isfinite(df["realized_L_over_Rg"]) &
        np.isfinite(df["fRg"]) &
        np.isfinite(df["fRg_err"]) &
        (df["realized_L_over_Rg"] >= xmin) &
        (df["realized_L_over_Rg"] <= xmax) &
        (df["fRg"] > 0.0) &
        (df["fRg_err"] > 0.0) &
        (df["quality_flag"] == "OK")
    ].copy()

    if len(d) < 3:
        return {
            "scope": "pooled",
            "xmin": xmin,
            "xmax": xmax,
            "n_points": int(len(d)),
            "exponent": np.nan,
            "exponent_err": np.nan,
            "formal_exponent_err": np.nan,
            "amplitude": np.nan,
            "amplitude_err": np.nan,
            "formal_amplitude_err": np.nan,
            "chi2": np.nan,
            "dof": 0,
            "reduced_chi2": np.nan,
            "error_inflation_factor": np.nan,
        }

    x = np.log(d["realized_L_over_Rg"].to_numpy(float))
    y = np.log(d["fRg"].to_numpy(float))
    sy = d["fRg_err"].to_numpy(float) / d["fRg"].to_numpy(float)
    sy = np.maximum(sy, 1e-12)
    weights = 1.0 / sy**2
    X = np.column_stack([np.ones_like(x), x])

    normal = X.T @ (weights[:, None] * X)
    cov = np.linalg.inv(normal)
    beta = cov @ (X.T @ (weights * y))
    intercept, slope = beta
    residual = y - X @ beta
    chi2 = float(np.sum((residual / sy) ** 2))
    dof = len(y) - 2
    redchi2 = float(chi2 / dof) if dof > 0 else np.nan
    inflation = math.sqrt(max(1.0, redchi2)) if np.isfinite(redchi2) else 1.0

    formal_slope_err = math.sqrt(cov[1, 1])
    formal_intercept_err = math.sqrt(cov[0, 0])
    slope_err = formal_slope_err * inflation
    intercept_err = formal_intercept_err * inflation

    return {
        "scope": "pooled",
        "xmin": xmin,
        "xmax": xmax,
        "n_points": int(len(d)),
        "exponent": float(slope),
        "exponent_err": float(slope_err),
        "formal_exponent_err": float(formal_slope_err),
        "amplitude": float(math.exp(intercept)),
        "amplitude_err": float(math.exp(intercept) * intercept_err),
        "formal_amplitude_err": float(math.exp(intercept) * formal_intercept_err),
        "chi2": chi2,
        "dof": int(dof),
        "reduced_chi2": redchi2,
        "error_inflation_factor": float(inflation),
    }

def fit_by_N(df: pd.DataFrame, xmin: float, xmax: float) -> List[dict]:
    rows = []
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        fit = weighted_loglog_fit(sub, xmin, xmax)
        fit["scope"] = f"N={N}"
        rows.append(fit)
    return rows


# =============================================================================
# Collapse residual diagnostics
# =============================================================================


def collapse_residuals(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["quality_flag"] == "OK") &
        np.isfinite(df["realized_L_over_Rg"]) &
        np.isfinite(df["fRg"]) &
        (df["fRg"] > 0.0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    # Fixed bins are a diagnostic only; no smoothing is used in the force data.
    bins = np.arange(0.75, 3.31, 0.25)
    d["x_bin"] = pd.cut(d["realized_L_over_Rg"], bins=bins, include_lowest=True)
    rows = []
    for _, g in d.groupby("x_bin", observed=True):
        y = g["fRg"].to_numpy(float)
        if y.size < 2:
            continue
        center = float(np.exp(np.mean(np.log(y))))
        for _, r in g.iterrows():
            rows.append({
                "N": int(r["N"]),
                "L": int(r["L"]),
                "realized_L_over_Rg": float(r["realized_L_over_Rg"]),
                "fRg": float(r["fRg"]),
                "bin_center_geometric": center,
                "ratio_to_bin_geometric_center": float(r["fRg"] / center),
                "log_residual": float(np.log(r["fRg"] / center)),
            })
    return pd.DataFrame(rows)


def summarize_collapse(resid: pd.DataFrame) -> dict:
    if resid.empty:
        return {
            "n_points": 0,
            "rms_log_residual": np.nan,
            "median_abs_percent_residual": np.nan,
        }
    lr = resid["log_residual"].to_numpy(float)
    pct = (np.exp(lr) - 1.0) * 100.0
    return {
        "n_points": int(len(resid)),
        "rms_log_residual": float(np.sqrt(np.mean(lr**2))),
        "median_abs_percent_residual": float(np.median(np.abs(pct))),
    }


# =============================================================================
# Publication plot
# =============================================================================


def configure_plot():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "legend.fontsize": 10,
        "axes.linewidth": 1.4,
        "savefig.dpi": 600,
    })


def pooled_binned_guide(points: pd.DataFrame, xmin: float, xmax: float) -> pd.DataFrame:
    d = points[
        (points["quality_flag"] == "OK") &
        (points["realized_L_over_Rg"] >= xmin) &
        (points["realized_L_over_Rg"] <= xmax) &
        np.isfinite(points["fRg"]) &
        (points["fRg"] > 0.0)
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=["x", "y", "n"])
    # Geometric-bin means are a guide to the pooled data only; they are not
    # used to estimate the primary force or uncertainty.
    edges = np.linspace(xmin, xmax, 8)
    rows = []
    d["bin"] = pd.cut(d["realized_L_over_Rg"], bins=edges, include_lowest=True)
    for _, g in d.groupby("bin", observed=True):
        if len(g) < 2:
            continue
        x = float(np.exp(np.mean(np.log(g["realized_L_over_Rg"].to_numpy(float)))))
        y = float(np.exp(np.mean(np.log(g["fRg"].to_numpy(float)))))
        rows.append({"x": x, "y": y, "n": int(len(g))})
    return pd.DataFrame(rows).sort_values("x") if rows else pd.DataFrame(columns=["x", "y", "n"])


def plot_figure(
    points: pd.DataFrame,
    fit: dict,
    out_png: Path,
    out_pdf: Path,
    show: bool,
) -> None:
    configure_plot()

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    N_list = sorted(points["N"].unique())
    # No explicit color choices: matplotlib default cycle is used.

    # Panel (a): dimensional force vs physical L.
    ax = axes[0]
    for N in N_list:
        d = points[points["N"] == N].sort_values("L")
        pos = d["force"] > 0.0
        if pos.any():
            ax.errorbar(
                d.loc[pos, "L"],
                d.loc[pos, "force"] * A / KBT,
                yerr=d.loc[pos, "force_err"] * A / KBT,
                fmt="o-", ms=5.8, lw=1.4, capsize=2.8,
                label=fr"$N={N}$",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Slit width $L/a$")
    ax.set_ylabel(r"$fa/k_BT$")
    ax.set_title("(a) Dimensional force")
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", alpha=0.08)
    ax.legend(loc="upper right", frameon=True)

    # Panel (b): scaled force vs L/Rg.
    ax = axes[1]
    for N in N_list:
        d = points[points["N"] == N].sort_values("realized_L_over_Rg")
        pos = d["fRg"] > 0.0
        if pos.any():
            ax.errorbar(
                d.loc[pos, "realized_L_over_Rg"],
                d.loc[pos, "fRg"],
                yerr=d.loc[pos, "fRg_err"],
                fmt="o", ms=5.8, capsize=2.8,
                label=fr"$N={N}$",
            )

    guide = pooled_binned_guide(points, max(0.9, fit.get("xmin", 0.9)), fit.get("xmax", 1.8))
    if len(guide) >= 2:
        ax.plot(guide["x"], guide["y"], "-", lw=2.0,
                label="pooled binned guide")

    # Pooled fit is an audit/reference curve, not a claim of exact universality.
    if np.isfinite(fit.get("exponent", np.nan)):
        xmin = max(0.75, fit["xmin"])
        xmax = fit["xmax"]
        xx = np.logspace(np.log10(xmin), np.log10(xmax), 250)
        yy = fit["amplitude"] * xx ** fit["exponent"]
        ax.loglog(
            xx, yy, "--", lw=2.0,
            label=rf"finite-range fit: $p={fit['exponent']:.3f}\pm{fit['exponent_err']:.3f}$",
        )

        # Reference slopes only; both are normalized at a common point.
        xref = min(1.15, xmax)
        yref0 = fit["amplitude"] * xref ** fit["exponent"]
        y_gauss = yref0 * (xx / xref) ** GAUSSIAN_FORCE_EXPONENT
        y_saw = yref0 * (xx / xref) ** SAW_FORCE_EXPONENT
        ax.loglog(
            xx, y_gauss, ":", lw=1.8,
            label=rf"Gaussian slope: $p={GAUSSIAN_FORCE_EXPONENT:.1f}$",
        )
        ax.loglog(
            xx, y_saw, "-.", lw=1.8,
            label=rf"SAW slope: $p={SAW_FORCE_EXPONENT:.3f}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/R_g$")
    ax.set_ylabel(r"$fR_g/k_BT$")
    ax.set_title("(b) Scaled force: approximate collapse")
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", alpha=0.08)
    ax.legend(loc="upper right", frameon=True)

    # Explicitly state the interpretation without overclaiming universality.
    fig.suptitle(
        r"Domb–Joyce confinement force at fixed $w=0.1$",
        fontsize=20, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.015,
        r"Unconfined 3D $R_g$; signed force estimates retained in the audit files; "
        r"log plots display only positive estimates.",
        ha="center", va="bottom", fontsize=11,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.88, wspace=0.22)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_bool_string(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Final Domb–Joyce Fig. 7(ii) PERM N-scaling workflow.",
    )
    p.add_argument("--rg-master", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("Fig7ii_DJ_NScaling_PERM_FINAL"))
    p.add_argument("--N-list", nargs="+", type=int, default=DEFAULT_N_LIST)
    p.add_argument("--target-ratios", nargs="+", type=float, default=DEFAULT_TARGET_RATIOS)
    p.add_argument("--w", type=float, default=W_TARGET)
    p.add_argument("--delta-L", type=int, default=DEFAULT_DELTA_L)
    p.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    p.add_argument("--roots-per-block", type=int, default=DEFAULT_ROOTS_PER_BLOCK)
    p.add_argument("--pilot-roots", type=int, default=DEFAULT_PILOT_ROOTS)
    p.add_argument("--max-population", type=int, default=DEFAULT_MAX_POPULATION)
    p.add_argument("--C-minus", type=float, default=DEFAULT_C_MINUS)
    p.add_argument("--C-plus", type=float, default=DEFAULT_C_PLUS)
    p.add_argument("--prune-probability", type=float, default=DEFAULT_PRUNE_P)
    p.add_argument("--max-clones", type=int, default=DEFAULT_MAX_CLONES)
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--show", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--pilot-thresholds", type=Path, default=None,
                    help="Frozen pilot-threshold CSV from a completed --pilot-only run.")
    p.add_argument("--fit-xmin", type=float, default=0.9)
    p.add_argument("--fit-xmax", type=float, default=1.8)
    return p.parse_args()


# =============================================================================
# Plan / pilot / production
# =============================================================================


def build_plan_dataframe(
    rg: Dict[int, dict],
    N_list: Sequence[int],
    target_ratios: Sequence[float],
    delta_L: int,
) -> pd.DataFrame:
    rows = []
    for N in N_list:
        plan = make_unique_L_plan(rg[N]["Rg"], target_ratios, delta_L)
        for pdef in plan:
            L = int(pdef["L"])
            rows.append({
                "N": int(N),
                "Rg": float(rg[N]["Rg"]),
                "Rg_err": float(rg[N]["Rg_err"]),
                "target_L_over_Rg": float(pdef["target_L_over_Rg"]),
                "L": L,
                "realized_L_over_Rg": float(pdef["realized_L_over_Rg"]),
                "L_minus": int(L - delta_L),
                "L_plus": int(L + delta_L),
                "delta_L": int(delta_L),
            })
    return pd.DataFrame(rows).sort_values(["N", "L"]).reset_index(drop=True)


def print_plan(
    rg: Dict[int, dict],
    N_list: Sequence[int],
    target_ratios,
    delta_L: int,
    outdir: Path,
) -> Tuple[List[Tuple[int, dict]], pd.DataFrame]:
    all_plans = []
    print("\nPHYSICAL EVEN-L PLAN")
    print("=" * 96)
    for N in N_list:
        plan = make_unique_L_plan(rg[N]["Rg"], target_ratios, delta_L)
        all_plans.extend((N, p) for p in plan)
        print(f"N={N:4d}  Rg={rg[N]['Rg']:.8f}  points={len(plan)}")
        for pdef in plan:
            L = pdef["L"]
            print(
                f"  target x={pdef['target_L_over_Rg']:.2f}  "
                f"L={L:2d}  realized x={pdef['realized_L_over_Rg']:.5f}  "
                f"derivative states=({L-delta_L},{L+delta_L})"
            )

    plan_df = build_plan_dataframe(rg, N_list, target_ratios, delta_L)
    plan_path = outdir / "Fig7ii_DJ_NScaling_LPlan.csv"
    plan_df.to_csv(plan_path, index=False)

    derivative_refs = []
    unique_states = set()
    for _, r in plan_df.iterrows():
        derivative_refs.extend([
            (int(r["N"]), int(r["L_minus"]), int(r["L"]), "minus"),
            (int(r["N"]), int(r["L_plus"]), int(r["L"]), "plus"),
        ])
        unique_states.add((int(r["N"]), int(r["L_minus"])))
        unique_states.add((int(r["N"]), int(r["L_plus"])))

    print(f"Total central points       : {len(all_plans)}")
    print(f"Total derivative references: {len(derivative_refs)}")
    print(f"Unique physical (N,L) states: {len(unique_states)}")
    print(f"Repeated derivative references: {len(derivative_refs) - len(unique_states)}")
    print(f"Saved plan CSV: {plan_path.resolve()}")
    return all_plans, plan_df


def run_pilots(
    args,
    rg: Dict[int, dict],
    point_plans: Sequence[Tuple[int, dict]],
    outdir: Path,
):
    """
    Run exactly one independent unpruned pilot for each unique physical
    derivative state (N, L_state). The same threshold schedule is therefore
    reused consistently whenever a state is referenced by two neighboring
    central-width derivatives.
    """
    state_refs: Dict[Tuple[int, int], list] = {}
    for N, pdef in point_plans:
        Lc = int(pdef["L"])
        for side, Lstate in [
            ("minus", Lc - args.delta_L),
            ("plus", Lc + args.delta_L),
        ]:
            state_refs.setdefault((int(N), int(Lstate)), []).append((Lc, side))

    unique_states = sorted(state_refs.keys())
    threshold_rows = []
    summary_rows = []
    threshold_store: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    print("\nBUILDING FROZEN INDEPENDENT PILOT THRESHOLDS")
    print("=" * 96)
    print(f"Unique physical pilot states : {len(unique_states)}")
    print(f"Pilot roots per state       : {args.pilot_roots}")
    print(f"Total pilot roots           : {len(unique_states) * args.pilot_roots:,}")
    print("Pilot key                   : (N, L_state) only")
    print("Repeated references         : reuse the same frozen threshold schedule")

    for idx, (N, Lstate) in enumerate(unique_states, start=1):
        seed = make_seed(args.seed, 1000, N, Lstate)
        raw, low, high = build_thresholds(
            N=N,
            L=Lstate,
            w=args.w,
            pilot_roots=args.pilot_roots,
            seed=seed,
            cminus=args.C_minus,
            cplus=args.C_plus,
        )
        threshold_store[(N, Lstate)] = (low, high)

        refs = state_refs[(N, Lstate)]
        refs_text = ";".join(f"L{lc}:{side}" for lc, side in refs)
        finite = np.isfinite(raw)
        finite_fraction = float(np.mean(finite[1:])) if raw.size > 1 else 1.0
        monotone_violations = int(np.sum(np.diff(raw[1:][np.isfinite(raw[1:])]) < 0.0)) if np.count_nonzero(finite[1:]) > 1 else 0
        threshold_gap = float(np.median(high[1:] - low[1:])) if raw.size > 1 else np.nan

        summary_rows.append({
            "N": N,
            "L_state": Lstate,
            "pilot_roots": args.pilot_roots,
            "seed": seed,
            "n_references": len(refs),
            "references": refs_text,
            "finite_fraction": finite_fraction,
            "pilot_monotone_violations": monotone_violations,
            "median_threshold_gap": threshold_gap,
            "expected_threshold_gap": math.log(args.C_plus / args.C_minus),
            "w": args.w,
            "C_minus": args.C_minus,
            "C_plus": args.C_plus,
        })

        for depth in range(1, N + 1):
            threshold_rows.append({
                "N": N,
                "L_state": Lstate,
                "depth": depth,
                "pilot_logZ_ref": float(raw[depth]),
                "log_low": float(low[depth]),
                "log_high": float(high[depth]),
                "pilot_roots": args.pilot_roots,
                "w": args.w,
                "C_minus": args.C_minus,
                "C_plus": args.C_plus,
                "seed": seed,
                "n_references": len(refs),
                "references": refs_text,
            })

        print(
            f"[{idx:2d}/{len(unique_states):2d}] N={N:3d} L={Lstate:2d} "
            f"refs={len(refs)} finite={finite_fraction:.3f} "
            f"median_gap={threshold_gap:.6f}"
        )

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["N", "L_state", "depth"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(["N", "L_state"])

    threshold_path = outdir / "Fig7ii_DJ_NScaling_PilotThresholds.csv"
    summary_path = outdir / "Fig7ii_DJ_NScaling_PilotStateSummary.csv"
    threshold_df.to_csv(threshold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved frozen threshold table: {threshold_path.resolve()}")
    print(f"Saved pilot state summary   : {summary_path.resolve()}")
    return threshold_store


def load_frozen_pilot_thresholds(
    path: Path,
    args,
    rg: Dict[int, dict],
    point_plans: Sequence[Tuple[int, dict]],
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen pilot-threshold file not found: {path}")
    df = pd.read_csv(path)
    required = {
        "N", "L_state", "depth", "pilot_logZ_ref", "log_low", "log_high",
        "pilot_roots", "w", "C_minus", "C_plus",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pilot-threshold CSV missing columns: {sorted(missing)}")

    df = df.copy()
    for col in ["N", "L_state", "depth", "pilot_roots"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["pilot_logZ_ref", "log_low", "log_high", "w", "C_minus", "C_plus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    expected_states = set()
    for N, pdef in point_plans:
        Lc = int(pdef["L"])
        expected_states.add((int(N), int(Lc - args.delta_L)))
        expected_states.add((int(N), int(Lc + args.delta_L)))

    actual_states = set((int(n), int(l)) for n, l in zip(df["N"], df["L_state"]) if np.isfinite(n) and np.isfinite(l))
    if actual_states != expected_states:
        missing_states = sorted(expected_states - actual_states)
        extra_states = sorted(actual_states - expected_states)
        raise ValueError(
            "Frozen pilot states do not exactly match the current physical plan. "
            f"Missing={missing_states}; extra={extra_states}"
        )

    threshold_store = {}
    for N, Lstate in sorted(expected_states):
        g = df[(df["N"] == N) & (df["L_state"] == Lstate)].copy()
        g = g.sort_values("depth")
        expected_depths = np.arange(1, N + 1, dtype=float)
        if len(g) != N or not np.array_equal(g["depth"].to_numpy(float), expected_depths):
            raise ValueError(f"Pilot threshold depths invalid for N={N}, L={Lstate}.")
        if not np.allclose(g["w"].to_numpy(float), args.w, atol=1e-12, rtol=0):
            raise ValueError(f"Pilot w mismatch for N={N}, L={Lstate}.")
        if not np.allclose(g["C_minus"].to_numpy(float), args.C_minus, atol=1e-12, rtol=0):
            raise ValueError(f"Pilot C-minus mismatch for N={N}, L={Lstate}.")
        if not np.allclose(g["C_plus"].to_numpy(float), args.C_plus, atol=1e-12, rtol=0):
            raise ValueError(f"Pilot C-plus mismatch for N={N}, L={Lstate}.")
        if not np.allclose(g["pilot_roots"].to_numpy(float), args.pilot_roots, atol=0, rtol=0):
            raise ValueError(
                f"Pilot roots mismatch for N={N}, L={Lstate}: "
                f"file has {sorted(set(g['pilot_roots'].astype(int)))}; expected {args.pilot_roots}."
            )
        low = np.zeros(N + 1, dtype=float)
        high = np.zeros(N + 1, dtype=float)
        low[1:] = g["log_low"].to_numpy(float)
        high[1:] = g["log_high"].to_numpy(float)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError(f"Non-finite frozen thresholds for N={N}, L={Lstate}.")
        if np.any(high[1:] <= low[1:]):
            raise ValueError(f"Invalid threshold ordering for N={N}, L={Lstate}.")
        threshold_store[(N, Lstate)] = (low, high)

    print("\nFROZEN PILOT THRESHOLDS LOADED")
    print("=" * 96)
    print(f"Source file        : {path.resolve()}")
    print(f"Unique states      : {len(threshold_store)}")
    print(f"Pilot roots/state  : {args.pilot_roots}")
    print("Thresholds         : frozen; no production adaptation")
    return threshold_store


def run_production(
    args,
    rg: Dict[int, dict],
    point_plans: Sequence[Tuple[int, dict]],
    threshold_store: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
):
    tasks = []
    for N, pdef in point_plans:
        Lc = int(pdef["L"])
        Rg = rg[N]["Rg"]
        Rg_err = rg[N]["Rg_err"]
        for side_code, side, Lstate in [
            (0, "minus", Lc - args.delta_L),
            (1, "plus", Lc + args.delta_L),
        ]:
            low, high = threshold_store[(N, Lstate)]
            for block_id in range(args.blocks):
                seed = make_seed(args.seed, 2000, N, Lc, side_code, block_id)
                tasks.append((
                    N, Lc, Lstate, side, block_id,
                    Rg, Rg_err, args.w,
                    args.roots_per_block, args.max_population,
                    args.prune_probability, args.max_clones,
                    low, high, seed,
                ))

    print("\nPRODUCTION PLAN")
    print("=" * 88)
    print(f"Central points       : {len(point_plans)}")
    print(f"Derivative states    : {2 * len(point_plans)}")
    print(f"Blocks/derivative    : {args.blocks}")
    print(f"PERM roots/block     : {args.roots_per_block}")
    print(f"Total PERM roots     : {len(tasks) * args.roots_per_block:,}")
    print(f"Workers              : {args.workers}")
    print("Overflow policy      : HARD INVALIDATION")
    print("Sign selection       : DISABLED")

    rows = []
    if args.workers == 1:
        for i, task in enumerate(tasks, start=1):
            row = run_block_task(task)
            rows.append(row)
            print(
                f"[{i:4d}/{len(tasks):4d}] N={row['N']:3d} "
                f"L={row['L_center']:2d} {row['side']:5s} "
                f"b={row['block_id']:2d} logZ={row['logZ']:.6f} "
                f"ESS={row['ESS']:.1f} qmax={row['max_weight_fraction']:.4f} "
                f"overflow={row['overflow_events']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_block_task, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                rows.append(row)
                print(
                    f"[{i:4d}/{len(tasks):4d}] N={row['N']:3d} "
                    f"L={row['L_center']:2d} {row['side']:5s} "
                    f"b={row['block_id']:2d} logZ={row['logZ']:.6f} "
                    f"ESS={row['ESS']:.1f} qmax={row['max_weight_fraction']:.4f} "
                    f"overflow={row['overflow_events']}"
                )

    block_df = pd.DataFrame(rows).sort_values(
        ["N", "L_center", "side", "block_id"]
    ).reset_index(drop=True)
    return block_df


def assemble_all_points(args, rg, point_plans, block_df):
    point_rows = []
    paired_rows = []
    convergence_rows = []

    for N, pdef in point_plans:
        Lc = int(pdef["L"])
        sub = block_df[
            (block_df["N"] == N) &
            (block_df["L_center"] == Lc)
        ].copy()
        point, paired = assemble_point(
            N=N,
            target_ratio=float(pdef["target_L_over_Rg"]),
            L_center=Lc,
            delta_L=args.delta_L,
            rg=rg[N],
            block_df=sub,
            bootstrap_reps=args.bootstrap_reps,
            seed=make_seed(args.seed, 3000, N, Lc),
        )
        point_rows.append(point)
        paired_rows.append(paired)

        conv = cumulative_convergence(paired, point)
        convergence_rows.append(conv)

        print(
            f"POINT N={N:3d} L={Lc:2d} "
            f"x={point['realized_L_over_Rg']:.5f} "
            f"fRg={point['fRg']:.6g} +/- {point['fRg_err']:.3g} "
            f"minESS={point['min_ESS']:.1f} "
            f"qmax={point['max_weight_fraction']:.4f} "
            f"QC={point['quality_flag']}"
        )

    points = pd.DataFrame(point_rows).sort_values(["N", "L"]).reset_index(drop=True)
    pair_df = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    conv_df = pd.concat(convergence_rows, ignore_index=True) if convergence_rows else pd.DataFrame()
    conv_df = add_relative_convergence_metrics(conv_df)
    return points, pair_df, conv_df


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    if abs(args.w - W_TARGET) > 1e-12:
        raise ValueError(
            "Fig. 7(ii) is defined at w=0.1. Use --w 0.1."
        )
    if args.delta_L <= 0 or args.delta_L % 2 != 0:
        raise ValueError("delta-L must be a positive even integer; default is 2.")
    if args.blocks < 4:
        raise ValueError("Use at least 4 blocks; production default is 20.")
    if args.roots_per_block < 2:
        raise ValueError("roots-per-block must be >= 2.")
    if args.pilot_roots < 100:
        raise ValueError("pilot-roots must be >= 100.")
    if args.max_population < args.roots_per_block:
        raise ValueError("max-population must be >= roots-per-block.")
    if not (0.0 < args.prune_probability < 1.0):
        raise ValueError("prune-probability must lie in (0,1).")
    if args.C_minus <= 0.0 or args.C_plus <= 1.0:
        raise ValueError("Require C-minus>0 and C-plus>1.")
    if args.max_clones < 2:
        raise ValueError("max-clones must be >=2.")
    if args.fit_xmin <= 0 or args.fit_xmax <= args.fit_xmin:
        raise ValueError("Require 0 < fit-xmin < fit-xmax.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    N_list = [int(n) for n in args.N_list]
    if sorted(N_list) != N_list or len(set(N_list)) != len(N_list):
        raise ValueError("N-list must be strictly increasing and unique.")

    # Smoke mode changes only run size, never interpretation of production defaults.
    if args.smoke:
        print("SMOKE MODE: output is for code validation only, not manuscript use.")
        args.blocks = min(args.blocks, 4)
        args.roots_per_block = min(args.roots_per_block, 16)
        args.pilot_roots = min(args.pilot_roots, 200)
        args.max_population = min(args.max_population, 512)
        args.workers = min(args.workers, 2)
        N_list = N_list[:2]
        args.target_ratios = args.target_ratios[:5]
        args.bootstrap_reps = min(args.bootstrap_reps, 500)

    print("=" * 96)
    print("FIGURE 7(ii) — DOMB–JOYCE N-SCALING CONFINEMENT FORCE (FINAL)")
    print("=" * 96)
    print(f"Script              : {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"Python              : {sys.version.split()[0]}")
    print(f"Numba               : {numba.__version__}")
    print(f"N list              : {N_list}")
    print(f"w                   : {args.w}")
    print(f"Target L/Rg         : {args.target_ratios}")
    print(f"Delta L             : {args.delta_L}")
    print(f"Blocks              : {args.blocks}")
    print(f"Roots/block         : {args.roots_per_block}")
    print(f"Pilot roots/state   : {args.pilot_roots}")
    print(f"Population cap      : {args.max_population}")
    print(f"C-/C+               : {args.C_minus}/{args.C_plus}")
    print(f"Prune probability    : {args.prune_probability}")
    print(f"Max clones          : {args.max_clones}")
    print(f"Workers             : {args.workers}")
    print(f"Seed                : {args.seed}")
    print(f"Output              : {outdir.resolve()}")

    run_selftest()

    rg = load_master_rg(args.rg_master, N_list)
    print("\nCANONICAL D-J R_g INPUTS")
    for N in N_list:
        print(
            f"N={N:4d}: Rg={rg[N]['Rg']:.8f} +/- {rg[N]['Rg_err']:.8f} "
            f"[{rg[N]['quality_flag']}]"
        )

    point_plans, plan_df = print_plan(
        rg=rg,
        N_list=N_list,
        target_ratios=args.target_ratios,
        delta_L=args.delta_L,
        outdir=outdir,
    )

    if args.plan_only:
        print("\nPLAN-ONLY requested; no pilot or production sampling performed.")
        return

    if args.pilot_only and args.pilot_thresholds is not None:
        raise ValueError("Do not combine --pilot-only with --pilot-thresholds. Run pilot-only to create the file.")

    if args.pilot_only:
        run_pilots(
            args=args,
            rg=rg,
            point_plans=point_plans,
            outdir=outdir,
        )
        print("\nPILOT-ONLY requested; production sampling not performed.")
        return

    if args.pilot_thresholds is not None:
        threshold_store = load_frozen_pilot_thresholds(
            path=args.pilot_thresholds,
            args=args,
            rg=rg,
            point_plans=point_plans,
        )
        pilot_source = str(args.pilot_thresholds.resolve())
    else:
        print("\nNo --pilot-thresholds supplied; generating a fresh unique-state pilot before production.")
        threshold_store = run_pilots(
            args=args,
            rg=rg,
            point_plans=point_plans,
            outdir=outdir,
        )
        pilot_source = str((outdir / "Fig7ii_DJ_NScaling_PilotThresholds.csv").resolve())

    block_df = run_production(
        args=args,
        rg=rg,
        point_plans=point_plans,
        threshold_store=threshold_store,
    )

    block_path = outdir / "Fig7ii_DJ_NScaling_BlockLevel.csv"
    block_df.to_csv(block_path, index=False)

    points, pair_df, convergence_df = assemble_all_points(
        args=args,
        rg=rg,
        point_plans=point_plans,
        block_df=block_df,
    )

    # Expand paired block file with all essential audit metadata.
    pair_path = outdir / "Fig7ii_DJ_NScaling_PairedBlockLevel.csv"
    pair_df.to_csv(pair_path, index=False)
    point_path = outdir / "Fig7ii_DJ_NScaling_PointLevel.csv"
    points.to_csv(point_path, index=False)

    conv_path = outdir / "Fig7ii_DJ_NScaling_Convergence.csv"
    convergence_df.to_csv(conv_path, index=False)

    # Fit diagnostics: pooled + each N for several windows.
    fit_rows = []
    for xmin, xmax in FIT_WINDOWS:
        pooled = weighted_loglog_fit(points, xmin, xmax)
        pooled["scope"] = "pooled"
        fit_rows.append(pooled)
        fit_rows.extend(fit_by_N(points, xmin, xmax))

    main_fit = weighted_loglog_fit(points, args.fit_xmin, args.fit_xmax)
    fit_df = pd.DataFrame(fit_rows)
    fit_path = outdir / "Fig7ii_DJ_NScaling_FitDiagnostics.csv"
    fit_df.to_csv(fit_path, index=False)

    # Collapse residual audit.
    residual_df = collapse_residuals(points)
    residual_path = outdir / "Fig7ii_DJ_NScaling_CollapseResiduals.csv"
    residual_df.to_csv(residual_path, index=False)
    collapse_summary = summarize_collapse(residual_df)

    # Plot.
    png_path = outdir / "Fig7ii_DJ_NScaling.png"
    pdf_path = outdir / "Fig7ii_DJ_NScaling.pdf"
    plot_figure(points, main_fit, png_path, pdf_path, args.show)

    # Summary QA.
    total_overflow = int(block_df["overflow_events"].sum())
    invalid_points = int((points["quality_flag"] != "OK").sum())
    nonpositive_points = int(np.sum(~(points["fRg"] > 0.0)))

    # Convergence summary at 12 and 16 blocks when available.
    conv_summary = {}
    for k in (12, 16):
        rows = convergence_df[convergence_df["blocks_used"] == k]
        if not rows.empty:
            rel_f = np.abs(rows["relative_change_force_vs_final"].to_numpy(float))
            rel_df = np.abs(rows["relative_change_deltaF_vs_final"].to_numpy(float))
            rel_f = rel_f[np.isfinite(rel_f)]
            rel_df = rel_df[np.isfinite(rel_df)]
            conv_summary[str(k)] = {
                "median_abs_relative_force_change_to_final": float(np.median(rel_f)) if rel_f.size else np.nan,
                "max_abs_relative_force_change_to_final": float(np.max(rel_f)) if rel_f.size else np.nan,
                "median_abs_relative_deltaF_change_to_final": float(np.median(rel_df)) if rel_df.size else np.nan,
                "max_abs_relative_deltaF_change_to_final": float(np.max(rel_df)) if rel_df.size else np.nan,
            }

    provenance = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulation_performed": True,
        "smoke_test": bool(args.smoke),
        "input_rg_master": str(args.rg_master.resolve()),
        "input_rg_master_sha256": sha256_file(args.rg_master),
        "pilot_threshold_source": pilot_source,
        "pilot_threshold_source_sha256": sha256_file(Path(pilot_source)),
        "canonical_Rg_definition": "sqrt(weighted mean of per-chain Rg^2), block-level Rg SEM/CI from Rg_MASTER_FINAL.py",
        "N_list": N_list,
        "w": args.w,
        "target_ratios": args.target_ratios,
        "delta_L": args.delta_L,
        "geometry": {
            "lattice": "3D simple cubic",
            "walls": "z=0 and z=L",
            "allowed_layers": "z=1,...,L-1",
            "tether": "z=L/2",
            "even_L_only": True,
            "symmetric_wall_displacement": True,
        },
        "ensemble": {
            "model": "Domb–Joyce",
            "w": args.w,
            "ensemble": "center-tethered survival/Rosenbluth ensemble in an absorbing slit",
            "overlap_weight": "exp(-w U)",
        },
        "force_definition": {
            "mechanical_definition": "f=d ln Z/dL=-dF/dL",
            "finite_difference": "[ln Z(L+delta_L)-ln Z(L-delta_L)]/(2 delta_L)",
            "delta_L": args.delta_L,
            "signed_block_force_retained": True,
            "sign_selection": False,
            "force_quality_does_not_depend_on_sign": True,
        },
        "perm": {
            "genuine_perm": True,
            "pilot": "independent unpruned Domb–Joyce Rosenbluth pilot per unique physical (N,L_state) derivative state",
            "unique_pilot_states": int(len(threshold_store)),
            "duplicate_derivative_references_reuse_same_thresholds": True,
            "thresholds_frozen_before_production": True,
            "pilot_roots": args.pilot_roots,
            "roots_per_block": args.roots_per_block,
            "blocks": args.blocks,
            "population_cap": args.max_population,
            "C_minus": args.C_minus,
            "C_plus": args.C_plus,
            "prune_probability": args.prune_probability,
            "max_clones": args.max_clones,
            "overflow_policy": "hard invalidation; no silent branch deletion",
            "total_overflow_events": total_overflow,
        },
        "uncertainty": {
            "published_force_error": "SEM across paired signed block-force estimates",
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_CI": "percentile 95% CI of block-force mean",
            "logZ_error": "SEM across independent blocks",
            "fRg_error": "quadrature of force-sampling term Rg*SEM(f) and independent master-Rg term |f|*Rg_err",
        },
        "convergence": {
            "cumulative_block_counts": list(CONVERGENCE_K),
            "summary": conv_summary,
        },
        "scaling_analysis": {
            "primary_fit_window": [args.fit_xmin, args.fit_xmax],
            "fit_method": "weighted least squares in ln(fRg) vs ln(L/Rg); reported exponent uncertainty inflated by sqrt(reduced chi^2) when reduced chi^2>1",
            "gaussian_reference_exponent": GAUSSIAN_FORCE_EXPONENT,
            "SAW_reference_exponent": SAW_FORCE_EXPONENT,
            "interpretation": "finite-range effective exponent and approximate collapse test; no exact universal-master-curve claim",
            "main_fit": main_fit,
            "fit_windows_audit": [list(w) for w in FIT_WINDOWS],
            "collapse_summary": collapse_summary,
        },
        "QA_summary": {
            "central_points": int(len(points)),
            "invalid_points": invalid_points,
            "nonpositive_force_points": nonpositive_points,
            "total_overflow_events": total_overflow,
            "all_points_OK": bool(invalid_points == 0),
            "all_overflow_zero": bool(total_overflow == 0),
            "sign_selected": False,
            "excluded": False,
        },
        "files": {
            "point_level": point_path.name,
            "block_level": block_path.name,
            "paired_block_level": pair_path.name,
            "pilot_thresholds": Path(pilot_source).name,
            "pilot_state_summary": (
                (Path(pilot_source).parent / "Fig7ii_DJ_NScaling_PilotStateSummary.csv").name
                if (Path(pilot_source).parent / "Fig7ii_DJ_NScaling_PilotStateSummary.csv").exists()
                else None
            ),
            "l_plan": (outdir / "Fig7ii_DJ_NScaling_LPlan.csv").name,
            "convergence": conv_path.name,
            "fit_diagnostics": fit_path.name,
            "collapse_residuals": residual_path.name,
            "figure_png": png_path.name,
            "figure_pdf": pdf_path.name,
        },
        "runtime_seconds": time.perf_counter() - start_time,
        "python": sys.version,
        "platform": platform.platform(),
        "numba": numba.__version__,
    }

    provenance_path = outdir / "Fig7ii_DJ_NScaling_Provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("FIGURE 7(ii) PRODUCTION COMPLETE")
    print("=" * 96)
    print(f"L-plan CSV         : {(outdir / 'Fig7ii_DJ_NScaling_LPlan.csv').resolve()}")
    print(f"Point CSV          : {point_path.resolve()}")
    print(f"Block CSV          : {block_path.resolve()}")
    print(f"Paired block CSV   : {pair_path.resolve()}")
    print(f"Convergence CSV    : {conv_path.resolve()}")
    print(f"Fit diagnostics    : {fit_path.resolve()}")
    print(f"Collapse residuals : {residual_path.resolve()}")
    print(f"Provenance JSON    : {provenance_path.resolve()}")
    print(f"PNG                : {png_path.resolve()}")
    print(f"PDF                : {pdf_path.resolve()}")
    print(f"Overflow events    : {total_overflow}")
    print(f"Invalid points     : {invalid_points}")
    print(f"Non-positive fRg   : {nonpositive_points}")
    if np.isfinite(main_fit.get("exponent", np.nan)):
        print(
            f"Primary pooled p   : {main_fit['exponent']:.6f} +/- "
            f"{main_fit['exponent_err']:.6f}"
        )
        print(f"Reduced chi^2      : {main_fit['reduced_chi2']:.4f}")
    print(
        f"Collapse RMS(log)  : {collapse_summary['rms_log_residual']:.5f}"
    )
    print("=" * 96)

    if not args.smoke and total_overflow != 0:
        raise RuntimeError(
            "Production completed with overflow events. Do not use the output "
            "for manuscript analysis; increase the population cap and rerun."
        )
    if not args.smoke and invalid_points != 0:
        raise RuntimeError(
            "One or more production points failed the quality gate. Do not "
            "freeze the figure until the failed points are resolved."
        )


if __name__ == "__main__":
    main()
