#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 — SAW and Gaussian chain-length scaling, reviewer-revised production
=============================================================================

Purpose
-------
This production workflow is designed to answer Reviewer #1(c) directly:

1. SAW chains with N = 100, 150, 200, 250 are shown both in physical and
   R_g^(0)-scaled variables.
2. The corresponding Gaussian chain is treated analytically for the same
   chain lengths, with R_g^(0)=a*sqrt(N/6).
3. Both confinement force and confinement free energy are reported in raw and
   R_g^(0)-scaled forms for SAW and Gaussian chains.
4. The SAW force exponent is treated only as a finite-range effective exponent.
5. All signed block-force estimates are retained. Log plots display only
   positive point estimates because negative values cannot be plotted on log
   axes; the CSV files retain every signed estimate.

Current numerical workflow
--------------------------
- Strict 3D simple-cubic SAW.
- One-coordinate slit confinement, walls z=0 and z=L.
- Center tether: z0=L/2.
- Even integer L only.
- Absorbing/survival condition: allowed sites z=1,...,L-1.
- Genuine PERM with frozen independent unpruned pilots.
- Production defaults match the current revised workflow:
      40 blocks x 384 roots/block
      pilot roots = 20,000 per state
      C_minus = 0.5, C_plus = 2.0
      prune probability = 0.5
      maximum clones = 4
      population cap = 32,768
- Force:
      f(L) = [ln Z(L+delta_L)-ln Z(L-delta_L)]/(2*delta_L)
  with delta_L=2 by default.
- Confinement free energy:
      Delta F(L) = -ln[Z(L)/Z(infinity)]
  for the SAW PERM ensemble.
- Gaussian free energy and force are evaluated from the exact
  center-tethered absorbing-boundary eigenmode series.
- No sign filtering, SNR-based point deletion, sigma clipping, or post-hoc
  outlier deletion is applied.
- Convergence audit reports cumulative log Z, Delta F, and signed force for
  a representative difficult state (worst endpoint ESS among final force
  points).

Important interpretation
------------------------
The output intentionally uses language such as "chain-length scaling" and
"approximate collapse" rather than claiming an unrestricted universal master
curve. This figure is a direct test of whether R_g^(0) removes the dominant
chain-length dependence within the specified model and geometry.

Outputs
-------
Fig3_FORCE_SCALING_REV2.png/.pdf
    2x2 force figure: SAW raw/scaled, Gaussian raw/scaled.

Fig3_FREE_ENERGY_SCALING_REV2.png/.pdf
    2x2 free-energy figure: SAW raw/scaled, Gaussian raw/scaled.

Fig3_SAW_PointLevel.csv
Fig3_SAW_BlockLevel.csv
Fig3_SAW_PilotThresholds.csv
Fig3_SAW_Convergence.csv
Fig3_SAW_ForceFit.csv
Fig3_Gaussian_PointLevel.csv
Fig3_Provenance.json
Fig3_RunSummary.txt

Kaggle production example
-------------------------
!python Fig3_SAW_Gaussian_SCALING_REV2_FINAL.py \\
  --rg-master /kaggle/input/datasets/soumyajyotikabi/mergedrg-table/Rg_MASTER_FINAL_MERGED.csv \\
  --outdir /kaggle/working/Fig3_REV2_FINAL \\
  --N-list 100 150 200 250 \\
  --target-ratios 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 \\
  --delta-L 2 \\
  --blocks 40 \\
  --block-counts 8 16 24 32 40 \\
  --roots-per-block 384 \\
  --unconfined-roots-per-block 384 \\
  --pilot-roots 20000 \\
  --max-population 32768 \\
  --C-minus 0.5 \\
  --C-plus 2.0 \\
  --prune-probability 0.5 \\
  --max-clones 4 \\
  --bootstrap-reps 5000 \\
  --fit-max-x 1.9 \\
  --seed 20260930

Self-test
---------
!python Fig3_SAW_Gaussian_SCALING_REV2_FINAL.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Numba is required for production SAW/PERM runs.") from exc


# ============================================================================
# Constants
# ============================================================================

A = 1.0
KBT = 1.0
NU = 0.587597
EXPECTED_SAW_EXPONENT = -(1.0 / NU + 1.0)

DEFAULT_N_LIST = [100, 150, 200, 250]
DEFAULT_TARGET_RATIOS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
DEFAULT_DELTA_L = 2
DEFAULT_BLOCKS = 40
DEFAULT_BLOCK_COUNTS = [8, 16, 24, 32, 40]
DEFAULT_ROOTS_PER_BLOCK = 384
DEFAULT_UNCONFINED_ROOTS_PER_BLOCK = 384
DEFAULT_PILOT_ROOTS = 20000
DEFAULT_MAX_POPULATION = 32768
DEFAULT_CMINUS = 0.5
DEFAULT_CPLUS = 2.0
DEFAULT_PRUNE_P = 0.5
DEFAULT_MAX_CLONES = 4
DEFAULT_BOOTSTRAP = 5000
DEFAULT_FIT_MAX_X = 1.9
DEFAULT_SEED = 20260930

ESS_CAUTION = 20.0
MAX_WEIGHT_FRACTION_CAUTION = 0.10

STEPS = np.asarray(
    [[1, 0, 0], [-1, 0, 0],
     [0, 1, 0], [0, -1, 0],
     [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


# ============================================================================
# Utility / reproducibility helpers
# ============================================================================

def make_seed(base_seed: int, *tokens: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), *[int(t) for t in tokens]])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def safe_sem(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def bootstrap_mean_ci(values: Sequence[float], reps: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or reps <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ============================================================================
# Coordinate hash table for SAW growth
# ============================================================================

@njit(cache=True)
def coord_key(x: int, y: int, z: int) -> np.uint64:
    bias = np.int64(1_000_000)
    mask = np.uint64((1 << 21) - 1)
    ux = np.uint64(np.int64(x) + bias) & mask
    uy = np.uint64(np.int64(y) + bias) & mask
    uz = np.uint64(np.int64(z) + bias) & mask
    return (ux << np.uint64(42)) | (uy << np.uint64(21)) | uz


@njit(cache=True)
def hidx(key: np.uint64, mask: int) -> int:
    return np.int64((key * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(mask))


@njit(cache=True)
def occupied(keys, used, table_size: int, x: int, y: int, z: int) -> int:
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
    for _ in range(2048):
        if used[idx] == 0:
            return 0
        if keys[idx] == key:
            return 1
        idx += 1
        if idx >= table_size:
            idx = 0
    return -1


@njit(cache=True)
def insert_site(keys, used, table_size: int, x: int, y: int, z: int) -> int:
    key = coord_key(x, y, z)
    idx = np.int64(hidx(key, table_size - 1))
    for _ in range(2048):
        if used[idx] == 0:
            used[idx] = 1
            keys[idx] = key
            return 1
        if keys[idx] == key:
            return 1
        idx += 1
        if idx >= table_size:
            idx = 0
    return 0


@njit(cache=True)
def logsumexp_array(values, n: int) -> float:
    m = -np.inf
    for i in range(n):
        v = values[i]
        if np.isfinite(v) and v > m:
            m = v
    if not np.isfinite(m):
        return -np.inf
    s = 0.0
    for i in range(n):
        v = values[i]
        if np.isfinite(v):
            s += math.exp(v - m)
    return m + math.log(s) if s > 0 else -np.inf


# ============================================================================
# Canonical R_g loading
# ============================================================================

def normalize_model(v: object) -> str:
    return str(v).strip().lower().replace("–", "-").replace(" ", "")


def load_rg_master(path: Path, N_list: Sequence[int]) -> Dict[int, dict]:
    df = pd.read_csv(path)
    required = {"model", "N", "Rg", "Rg_err"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"R_g master missing columns: {sorted(missing)}")
    df = df.copy()
    df["_model"] = df["model"].map(normalize_model)
    saw = df[df["_model"] == "saw"].copy()
    out: Dict[int, dict] = {}
    for N in N_list:
        rows = saw[saw["N"].astype(int) == int(N)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one canonical SAW row for N={N}; found {len(rows)}."
            )
        r = rows.iloc[0]
        rg = float(r["Rg"])
        rg_err = float(r["Rg_err"])
        if not np.isfinite(rg) or rg <= 0:
            raise ValueError(f"Invalid Rg for N={N}: {rg}")
        out[int(N)] = {
            "N": int(N),
            "Rg": rg,
            "Rg_err": rg_err,
            "quality_flag": str(r.get("quality_flag", "UNKNOWN")),
            "model": str(r["model"]),
        }
    return out


# ============================================================================
# Unique even-L planning
# ============================================================================

def nearest_even(x: float) -> int:
    n = int(math.floor(x / 2.0 + 0.5))
    return max(4, 2 * n)


def make_center_plan(Rg: float, target_ratios: Sequence[float], delta_L: int) -> List[dict]:
    by_L: Dict[int, dict] = {}
    for target in target_ratios:
        L = nearest_even(float(target) * Rg)
        if L - delta_L < 2:
            continue
        realized = L / Rg
        cand = {
            "target_L_over_Rg": float(target),
            "L": int(L),
            "realized_L_over_Rg": float(realized),
        }
        if L not in by_L:
            by_L[L] = cand
        else:
            old = by_L[L]
            if abs(target - realized) < abs(old["target_L_over_Rg"] - realized):
                by_L[L] = cand
    plan = sorted(by_L.values(), key=lambda r: r["L"])
    return plan


# ============================================================================
# Unpruned pilot for both confined and unconfined SAW
# ============================================================================

@njit(cache=True)
def pilot_partial_logz(N: int, L: int, roots: int, seed: int, confined: int) -> np.ndarray:
    """
    Unpruned Rosenbluth pilot.

    If confined=1, z must stay in 1,...,L-1.
    If confined=0, there is no wall restriction.
    """
    np.random.seed(seed)
    table_size = 1
    while table_size < 4 * (N + 64):
        table_size *= 2

    logsum = np.full(N + 1, -np.inf, dtype=np.float64)
    dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
    dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
    dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

    for r in range(roots):
        keys = np.zeros(table_size, dtype=np.uint64)
        used = np.zeros(table_size, dtype=np.uint8)
        x = 0
        y = 0
        z = L // 2 if confined else 0
        if insert_site(keys, used, table_size, x, y, z) == 0:
            continue

        lw = 0.0
        logsum[1] = np.logaddexp(logsum[1], lw)

        alive = 1
        for k in range(1, N):
            if alive == 0:
                break
            dirs = np.full(6, -1, dtype=np.int64)
            m = 0
            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]
                if confined and (zz <= 0 or zz >= L):
                    continue
                occ = occupied(keys, used, table_size, xx, yy, zz)
                if occ < 0:
                    alive = 0
                    break
                if occ == 0:
                    dirs[m] = d
                    m += 1
            if alive == 0 or m == 0:
                alive = 0
                break
            lw += math.log(m)
            pick = min(int(np.random.random() * m), m - 1)
            d = dirs[pick]
            x += dx[d]
            y += dy[d]
            z += dz[d]
            if insert_site(keys, used, table_size, x, y, z) == 0:
                alive = 0
                break
            logsum[k + 1] = np.logaddexp(logsum[k + 1], lw)

    log_roots = math.log(roots)
    for k in range(1, N + 1):
        if np.isfinite(logsum[k]):
            logsum[k] -= log_roots
    return logsum


def complete_pilot_thresholds(logz: np.ndarray) -> np.ndarray:
    out = logz.copy()
    n = len(out)
    valid = np.isfinite(out)
    valid[0] = False
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.arange(n, dtype=float) * math.log(2.5)
    out[~valid] = np.interp(np.flatnonzero(~valid), idx, out[idx])
    if idx.size >= 2:
        take = min(5, idx.size)
        xx = idx[-take:].astype(float)
        yy = out[idx[-take:]]
        slope = np.polyfit(xx, yy, 1)[0]
        last_idx = int(idx[-1])
        for k in range(last_idx + 1, n):
            out[k] = out[last_idx] + slope * (k - last_idx)
    out[0] = 0.0
    return out


def build_thresholds(
    N: int,
    L: int,
    roots: int,
    seed: int,
    cminus: float,
    cplus: float,
    confined: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pilot_partial_logz(N, L, roots, seed, 1 if confined else 0)
    ref = complete_pilot_thresholds(raw)
    low = ref + math.log(cminus)
    high = ref + math.log(cplus)
    return raw, ref, low, high


# ============================================================================
# Genuine PERM block
# ============================================================================

@njit(cache=True)
def perm_block_saw(
    N: int,
    L: int,
    roots: int,
    max_population: int,
    seed: int,
    prune_p: float,
    max_clones: int,
    log_low: np.ndarray,
    log_high: np.ndarray,
    confined: int,
) -> tuple:
    np.random.seed(seed)
    table_size = 1
    while table_size < 4 * (N + 64):
        table_size *= 2

    coords_a = np.zeros((max_population, N, 3), dtype=np.int16)
    coords_b = np.zeros((max_population, N, 3), dtype=np.int16)
    keys_a = np.zeros((max_population, table_size), dtype=np.uint64)
    keys_b = np.zeros((max_population, table_size), dtype=np.uint64)
    used_a = np.zeros((max_population, table_size), dtype=np.uint8)
    used_b = np.zeros((max_population, table_size), dtype=np.uint8)
    logw_a = np.full(max_population, -np.inf, dtype=np.float64)
    logw_b = np.full(max_population, -np.inf, dtype=np.float64)

    dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
    dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
    dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

    if roots > max_population:
        return -np.inf, 0, 0, 0, 0, 0, 1

    z0 = L // 2 if confined else 0
    for r in range(roots):
        coords_a[r, 0, 0] = 0
        coords_a[r, 0, 1] = 0
        coords_a[r, 0, 2] = z0
        logw_a[r] = 0.0
        if insert_site(keys_a[r], used_a[r], table_size, 0, 0, z0) == 0:
            return -np.inf, 0, 0, 0, 0, 0, 1

    pop = roots
    min_pop_seen = pop
    max_pop_seen = pop
    prune_events = 0
    clone_events = 0

    for k in range(1, N):
        new_pop = 0
        for parent in range(pop):
            if not np.isfinite(logw_a[parent]):
                continue
            x = int(coords_a[parent, k - 1, 0])
            y = int(coords_a[parent, k - 1, 1])
            z = int(coords_a[parent, k - 1, 2])

            dirs = np.full(6, -1, dtype=np.int64)
            m = 0
            bad = False
            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]
                if confined and (zz <= 0 or zz >= L):
                    continue
                occ = occupied(keys_a[parent], used_a[parent], table_size, xx, yy, zz)
                if occ < 0:
                    bad = True
                    break
                if occ == 0:
                    dirs[m] = d
                    m += 1
            if bad or m == 0:
                continue

            new_logw = logw_a[parent] + math.log(m)
            if new_logw < log_low[k + 1]:
                if np.random.random() < prune_p:
                    prune_events += 1
                    continue
                new_logw -= math.log(1.0 - prune_p)

            if new_logw > log_high[k + 1]:
                b = int(math.ceil(math.exp(min(20.0, new_logw - log_high[k + 1]))))
                if b < 2:
                    b = 2
                if b > max_clones:
                    b = max_clones
                if b > m:
                    b = m
            else:
                b = 1

            if new_pop + b > max_population:
                return (-np.inf, new_pop, min_pop_seen,
                        max(max_pop_seen, new_pop),
                        prune_events, clone_events, 1)

            child_logw = new_logw - math.log(b)

            available = dirs.copy()
            for j in range(b):
                remaining = m - j
                pick = j + int(np.random.random() * remaining)
                tmp = available[j]
                available[j] = available[pick]
                available[pick] = tmp
                d = available[j]

                nx = x + dx[d]
                ny = y + dy[d]
                nz = z + dz[d]
                child = new_pop
                new_pop += 1

                for q in range(k):
                    coords_b[child, q, 0] = coords_a[parent, q, 0]
                    coords_b[child, q, 1] = coords_a[parent, q, 1]
                    coords_b[child, q, 2] = coords_a[parent, q, 2]
                coords_b[child, k, 0] = nx
                coords_b[child, k, 1] = ny
                coords_b[child, k, 2] = nz

                for q in range(table_size):
                    keys_b[child, q] = keys_a[parent, q]
                    used_b[child, q] = used_a[parent, q]
                if insert_site(keys_b[child], used_b[child], table_size, nx, ny, nz) == 0:
                    return (-np.inf, new_pop, min_pop_seen,
                            max(max_pop_seen, new_pop),
                            prune_events, clone_events, 1)
                logw_b[child] = child_logw
                if b > 1:
                    clone_events += 1

        tmp = coords_a; coords_a = coords_b; coords_b = tmp
        tmp = keys_a; keys_a = keys_b; keys_b = tmp
        tmp = used_a; used_a = used_b; used_b = tmp
        tmp = logw_a; logw_a = logw_b; logw_b = tmp

        for i in range(max_population):
            logw_b[i] = -np.inf

        pop = new_pop
        if pop == 0:
            return (-np.inf, 0, min_pop_seen, max_pop_seen,
                    prune_events, clone_events, 0)
        min_pop_seen = min(min_pop_seen, pop)
        max_pop_seen = max(max_pop_seen, pop)

    logZ_sum = logsumexp_array(logw_a, pop)
    logZ = logZ_sum - math.log(roots)
    return (logZ, pop, min_pop_seen, max_pop_seen,
            prune_events, clone_events, 0)


# ============================================================================
# Production bookkeeping
# ============================================================================

@dataclass
class StateTask:
    N: int
    L: int
    side: str
    block_id: int
    Rg: float
    Rg_err: float
    roots: int
    max_population: int
    prune_p: float
    max_clones: int
    confined: bool
    log_low: np.ndarray
    log_high: np.ndarray
    seed: int


def run_state_block(task: StateTask) -> dict:
    t0 = time.perf_counter()
    out = perm_block_saw(
        task.N,
        task.L,
        task.roots,
        task.max_population,
        task.seed,
        task.prune_p,
        task.max_clones,
        task.log_low,
        task.log_high,
        1 if task.confined else 0,
    )
    return {
        "N": task.N,
        "L": task.L,
        "side": task.side,
        "block_id": task.block_id,
        "Rg": task.Rg,
        "Rg_err": task.Rg_err,
        "realized_L_over_Rg": task.L / task.Rg if task.confined else np.inf,
        "logZ": float(out[0]),
        "roots": task.roots,
        "final_population": int(out[1]),
        "min_population": int(out[2]),
        "max_population_seen": int(out[3]),
        "prune_events": int(out[4]),
        "clone_events": int(out[5]),
        "overflow": int(out[6]),
        "runtime_seconds": time.perf_counter() - t0,
        "confined": bool(task.confined),
    }


def build_state_tasks(
    rg: Dict[int, dict],
    center_plans: Dict[int, List[dict]],
    delta_L: int,
    blocks: int,
    roots: int,
    unconfined_roots: int,
    max_population: int,
    prune_p: float,
    max_clones: int,
    pilot_roots: int,
    cminus: float,
    cplus: float,
    seed: int,
    outdir: Path,
) -> Tuple[List[dict], Dict[Tuple[int, int, bool], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    # All confined widths needed for center deltaF points and force endpoints.
    widths: Dict[int, set] = {}
    for N, plan in center_plans.items():
        widths[N] = set()
        for p in plan:
            L = int(p["L"])
            widths[N].add(L)
            widths[N].add(L - delta_L)
            widths[N].add(L + delta_L)

    threshold_cache = {}
    pilot_rows = []

    for N in sorted(rg):
        # Unconfined reference state L=0 is used only as a bookkeeping label.
        state_key = (N, 0, False)
        raw, ref, low, high = build_thresholds(
            N, 0, pilot_roots, make_seed(seed, 1000, N, 0), cminus, cplus, False
        )
        threshold_cache[state_key] = (raw, ref, low, high)
        pilot_rows.append({
            "N": N, "L": 0, "confined": False,
            "pilot_roots": pilot_roots,
            "raw_terminal_logz": float(raw[-1]),
            "reference_terminal_logz": float(ref[-1]),
            "C_minus": cminus, "C_plus": cplus,
        })

        for L in sorted(widths[N]):
            state_key = (N, L, True)
            raw, ref, low, high = build_thresholds(
                N, L, pilot_roots, make_seed(seed, 2000, N, L), cminus, cplus, True
            )
            threshold_cache[state_key] = (raw, ref, low, high)
            pilot_rows.append({
                "N": N, "L": L, "confined": True,
                "pilot_roots": pilot_roots,
                "raw_terminal_logz": float(raw[-1]),
                "reference_terminal_logz": float(ref[-1]),
                "C_minus": cminus, "C_plus": cplus,
            })

    pd.DataFrame(pilot_rows).to_csv(outdir / "Fig3_SAW_PilotThresholds.csv", index=False)

    tasks: List[StateTask] = []
    for N in sorted(rg):
        rec = rg[N]
        raw, ref, low, high = threshold_cache[(N, 0, False)]
        for b in range(blocks):
            tasks.append(StateTask(
                N=N, L=0, side="unconfined", block_id=b,
                Rg=rec["Rg"], Rg_err=rec["Rg_err"], roots=unconfined_roots,
                max_population=max_population, prune_p=prune_p,
                max_clones=max_clones, confined=False,
                log_low=low, log_high=high,
                seed=make_seed(seed, 3000, N, b),
            ))

        # Unique confined widths.
        for L in sorted(widths[N]):
            raw, ref, low, high = threshold_cache[(N, L, True)]
            for b in range(blocks):
                side = "confined"
                tasks.append(StateTask(
                    N=N, L=L, side=side, block_id=b,
                    Rg=rec["Rg"], Rg_err=rec["Rg_err"], roots=roots,
                    max_population=max_population, prune_p=prune_p,
                    max_clones=max_clones, confined=True,
                    log_low=low, log_high=high,
                    seed=make_seed(seed, 4000, N, L, b),
                ))
    return [asdict(t) for t in tasks], threshold_cache


# ============================================================================
# Point construction
# ============================================================================

def state_logz(df: pd.DataFrame, N: int, L: int, side: str) -> pd.DataFrame:
    return df[(df["N"] == N) & (df["L"] == L) & (df["side"] == side)].sort_values("block_id").copy()


def build_points(
    block_df: pd.DataFrame,
    center_plans: Dict[int, List[dict]],
    rg: Dict[int, dict],
    delta_L: int,
    bootstrap_reps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point_rows = []
    force_block_rows = []

    for N in sorted(center_plans):
        Rg = rg[N]["Rg"]
        for pdef in center_plans[N]:
            L = int(pdef["L"])
            minus = state_logz(block_df, N, L - delta_L, "confined")
            plus = state_logz(block_df, N, L + delta_L, "confined")
            center = state_logz(block_df, N, L, "confined")
            inf = state_logz(block_df, N, 0, "unconfined")

            common = sorted(set(minus.block_id) & set(plus.block_id))
            f_blocks = []
            for b in common:
                bm = minus[minus.block_id == b].iloc[0]
                bp = plus[plus.block_id == b].iloc[0]
                if np.isfinite(bm.logZ) and np.isfinite(bp.logZ):
                    f = (float(bp.logZ) - float(bm.logZ)) / (2.0 * delta_L)
                else:
                    f = np.nan
                f_blocks.append(f)
                force_block_rows.append({
                    "N": N,
                    "L_center": L,
                    "L_minus": L-delta_L,
                    "L_plus": L+delta_L,
                    "block_id": int(b),
                    "logZ_minus": float(bm.logZ),
                    "logZ_plus": float(bp.logZ),
                    "force_block": float(f),
                    "force_Rg_block": float(f*Rg) if np.isfinite(f) else np.nan,
                    "overflow_minus": int(bm.overflow),
                    "overflow_plus": int(bp.overflow),
                })

            f_arr = np.asarray(f_blocks, dtype=float)
            valid_f = np.isfinite(f_arr)
            fv = f_arr[valid_f]
            overflow = int(minus.overflow.sum() + plus.overflow.sum())

            # Free-energy blocks: pair block ids across independent ensembles.
            common_df = sorted(set(center.block_id) & set(inf.block_id))
            dF_blocks = []
            for b in common_df:
                bc = center[center.block_id == b].iloc[0]
                bi = inf[inf.block_id == b].iloc[0]
                if np.isfinite(bc.logZ) and np.isfinite(bi.logZ):
                    dF = float(bi.logZ - bc.logZ)
                else:
                    dF = np.nan
                dF_blocks.append(dF)

            dF_arr = np.asarray(dF_blocks, dtype=float)
            valid_dF = np.isfinite(dF_arr)
            dFv = dF_arr[valid_dF]

            if overflow > 0:
                quality = "INVALID_OVERFLOW"
            elif fv.size < max(8, int(0.75 * len(common))):
                quality = "LOW_VALID_FORCE_BLOCK_FRACTION"
            elif dFv.size < max(8, int(0.75 * len(common_df))):
                quality = "LOW_VALID_DF_BLOCK_FRACTION"
            else:
                quality = "OK"

            mean_f = float(np.mean(fv)) if fv.size else np.nan
            f_sem = safe_sem(fv)
            f_ci_lo, f_ci_hi = bootstrap_mean_ci(
                fv, bootstrap_reps, make_seed(seed, 9000, N, L)
            ) if fv.size >= 2 else (np.nan, np.nan)

            mean_dF = float(np.mean(dFv)) if dFv.size else np.nan
            dF_sem = safe_sem(dFv)
            dF_ci_lo, dF_ci_hi = bootstrap_mean_ci(
                dFv, bootstrap_reps, make_seed(seed, 10000, N, L)
            ) if dFv.size >= 2 else (np.nan, np.nan)

            # State diagnostics: ESS-like measure across block logZ values.
            ess_parts = []
            maxfrac_parts = []
            for s in (minus, plus, center, inf):
                z = s.loc[np.isfinite(s.logZ), "logZ"].to_numpy(float)
                if z.size:
                    rw = np.exp(z - np.max(z))
                    sw = rw.sum()
                    ess = (sw*sw)/np.sum(rw*rw)
                    maxfrac = float(np.max(rw)/sw)
                    ess_parts.append(float(ess))
                    maxfrac_parts.append(maxfrac)
            min_ess_like = float(np.min(ess_parts)) if ess_parts else np.nan
            maxfrac = float(np.max(maxfrac_parts)) if maxfrac_parts else np.nan

            point_rows.append({
                "N": N,
                "Rg": Rg,
                "Rg_err": rg[N]["Rg_err"],
                "Rg_quality": rg[N]["quality_flag"],
                "target_L_over_Rg": pdef["target_L_over_Rg"],
                "L": L,
                "realized_L_over_Rg": L/Rg,
                "L_minus": L-delta_L,
                "L_plus": L+delta_L,
                "delta_L": delta_L,
                "force": mean_f,
                "force_err": f_sem,
                "fRg": mean_f*Rg if np.isfinite(mean_f) else np.nan,
                "fRg_err": f_sem*Rg if np.isfinite(f_sem) else np.nan,
                "fRg_bootstrap95_low": f_ci_lo*Rg if np.isfinite(f_ci_lo) else np.nan,
                "fRg_bootstrap95_high": f_ci_hi*Rg if np.isfinite(f_ci_hi) else np.nan,
                "DeltaF": mean_dF,
                "DeltaF_err": dF_sem,
                "DeltaF_bootstrap95_low": dF_ci_lo,
                "DeltaF_bootstrap95_high": dF_ci_hi,
                "valid_force_blocks": int(fv.size),
                "valid_DeltaF_blocks": int(dFv.size),
                "n_force_blocks": int(len(common)),
                "n_DeltaF_blocks": int(len(common_df)),
                "min_ESS_like": min_ess_like,
                "max_block_weight_fraction_like": maxfrac,
                "overflow_count_for_point": overflow,
                "quality_flag": quality,
            })

    return pd.DataFrame(point_rows), pd.DataFrame(force_block_rows)


# ============================================================================
# Gaussian exact results
# ============================================================================

def gaussian_rg(N: int) -> float:
    return A * math.sqrt(N / 6.0)


def gaussian_logZ_and_force(L: np.ndarray, N: int, nmax: int = 401) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact midpoint-tethered absorbing Gaussian chain.

    With Rg^2 = a^2 N / 6,

      Z(L) = sum_n 4/(n*pi) sin(n*pi/2) exp[-(n*pi*Rg/L)^2]

    and f Rg = Rg d(ln Z)/dL.
    """
    L = np.asarray(L, dtype=float)
    rg = gaussian_rg(N)
    n = np.arange(1, nmax + 1, dtype=float)[:, None]
    LL = L[None, :]
    amp = 4.0/(n*np.pi) * np.sin(n*np.pi/2.0)
    exponent = -((n*np.pi*rg/LL)**2)
    terms = amp * np.exp(exponent)
    Z = np.sum(terms, axis=0)
    if np.any(Z <= 0) or np.any(~np.isfinite(Z)):
        raise RuntimeError("Gaussian exact Z is non-positive/non-finite.")
    # d[-(n*pi*rg/L)^2]/dL = +2(n*pi)^2 rg^2/L^3
    dZ = np.sum(terms * (2.0*(n*np.pi)**2*rg**2/LL**3), axis=0)
    fRg = rg * dZ / Z
    return np.log(Z), fRg


def gaussian_exact_points(target_ratios: Sequence[float], delta_L: int, N_list: Sequence[int]) -> pd.DataFrame:
    """Generate an independent even-L Gaussian plan for the requested scaled x-grid."""
    rows = []
    for N in sorted(N_list):
        rg = gaussian_rg(N)
        plan_by_L = {}
        for target in target_ratios:
            L = nearest_even(float(target) * rg)
            if L - delta_L < 2:
                continue
            realized = L / rg
            cand = {
                "target_L_over_Rg": float(target),
                "L": int(L),
                "realized_L_over_Rg": float(realized),
            }
            if L not in plan_by_L or abs(target-realized) < abs(plan_by_L[L]["target_L_over_Rg"]-realized):
                plan_by_L[L] = cand
        for pdef in sorted(plan_by_L.values(), key=lambda q: q["L"]):
            L = int(pdef["L"])
            logZ_L, fRg = gaussian_logZ_and_force(np.array([L], float), N)
            logZ = float(logZ_L[0])
            # Free reference Z(infinity)=1 for the normalized continuum Gaussian survival probability.
            dF = -logZ
            rows.append({
                "N": N,
                "Rg": rg,
                "target_L_over_Rg": pdef["target_L_over_Rg"],
                "L": L,
                "realized_L_over_Rg": L/rg,
                "DeltaF": dF,
                "fRg": float(fRg[0]),
                "force": float(fRg[0]/rg),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Fits / convergence
# ============================================================================

def weighted_loglog_fit(points: pd.DataFrame, xmax: float) -> dict:
    d = points[
        np.isfinite(points["realized_L_over_Rg"]) &
        np.isfinite(points["fRg"]) & np.isfinite(points["fRg_err"]) &
        (points["fRg"] > 0) & (points["fRg_err"] > 0) &
        (points["realized_L_over_Rg"] <= xmax) &
        (points["quality_flag"].astype(str) == "OK")
    ].copy()
    if len(d) < 3:
        return {"xmax": xmax, "n_points": int(len(d)), "exponent": np.nan,
                "exponent_err": np.nan, "amplitude": np.nan,
                "chi2": np.nan, "dof": 0, "reduced_chi2": np.nan}
    x = np.log(d["realized_L_over_Rg"].to_numpy(float))
    y = np.log(d["fRg"].to_numpy(float))
    sy = d["fRg_err"].to_numpy(float) / d["fRg"].to_numpy(float)
    w = 1.0/(sy*sy)
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ (X.T @ (w*y))
    c, p = beta
    resid = y - X@beta
    chi2 = float(np.sum((resid/sy)**2))
    dof = len(y)-2
    return {
        "xmax": xmax,
        "n_points": int(len(d)),
        "exponent": float(p),
        "exponent_err": float(math.sqrt(cov[1,1])),
        "amplitude": float(math.exp(c)),
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2/dof) if dof>0 else np.nan,
    }


def choose_worst_force_point(points: pd.DataFrame) -> pd.Series:
    d = points[np.isfinite(points["min_ESS_like"])].copy()
    if d.empty:
        return points.iloc[0]
    return d.sort_values("min_ESS_like", ascending=True).iloc[0]


def build_convergence(block_df: pd.DataFrame, point: pd.Series, block_counts: Sequence[int], rg: Dict[int,dict]) -> pd.DataFrame:
    N = int(point["N"])
    L = int(point["L"])
    dm = state_logz(block_df, N, int(point["L_minus"]), "confined").sort_values("block_id")
    dp = state_logz(block_df, N, int(point["L_plus"]), "confined").sort_values("block_id")
    dc = state_logz(block_df, N, L, "confined").sort_values("block_id")
    di = state_logz(block_df, N, 0, "unconfined").sort_values("block_id")
    rows = []
    for k in block_counts:
        k = min(int(k), len(di), len(dm), len(dp), len(dc))
        lm = dm.iloc[:k]["logZ"].to_numpy(float)
        lp = dp.iloc[:k]["logZ"].to_numpy(float)
        lc = dc.iloc[:k]["logZ"].to_numpy(float)
        li = di.iloc[:k]["logZ"].to_numpy(float)
        logZm = float(np.mean(lm[np.isfinite(lm)]))
        logZp = float(np.mean(lp[np.isfinite(lp)]))
        logZc = float(np.mean(lc[np.isfinite(lc)]))
        logZi = float(np.mean(li[np.isfinite(li)]))
        force = (logZp-logZm)/(2.0*point["delta_L"])
        dF = logZi-logZc
        rows.append({
            "N": N,
            "L_center": L,
            "L_over_Rg": L/rg[N]["Rg"],
            "blocks_used": k,
            "mean_logZ_minus": logZm,
            "mean_logZ_plus": logZp,
            "mean_logZ_center": logZc,
            "mean_logZ_infinity": logZi,
            "DeltaF": dF,
            "force": force,
            "fRg": force*rg[N]["Rg"],
        })
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================

COLORS = {
    100: "#4C78A8",
    150: "#59A14F",
    200: "#F28E2B",
    250: "#B07AA1",
}


def set_pub_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "legend.fontsize": 10.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "axes.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_force_figure(saw: pd.DataFrame, gauss: pd.DataFrame, fit: dict, outdir: Path):
    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # SAW raw
    ax = axes[0,0]
    for N in sorted(saw.N.unique()):
        d = saw[saw.N==N].sort_values("L")
        pos = d.force > 0
        ax.errorbar(d.loc[pos,"L"], d.loc[pos,"force"], yerr=d.loc[pos,"force_err"],
                    fmt="o-", ms=4.8, lw=1.4, capsize=2.5, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 20, 30])
    ax.set_xticklabels(["10", "20", "30"])
    ax.set_xlabel("Physical slit width $L$")
    ax.set_ylabel(r"Entropic force $f\;(k_BT/a)$")
    ax.set_title("(a) SAW: unscaled force")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    # SAW scaled
    ax = axes[0,1]
    for N in sorted(saw.N.unique()):
        d = saw[saw.N==N].sort_values("realized_L_over_Rg")
        pos = d.fRg > 0
        ax.errorbar(d.loc[pos,"realized_L_over_Rg"], d.loc[pos,"fRg"], yerr=d.loc[pos,"fRg_err"],
                    fmt="o", ms=5, capsize=2.5, color=COLORS.get(int(N)), label=f"N={N}")
    if np.isfinite(fit.get("exponent", np.nan)):
        xx = np.logspace(np.log10(1.15), np.log10(float(fit["xmax"])), 250)
        yy = fit["amplitude"] * xx**fit["exponent"]
        ax.loglog(xx, yy, "--", color="black", lw=2.0,
                  label=rf"finite-range fit: $p={fit['exponent']:.3f}\pm{fit['exponent_err']:.3f}$")
    xa = np.array([1.2, 1.9])
    if np.isfinite(fit.get("amplitude", np.nan)):
        x0 = 1.5
        y0 = fit["amplitude"]*x0**fit["exponent"]
        ya = y0*(xa/x0)**EXPECTED_SAW_EXPONENT
        ax.loglog(xa, ya, ":", color="gray", lw=2.0,
                  label=rf"SAW reference slope $p={EXPECTED_SAW_EXPONENT:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    ax.set_xticklabels(["1.2","1.4","1.6","1.8","2.0","2.2","2.4","2.6","2.8","3.0"])
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$fR_g^{(0)}\;(k_BT)$")
    ax.set_title("(b) SAW: scaled force")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    # Gaussian raw
    ax = axes[1,0]
    for N in sorted(gauss.N.unique()):
        d = gauss[gauss.N==N].sort_values("L")
        ax.loglog(d.L, d.force, "o-", ms=4.8, lw=1.4, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 20, 30])
    ax.set_xticklabels(["10", "20", "30"])
    ax.set_xlabel("Physical slit width $L$")
    ax.set_ylabel(r"Exact Gaussian force $f\;(k_BT/a)$")
    ax.set_title("(c) Gaussian: unscaled force")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    # Gaussian scaled
    ax = axes[1,1]
    for N in sorted(gauss.N.unique()):
        d = gauss[gauss.N==N].sort_values("realized_L_over_Rg")
        ax.loglog(d.realized_L_over_Rg, d.fRg, "o", ms=5, color=COLORS.get(int(N)), label=f"N={N}")
    xx = np.logspace(np.log10(1.15), np.log10(3.15), 250)
    ax.loglog(xx, 2*np.pi**2*xx**(-3), "--", color="black", lw=2.0,
              label=r"strong-confinement reference $2\pi^2(L/R_g^{(0)})^{-3}$")
    ax.set_xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    ax.set_xticklabels(["1.2","1.4","1.6","1.8","2.0","2.2","2.4","2.6","2.8","3.0"])
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"Exact $fR_g^{(0)}\;(k_BT)$")
    ax.set_title("(d) Gaussian: scaled force")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Chain-length scaling of confinement force: SAW and Gaussian reference", fontsize=20, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.92, wspace=0.25, hspace=0.30)
    png = outdir/"Fig3_FORCE_SCALING_REV2.png"; pdf = outdir/"Fig3_FORCE_SCALING_REV2.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_free_energy_figure(saw: pd.DataFrame, gauss: pd.DataFrame, outdir: Path):
    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    ax = axes[0,0]
    for N in sorted(saw.N.unique()):
        d = saw[saw.N==N].sort_values("L")
        pos = d.DeltaF > 0
        ax.errorbar(d.loc[pos,"L"], d.loc[pos,"DeltaF"], yerr=d.loc[pos,"DeltaF_err"],
                    fmt="o-", ms=4.8, lw=1.4, capsize=2.5, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 20, 30])
    ax.set_xticklabels(["10", "20", "30"])
    ax.set_xlabel("Physical slit width $L$")
    ax.set_ylabel(r"Confinement free energy $\Delta F\;(k_BT)$")
    ax.set_title("(a) SAW: unscaled free energy")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[0,1]
    for N in sorted(saw.N.unique()):
        d = saw[saw.N==N].sort_values("realized_L_over_Rg")
        pos = d.DeltaF > 0
        ax.errorbar(d.loc[pos,"realized_L_over_Rg"], d.loc[pos,"DeltaF"], yerr=d.loc[pos,"DeltaF_err"],
                    fmt="o", ms=5, capsize=2.5, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    ax.set_xticklabels(["1.2","1.4","1.6","1.8","2.0","2.2","2.4","2.6","2.8","3.0"])
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$\Delta F\;(k_BT)$")
    ax.set_title("(b) SAW: scaled free energy")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1,0]
    for N in sorted(gauss.N.unique()):
        d = gauss[gauss.N==N].sort_values("L")
        ax.loglog(d.L, d.DeltaF, "o-", ms=4.8, lw=1.4, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 20, 30])
    ax.set_xticklabels(["10", "20", "30"])
    ax.set_xlabel("Physical slit width $L$")
    ax.set_ylabel(r"Exact Gaussian $\Delta F\;(k_BT)$")
    ax.set_title("(c) Gaussian: unscaled free energy")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1,1]
    for N in sorted(gauss.N.unique()):
        d = gauss[gauss.N==N].sort_values("realized_L_over_Rg")
        ax.loglog(d.realized_L_over_Rg, d.DeltaF, "o", ms=5, color=COLORS.get(int(N)), label=f"N={N}")
    ax.set_xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    ax.set_xticklabels(["1.2","1.4","1.6","1.8","2.0","2.2","2.4","2.6","2.8","3.0"])
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"Exact $\Delta F\;(k_BT)$")
    ax.set_title("(d) Gaussian: scaled free energy")
    ax.grid(True, which="major", alpha=0.22); ax.grid(True, which="minor", alpha=0.07, linestyle=":")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Chain-length scaling of confinement free energy: SAW and Gaussian reference", fontsize=20, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.92, wspace=0.25, hspace=0.30)
    png = outdir/"Fig3_FREE_ENERGY_SCALING_REV2.png"; pdf = outdir/"Fig3_FREE_ENERGY_SCALING_REV2.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# ============================================================================
# Self-test
# ============================================================================

def self_test():
    # Exact Gaussian strong-confinement prefactor test.
    rg = gaussian_rg(200)
    x = np.array([1.25, 1.50, 1.75], float)
    L = x*rg
    logz, frg = gaussian_logZ_and_force(L, 200)
    approx = 2*np.pi**2*x**(-3)
    if not (np.all(np.isfinite(logz)) and np.all(frg > 0)):
        raise AssertionError("Gaussian exact self-test failed.")
    if float(np.max(np.abs(frg-approx)/approx)) > 0.30:
        raise AssertionError("Gaussian exact strong-confinement sanity check failed.")

    a,b,c = insert_site(np.zeros(256,dtype=np.uint64), np.zeros(256,dtype=np.uint8), 256, 2,3,4), None, None
    # Compile/execute a tiny pilot and PERM state.
    raw, ref, low, high = build_thresholds(20, 8, 50, 1234, 0.5, 2.0, True)
    out = perm_block_saw(20, 8, 8, 128, 5678, 0.5, 4, low, high, 1)
    if len(raw) != 21 or out[6] not in (0,1):
        raise AssertionError("SAW/PERM self-test failed.")
    print("Self-test: PASS")
    print(f"  Gaussian Rg(N=200) = {rg:.8f}")
    print(f"  Expected SAW exponent = {EXPECTED_SAW_EXPONENT:.8f}")
    print(f"  Test PERM logZ = {out[0]:.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Reviewer-revised Fig. 3 SAW/Gaussian scaling production workflow")
    ap.add_argument("--rg-master", default="Rg_MASTER_FINAL_MERGED.csv", type=Path)
    ap.add_argument("--N-list", nargs="+", type=int, default=DEFAULT_N_LIST)
    ap.add_argument("--target-ratios", nargs="+", type=float, default=DEFAULT_TARGET_RATIOS)
    ap.add_argument("--delta-L", type=int, default=DEFAULT_DELTA_L)
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--block-counts", nargs="+", type=int, default=DEFAULT_BLOCK_COUNTS)
    ap.add_argument("--roots-per-block", type=int, default=DEFAULT_ROOTS_PER_BLOCK)
    ap.add_argument("--unconfined-roots-per-block", type=int, default=DEFAULT_UNCONFINED_ROOTS_PER_BLOCK)
    ap.add_argument("--pilot-roots", type=int, default=DEFAULT_PILOT_ROOTS)
    ap.add_argument("--max-population", type=int, default=DEFAULT_MAX_POPULATION)
    ap.add_argument("--C-minus", type=float, default=DEFAULT_CMINUS)
    ap.add_argument("--C-plus", type=float, default=DEFAULT_CPLUS)
    ap.add_argument("--prune-probability", type=float, default=DEFAULT_PRUNE_P)
    ap.add_argument("--max-clones", type=int, default=DEFAULT_MAX_CLONES)
    ap.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--fit-max-x", type=float, default=DEFAULT_FIT_MAX_X)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--outdir", type=Path, default=Path("Fig3_REV2_FINAL"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(); return

    args.N_list = sorted(set(int(n) for n in args.N_list))
    if any(n <= 2 for n in args.N_list):
        raise ValueError("N must be >2.")
    if args.delta_L <= 0 or args.delta_L % 2 != 0:
        raise ValueError("delta_L must be a positive even integer.")
    if args.blocks < 2 or any(k < 2 for k in args.block_counts):
        raise ValueError("Need at least 2 production blocks and convergence counts >=2.")
    if args.C_minus <= 0 or args.C_plus <= args.C_minus:
        raise ValueError("Require 0 < C_minus < C_plus.")
    if not (0 < args.prune_probability < 1):
        raise ValueError("prune probability must lie in (0,1).")

    outdir = args.outdir.resolve(); outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.N_list = [100, 150]
        args.target_ratios = [1.4, 1.8, 2.2]
        args.blocks = 4
        args.block_counts = [2, 4]
        args.roots_per_block = 32
        args.unconfined_roots_per_block = 32
        args.pilot_roots = 300
        args.max_population = 1024
        args.bootstrap_reps = 500
        args.workers = 1

    rg = load_rg_master(args.rg_master.resolve(), args.N_list)
    center_plans = {N: make_center_plan(rg[N]["Rg"], args.target_ratios, args.delta_L) for N in args.N_list}
    if any(len(v) < 3 for v in center_plans.values()):
        raise ValueError("At least 3 unique center widths are required for every N.")

    # Plan CSV.
    plan_rows = []
    for N, plan in center_plans.items():
        for p in plan:
            plan_rows.append({"N": N, **p, "L_minus": p["L"]-args.delta_L, "L_plus": p["L"]+args.delta_L})
    pd.DataFrame(plan_rows).to_csv(outdir/"Fig3_SAW_LPlan.csv", index=False)

    print("="*94)
    print("FIGURE 3 — REVIEWER-REVISED SAW + GAUSSIAN SCALING PRODUCTION")
    print("="*94)
    print(f"N list: {args.N_list}")
    for N in args.N_list:
        print(f"  N={N}: Rg={rg[N]['Rg']:.8f} +/- {rg[N]['Rg_err']:.8f} [{rg[N]['quality_flag']}], centers={len(center_plans[N])}")
    print(f"blocks x roots = {args.blocks} x {args.roots_per_block}")
    print(f"unconfined roots/block = {args.unconfined_roots_per_block}")
    print(f"pilot roots/state = {args.pilot_roots}")
    print(f"delta_L = {args.delta_L}")
    print("signed block forces retained; no sign-based selection")
    print("="*94)

    t0 = time.perf_counter()
    tasks, threshold_cache = build_state_tasks(
        rg, center_plans, args.delta_L, args.blocks,
        args.roots_per_block, args.unconfined_roots_per_block,
        args.max_population, args.prune_probability, args.max_clones,
        args.pilot_roots, args.C_minus, args.C_plus,
        args.seed, outdir,
    )

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_state_block, StateTask(**t)) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result(); all_rows.append(row)
            if i % max(1, len(futures)//20) == 0 or i == len(futures):
                print(f"Completed state-block {i}/{len(futures)}")

    block_df = pd.DataFrame(all_rows).sort_values(["N","side","L","block_id"]).reset_index(drop=True)
    block_df.to_csv(outdir/"Fig3_SAW_BlockLevel.csv", index=False)

    saw_points, force_block = build_points(
        block_df, center_plans, rg, args.delta_L,
        args.bootstrap_reps, args.seed,
    )
    saw_points.to_csv(outdir/"Fig3_SAW_PointLevel.csv", index=False)
    force_block.to_csv(outdir/"Fig3_SAW_ForceBlockLevel.csv", index=False)

    # Fit over the predeclared finite range.
    fits = [weighted_loglog_fit(saw_points, x) for x in [1.7, 1.8, 1.9]]
    fit = weighted_loglog_fit(saw_points, args.fit_max_x)
    pd.DataFrame(fits).to_csv(outdir/"Fig3_SAW_ForceFit.csv", index=False)

    worst = choose_worst_force_point(saw_points)
    conv = build_convergence(block_df, worst, args.block_counts, rg)
    conv.to_csv(outdir/"Fig3_SAW_Convergence.csv", index=False)

    gauss_points = gaussian_exact_points(args.target_ratios, args.delta_L, args.N_list)
    gauss_points.to_csv(outdir/"Fig3_Gaussian_PointLevel.csv", index=False)

    force_png, force_pdf = plot_force_figure(saw_points, gauss_points, fit, outdir)
    fe_png, fe_pdf = plot_free_energy_figure(saw_points, gauss_points, outdir)

    overflow_blocks = int(block_df["overflow"].sum())
    quality_counts = saw_points["quality_flag"].value_counts(dropna=False).to_dict()
    provenance = {
        "script": "Fig3_SAW_Gaussian_SCALING_REV2_FINAL.py",
        "version": "REV2",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Direct response to Reviewer #1(c): chain-length scaling for SAW and exact Gaussian, including force and free-energy scaling.",
        "reviewer_response": {
            "saw_chain_lengths": args.N_list,
            "gaussian_chain_lengths": args.N_list,
            "force_raw_and_scaled": True,
            "free_energy_raw_and_scaled": True,
            "explicit_numeric_scaled_x_ticks": [1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0],
            "claim_scope": "approximate chain-length scaling within the specified model/geometry; not an unrestricted universal master curve",
        },
        "input_rg_master": str(args.rg_master.resolve()),
        "canonical_rg": {str(N): rg[N] for N in args.N_list},
        "geometry": {
            "lattice": "3D simple cubic",
            "confinement": "one Cartesian coordinate z",
            "walls": "z=0 and z=L",
            "tether": "z0=L/2",
            "allowed_layers": "1,...,L-1",
            "L_grid": "even integer widths",
            "boundary": "absorbing/survival",
        },
        "force_definition": "f = d ln Z_conf / dL",
        "force_finite_difference": f"[ln Z(L+{args.delta_L})-ln Z(L-{args.delta_L})]/({2*args.delta_L})",
        "deltaF_definition": "Delta F = -ln[Z(L)/Z(infinity)]",
        "statistics": {
            "blocks": args.blocks,
            "roots_per_block": args.roots_per_block,
            "unconfined_roots_per_block": args.unconfined_roots_per_block,
            "pilot_roots": args.pilot_roots,
            "C_minus": args.C_minus,
            "C_plus": args.C_plus,
            "prune_probability": args.prune_probability,
            "max_clones": args.max_clones,
            "max_population": args.max_population,
            "bootstrap_reps": args.bootstrap_reps,
            "signed_force_retained": True,
            "sign_based_selection": False,
            "point_deletion": False,
        },
        "force_fit": fit,
        "worst_force_point_for_convergence": worst.to_dict(),
        "overflow_blocks": overflow_blocks,
        "quality_counts": quality_counts,
        "outputs": {
            "force_png": str(force_png), "force_pdf": str(force_pdf),
            "free_energy_png": str(fe_png), "free_energy_pdf": str(fe_pdf),
            "saw_point_csv": str(outdir/"Fig3_SAW_PointLevel.csv"),
            "saw_block_csv": str(outdir/"Fig3_SAW_BlockLevel.csv"),
            "saw_force_block_csv": str(outdir/"Fig3_SAW_ForceBlockLevel.csv"),
            "saw_convergence_csv": str(outdir/"Fig3_SAW_Convergence.csv"),
            "gaussian_point_csv": str(outdir/"Fig3_Gaussian_PointLevel.csv"),
        },
        "runtime_seconds": time.perf_counter() - t0,
        "system": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    (outdir/"Fig3_Provenance.json").write_text(json.dumps(jsonable(provenance), indent=2), encoding="utf-8")

    summary = [
        "FIGURE 3 REVIEWER-REVISED PRODUCTION COMPLETE",
        f"N list: {args.N_list}",
        f"Force fit: p={fit.get('exponent')} +/- {fit.get('exponent_err')}",
        f"Expected SAW reference exponent: {EXPECTED_SAW_EXPONENT}",
        f"Overflow blocks: {overflow_blocks}",
        f"Worst convergence point: N={int(worst['N'])}, L={int(worst['L'])}, x={worst['realized_L_over_Rg']:.5f}",
        f"Force figure: {force_png}",
        f"Free-energy figure: {fe_png}",
        f"Runtime: {(time.perf_counter()-t0)/60:.2f} min",
        "",
        "Important: do not describe these plots as proving unrestricted universality across models, interaction strengths, or boundary conditions.",
    ]
    (outdir/"Fig3_RunSummary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "="*94)
    print("FIGURE 3 PRODUCTION COMPLETE")
    print("="*94)
    print(f"Force figure      : {force_png}")
    print(f"Free-energy figure: {fe_png}")
    print(f"SAW points        : {outdir/'Fig3_SAW_PointLevel.csv'}")
    print(f"SAW blocks        : {outdir/'Fig3_SAW_BlockLevel.csv'}")
    print(f"Gaussian points   : {outdir/'Fig3_Gaussian_PointLevel.csv'}")
    print(f"Convergence       : {outdir/'Fig3_SAW_Convergence.csv'}")
    print(f"Overflow blocks   : {overflow_blocks}")
    print("="*94)


if __name__ == "__main__":
    main()
