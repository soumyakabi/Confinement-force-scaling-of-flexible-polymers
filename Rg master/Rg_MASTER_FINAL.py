#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rg_MASTER_FINAL.py
==================

Canonical unconfined radius-of-gyration master calculator for the manuscript.

Scientific definition
---------------------
For every polymer ensemble we report

    R_g = sqrt(< R_g^2 >_W)

where, for each chain,

    R_g^2 = (1/N) sum_i |r_i - r_cm|^2,

and <...>_W is the appropriate Rosenbluth/sequential-importance weighted
ensemble average.

For finite-w Domb-Joyce:
    target weight = exp(-w U)
    proposal at each growth step is proportional to exp(-w * DeltaU)
    Rosenbluth/SIS path weight is the product of the corresponding
    normalization factors.

For strict SAW:
    target ensemble forbids site revisits
    proposal is uniform among allowed neighbours
    Rosenbluth weight = product(number of allowed neighbours).

Dead/trapped chains therefore have zero statistical weight in the corresponding
partition-function estimator; they are not treated as missing attempts.

Independent blocks
------------------
All production estimates are divided into independent blocks.  The published
uncertainty is the SEM of the block-level R_g estimates.  The code also reports
block ESS, maximum normalized block weight fraction, valid/finite fraction, and
a block bootstrap 95% CI as an audit diagnostic.

Tasks
-----
1) dj_w_sweep:
      N=200, w = 0,0.1,0.2,0.3,0.4,0.5,1,2,4,8
      This is the authoritative R_g(w) dataset for Fig. 5 and downstream use.

2) dj_chainlength:
      fixed w=0.1, N = 100,150,200,250,500,1000,2000
      Intended for Fig. 7(i), S14, and chain-length-dependent scaling.

3) saw_chainlength:
      strict SAW, N = 100,150,200,250 by default.
      Add other N with --N-list when needed.

4) gaussian_exact:
      analytic continuous-Gaussian reference R_g = a*sqrt(N/6).

The code deliberately does NOT mix the Gaussian analytic convention with the
numerical DJ/SAW master estimates.

Examples
--------
Smoke tests:
    python Rg_MASTER_FINAL.py --task dj_w_sweep --smoke --workers 4
    python Rg_MASTER_FINAL.py --task saw_chainlength --smoke --workers 4

Production Fig. 5 R_g sweep:
    python Rg_MASTER_FINAL.py --task dj_w_sweep \
        --N 200 --attempts 120000 --blocks 30 --workers 8 \
        --seed 20260921 --outdir Rg_MASTER_DJ_N200

Production fixed-w chain-length dataset:
    python Rg_MASTER_FINAL.py --task dj_chainlength \
        --w 0.1 --N-list 100 150 200 250 500 1000 2000 \
        --attempts 60000 --blocks 20 --workers 8 \
        --seed 20260922 --outdir Rg_MASTER_DJ_w0p1

Production SAW:
    python Rg_MASTER_FINAL.py --task saw_chainlength \
        --N-list 100 150 200 250 \
        --attempts 60000 --blocks 20 --workers 8 \
        --seed 20260923 --outdir Rg_MASTER_SAW

Notes on reproducibility
------------------------
Block seeds are generated deterministically from a master SeedSequence and are
independent of worker completion order.  Do not use Python's hash() to seed
production runs.

The code is intended to be robust enough for the current revision deadline,
not as a new general-purpose polymer-growth framework.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A = 1.0

STEPS = np.asarray(
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

DEFAULT_DJ_W_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_DJ_N_LIST = [100, 150, 200, 250, 500, 1000, 2000]
DEFAULT_SAW_N_LIST = [100, 150, 200, 250]

QUALITY_ESS_MIN = 20.0
QUALITY_MAX_WEIGHT_FRAC = 0.10
QUALITY_VALID_FRAC = 0.80
BOOTSTRAP_REPS = 2000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def trajectory_rg2(positions: np.ndarray) -> float:
    """Return single-chain R_g^2."""
    cm = positions.mean(axis=0)
    dr = positions - cm
    return float(np.mean(np.sum(dr * dr, axis=1)))


def weighted_mean_stable(logw: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """
    Weighted mean and ESS diagnostics using log-weights.

    Returns:
        mean_value, ESS, max_normalized_weight_fraction
    """
    finite = np.isfinite(logw) & np.isfinite(values)
    if not np.any(finite):
        return math.nan, 0.0, math.nan

    lw = logw[finite]
    vals = values[finite]
    m = float(np.max(lw))
    rw = np.exp(lw - m)
    sw = float(np.sum(rw))
    if sw <= 0.0 or not np.isfinite(sw):
        return math.nan, 0.0, math.nan

    mean = float(np.sum(rw * vals) / sw)
    ess = float((sw * sw) / np.sum(rw * rw))
    max_frac = float(np.max(rw) / sw)
    return mean, ess, max_frac


def safe_sem(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def bootstrap_ci(block_values: Sequence[float], reps: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(block_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or reps <= 0:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(reps, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_block_seeds(master_seed: int, n_blocks: int, stream: int) -> list[int]:
    """
    Deterministic block seeds independent of ProcessPool completion order.
    `stream` separates distinct (model, N, w) jobs.
    """
    root = np.random.SeedSequence([int(master_seed), int(stream)])
    children = root.spawn(int(n_blocks))
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("no_rows\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def quality_flag(neff_min: float, max_weight_frac_max: float, valid_frac: float) -> str:
    if (
        np.isfinite(neff_min)
        and np.isfinite(max_weight_frac_max)
        and np.isfinite(valid_frac)
        and neff_min >= QUALITY_ESS_MIN
        and max_weight_frac_max <= QUALITY_MAX_WEIGHT_FRAC
        and valid_frac >= QUALITY_VALID_FRAC
    ):
        return "OK"
    return "CAUTION"


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def grow_dj(N: int, w: float, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """
    Unconfined Domb-Joyce sequential-importance sampler.

    At each step:
      candidate probability ~ exp(-w * overlap_increment)
      Rosenbluth log weight += log(sum candidate Boltzmann factors)

    This gives target path weight exp(-w U) after summing over all lattice walks.
    """
    positions = np.empty((N, 3), dtype=np.int32)
    pos = np.zeros(3, dtype=np.int32)
    positions[0] = pos

    visited = {tuple(pos): 1}
    logW = 0.0

    for i in range(1, N):
        increments = np.empty(6, dtype=np.int32)
        candidates = []
        boltz = np.empty(6, dtype=float)

        for j, step in enumerate(STEPS):
            new_pos = pos + step
            key = tuple(int(v) for v in new_pos)
            inc = int(visited.get(key, 0))
            increments[j] = inc
            candidates.append(new_pos.copy())
            boltz[j] = -w * inc

        m = float(np.max(boltz))
        rel = np.exp(boltz - m)
        total_scaled = float(np.sum(rel))
        if total_scaled <= 0.0 or not np.isfinite(total_scaled):
            return -np.inf, positions

        probs = rel / total_scaled
        chosen = int(rng.choice(6, p=probs))

        # IMPORTANT: add the proposal normalization once for this step.
        # The actual overlap count is encoded in the proposal; no repeated
        # cumulative penalty is applied here.
        logW += m + math.log(total_scaled)

        pos = candidates[chosen]
        positions[i] = pos
        key = tuple(int(v) for v in pos)
        visited[key] = visited.get(key, 0) + 1

    return float(logW), positions


def grow_saw(N: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Unconfined strict-SAW Rosenbluth growth."""
    positions = np.empty((N, 3), dtype=np.int32)
    pos = np.zeros(3, dtype=np.int32)
    positions[0] = pos

    visited = {tuple(pos)}
    logW = 0.0

    for i in range(1, N):
        candidates = []
        for step in STEPS:
            new_pos = pos + step
            key = tuple(int(v) for v in new_pos)
            if key not in visited:
                candidates.append(new_pos.copy())

        m_free = len(candidates)
        if m_free == 0:
            return -np.inf, positions

        logW += math.log(m_free)
        pos = candidates[int(rng.integers(m_free))]
        positions[i] = pos
        visited.add(tuple(int(v) for v in pos))

    return float(logW), positions


# ---------------------------------------------------------------------------
# Block worker
# ---------------------------------------------------------------------------

def run_block(
    model: str,
    N: int,
    w: float,
    attempts: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    logw = np.full(attempts, -np.inf, dtype=float)
    rg2 = np.full(attempts, np.nan, dtype=float)

    valid = 0
    for i in range(attempts):
        if model == "DJ":
            lw, positions = grow_dj(N, w, rng)
        elif model == "SAW":
            lw, positions = grow_saw(N, rng)
        else:
            raise ValueError(f"Unknown model: {model}")

        if np.isfinite(lw):
            logw[i] = lw
            rg2[i] = trajectory_rg2(positions)
            valid += 1

    rg2_weighted, ess, max_frac = weighted_mean_stable(logw, rg2)

    if np.isfinite(rg2_weighted) and rg2_weighted >= 0.0:
        rg_block = math.sqrt(rg2_weighted)
    else:
        rg_block = math.nan

    valid_fraction = valid / attempts if attempts else 0.0

    # Unweighted chainwise mean is diagnostic only.
    finite_rg2 = rg2[np.isfinite(rg2)]
    unweighted_rg2 = float(np.mean(finite_rg2)) if finite_rg2.size else math.nan
    unweighted_rg = math.sqrt(unweighted_rg2) if np.isfinite(unweighted_rg2) else math.nan

    return {
        "seed": int(seed),
        "attempts": int(attempts),
        "valid_chains": int(valid),
        "valid_fraction": float(valid_fraction),
        "weighted_rg2": float(rg2_weighted),
        "rg_block": float(rg_block),
        "unweighted_rg": float(unweighted_rg),
        "ESS": float(ess),
        "max_weight_fraction": float(max_frac),
    }


# ---------------------------------------------------------------------------
# One master estimate
# ---------------------------------------------------------------------------

def estimate_master(
    model: str,
    N: int,
    w: float,
    attempts: int,
    blocks: int,
    master_seed: int,
    stream: int,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> tuple[dict, list[dict]]:
    if attempts < blocks:
        raise ValueError("attempts must be >= blocks")

    block_size = attempts // blocks
    attempts_used = block_size * blocks

    seeds = make_block_seeds(master_seed, blocks, stream)

    block_results: list[dict] = []

    # Run blocks. The main script may dispatch them in parallel.
    for seed in seeds:
        block_results.append(run_block(model, N, w, block_size, seed))

    return summarize_blocks(
        model=model,
        N=N,
        w=w,
        attempts=attempts_used,
        blocks=blocks,
        master_seed=master_seed,
        stream=stream,
        block_results=block_results,
        bootstrap_reps=bootstrap_reps,
    )


def summarize_blocks(
    model: str,
    N: int,
    w: float,
    attempts: int,
    blocks: int,
    master_seed: int,
    stream: int,
    block_results: list[dict],
    bootstrap_reps: int,
) -> tuple[dict, list[dict]]:
    rg_blocks = np.asarray([b["rg_block"] for b in block_results], dtype=float)
    rg2_blocks = np.asarray([b["weighted_rg2"] for b in block_results], dtype=float)
    ess = np.asarray([b["ESS"] for b in block_results], dtype=float)
    max_frac = np.asarray([b["max_weight_fraction"] for b in block_results], dtype=float)
    valid_frac = np.asarray([b["valid_fraction"] for b in block_results], dtype=float)

    finite_rg = np.isfinite(rg_blocks)
    n_valid_blocks = int(np.sum(finite_rg))

    if n_valid_blocks < max(3, blocks // 2):
        raise RuntimeError(
            f"{model} N={N} w={w}: too few valid blocks "
            f"({n_valid_blocks}/{blocks})."
        )

    rg = float(np.mean(rg_blocks[finite_rg]))
    rg_err = safe_sem(rg_blocks[finite_rg])

    rg2 = float(np.mean(rg2_blocks[finite_rg]))
    rg2_err = safe_sem(rg2_blocks[finite_rg])

    boot_seed = int(np.random.SeedSequence([master_seed, stream, 99991]).generate_state(1, dtype=np.uint64)[0])
    ci_lo, ci_hi = bootstrap_ci(rg_blocks[finite_rg], bootstrap_reps, boot_seed)

    min_ess = float(np.nanmin(ess))
    max_weight_fraction_max = float(np.nanmax(max_frac))
    valid_fraction_mean = float(np.nanmean(valid_frac))
    valid_fraction_min = float(np.nanmin(valid_frac))

    quality = quality_flag(min_ess, max_weight_fraction_max, valid_fraction_min)

    result = {
        "model": model,
        "N": int(N),
        "w": (float(w) if model == "DJ" else "inf"),
        "Rg": rg,
        "Rg_err": rg_err,
        "Rg2": rg2,
        "Rg2_err": rg2_err,
        "Rg_bootstrap_CI95_low": ci_lo,
        "Rg_bootstrap_CI95_high": ci_hi,
        "attempts": int(attempts),
        "blocks": int(blocks),
        "block_size": int(attempts // blocks),
        "valid_blocks": n_valid_blocks,
        "valid_block_fraction": n_valid_blocks / blocks,
        "min_valid_fraction_within_block": valid_fraction_min,
        "min_ESS": min_ess,
        "median_ESS": float(np.nanmedian(ess)),
        "mean_ESS": float(np.nanmean(ess)),
        "max_block_weight_fraction": max_weight_fraction_max,
        "mean_block_valid_fraction": valid_fraction_mean,
        "quality_flag": quality,
        "estimator": "sqrt(weighted mean of per-chain Rg^2)",
        "single_chain_definition": "(1/N) sum_i |r_i-r_cm|^2",
        "ensemble_weighting": (
            "DJ SIS/Rosenbluth" if model == "DJ"
            else "strict-SAW Rosenbluth"
        ),
        "dead_chain_treatment": "zero statistical weight; denominator remains total attempted chains",
        "seed": int(master_seed),
        "stream": int(stream),
    }

    return result, block_results


def dispatch_blocks_parallel(
    model: str,
    N: int,
    w: float,
    attempts: int,
    blocks: int,
    master_seed: int,
    stream: int,
    workers: int,
) -> tuple[list[dict], int]:
    block_size = attempts // blocks
    attempts_used = block_size * blocks
    seeds = make_block_seeds(master_seed, blocks, stream)

    tasks = [
        (model, N, w, block_size, seeds[b])
        for b in range(blocks)
    ]

    if workers <= 1:
        results = [run_block(*task) for task in tasks]
        return results, attempts_used

    results_by_index: dict[int, dict] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(run_block, *task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results_by_index[idx] = future.result()

    ordered = [results_by_index[i] for i in range(blocks)]
    return ordered, attempts_used


def run_master_job(
    model: str,
    N: int,
    w: float,
    attempts: int,
    blocks: int,
    seed: int,
    stream: int,
    workers: int,
    bootstrap_reps: int,
) -> tuple[dict, list[dict]]:
    t0 = time.time()
    block_results, attempts_used = dispatch_blocks_parallel(
        model=model,
        N=N,
        w=w,
        attempts=attempts,
        blocks=blocks,
        master_seed=seed,
        stream=stream,
        workers=workers,
    )

    result, block_rows = summarize_blocks(
        model=model,
        N=N,
        w=w,
        attempts=attempts_used,
        blocks=blocks,
        master_seed=seed,
        stream=stream,
        block_results=block_results,
        bootstrap_reps=bootstrap_reps,
    )
    result["runtime_seconds"] = float(time.time() - t0)

    # Attach row-level block metadata.
    enriched_blocks = []
    for i, b in enumerate(block_rows):
        row = {
            "model": model,
            "N": int(N),
            "w": (float(w) if model == "DJ" else "inf"),
            "block": int(i),
            **b,
        }
        enriched_blocks.append(row)

    return result, enriched_blocks


# ---------------------------------------------------------------------------
# Exact Gaussian rows
# ---------------------------------------------------------------------------

def gaussian_exact_row(N: int, a: float = A) -> dict:
    rg = a * math.sqrt(N / 6.0)
    return {
        "model": "Gaussian_exact",
        "N": int(N),
        "w": 0.0,
        "Rg": rg,
        "Rg_err": 0.0,
        "Rg2": rg * rg,
        "Rg2_err": 0.0,
        "Rg_bootstrap_CI95_low": rg,
        "Rg_bootstrap_CI95_high": rg,
        "attempts": 0,
        "blocks": 0,
        "block_size": 0,
        "valid_blocks": 0,
        "valid_block_fraction": 1.0,
        "min_valid_fraction_within_block": 1.0,
        "min_ESS": math.inf,
        "median_ESS": math.inf,
        "mean_ESS": math.inf,
        "max_block_weight_fraction": 0.0,
        "mean_block_valid_fraction": 1.0,
        "quality_flag": "EXACT",
        "estimator": "analytic continuous-Gaussian Rg=a*sqrt(N/6)",
        "single_chain_definition": "continuous Gaussian-chain reference",
        "ensemble_weighting": "analytic",
        "dead_chain_treatment": "not applicable",
        "seed": "",
        "stream": "",
        "runtime_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def run_self_tests() -> None:
    print("=" * 80)
    print("Rg_MASTER_FINAL self-test")
    print("=" * 80)

    # Gaussian exact
    g = gaussian_exact_row(200)
    assert abs(g["Rg"] - math.sqrt(200.0 / 6.0)) < 1e-14

    # DJ w=0 must have zero-overlap weighting structure and finite Rg.
    result, blocks = run_master_job(
        model="DJ",
        N=20,
        w=0.0,
        attempts=400,
        blocks=4,
        seed=20260921,
        stream=1,
        workers=1,
        bootstrap_reps=100,
    )
    assert np.isfinite(result["Rg"])
    assert result["quality_flag"] in {"OK", "CAUTION"}
    assert len(blocks) == 4

    # Strict SAW smoke.
    result_saw, blocks_saw = run_master_job(
        model="SAW",
        N=20,
        w=0.0,
        attempts=400,
        blocks=4,
        seed=20260922,
        stream=2,
        workers=1,
        bootstrap_reps=100,
    )
    assert np.isfinite(result_saw["Rg"])
    assert len(blocks_saw) == 4

    # Exact definition must appear in output.
    assert result["estimator"] == "sqrt(weighted mean of per-chain Rg^2)"

    print("PASS: Gaussian exact reference")
    print(f"PASS: DJ N=20, w=0: Rg={result['Rg']:.6f} ± {result['Rg_err']:.6f}")
    print(
        f"PASS: SAW N=20: Rg={result_saw['Rg']:.6f} ± "
        f"{result_saw['Rg_err']:.6f}"
    )
    print("All self-tests passed.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical unconfined Rg master calculator for DJ and strict SAW."
    )

    p.add_argument(
        "--task",
        choices=["dj_w_sweep", "dj_chainlength", "saw_chainlength", "gaussian_exact", "all"],
        default="dj_w_sweep",
    )
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--N-list", type=int, nargs="+", default=None)
    p.add_argument("--w", type=float, default=0.1)
    p.add_argument("--w-values", type=float, nargs="+", default=DEFAULT_DJ_W_VALUES)

    p.add_argument("--attempts", type=int, default=120000)
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)

    p.add_argument("--outdir", type=str, default="Rg_MASTER_FINAL")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")

    return p.parse_args()


def build_jobs(args: argparse.Namespace) -> list[tuple[str, int, float, int]]:
    jobs: list[tuple[str, int, float, int]] = []

    if args.task == "dj_w_sweep":
        for idx, w in enumerate(sorted(set(args.w_values))):
            jobs.append(("DJ", args.N, float(w), idx + 1))

    elif args.task == "dj_chainlength":
        nlist = args.N_list or DEFAULT_DJ_N_LIST
        for idx, N in enumerate(nlist):
            jobs.append(("DJ", int(N), float(args.w), 1000 + idx))

    elif args.task == "saw_chainlength":
        nlist = args.N_list or DEFAULT_SAW_N_LIST
        for idx, N in enumerate(nlist):
            jobs.append(("SAW", int(N), 0.0, 2000 + idx))

    elif args.task == "gaussian_exact":
        nlist = args.N_list or [args.N]
        for idx, N in enumerate(nlist):
            jobs.append(("Gaussian_exact", int(N), 0.0, 3000 + idx))

    elif args.task == "all":
        for idx, w in enumerate(sorted(set(args.w_values))):
            jobs.append(("DJ", args.N, float(w), idx + 1))
        for idx, N in enumerate(args.N_list or DEFAULT_DJ_N_LIST):
            jobs.append(("DJ", int(N), float(args.w), 1000 + idx))
        for idx, N in enumerate(args.N_list or DEFAULT_SAW_N_LIST):
            jobs.append(("SAW", int(N), 0.0, 2000 + idx))

    return jobs


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_tests()
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Smoke settings are intentionally small and should finish quickly.
    if args.smoke:
        attempts = min(args.attempts, 800)
        blocks = min(args.blocks, 4)
        workers = min(args.workers, 4)
        bootstrap_reps = min(args.bootstrap_reps, 200)
        if args.task == "dj_w_sweep":
            w_values = sorted(set(args.w_values[:3]))
            args.w_values = w_values
        if args.task in {"dj_chainlength", "saw_chainlength"}:
            if args.N_list is not None:
                args.N_list = args.N_list[:2]
    else:
        attempts = args.attempts
        blocks = args.blocks
        workers = args.workers
        bootstrap_reps = args.bootstrap_reps

    jobs = build_jobs(args)

    print("=" * 86)
    print("Rg_MASTER_FINAL")
    print("=" * 86)
    print(f"Task              : {args.task}")
    print(f"Platform           : {platform.platform()}")
    print(f"Python             : {sys.version.split()[0]}")
    print(f"CPU count          : {os.cpu_count()}")
    print(f"Workers             : {workers}")
    print(f"Attempts/job        : {attempts}")
    print(f"Blocks/job          : {blocks}")
    print(f"Bootstrap reps      : {bootstrap_reps}")
    print(f"Output directory    : {outdir.resolve()}")
    print("-" * 86)

    started = datetime.now().isoformat(timespec="seconds")
    master_rows = []
    all_block_rows = []

    for job_index, (model, N, w, stream) in enumerate(jobs, start=1):
        print()
        print(f"[{job_index}/{len(jobs)}] model={model}  N={N}  w={w}")

        if model == "Gaussian_exact":
            row = gaussian_exact_row(N)
            master_rows.append(row)
            print(f"  Exact Gaussian Rg = {row['Rg']:.8f}")
            continue

        t0 = time.time()
        row, block_rows = run_master_job(
            model=model,
            N=N,
            w=w,
            attempts=attempts,
            blocks=blocks,
            seed=args.seed,
            stream=stream,
            workers=workers,
            bootstrap_reps=bootstrap_reps,
        )

        master_rows.append(row)
        all_block_rows.extend(block_rows)

        print(
            f"  Rg = {row['Rg']:.8f} ± {row['Rg_err']:.8f} | "
            f"Rg² = {row['Rg2']:.8f} ± {row['Rg2_err']:.8f}"
        )
        print(
            f"  ESS min/median = {row['min_ESS']:.1f} / {row['median_ESS']:.1f}; "
            f"max weight frac = {row['max_block_weight_fraction']:.4f}; "
            f"valid blocks = {row['valid_blocks']}/{row['blocks']}; "
            f"status = {row['quality_flag']}"
        )
        print(f"  runtime = {row['runtime_seconds'] / 60.0:.2f} min")

    # Sort master rows in a stable, manuscript-friendly order.
    def sort_key(r):
        model_order = {"Gaussian_exact": 0, "DJ": 1, "SAW": 2}
        w = r["w"]
        w_num = float(w) if w != "inf" else math.inf
        return (model_order.get(r["model"], 9), int(r["N"]), w_num)

    master_rows.sort(key=sort_key)

    # Main files.
    master_csv = outdir / "Rg_MASTER_FINAL.csv"
    block_csv = outdir / "Rg_MASTER_FINAL_BLOCKS.csv"
    report_json = outdir / "Rg_MASTER_FINAL_REPORT.json"
    provenance_json = outdir / "Rg_MASTER_FINAL_PROVENANCE.json"

    save_csv(master_rows, master_csv)
    save_csv(all_block_rows, block_csv)

    quality_counts = {}
    for row in master_rows:
        quality_counts[row["quality_flag"]] = quality_counts.get(row["quality_flag"], 0) + 1

    report = {
        "script": "Rg_MASTER_FINAL.py",
        "version": "1.0",
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "task": args.task,
        "jobs": jobs,
        "attempts_per_job": attempts,
        "blocks_per_job": blocks,
        "workers": workers,
        "bootstrap_reps": bootstrap_reps,
        "quality_counts": quality_counts,
        "master_rows": master_rows,
    }

    provenance = {
        "scientific_definition": {
            "Rg": "sqrt(<Rg^2>_W)",
            "single_chain_Rg2": "(1/N) sum_i |r_i-r_cm|^2",
            "DJ_target_weight": "exp(-w U)",
            "DJ_sampling": "sequential importance sampling; candidate probability proportional to exp(-w DeltaU)",
            "SAW_sampling": "strict SAW Rosenbluth growth; weight is product of allowed-neighbour counts",
        },
        "dead_chain_treatment": "zero statistical weight; denominator remains total attempted chains",
        "uncertainty": "SEM across independent block-level Rg estimates",
        "bootstrap": "percentile 95% CI across block means",
        "seed_policy": "numpy SeedSequence; deterministic independent block seeds; no Python hash()",
        "geometry": "unconfined 3D cubic lattice",
        "Gaussian_reference": "analytic continuous Gaussian Rg=a*sqrt(N/6), kept separate from numerical master data",
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }

    with report_json.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(report), fh, indent=2)

    with provenance_json.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(provenance), fh, indent=2)

    print()
    print("=" * 86)
    print("MASTER Rg DATASET WRITTEN")
    print("=" * 86)
    print(master_csv.resolve())
    print(block_csv.resolve())
    print(report_json.resolve())
    print(provenance_json.resolve())
    print()
    print("Important: use Rg_MASTER_FINAL.csv as the sole numerical Rg source for")
    print("Fig. 5, Fig. 6, S3, S5-S8, S10-S12 and the numerical part of S14.")
    print("Keep the analytic Gaussian reference separate.")
    print()


if __name__ == "__main__":
    main()
