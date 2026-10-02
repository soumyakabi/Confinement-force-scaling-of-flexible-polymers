#!/usr/bin/env python3
"""
S14 v0.9 — genuine PERM population-control production sampler.

Model:
  Domb–Joyce chain on a cubic lattice, absorbing walls z=0,L,
  center tether z=L/2, even integer L.

Growth:
  q_j = exp(-w*n_j) / sum_k exp(-w*n_k)
  Rosenbluth factor per growth step = sum_k exp(-w*n_k)
  log W += log(Rosenbluth factor)

PERM control:
  fixed C- and C+ thresholds derived from an independent unpruned pilot.
  Low weight: Russian roulette with probability p; survivor weight /= (1-p).
  High weight: deterministic cloning into k copies, each weight /= k.
  No force-sign selection.

Important:
  This is a new sampler and must be validated against the previously
  validated unpruned v0.8.2 reference before final production.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import numba
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    numba = None

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


SCRIPT_VERSION = "0.9.2-PERM"

# Archived/current Rg values used in S14.
DEFAULT_RG = {
    500: 11.165841041,
    1000: 16.579163646,
    2000: 24.357089468,
    3000: 29.2248410156,
}

DEFAULT_LR = [1.2, 1.4, 1.6, 1.8, 2.0,
              2.2, 2.4, 2.6, 2.8, 3.0]


def nearest_even(x):
    return max(4, int(math.floor(x / 2.0 + 0.5) * 2))


def quality_flag(min_ess, max_w):
    if min_ess < 2.0 or max_w >= 0.80:
        return "CRITICAL_ESS"
    if min_ess < 5.0:
        return "VERY_LOW_ESS"
    if min_ess < 10.0:
        return "LOW_ESS"
    if max_w >= 0.40:
        return "HIGH_WEIGHT_CONCENTRATION"
    return "OK"


@njit(cache=True)
def coord_key(x, y, z):
    # 21 bits per coordinate, sufficient for the S14 coordinate range.
    bias = np.int64(1_000_000)
    mask = np.uint64((1 << 21) - 1)
    ux = np.uint64(np.int64(x) + bias) & mask
    uy = np.uint64(np.int64(y) + bias) & mask
    uz = np.uint64(np.int64(z) + bias) & mask
    return ((ux << np.uint64(42))
            | (uy << np.uint64(21))
            | uz)


@njit(cache=True)
def hidx(key, mask):
    # Power-of-two table: avoids uint64 modulo typing problems on Kaggle.
    return int((key * np.uint64(0x9E3779B97F4A7C15))
               & np.uint64(mask))


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
def logsumexp6(v):
    m = -np.inf
    for i in range(6):
        if np.isfinite(v[i]) and v[i] > m:
            m = v[i]
    if not np.isfinite(m):
        return -np.inf
    s = 0.0
    for i in range(6):
        if np.isfinite(v[i]):
            s += math.exp(v[i] - m)
    return m + math.log(s) if s > 0.0 else -np.inf


@njit(cache=True)
def selftest_kernel():
    size = 256
    keys = np.zeros(size, dtype=np.uint64)
    counts = np.zeros(size, dtype=np.int16)
    used = np.zeros(size, dtype=np.uint8)
    a = ins(keys, counts, used, size, 0, 0, 10)
    c1 = occ(keys, counts, used, size, 0, 0, 10)
    b = ins(keys, counts, used, size, 0, 0, 10)
    c2 = occ(keys, counts, used, size, 0, 0, 10)
    return a, b, c1, c2


def numba_selftest():
    a, b, c1, c2 = selftest_kernel()
    if not a or not b or c1 != 1 or c2 != 2:
        raise RuntimeError(
            f"PERM self-test failed: {a=} {b=} {c1=} {c2=}"
        )
    print("Numba PERM self-test: PASS")


@njit(cache=True)
def pilot_partial(N, L, w, roots, seed):
    """
    Independent unpruned pilot.

    pilot_logZ[k] = log(mean_root W_k * I_alive) for chains with k monomers.

    It is used only to define fixed PERM thresholds for a separate
    production realization.
    """
    np.random.seed(seed)

    table_size = 1
    while table_size < 3 * (N + 64):
        table_size *= 2

    dx = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
    dy = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
    dz = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)

    logsum = np.full(N + 1, -np.inf, dtype=np.float64)
    logsum[0] = 0.0  # mean weight of the root ensemble.

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

        # k = current number of monomers. Before adding the next monomer,
        # the current partial weight is W_k.
        for k in range(1, N):
            # Add current partial weight W_k to the pilot average.
            a = logsum[k]
            b = lw
            if not np.isfinite(a):
                logsum[k] = b
            elif b > a:
                logsum[k] = b + math.log1p(math.exp(a - b))
            else:
                logsum[k] = a + math.log1p(math.exp(b - a))

            trial = np.full(6, -np.inf, dtype=np.float64)
            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]
                if zz <= 0 or zz >= L:
                    continue
                ov = occ(keys, counts, used, table_size,
                         xx, yy, zz)
                if ov < 0:
                    break
                trial[d] = -w * ov

            norm = logsumexp6(trial)
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
            acc = 0.0
            chosen = -1
            for d in range(6):
                if np.isfinite(trial[d]):
                    acc += math.exp(trial[d] - m)
                    if u <= acc:
                        chosen = d
                        break

            if chosen < 0:
                break

            lw += norm
            x += dx[chosen]
            y += dy[chosen]
            z += dz[chosen]

            if not ins(keys, counts, used, table_size,
                       x, y, z):
                break

        # Add W_N when the complete chain survives.
        if np.isfinite(lw):
            a = logsum[N]
            b = lw
            if not np.isfinite(a):
                logsum[N] = b
            elif b > a:
                logsum[N] = b + math.log1p(math.exp(a - b))
            else:
                logsum[N] = a + math.log1p(math.exp(b - a))

    ln_roots = math.log(roots)
    for k in range(N + 1):
        if np.isfinite(logsum[k]):
            logsum[k] -= ln_roots

    return logsum


@njit(cache=True)
def perm_block(
    N, L, w, roots, max_pop, seed,
    prune_p, cminus, cplus, pilot_logz
):
    """
    Genuine PERM population growth.

    Two population buffers are used so parents are never overwritten while
    they are still being processed.
    """
    np.random.seed(seed)

    table_size = 1
    while table_size < 3 * (N + 64):
        table_size *= 2

    # Two buffers: safe in-place population transition without overwriting
    # unprocessed parents.
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

    for r in range(roots):
        coords_a[r, 0, 0] = 0
        coords_a[r, 0, 1] = 0
        coords_a[r, 0, 2] = L // 2
        logw_a[r] = 0.0
        ins(keys_a[r], counts_a[r], used_a[r],
            table_size, 0, 0, L // 2)

    prune_deaths = 0
    clone_events = 0
    overflow_events = 0

    for k in range(1, N):
        new_pop = 0

        # pilot_logz[k] is the reference scale for k monomers,
        # so population control is applied AFTER growth to k+1 monomers
        # using pilot_logz[k+1]. We therefore compute the next threshold
        # after obtaining the new weight.
        threshold = pilot_logz[k + 1]
        low_log = (-np.inf if not np.isfinite(threshold)
                   else threshold + math.log(cminus))
        high_log = (np.inf if not np.isfinite(threshold)
                    else threshold + math.log(cplus))

        for parent in range(pop):
            if not np.isfinite(logw_a[parent]):
                continue

            x = coords_a[parent, k - 1, 0]
            y = coords_a[parent, k - 1, 1]
            z = coords_a[parent, k - 1, 2]

            trial = np.full(6, -np.inf, dtype=np.float64)
            bad_table = False

            for d in range(6):
                xx = x + dx[d]
                yy = y + dy[d]
                zz = z + dz[d]

                if zz <= 0 or zz >= L:
                    continue

                ov = occ(keys_a[parent], counts_a[parent],
                         used_a[parent], table_size,
                         xx, yy, zz)
                if ov < 0:
                    bad_table = True
                    break
                trial[d] = -w * ov

            if bad_table:
                continue

            norm = logsumexp6(trial)
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

            if new_pop >= max_pop:
                # Retain the parent contribution without enrichment rather
                # than dropping it. This preserves unbiasedness; overflow is
                # only an efficiency diagnostic.
                overflow_events += 1
                continue

            child = new_pop
            new_pop += 1

            # Copy parent state into output buffer.
            for q in range(k):
                coords_b[child, q, 0] = coords_a[parent, q, 0]
                coords_b[child, q, 1] = coords_a[parent, q, 1]
                coords_b[child, q, 2] = coords_a[parent, q, 2]

            coords_b[child, k, 0] = nx
            coords_b[child, k, 1] = ny
            coords_b[child, k, 2] = nz

            for q in range(table_size):
                keys_b[child, q] = keys_a[parent, q]
                counts_b[child, q] = counts_a[parent, q]
                used_b[child, q] = used_a[parent, q]

            if not ins(keys_b[child], counts_b[child],
                       used_b[child], table_size, nx, ny, nz):
                logw_b[child] = -np.inf
                continue

            logw_b[child] = new_logw

            # Russian roulette pruning.
            if logw_b[child] < low_log:
                if np.random.random() < prune_p:
                    logw_b[child] = -np.inf
                    prune_deaths += 1
                    continue
                logw_b[child] -= math.log(1.0 - prune_p)

            # Deterministic enrichment. Parent total weight is exactly
            # preserved by assigning W/k to every clone.
            if logw_b[child] > high_log:
                ratio = math.exp(min(20.0,
                                     logw_b[child] - high_log))
                nk = int(math.ceil(ratio))
                if nk < 2:
                    nk = 2
                if nk > 4:
                    nk = 4

                split_logw = logw_b[child] - math.log(nk)
                logw_b[child] = split_logw

                made = 1
                for c in range(1, nk):
                    if new_pop >= max_pop:
                        overflow_events += 1
                        break

                    child2 = new_pop
                    new_pop += 1
                    clone_events += 1
                    made += 1

                    for q in range(k + 1):
                        coords_b[child2, q, 0] = coords_b[child, q, 0]
                        coords_b[child2, q, 1] = coords_b[child, q, 1]
                        coords_b[child2, q, 2] = coords_b[child, q, 2]

                    for q in range(table_size):
                        keys_b[child2, q] = keys_b[child, q]
                        counts_b[child2, q] = counts_b[child, q]
                        used_b[child2, q] = used_b[child, q]

                    logw_b[child2] = split_logw

        # Swap buffers for next growth level.
        tmp = coords_a
        coords_a = coords_b
        coords_b = tmp

        tmp = keys_a
        keys_a = keys_b
        keys_b = tmp

        tmp = counts_a
        counts_a = counts_b
        counts_b = tmp

        tmp = used_a
        used_a = used_b
        used_b = tmp

        tmp2 = logw_a
        logw_a = logw_b
        logw_b = tmp2

        # Clear output log weights for the next iteration.
        for i in range(max_pop):
            logw_b[i] = -np.inf

        pop = new_pop

        if pop == 0:
            break

    if pop == 0:
        return (
            -np.inf, 0.0, 1.0, 0.0,
            0, prune_deaths, clone_events, overflow_events
        )

    vals = logw_a[:pop]
    finite = np.isfinite(vals)
    vals = vals[finite]

    if vals.size == 0:
        return (
            -np.inf, 0.0, 1.0, 0.0,
            0, prune_deaths, clone_events, overflow_events
        )

    vmax = np.max(vals)
    ww = np.exp(vals - vmax)
    s1 = np.sum(ww)
    s2 = np.sum(ww * ww)

    logz = vmax + math.log(s1 / roots)
    ess = s1 * s1 / s2 if s2 > 0.0 else 0.0
    maxfrac = np.max(ww / s1) if s1 > 0.0 else 1.0

    # Population amplification relative to initial roots.
    pop_ratio = vals.size / roots

    return (
        logz, ess, maxfrac, pop_ratio,
        vals.size, prune_deaths, clone_events, overflow_events
    )


def run_perm_side(N, L, w, roots, max_pop, blocks, seed,
                  prune_p, cminus, cplus, pilot, workers):
    tasks = []
    for b in range(blocks):
        tasks.append((
            N, L, w, roots, max_pop,
            seed + 100000 * b,
            prune_p, cminus, cplus, pilot
        ))

    # Memory is dominated by the per-worker population arrays. Use one process
    # by default; explicit workers>1 is allowed but should be used cautiously.
    if workers == 1:
        return [perm_block(*task) for task in tasks]

    with ProcessPoolExecutor(max_workers=min(workers, blocks)) as pool:
        futures = [pool.submit(perm_block, *task) for task in tasks]
        return [f.result() for f in futures]


def summarise(results):
    logs = np.array([r[0] for r in results], float)
    ess = np.array([r[1] for r in results], float)
    maxw = np.array([r[2] for r in results], float)
    popratio = np.array([r[3] for r in results], float)
    finalpop = np.array([r[4] for r in results], int)
    pruned = np.array([r[5] for r in results], int)
    clones = np.array([r[6] for r in results], int)
    overflow = np.array([r[7] for r in results], int)

    good = np.isfinite(logs)
    if not np.any(good):
        return {
            "logZ": -np.inf,
            "logZ_err": np.nan,
            "block_logZ": logs.tolist(),
            "block_ESS": ess.tolist(),
            "block_max_weight_fraction": maxw.tolist(),
            "block_population_ratio": popratio.tolist(),
            "block_final_population": finalpop.tolist(),
            "block_prune_deaths": pruned.tolist(),
            "block_clone_events": clones.tolist(),
            "block_overflow_events": overflow.tolist(),
            "min_ESS": 0.0,
            "max_weight_fraction": 1.0,
        }

    g = logs[good]
    return {
        "logZ": float(np.mean(g)),
        "logZ_err":
            float(np.std(g, ddof=1) / math.sqrt(len(g)))
            if len(g) > 1 else np.nan,
        "block_logZ": logs.tolist(),
        "block_ESS": ess.tolist(),
        "block_max_weight_fraction": maxw.tolist(),
        "block_population_ratio": popratio.tolist(),
        "block_final_population": finalpop.tolist(),
        "block_prune_deaths": pruned.tolist(),
        "block_clone_events": clones.tolist(),
        "block_overflow_events": overflow.tolist(),
        "min_ESS": float(np.nanmin(ess)),
        "max_weight_fraction": float(np.nanmax(maxw)),
    }


def make_pilot(args):
    Lc = nearest_even(args.Lr * args.Rg)
    Lm = Lc - args.delta_L
    Lp = Lc + args.delta_L

    t0 = time.time()
    zm = pilot_partial(
        args.N, Lm, args.w, args.pilot_roots, args.seed + 1
    )
    zp = pilot_partial(
        args.N, Lp, args.w, args.pilot_roots, args.seed + 2
    )
    dt = time.time() - t0

    payload = {
        "script_name": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "run_type": "PERM_threshold_pilot",
        "N": args.N,
        "w": args.w,
        "Rg": args.Rg,
        "Lr_requested": args.Lr,
        "Lr_realized": Lc / args.Rg,
        "L_center": Lc,
        "L_minus": Lm,
        "L_plus": Lp,
        "delta_L": args.delta_L,
        "pilot_roots": args.pilot_roots,
        "seed_minus": args.seed + 1,
        "seed_plus": args.seed + 2,
        "pilot_logZ_minus": zm.tolist(),
        "pilot_logZ_plus": zp.tolist(),
        "definition":
            "logZ_partial[k]=log(mean_root W_k I_alive) for k monomers",
        "runtime_seconds": dt,
    }

    path = Path(args.pilot_file)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8"
    )
    print(f"Pilot written: {path.resolve()}")


def load_pilot(path, args, Lc, Lm, Lp):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pilot file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    checks = {
        "N": args.N,
        "w": args.w,
        "Rg": args.Rg,
        "Lr_requested": args.Lr,
        "L_center": Lc,
        "L_minus": Lm,
        "L_plus": Lp,
        "delta_L": args.delta_L,
    }
    tol = 1e-12
    for key, expected in checks.items():
        if key not in data:
            raise RuntimeError(
                f"Pilot metadata missing '{key}': {path}"
            )
        actual = data[key]
        if isinstance(expected, (int, np.integer)):
            if int(actual) != int(expected):
                raise RuntimeError(
                    f"Pilot mismatch for {key}: pilot={actual}, "
                    f"production={expected}\nFile: {path}"
                )
        else:
            if not np.isclose(float(actual), float(expected),
                              rtol=tol, atol=tol):
                raise RuntimeError(
                    f"Pilot mismatch for {key}: pilot={actual}, "
                    f"production={expected}\nFile: {path}"
                )

    if data.get("run_type") != "PERM_threshold_pilot":
        raise RuntimeError(
            f"Invalid pilot run_type={data.get('run_type')!r}: {path}"
        )

    if data.get("script_version") != SCRIPT_VERSION:
        raise RuntimeError(
            "Pilot script version does not match production script: "
            f"pilot={data.get('script_version')!r}, "
            f"production={SCRIPT_VERSION!r}\nFile: {path}"
        )

    zm = np.asarray(data["pilot_logZ_minus"], dtype=np.float64)
    zp = np.asarray(data["pilot_logZ_plus"], dtype=np.float64)
    if zm.shape[0] != args.N + 1 or zp.shape[0] != args.N + 1:
        raise RuntimeError(
            f"Pilot array length mismatch: minus={zm.shape[0]}, "
            f"plus={zp.shape[0]}, expected={args.N + 1}\nFile: {path}"
        )

    return zm, zp


def run_perm(args):
    Lc = nearest_even(args.Lr * args.Rg)
    Lm = Lc - args.delta_L
    Lp = Lc + args.delta_L

    pilot_m, pilot_p = load_pilot(args.pilot_file, args, Lc, Lm, Lp)

    t0 = time.time()

    rm = run_perm_side(
        args.N, Lm, args.w, args.roots, args.max_population,
        args.blocks, args.seed + 1_000_000,
        args.prune_probability, args.C_minus, args.C_plus,
        pilot_m, args.workers
    )
    rp = run_perm_side(
        args.N, Lp, args.w, args.roots, args.max_population,
        args.blocks, args.seed + 2_000_000,
        args.prune_probability, args.C_minus, args.C_plus,
        pilot_p, args.workers
    )

    sm = summarise(rm)
    sp = summarise(rp)

    bf = np.array([
        (rp[i][0] - rm[i][0]) / (2.0 * args.delta_L)
        for i in range(args.blocks)
        if np.isfinite(rp[i][0]) and np.isfinite(rm[i][0])
    ], dtype=float)

    force = float(np.mean(bf)) if bf.size else np.nan
    ferr = (
        float(np.std(bf, ddof=1) / math.sqrt(bf.size))
        if bf.size > 1 else np.nan
    )

    total_overflow = int(np.sum(sm["block_overflow_events"])
                         + np.sum(sp["block_overflow_events"]))
    if total_overflow != 0:
        raise RuntimeError(
            "PERM population overflow detected. This run is invalid for "
            "production because the current implementation discards branches "
            "when the population cap is reached. "
            f"Total overflow events={total_overflow}"
        )

    result = {
        "N": args.N,
        "w": args.w,
        "L_ratio_requested": args.Lr,
        "L_ratio_realized": Lc / args.Rg,
        "L_center": Lc,
        "L_minus": Lm,
        "L_plus": Lp,
        "delta_L": args.delta_L,
        "Rg": args.Rg,
        "Rg_err": args.Rg_err,
        "roots_per_block": args.roots,
        "blocks": args.blocks,
        "max_population": args.max_population,
        "pruning_enabled": True,
        "prune_probability": args.prune_probability,
        "C_minus": args.C_minus,
        "C_plus": args.C_plus,
        "pilot_file": str(args.pilot_file),
        "logZ_minus": sm["logZ"],
        "logZ_minus_err": sm["logZ_err"],
        "logZ_plus": sp["logZ"],
        "logZ_plus_err": sp["logZ_err"],
        "deltaF": sp["logZ"] - sm["logZ"],
        "force": force,
        "force_err": ferr,
        "fRg": force * args.Rg,
        "fRg_err": ferr * args.Rg if np.isfinite(ferr) else np.nan,
        "block_force": bf.tolist(),
        "quality_flag": quality_flag(
            min(sm["min_ESS"], sp["min_ESS"]),
            max(sm["max_weight_fraction"],
                sp["max_weight_fraction"])
        ),
        "sign_selected": False,
        "excluded": False,
        "min_ESS_minus": sm["min_ESS"],
        "min_ESS_plus": sp["min_ESS"],
        "max_weight_fraction_minus": sm["max_weight_fraction"],
        "max_weight_fraction_plus": sp["max_weight_fraction"],
        "minus_diagnostics": sm,
        "plus_diagnostics": sp,
        "runtime_seconds": time.time() - t0,
        "sampler":
            "genuine PERM population control with fixed independent "
            "pilot thresholds",
        "geometry_convention":
            "absorbing walls z=0,L; even integer L; tether z=L/2; "
            "accessible sites z=1,...,L-1",
        "derivative_definition":
            "[lnZ(L+delta_L)-lnZ(L-delta_L)]/(2*delta_L)",
        "ensemble_definition":
            "Center-tethered Domb–Joyce chain in absorbing slit; "
            "survival/Rosenbluth ensemble",
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "master_force_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    (out / "production_manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=True),
        encoding="utf-8"
    )

    print(
        f"N={args.N} L/Rg={Lc/args.Rg:.6f} "
        f"fRg={result['fRg']:.8g} +/- {result['fRg_err']:.4g} "
        f"quality={result['quality_flag']} "
        f"runtime={result['runtime_seconds']:.1f}s"
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mode", choices=["selftest", "pilot", "perm"],
                   default="selftest")
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--Lr", type=float, default=2.8)
    p.add_argument("--w", type=float, default=0.1)
    p.add_argument("--Rg", type=float, default=24.357089468)
    p.add_argument("--Rg-err", type=float, default=0.0)
    p.add_argument("--delta-L", type=int, default=2)

    p.add_argument("--pilot-roots", type=int, default=5000)
    p.add_argument("--pilot-file",
                   default="S14_PERM_pilot_N2000_Lr2p8.json")

    p.add_argument("--roots", type=int, default=256)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--max-population", type=int, default=1024)

    p.add_argument("--prune-probability", type=float, default=0.5)
    p.add_argument("--C-minus", type=float, default=0.5)
    p.add_argument("--C-plus", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=50_000_000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-dir", default="S14_PERM_v09_output")

    args = p.parse_args()

    numba_selftest()

    if args.mode == "selftest":
        return

    if args.N < 2:
        raise SystemExit("N must be > 1")
    if args.blocks < 2:
        raise SystemExit("blocks must be >= 2")
    if args.roots < 1:
        raise SystemExit("roots must be >= 1")
    if args.max_population < args.roots:
        raise SystemExit("max-population must be >= roots")
    if not (0.0 < args.prune_probability < 1.0):
        raise SystemExit("prune-probability must be in (0,1)")
    if args.C_minus <= 0 or args.C_plus <= 1:
        raise SystemExit("Require C-minus>0 and C-plus>1")

    if args.mode == "pilot":
        make_pilot(args)
    elif args.mode == "perm":
        run_perm(args)


if __name__ == "__main__":
    main()
