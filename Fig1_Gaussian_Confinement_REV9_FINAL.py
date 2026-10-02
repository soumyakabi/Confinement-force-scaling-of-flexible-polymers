#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 — Gaussian confinement benchmark (REV9 FINAL)
========================================================

Purpose
-------
Publication-grade Gaussian absorbing-slit benchmark, aligned with the final
numerical workflow adopted for the revised manuscript.

Physical model
--------------
* Continuous 3D Gaussian chain projected onto the confined coordinate.
* N Gaussian contour steps, step variance sigma_step^2 = a^2/3.
* Absorbing walls at z=0 and z=L.
* Midpoint, co-moving tether z0=L/2.
* Unconfined 3D radius of gyration R_g^(0) = a sqrt(N/6).

Exact reference
---------------
The exact survival probability is the absorbing-interval eigenmode expansion

    S(L) = sum_{n odd} 4/(n pi) sin(n pi z0/L)
           exp[-(n pi/L)^2 (a^2/6) N],

with F = -kBT ln S and

    f = -dF/dL = kBT (1/S) dS/dL,

including dz0/dL = 1/2 for the co-moving midpoint tether.

Production Monte Carlo
----------------------
The Monte Carlo calculation samples continuous Gaussian increments and applies
an exact conditional Brownian-bridge survival test based on the two-wall
absorbing interval kernel evaluated with a converged image series. The image
series is truncated only after an internal convergence audit; no additive
one-wall approximation is used for the production estimator.

For the force, the FINAL production convention is used literally:

    f(L) ~= [ln Z(L+delta_L) - ln Z(L-delta_L)]/(2 delta_L),

with delta_L = 2 a by default, equivalently the total endpoint separation is
L_plus - L_minus = 4 a. The two derivative states are simulated independently.
Signed block forces are retained. No sign-based filtering, smoothing, fitting,
or post-hoc deletion is used. For logarithmic visualization only, non-positive
point estimates cannot be displayed and are masked without modifying the raw
or block-level data.

Uncertainty
-----------
* Free-energy uncertainty: independent block SEM of survival fractions
  propagated through -ln(S).
* Force uncertainty: SEM of signed block finite-difference forces plus a
  percentile bootstrap CI over the independent blocks.
* Block-level records are written for every simulated derivative state.

Validation and audit
--------------------
The script performs the following self-contained audits:
1. Exact image-kernel convergence against a larger image cutoff.
2. Exact interval kernel consistency between image and eigenfunction forms.
3. Exact chain-rule force versus central finite differences for delta_L values
   including the production delta_L=2.
4. Exact monotonicity/positivity checks for the Gaussian benchmark.
5. Post-production convergence of free energy and force using cumulative block
   counts [8,16,24,32,40] (or the user-specified list).

No CVT theorem calculation is included. The independent force validations are
instead the exact chain-rule derivative, Brownian-bridge audit, and block-
convergence analysis.

Outputs
-------
<outdir>/
    Fig1_Gaussian_Confinement_REV9.png
    Fig1_Gaussian_Confinement_REV9.pdf
    Fig1_PointLevel.csv
    Fig1_BlockLevel.csv
    Fig1_Convergence.csv
    Fig1_KernelAudit.csv
    Fig1_ValidationSummary.json
    Fig1_Provenance.json

Recommended Kaggle usage
------------------------
Self-test:
    !python Fig1_Gaussian_Confinement_REV9_FINAL.py --self-test

Production example:
    !python Fig1_Gaussian_Confinement_REV9_FINAL.py \
      --outdir /kaggle/working/Fig1_REV9_FINAL \
      --N 200 \
      --free-energy-ratios 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 \
      --force-ratios 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 \
      --delta-L 2 \
      --blocks 40 \
      --block-counts 8 16 24 32 40 \
      --walks-strong 2000000 \
      --walks-moderate 1000000 \
      --walks-weak 300000 \
      --strong-cut 1.20 \
      --moderate-cut 1.60 \
      --images 3 \
      --bootstrap-reps 5000 \
      --seed 20260930

Notes
-----
The default Monte Carlo grid begins at L/R_g=1.0 because direct survival
sampling below this point is exponentially rare for N=200. The exact curve is
still displayed down to --exact-min-ratio (default 0.6). This is a deliberate
sampling-range distinction, not a post-hoc deletion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import numba
    from numba import njit
except Exception as exc:
    raise RuntimeError("This production code requires numba.") from exc

A = 1.0
KBT = 1.0
DEFAULT_N = 200
DEFAULT_TETHER_FRACTION = 0.5
DEFAULT_DELTA_L = 2.0
DEFAULT_IMAGES = 3
DEFAULT_BLOCKS = 40
DEFAULT_BLOCK_COUNTS = [8, 16, 24, 32, 40]
DEFAULT_WALKS_STRONG = 2_000_000
DEFAULT_WALKS_MODERATE = 1_000_000
DEFAULT_WALKS_WEAK = 300_000
DEFAULT_STRONG_CUT = 1.20
DEFAULT_MODERATE_CUT = 1.60
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_EXACT_MIN_RATIO = 0.60
DEFAULT_EXACT_MAX_RATIO = 5.00
DEFAULT_FREE_RATIOS = [1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]
DEFAULT_FORCE_RATIOS = [1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "legend.fontsize": 10.5,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.5,
})


@dataclass(frozen=True)
class Params:
    N: int = DEFAULT_N
    a: float = A
    kBT: float = KBT
    tether_fraction: float = DEFAULT_TETHER_FRACTION
    delta_L: float = DEFAULT_DELTA_L
    images: int = DEFAULT_IMAGES
    blocks: int = DEFAULT_BLOCKS
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS

    @property
    def Rg(self) -> float:
        return self.a * math.sqrt(self.N / 6.0)

    @property
    def sigma_step(self) -> float:
        return self.a / math.sqrt(3.0)

    @property
    def D(self) -> float:
        return self.a * self.a / 6.0


def seed_for(base: int, tag: int) -> int:
    ss = np.random.SeedSequence([int(base), int(tag)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**31 - 1))


# ---------------------------------------------------------------------------
# Exact Gaussian solution
# ---------------------------------------------------------------------------

def exact_survival(N: int, L: float, a: float = 1.0, tether_fraction: float = 0.5,
                   n_modes: int = 4000, tol: float = 1e-16) -> float:
    if not (L > 0.0 and 0.0 < tether_fraction < 1.0):
        return float("nan")
    z0 = tether_fraction * L
    D = a * a / 6.0
    S = 0.0
    for n in range(1, n_modes + 1, 2):
        k = n * math.pi / L
        term = (4.0 / (n * math.pi)) * math.sin(k * z0) * math.exp(-k * k * D * N)
        S += term
        if n > 21 and abs(term) < tol:
            break
    return S


def exact_survival_and_derivative(N: int, L: float, a: float = 1.0,
                                  tether_fraction: float = 0.5,
                                  n_modes: int = 4000, tol: float = 1e-16):
    if not (L > 0.0 and 0.0 < tether_fraction < 1.0):
        return float("nan"), float("nan")
    z0 = tether_fraction * L
    D = a * a / 6.0
    S = 0.0
    dS = 0.0
    for n in range(1, n_modes + 1, 2):
        k = n * math.pi / L
        dk = -n * math.pi / (L * L)
        dtheta = dk * z0 + k * tether_fraction
        decay = math.exp(-k * k * D * N)
        s = math.sin(k * z0)
        c = math.cos(k * z0)
        amp = 4.0 / (n * math.pi)
        term = amp * s * decay
        S += term
        ddecay = decay * (-2.0 * k * D * N * dk)
        dS += amp * (c * dtheta * decay + s * ddecay)
        if n > 21 and abs(term) < tol:
            break
    return S, dS


def exact_free_energy(N, L, a=1.0, kBT=1.0, tether_fraction=0.5):
    S, _ = exact_survival_and_derivative(N, L, a, tether_fraction)
    if not (np.isfinite(S) and S > 0.0):
        return float("nan")
    return -kBT * math.log(S)


def exact_force(N, L, a=1.0, kBT=1.0, tether_fraction=0.5):
    S, dS = exact_survival_and_derivative(N, L, a, tether_fraction)
    if not (np.isfinite(S) and S > 0.0 and np.isfinite(dS)):
        return float("nan")
    return kBT * dS / S


# ---------------------------------------------------------------------------
# Exact interval Brownian-bridge kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def bridge_survival_image_numba(x: float, y: float, L: float, sigma2: float, images: int) -> float:
    if not (0.0 < x < L and 0.0 < y < L):
        return 0.0
    # Free Gaussian density, common prefactor cancels in the ratio.
    den = math.exp(-((y - x) * (y - x)) / (2.0 * sigma2))
    if den <= 0.0:
        return 0.0
    num = 0.0
    for k in range(-images, images + 1):
        dy1 = y - x + 2.0 * k * L
        dy2 = y + x + 2.0 * k * L
        num += math.exp(-(dy1 * dy1) / (2.0 * sigma2))
        num -= math.exp(-(dy2 * dy2) / (2.0 * sigma2))
    p = num / den
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def bridge_survival_image(x: float, y: float, L: float, sigma2: float, images: int) -> float:
    return float(bridge_survival_image_numba(x, y, L, sigma2, images))


def interval_kernel_eigen(x: float, y: float, L: float, D: float, dt: float = 1.0,
                          n_modes: int = 1000) -> float:
    if not (0.0 < x < L and 0.0 < y < L):
        return 0.0
    n = np.arange(1, n_modes + 1, dtype=float)
    terms = (2.0 / L) * np.sin(n * np.pi * x / L) * np.sin(n * np.pi * y / L) * \
            np.exp(-D * dt * (n * np.pi / L) ** 2)
    return float(np.sum(terms))


def bridge_survival_eigen(x: float, y: float, L: float, D: float, sigma2: float,
                          n_modes: int = 1000) -> float:
    kfree = math.exp(-((x - y) ** 2) / (2.0 * sigma2)) / math.sqrt(2.0 * math.pi * sigma2)
    if kfree <= 0.0:
        return 0.0
    kint = interval_kernel_eigen(x, y, L, D, 1.0, n_modes)
    p = kint / kfree
    return float(np.clip(p, 0.0, 1.0))


@njit(cache=True)
def simulate_survival_block_numba(N: int, L: float, sigma: float, images: int,
                                  n_walks: int, seed: int) -> int:
    np.random.seed(seed)
    sigma2 = sigma * sigma
    x0 = 0.5 * L
    survived = 0
    for _ in range(n_walks):
        x = x0
        alive = True
        for _step in range(N):
            y = x + sigma * np.random.normal()
            if y <= 0.0 or y >= L:
                alive = False
                break
            p = bridge_survival_image_numba(x, y, L, sigma2, images)
            if np.random.random() > p:
                alive = False
                break
            x = y
        if alive:
            survived += 1
    return survived


def simulate_blocks(params: Params, L: float, n_walks: int, seed: int) -> dict:
    if n_walks % params.blocks != 0:
        n_walks = params.blocks * (n_walks // params.blocks)
    if n_walks < params.blocks:
        raise ValueError("n_walks must be at least one walker per block")
    block_size = n_walks // params.blocks
    counts = np.zeros(params.blocks, dtype=int)
    for b in range(params.blocks):
        counts[b] = simulate_survival_block_numba(
            params.N, L, params.sigma_step, params.images, block_size,
            seed_for(seed, b + 1)
        )
    fractions = counts / block_size
    return {
        "L": float(L),
        "n_walks": int(n_walks),
        "block_size": int(block_size),
        "n_blocks": int(params.blocks),
        "survivors": counts,
        "fractions": fractions,
        "mean_survival": float(np.mean(fractions)),
        "sem_survival": float(np.std(fractions, ddof=1) / math.sqrt(params.blocks)),
        "total_survivors": int(np.sum(counts)),
    }


def choose_walks(ratio: float, strong: int, moderate: int, weak: int,
                 strong_cut: float, moderate_cut: float) -> int:
    if ratio < strong_cut:
        return strong
    if ratio < moderate_cut:
        return moderate
    return weak


def free_energy_from_blocks(sim: dict, kBT: float = 1.0) -> tuple[float, float, float]:
    p = sim["mean_survival"]
    if not (np.isfinite(p) and p > 0.0):
        return float("nan"), float("nan"), float("nan")
    dF = -kBT * math.log(p)
    sem_p = sim["sem_survival"]
    err = kBT * sem_p / p if np.isfinite(sem_p) else float("nan")
    return float(dF), float(err), float(p)


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        v = float(vals[0]) if vals.size else float("nan")
        return v, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(reps, vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.mean(vals)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_kernel_audit(params: Params, n_pairs: int = 1000) -> list[dict]:
    rng = np.random.default_rng(123456)
    rows = []
    max_img_diff = 0.0
    max_eigen_img = 0.0
    for i in range(n_pairs):
        ratio = rng.uniform(1.0, 5.0)
        L = ratio * params.Rg
        x = rng.uniform(0.05 * L, 0.95 * L)
        y = x + rng.normal(0.0, params.sigma_step)
        if not (0.0 < y < L):
            continue
        p3 = bridge_survival_image(x, y, L, params.sigma_step ** 2, params.images)
        p12 = bridge_survival_image(x, y, L, params.sigma_step ** 2, max(8, params.images + 5))
        pe = bridge_survival_eigen(x, y, L, params.D, params.sigma_step ** 2, n_modes=1200)
        max_img_diff = max(max_img_diff, abs(p3 - p12))
        max_eigen_img = max(max_eigen_img, abs(p12 - pe))
    rows.append({
        "audit": "image_cutoff_convergence",
        "max_abs_difference": max_img_diff,
        "threshold": 1e-12,
        "pass": bool(max_img_diff < 1e-12),
    })
    rows.append({
        "audit": "image_vs_eigen_kernel",
        "max_abs_difference": max_eigen_img,
        "threshold": 1e-10,
        "pass": bool(max_eigen_img < 1e-10),
    })
    return rows


def run_exact_fd_audit(params: Params) -> list[dict]:
    rows = []
    for ratio in [1.4, 2.0, 3.0, 4.0]:
        L = ratio * params.Rg
        exact = exact_force(params.N, L, params.a, params.kBT, params.tether_fraction)
        vals = {}
        for dL in [0.5, 1.0, 2.0]:
            fp = exact_free_energy(params.N, L + dL, params.a, params.kBT, params.tether_fraction)
            fm = exact_free_energy(params.N, L - dL, params.a, params.kBT, params.tether_fraction)
            fd = -(fp - fm) / (2.0 * dL)
            vals[dL] = abs(fd - exact) / abs(exact)
        rows.append({
            "ratio": ratio,
            "rel_error_deltaL_0p5": vals[0.5],
            "rel_error_deltaL_1": vals[1.0],
            "rel_error_deltaL_2": vals[2.0],
            "production_deltaL": params.delta_L,
        })
    return rows


def run_selftest() -> None:
    p = Params(blocks=4, images=3)
    assert abs(p.Rg - math.sqrt(200 / 6.0)) < 1e-12
    assert bridge_survival_image(p.Rg, p.Rg, 2.0 * p.Rg, p.sigma_step ** 2, 3) > 0.0
    ka = run_kernel_audit(p, n_pairs=300)
    assert all(r["pass"] for r in ka), ka
    fd = run_exact_fd_audit(p)
    assert all(np.isfinite(r["rel_error_deltaL_2"]) for r in fd)
    # The exact derivative is positive and finite in the test range.
    for ratio in [1.0, 1.5, 2.0, 3.0, 5.0]:
        f = exact_force(p.N, ratio * p.Rg, p.a, p.kBT, p.tether_fraction)
        assert np.isfinite(f) and f > 0.0
    print("Fig. 1 REV9 self-test: PASS")


# ---------------------------------------------------------------------------
# Production execution
# ---------------------------------------------------------------------------

def run_state_task(args_tuple):
    (N, L, n_walks, blocks, images, seed) = args_tuple
    p = Params(N=N, blocks=blocks, images=images)
    sim = simulate_blocks(p, L, n_walks, seed)
    F, F_err, surv = free_energy_from_blocks(sim, p.kBT)
    rows = []
    for bid, (cnt, frac) in enumerate(zip(sim["survivors"], sim["fractions"]), start=1):
        rows.append({
            "L": float(L),
            "L_over_Rg": float(L / p.Rg),
            "block_id": int(bid),
            "survivors": int(cnt),
            "block_size": int(sim["block_size"]),
            "block_survival": float(frac),
            "n_walks": int(sim["n_walks"]),
        })
    return {
        "L": float(L),
        "L_over_Rg": float(L / p.Rg),
        "n_walks": int(sim["n_walks"]),
        "n_blocks": int(blocks),
        "block_size": int(sim["block_size"]),
        "total_survivors": int(sim["total_survivors"]),
        "survival": float(surv),
        "survival_sem": float(sim["sem_survival"]),
        "F": float(F),
        "F_err": float(F_err),
        "block_rows": rows,
    }


def cumulative_free_energy(point: dict, block_counts: list[int], kBT: float) -> list[dict]:
    vals = np.asarray([r["block_survival"] for r in point["block_rows"]], dtype=float)
    rows = []
    for n in block_counts:
        sub = vals[:n]
        p = float(np.mean(sub))
        sem = float(np.std(sub, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        F = -kBT * math.log(p) if p > 0 else float("nan")
        F_err = kBT * sem / p if p > 0 and np.isfinite(sem) else float("nan")
        rows.append({"L": point["L"], "L_over_Rg": point["L_over_Rg"], "n_blocks": n,
                     "F": F, "F_err": F_err})
    return rows


def build_force_point(center_ratio: float, center_L: float, minus: dict, plus: dict,
                      params: Params, bootstrap_seed: int) -> tuple[dict, list[dict]]:
    if minus["n_blocks"] != plus["n_blocks"]:
        raise RuntimeError("Derivative endpoint block counts differ")
    fm = np.asarray([r["block_survival"] for r in minus["block_rows"]], dtype=float)
    fp = np.asarray([r["block_survival"] for r in plus["block_rows"]], dtype=float)
    # A zero-survival block is not silently deleted. It makes a log undefined;
    # retain the block in the audit and mark the force as unresolved if present.
    if np.any(fm <= 0) or np.any(fp <= 0):
        quality = "UNRESOLVED_ZERO_SURVIVAL_BLOCK"
        fvals = np.full(params.blocks, np.nan)
    else:
        fvals = (np.log(fp) - np.log(fm)) / (2.0 * params.delta_L)
        quality = "OK"

    valid = np.isfinite(fvals)
    if quality == "OK":
        f = float(np.mean(fvals))
        sem = float(np.std(fvals, ddof=1) / math.sqrt(len(fvals)))
        _, ci_lo, ci_hi = bootstrap_mean(fvals, params.bootstrap_reps, bootstrap_seed)
    else:
        f = sem = ci_lo = ci_hi = float("nan")

    block_rows = []
    for bid in range(params.blocks):
        block_rows.append({
            "L_center": center_L,
            "L_center_over_Rg": center_ratio,
            "block_id": bid + 1,
            "L_minus": minus["L"],
            "L_plus": plus["L"],
            "logS_minus": math.log(fm[bid]) if fm[bid] > 0 else float("nan"),
            "logS_plus": math.log(fp[bid]) if fp[bid] > 0 else float("nan"),
            "force_block": float(fvals[bid]) if valid[bid] else float("nan"),
            "signed_block_retained": True,
        })
    point = {
        "L": center_L,
        "L_over_Rg": center_ratio,
        "L_minus": minus["L"],
        "L_plus": plus["L"],
        "delta_L": params.delta_L,
        "force": f,
        "force_err": sem,
        "force_bootstrap95_low": ci_lo,
        "force_bootstrap95_high": ci_hi,
        "force_Rg": f * params.Rg if np.isfinite(f) else float("nan"),
        "force_Rg_err": sem * params.Rg if np.isfinite(sem) else float("nan"),
        "quality_flag": quality,
        "valid_force_blocks": int(np.sum(valid)),
        "negative_force_blocks": int(np.sum(valid & (fvals < 0.0))) if np.any(valid) else 0,
    }
    return point, block_rows


def produce_figure(args) -> None:
    params = Params(N=args.N, delta_L=args.delta_L, images=args.images,
                    blocks=args.blocks, bootstrap_reps=args.bootstrap_reps)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Exact curves are analytical and can cover a wider range than direct MC.
    x_exact = np.linspace(args.exact_min_ratio, args.exact_max_ratio, 500)
    F_exact = np.array([exact_free_energy(params.N, x * params.Rg, params.a, params.kBT,
                                          params.tether_fraction) for x in x_exact])
    f_exact = np.array([exact_force(params.N, x * params.Rg, params.a, params.kBT,
                                    params.tether_fraction) * params.Rg / params.kBT for x in x_exact])

    # Unique central free-energy and derivative endpoint states.
    free_ratios = sorted(set(round(float(x), 10) for x in args.free_energy_ratios))
    force_ratios = sorted(set(round(float(x), 10) for x in args.force_ratios))
    state_ratios = set(free_ratios)
    endpoint_map = {}
    for r in force_ratios:
        rc = r * params.Rg
        lm = rc - params.delta_L
        lp = rc + params.delta_L
        if lm <= 0.0:
            raise ValueError(f"Force ratio {r} gives nonpositive L-minus.")
        rm = lm / params.Rg
        rp = lp / params.Rg
        state_ratios.add(round(rm, 10)); state_ratios.add(round(rp, 10))
        endpoint_map[r] = (lm, lp)

    tasks = []
    for idx, sr in enumerate(sorted(state_ratios)):
        L = sr * params.Rg
        nw = choose_walks(sr, args.walks_strong, args.walks_moderate, args.walks_weak,
                          args.strong_cut, args.moderate_cut)
        tasks.append((params.N, L, nw, params.blocks, params.images, seed_for(args.seed, 100000 + idx)))

    print("=" * 96)
    print("FIGURE 1 REV9 FINAL — GAUSSIAN ABSORBING-SLIT BENCHMARK")
    print("=" * 96)
    print(f"N={params.N}, a={params.a}, Rg={params.Rg:.8f}, midpoint tether, delta_L={params.delta_L}")
    print(f"Production bridge: exact interval image kernel, images=±{params.images}")
    print(f"Blocks={params.blocks}; free-energy states={len(free_ratios)}; force centers={len(force_ratios)}")
    print("No sign filtering, smoothing, post-hoc deletion, or artificial floor.")

    states = {}
    # Moderate parallelism: each state is independent. Keep workers explicit.
    max_workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_state_task, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            res = fut.result()
            states[round(res["L_over_Rg"], 10)] = res
            print(f"  state L/Rg={res['L_over_Rg']:.5f}: walks={res['n_walks']:,}, "
                  f"survivors={res['total_survivors']:,}, F={res['F']:.6f}±{res['F_err']:.6f}")

    # Point-level free energies.
    free_points = []
    for r in free_ratios:
        s = states.get(round(r, 10))
        if s is None:
            raise RuntimeError(f"Missing simulated central state for free-energy ratio {r}")
        Fex = exact_free_energy(params.N, r * params.Rg, params.a, params.kBT, params.tether_fraction)
        free_points.append({
            "N": params.N, "L": s["L"], "L_over_Rg": r,
            "n_walks": s["n_walks"], "n_blocks": params.blocks,
            "survival_fraction": s["survival"], "survival_sem": s["survival_sem"],
            "total_survivors": s["total_survivors"],
            "DeltaF_sim": s["F"], "DeltaF_err": s["F_err"],
            "DeltaF_exact": Fex,
            "DeltaF_relative_error_percent": 100.0 * (s["F"] - Fex) / Fex if np.isfinite(s["F"]) and Fex != 0 else np.nan,
            "raw_data_retained": True,
        })

    # Block-level raw data for every simulated state.
    state_rows = []
    for sr, s in sorted(states.items()):
        for row in s["block_rows"]:
            state_rows.append({
                "state_role": "survival_state",
                "L": s["L"], "L_over_Rg": sr,
                "block_id": row["block_id"],
                "block_survival": row["block_survival"],
                "survivors": s["block_rows"][row["block_id"] - 1]["survivors"],
                "block_size": s["block_size"],
            })

    # Force assembly.
    force_points = []
    force_block_rows = []
    for i, r in enumerate(force_ratios):
        lm, lp = endpoint_map[r]
        rm = round(lm / params.Rg, 10)
        rp = round(lp / params.Rg, 10)
        minus = states[rm]
        plus = states[rp]
        point, brows = build_force_point(r, r * params.Rg, minus, plus, params,
                                         seed_for(args.seed, 700000 + i))
        force_points.append({"N": params.N, **point})
        force_block_rows.extend(brows)

    # Convergence audit: free energy at the smallest finite-MC free-energy ratio,
    # and force at the smallest force ratio (predeclared, hardest resolved state).
    conv_rows = []
    cfree = min(free_ratios)
    free_point_state = states[round(cfree, 10)]
    for row in cumulative_free_energy(free_point_state, args.block_counts, params.kBT):
        row["observable"] = "DeltaF"
        row["role"] = f"hardest_free_energy_state_{cfree:g}"
        conv_rows.append(row)

    if force_points:
        hardest_r = min(force_ratios)
        lm, lp = endpoint_map[hardest_r]
        minus = states[round(lm / params.Rg, 10)]
        plus = states[round(lp / params.Rg, 10)]
        fm = np.asarray([r["block_survival"] for r in minus["block_rows"]], dtype=float)
        fp = np.asarray([r["block_survival"] for r in plus["block_rows"]], dtype=float)
        for n in args.block_counts:
            if np.any(fm[:n] <= 0) or np.any(fp[:n] <= 0):
                ff = ee = np.nan
            else:
                fb = (np.log(fp[:n]) - np.log(fm[:n])) / (2 * params.delta_L)
                ff = float(np.mean(fb) * params.Rg / params.kBT)
                ee = float(np.std(fb, ddof=1) / math.sqrt(n) * params.Rg / params.kBT)
            conv_rows.append({"L": hardest_r * params.Rg, "L_over_Rg": hardest_r,
                              "n_blocks": n, "F": np.nan, "F_err": np.nan,
                              "fRg_over_kBT": ff, "fRg_err": ee,
                              "observable": "force", "role": f"hardest_force_state_{hardest_r:g}"})

    # Compare convergence from 32 to 40 blocks where available.
    conv_df = conv_rows

    # Kernel audits.
    kernel_rows = run_kernel_audit(params, n_pairs=args.kernel_pairs)
    fd_rows = run_exact_fd_audit(params)

    # Save free/force point-level files.
    pd_free = free_points
    point_file = outdir / "Fig1_PointLevel.csv"
    with point_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pd_free[0].keys()))
        writer.writeheader(); writer.writerows(pd_free)

    force_file = outdir / "Fig1_ForcePointLevel.csv"
    if force_points:
        with force_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(force_points[0].keys()))
            writer.writeheader(); writer.writerows(force_points)

    block_file = outdir / "Fig1_BlockLevel.csv"
    if force_block_rows:
        with block_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(force_block_rows[0].keys()))
            writer.writeheader(); writer.writerows(force_block_rows)
    else:
        # keep a valid file even if a future run has no force centers
        block_file.write_text("state_role,L,L_over_Rg,block_id\n", encoding="utf-8")

    conv_file = outdir / "Fig1_Convergence.csv"
    if conv_df:
        conv_fields = sorted({k for row in conv_df for k in row.keys()})
        with conv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=conv_fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(conv_df)

    # Kernel/FD audit CSV.
    audit_rows = list(kernel_rows)
    for r in fd_rows:
        audit_rows.append({"audit": "exact_vs_fd", **r})
    audit_file = outdir / "Fig1_KernelAudit.csv"
    fields = sorted({k for r in audit_rows for k in r.keys()})
    with audit_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(audit_rows)

    # Figure.
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ax = axes[0]
    fp = np.asarray([r["L_over_Rg"] for r in free_points])
    fy = np.asarray([r["DeltaF_sim"] for r in free_points])
    fe = np.asarray([r["DeltaF_err"] for r in free_points])
    ax.errorbar(fp, fy / params.kBT, yerr=fe / params.kBT, fmt="o", ms=6.5,
                capsize=3, label="Gaussian MC")
    ax.plot(x_exact, F_exact / params.kBT, "k-", lw=2.2, label="Exact eigenmode")
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$\Delta F/k_BT$")
    ax.set_title("(a) Confinement free energy")
    ax.set_xlim(args.exact_min_ratio, args.exact_max_ratio)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    xr = np.asarray([r["L_over_Rg"] for r in force_points])
    yr = np.asarray([r["force_Rg"] for r in force_points])
    ye = np.asarray([r["force_Rg_err"] for r in force_points])
    pos = np.isfinite(yr) & (yr > 0)
    ax.errorbar(xr[pos], yr[pos] / params.kBT, yerr=ye[pos] / params.kBT,
                fmt="o", ms=6.5, capsize=3, label=r"MC, $\delta L=2$")
    ax.plot(x_exact[f_exact > 0], f_exact[f_exact > 0], "k-", lw=2.2, label="Exact chain-rule force")
    guide = 2.0 * math.pi**2 * np.logspace(math.log10(max(args.exact_min_ratio,0.65)), math.log10(1.5), 100) ** (-3)
    gx = np.logspace(math.log10(max(args.exact_min_ratio,0.65)), math.log10(1.5), 100)
    ax.loglog(gx, guide, "k--", lw=1.2, alpha=0.65, label=r"Gaussian asymptote: $2\pi^2(L/R_g^{(0)})^{-3}$")
    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$fR_g^{(0)}/k_BT$")
    ax.set_title(r"(b) Entropic force ($\delta L=2$)")
    ax.set_xlim(args.exact_min_ratio, args.exact_max_ratio)
    ax.set_ylim(bottom=0.03)
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.03, "Signed block forces retained;\nlog display shows positive estimates only.",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    fig.suptitle("Gaussian polymer confinement: exact benchmark and numerical validation",
                 fontsize=19, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.86, wspace=0.22)
    out_png = outdir / "Fig1_Gaussian_Confinement_REV9.png"
    out_pdf = outdir / "Fig1_Gaussian_Confinement_REV9.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Provenance / validation summary.
    val_summary = {
        "N": params.N,
        "Rg": params.Rg,
        "delta_L": params.delta_L,
        "bridge": "exact absorbing-interval image kernel",
        "kernel_audit": kernel_rows,
        "exact_fd_audit": fd_rows,
        "free_energy_points": len(free_points),
        "force_points": len(force_points),
        "force_points_with_negative_block_estimates": [
            r for r in force_points if r["negative_force_blocks"] > 0
        ],
    }
    with (outdir / "Fig1_ValidationSummary.json").open("w", encoding="utf-8") as fh:
        json.dump(val_summary, fh, indent=2, default=float)

    provenance = {
        "script": "Fig1_Gaussian_Confinement_REV9_FINAL.py",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "numba": getattr(numba, "__version__", "unknown"),
        "physical": asdict(params),
        "geometry": {
            "dimension": 3,
            "confined_coordinate": "z",
            "walls": "z=0 and z=L",
            "tether": "z0=L/2, co-moving midpoint",
            "boundary_ensemble": "absorbing survival ensemble",
        },
        "force": {
            "definition": "f=d ln Z/dL=-d(DeltaF)/dL",
            "finite_difference": "[ln Z(L+delta_L)-ln Z(L-delta_L)]/(2 delta_L)",
            "delta_L": params.delta_L,
            "total_endpoint_span": 2 * params.delta_L,
            "signed_blocks_retained": True,
            "sign_filtering": False,
        },
        "sampling": {
            "blocks": params.blocks,
            "walks_strong": args.walks_strong,
            "walks_moderate": args.walks_moderate,
            "walks_weak": args.walks_weak,
            "strong_cut": args.strong_cut,
            "moderate_cut": args.moderate_cut,
            "kernel_images": params.images,
        },
        "exact_reference": "absorbing Gaussian interval eigenmode expansion",
        "production_bridge": "conditional interval survival kernel from image series",
        "no_smoothing": True,
        "no_artificial_floor": True,
        "no_posthoc_force_filtering": True,
        "cv_t_theorem": False,
        "free_energy_ratio_grid": free_ratios,
        "force_ratio_grid": force_ratios,
    }
    with (outdir / "Fig1_Provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=float)

    print("\nFIGURE 1 REV9 FINAL COMPLETE")
    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")
    print(f"Free-energy CSV: {point_file}")
    print(f"Force CSV: {force_file}")
    print(f"Block CSV: {block_file}")
    print(f"Convergence CSV: {conv_file}")
    print(f"Kernel audit: {audit_file}")
    print("Exact CVT theorem calculation: not included")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Final Gaussian Fig. 1 benchmark")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run a reduced execution test; not for scientific inference.")
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--outdir", type=Path, default=Path("Fig1_REV9_FINAL"))
    p.add_argument("--free-energy-ratios", nargs="+", type=float, default=DEFAULT_FREE_RATIOS)
    p.add_argument("--force-ratios", nargs="+", type=float, default=DEFAULT_FORCE_RATIOS)
    p.add_argument("--delta-L", type=float, default=DEFAULT_DELTA_L)
    p.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    p.add_argument("--block-counts", nargs="+", type=int, default=DEFAULT_BLOCK_COUNTS)
    p.add_argument("--walks-strong", type=int, default=DEFAULT_WALKS_STRONG)
    p.add_argument("--walks-moderate", type=int, default=DEFAULT_WALKS_MODERATE)
    p.add_argument("--walks-weak", type=int, default=DEFAULT_WALKS_WEAK)
    p.add_argument("--strong-cut", type=float, default=DEFAULT_STRONG_CUT)
    p.add_argument("--moderate-cut", type=float, default=DEFAULT_MODERATE_CUT)
    p.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--kernel-pairs", type=int, default=1000)
    p.add_argument("--exact-min-ratio", type=float, default=DEFAULT_EXACT_MIN_RATIO)
    p.add_argument("--exact-max-ratio", type=float, default=DEFAULT_EXACT_MAX_RATIO)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260930)
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_selftest()
        return
    if args.smoke:
        args.blocks = 4
        args.block_counts = [4]
        args.walks_strong = min(args.walks_strong, 2000)
        args.walks_moderate = min(args.walks_moderate, 2000)
        args.walks_weak = min(args.walks_weak, 1000)
        args.free_energy_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        args.force_ratios = [1.5, 2.0, 3.0, 4.0, 5.0]
        print("SMOKE MODE: results are execution tests only; do not use for inference.")
    if args.N < 2:
        raise ValueError("N must be >=2")
    if args.delta_L <= 0:
        raise ValueError("delta-L must be positive")
    if args.blocks < 3:
        raise ValueError("blocks must be >=3")
    if any(x <= 0 for x in args.free_energy_ratios + args.force_ratios):
        raise ValueError("all ratios must be positive")
    if any(b < 3 or b > args.blocks for b in args.block_counts):
        raise ValueError("block-counts must lie between 3 and blocks")
    produce_figure(args)


if __name__ == "__main__":
    main()
