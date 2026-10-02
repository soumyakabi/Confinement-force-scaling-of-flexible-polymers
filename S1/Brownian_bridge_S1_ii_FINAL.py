#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S1(ii) — Brownian-bridge correction audit
==============================================================

Purpose
-------
Audit the discrete-step Brownian survival correction used for the Gaussian
absorbing-slit benchmark.  The calculation compares:

  (1) endpoint-only survival;
  (2) exact conditional interval survival at each step, using
      K_interval(x,y;dt) / K_free(x,y;dt);
  (3) the exact N-step absorbing-interval eigenmode survival probability.

The exact interval kernel is evaluated with the two-wall image series.  An
independent eigenfunction-series calculation is used as a numerical audit of
the image kernel.

Conventions
-----------
- N = number of Gaussian contour steps (default 200)
- step variance per coordinate: a^2/3
- D_z = a^2/6 for one-dimensional z projection
- absorbing walls: z=0 and z=L
- co-moving midpoint tether: z0=L/2
- Rg = a*sqrt(N/6)
- confinement free energy: F = -kBT ln S

No smoothing, artificial floor, sign filtering, or post-hoc point selection is
used.  This script is a validation/audit tool; it is not a production
polymer-force estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = "Brownian_bridge_S1_ii_FINAL.py"
SCRIPT_VERSION = "2.0"
DEFAULT_RATIOS = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0)
DEFAULT_BLOCKS = 20
DEFAULT_WALKS = 200_000
DEFAULT_IMAGES = 40
DEFAULT_MODES = 2000
DEFAULT_SEED = 24680


def gaussian_rg(N: int, a: float = 1.0) -> float:
    return float(a * np.sqrt(N / 6.0))


def safe_sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size >= 2 else np.nan


def jackknife_error(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        return np.nan
    loo = (np.sum(v) - v) / (n - 1)
    return float(np.sqrt((n - 1) / n * np.sum((loo - np.mean(loo)) ** 2)))


def exact_survival_eigenmode(N: int, L: float, a: float = 1.0,
                             tether_fraction: float = 0.5,
                             n_modes: int = DEFAULT_MODES,
                             tol: float = 1e-15) -> float:
    """Exact N-step Gaussian survival probability for midpoint tethering."""
    if L <= 0 or not (0.0 < tether_fraction < 1.0):
        return np.nan
    x0 = tether_fraction * L
    D = a * a / 6.0
    S = 0.0
    for n in range(1, n_modes + 1, 2):
        k = n * np.pi / L
        term = (4.0 / (n * np.pi)) * np.sin(k * x0) * np.exp(-k * k * D * N)
        S += term
        if n > 21 and abs(term) < tol:
            break
    return float(S)


def interval_kernel_images(x: float, y: float, L: float, sigma2: float,
                           n_images: int = DEFAULT_IMAGES) -> float:
    """Exact absorbing-interval transition kernel divided by free kernel."""
    if not (0.0 < x < L and 0.0 < y < L):
        return 0.0
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")

    e_free = -((y - x) ** 2) / (2.0 * sigma2)
    if e_free < -745.0:
        return 0.0

    image_sum = 0.0
    for m in range(-n_images, n_images + 1):
        direct = np.exp(-((y - x + 2.0 * m * L) ** 2) / (2.0 * sigma2))
        reflected = np.exp(-((y + x + 2.0 * m * L) ** 2) / (2.0 * sigma2))
        image_sum += direct - reflected

    free_part = np.exp(e_free)
    ratio = image_sum / free_part
    # Roundoff can produce tiny excursions outside [0,1].
    return float(np.clip(ratio, 0.0, 1.0))


def exact_interval_kernel_eigen(x: float, y: float, L: float, sigma2: float,
                                n_modes: int = 2000) -> float:
    """Exact absorbing interval kernel divided by free kernel via eigenmodes."""
    if not (0.0 < x < L and 0.0 < y < L):
        return 0.0
    D = sigma2 / 2.0
    n = np.arange(1, n_modes + 1, dtype=float)
    k = n * np.pi / L
    kernel = (2.0 / L) * np.sum(
        np.sin(k * x) * np.sin(k * y) * np.exp(-D * k * k)
    )
    free = np.exp(-((y - x) ** 2) / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2)
    kernel *= 1.0  # kernel and free share consistent units; ratio below
    kernel = np.real(kernel)
    # The eigenfunction expansion above is a density with unit prefactor 2/L;
    # free is the corresponding normalized Gaussian density.
    free_density = free
    return float(np.clip(kernel / free_density, 0.0, 1.0))


def run_survival_blocks(N: int, L: float, sigma_step: float, n_walks: int,
                        n_blocks: int, mode: str, seed: int,
                        tether_fraction: float = 0.5,
                        n_images: int = DEFAULT_IMAGES) -> dict:
    if mode not in {"endpoint", "interval"}:
        raise ValueError("mode must be 'endpoint' or 'interval'")
    if n_walks < n_blocks or n_walks % n_blocks:
        raise ValueError("n_walks must be divisible by n_blocks and >= n_blocks")

    rng = np.random.default_rng(seed)
    block_size = n_walks // n_blocks
    sigma2 = sigma_step * sigma_step
    x0 = tether_fraction * L
    block_survival = np.empty(n_blocks, dtype=float)
    block_survivors = np.zeros(n_blocks, dtype=int)

    for b in range(n_blocks):
        survivors = 0
        for _ in range(block_size):
            x = x0
            alive = True
            for _step in range(N):
                y = x + rng.normal(0.0, sigma_step)
                if y <= 0.0 or y >= L:
                    alive = False
                    break
                if mode == "interval":
                    p_survive = interval_kernel_images(x, y, L, sigma2, n_images)
                    if rng.random() > p_survive:
                        alive = False
                        break
                x = y
            if alive:
                survivors += 1
        block_survivors[b] = survivors
        block_survival[b] = survivors / block_size

    return {
        "mode": mode,
        "n_walks": n_walks,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "block_survival": block_survival,
        "block_survivors": block_survivors,
        "survival": float(np.mean(block_survival)),
        "survival_sem": safe_sem(block_survival),
        "survival_jackknife_err": jackknife_error(block_survival),
        "n_survived": int(np.sum(block_survivors)),
    }


def free_energy_from_survival(result: dict, kBT: float = 1.0):
    p = float(result["survival"])
    if p <= 0 or not np.isfinite(p):
        return np.nan, np.nan, "UNRESOLVED_ZERO_SURVIVORS"
    ep = max(result["survival_sem"], result["survival_jackknife_err"])
    F = -kBT * np.log(p)
    Ferr = kBT * ep / p if np.isfinite(ep) else np.nan
    return float(F), float(Ferr), "RESOLVED"


def kernel_self_test(seed: int = DEFAULT_SEED) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        L = rng.uniform(4.0, 20.0)
        x = rng.uniform(0.05 * L, 0.95 * L)
        y = np.clip(x + rng.normal(0.0, 1.0 / np.sqrt(3.0)), 1e-6, L - 1e-6)
        p_img = interval_kernel_images(x, y, L, 1.0 / 3.0, 60)
        p_eig = exact_interval_kernel_eigen(x, y, L, 1.0 / 3.0, 1000)
        if not np.isfinite(p_img) or not np.isfinite(p_eig):
            raise RuntimeError("Non-finite kernel self-test value")
        if abs(p_img - p_eig) > 5e-8:
            raise RuntimeError(
                f"Image/eigen bridge-kernel mismatch: {abs(p_img-p_eig):.3e}"
            )

    # Exact survival must lie in (0,1) for representative widths.
    for ratio in (1.2, 2.0, 4.0):
        L = ratio * gaussian_rg(200)
        S = exact_survival_eigenmode(200, L)
        if not (0.0 < S < 1.0):
            raise RuntimeError(f"Invalid exact survival probability at L/Rg={ratio}: {S}")
    print("Self-test passed: interval-kernel and eigenmode checks are consistent.")


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Brownian-bridge correction audit for Gaussian absorbing-slit benchmark.",
    )
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--kBT", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--walks", type=int, default=DEFAULT_WALKS)
    ap.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    ap.add_argument("--modes", type=int, default=DEFAULT_MODES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output-dir", type=Path, default=Path("BrownianBridge_S1ii_FINAL"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        kernel_self_test(args.seed)
        return
    if args.N < 2 or args.a <= 0 or args.kBT <= 0:
        raise ValueError("Require N>=2, a>0, kBT>0")
    if args.blocks < 3:
        raise ValueError("Require at least 3 blocks")
    if args.images < 5 or args.modes < 50:
        raise ValueError("Use at least 5 image terms and 50 eigenmodes")

    n_walks = 20_000 if args.smoke else args.walks
    n_blocks = 10 if args.smoke else args.blocks
    n_walks -= n_walks % n_blocks

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rg = gaussian_rg(args.N, args.a)
    sigma_step = args.a / np.sqrt(3.0)
    ratios = np.array(DEFAULT_RATIOS, dtype=float)

    print("=" * 90)
    print("SUPPLEMENTARY FIG. S1(ii) — BROWNIAN-BRIDGE CORRECTION AUDIT")
    print("=" * 90)
    print(f"N={args.N}, a={args.a}, Rg={rg:.8f}")
    print(f"walks/point={n_walks}, blocks={n_blocks}, images={args.images}, modes={args.modes}")
    print("Absorbing walls z=0,L; co-moving midpoint tether z0=L/2")
    if args.smoke:
        print("SMOKE MODE: not for quantitative inference")

    kernel_self_test(args.seed)

    rows = []
    for i, ratio in enumerate(ratios):
        L = ratio * rg
        S_exact = exact_survival_eigenmode(args.N, L, args.a, 0.5, args.modes)
        F_exact = -args.kBT * np.log(S_exact) if S_exact > 0 else np.nan

        endpoint = run_survival_blocks(
            args.N, L, sigma_step, n_walks, n_blocks, "endpoint",
            args.seed + 10_000 * i + 1, 0.5, args.images
        )
        interval = run_survival_blocks(
            args.N, L, sigma_step, n_walks, n_blocks, "interval",
            args.seed + 10_000 * i + 2, 0.5, args.images
        )

        F_ep, Ferr_ep, status_ep = free_energy_from_survival(endpoint, args.kBT)
        F_int, Ferr_int, status_int = free_energy_from_survival(interval, args.kBT)

        rows.append({
            "N": args.N,
            "a": args.a,
            "kBT": args.kBT,
            "Rg": rg,
            "L": L,
            "L_over_Rg": ratio,
            "boundary": "absorbing",
            "tether": "z0=L/2 (co-moving midpoint)",
            "walks_per_method": n_walks,
            "n_blocks": n_blocks,
            "n_images": args.images,
            "n_modes": args.modes,
            "S_exact": S_exact,
            "F_exact": F_exact,
            "S_endpoint": endpoint["survival"],
            "S_endpoint_sem": endpoint["survival_sem"],
            "S_endpoint_jackknife_err": endpoint["survival_jackknife_err"],
            "n_survived_endpoint": endpoint["n_survived"],
            "F_endpoint": F_ep,
            "F_endpoint_err": Ferr_ep,
            "F_endpoint_status": status_ep,
            "F_endpoint_percent_error": 100.0 * (F_ep - F_exact) / F_exact if np.isfinite(F_ep) else np.nan,
            "S_interval": interval["survival"],
            "S_interval_sem": interval["survival_sem"],
            "S_interval_jackknife_err": interval["survival_jackknife_err"],
            "n_survived_interval": interval["n_survived"],
            "F_interval": F_int,
            "F_interval_err": Ferr_int,
            "F_interval_status": status_int,
            "F_interval_percent_error": 100.0 * (F_int - F_exact) / F_exact if np.isfinite(F_int) else np.nan,
        })
        print(
            f"L/Rg={ratio:4.1f}  F_exact={F_exact:8.5f}  "
            f"endpoint err={rows[-1]['F_endpoint_percent_error']:7.3f}%  "
            f"bridge err={rows[-1]['F_interval_percent_error']:7.3f}%"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir / f"FigS1ii_BrownianBridge_Audit_{stamp}"
    csv_path = base.with_name(base.name + "_points.csv")
    json_path = base.with_name(base.name + "_provenance.json")
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    ratio_arr = np.array([r["L_over_Rg"] for r in rows])
    ep_err = np.abs(np.array([r["F_endpoint_percent_error"] for r in rows]))
    int_err = np.abs(np.array([r["F_interval_percent_error"] for r in rows]))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.semilogy(ratio_arr, ep_err, "o-", label="Endpoint-only")
    ax.semilogy(ratio_arr, int_err, "s-", label="Exact interval-kernel bridge")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, label="1% reference")
    ax.set_xlabel(r"$L/R_g$")
    ax.set_ylabel(r"$|F_{\rm MC}-F_{\rm exact}|/F_{\rm exact}$ (%)")
    ax.set_title("Supplementary Fig. S1(ii): Brownian-bridge correction audit", fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "N": args.N, "a": args.a, "kBT": args.kBT, "Rg": rg,
        "sigma_step": sigma_step,
        "geometry": {
            "boundary": "absorbing z=0,L",
            "tether": "z0=L/2, co-moving midpoint",
            "increments": "Gaussian, variance a^2/3 per Cartesian coordinate per contour step",
        },
        "exact_reference": "absorbing-interval eigenmode survival probability",
        "bridge_correction": "K_interval/K_free from exact two-wall image kernel",
        "kernel_audit": "independent interval-kernel eigenfunction evaluation",
        "uncertainty": "independent block SEM and leave-one-block-out jackknife; larger propagated F error used",
        "no_smoothing": True,
        "no_artificial_floor": True,
        "sign_filtering": False,
        "smoke_mode": bool(args.smoke),
        "outputs": {"points_csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)},
        "points": rows,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=float)

    print("\nSaved:")
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
