#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S2(i) — Gaussian force finite-difference audit
==================================================================

Purpose
-------
Validate the finite-difference force convention against the exact Gaussian
chain-rule force for the same co-moving midpoint tether used throughout the
final workflow.

The final production convention uses a fixed absolute lattice-width increment
Delta L = 2.  Therefore the audit compares absolute Delta L values, including
Delta L=2 explicitly, rather than relying only on fractional Delta L/R_g.

Model
-----
- N Gaussian contour steps (default 200)
- a=1, kBT=1
- absorbing walls z=0,L
- co-moving midpoint tether z0=L/2
- Rg=a*sqrt(N/6)
- F=-kBT ln S
- f=d ln Z/dL=-dF/dL

No smoothing, fitting, sign filtering, artificial floors, or post-hoc
selection is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = "Gaussian_force_vs_reduced_separation_S2_i_FINAL.py"
SCRIPT_VERSION = "2.0"
DEFAULT_N = 200
DEFAULT_A = 1.0
DEFAULT_KBT = 1.0
DEFAULT_TETHER_FRACTION = 0.5
DEFAULT_MODES = 4000
DEFAULT_RATIOS = np.linspace(0.80, 4.00, 33)
DEFAULT_DELTAS = (1.0, 2.0, 4.0)


def gaussian_rg(N: int, a: float = DEFAULT_A) -> float:
    return float(a * np.sqrt(N / 6.0))


def survival_and_derivative(N: int, L: float, a: float = 1.0,
                            tether_fraction: float = 0.5,
                            n_modes: int = DEFAULT_MODES,
                            tolerance: float = 1e-15):
    """Exact S and dS/dL for a co-moving tether z0=tether_fraction*L."""
    if L <= 0 or not (0.0 < tether_fraction < 1.0):
        return np.nan, np.nan

    D = a * a / 6.0
    S = 0.0
    dS_dL = 0.0

    # For z0=t*L, theta=n*pi*z0/L=n*pi*t is L-independent.
    # Hence d(theta)/dL=0 exactly; only the exponential contributes.
    theta_factor = tether_fraction
    for n in range(1, n_modes + 1, 2):
        k = n * np.pi / L
        theta = n * np.pi * theta_factor
        decay = np.exp(-(k * k) * D * N)
        amplitude = 4.0 / (n * np.pi)
        sine = np.sin(theta)
        term = amplitude * sine * decay
        S += term

        dk_dL = -n * np.pi / (L * L)
        d_decay_dL = decay * (-2.0 * k * D * N * dk_dL)
        dS_dL += amplitude * sine * d_decay_dL

        if n > 21 and abs(term) < tolerance:
            break
    return float(S), float(dS_dL)


def exact_free_energy_and_force(N: int, L: float, a: float = 1.0,
                                kBT: float = 1.0,
                                tether_fraction: float = 0.5,
                                n_modes: int = DEFAULT_MODES):
    S, dS_dL = survival_and_derivative(
        N, L, a, tether_fraction, n_modes=n_modes
    )
    if not np.isfinite(S) or S <= 0:
        return np.nan, np.nan, S
    F = -kBT * np.log(S)
    force = kBT * dS_dL / S
    return float(F), float(force), float(S)


def central_difference_force(N: int, L: float, delta_L: float,
                             a: float = 1.0, kBT: float = 1.0,
                             tether_fraction: float = 0.5,
                             n_modes: int = DEFAULT_MODES):
    if delta_L <= 0 or L - delta_L <= 0:
        return np.nan
    F_plus, _, _ = exact_free_energy_and_force(
        N, L + delta_L, a, kBT, tether_fraction, n_modes
    )
    F_minus, _, _ = exact_free_energy_and_force(
        N, L - delta_L, a, kBT, tether_fraction, n_modes
    )
    if not (np.isfinite(F_plus) and np.isfinite(F_minus)):
        return np.nan
    return float(-(F_plus - F_minus) / (2.0 * delta_L))


def self_test() -> None:
    for ratio in (0.8, 1.5, 2.5, 4.0):
        rg = gaussian_rg(DEFAULT_N)
        L = ratio * rg
        F, f, S = exact_free_energy_and_force(DEFAULT_N, L)
        if not (np.isfinite(F) and np.isfinite(f) and 0 < S < 1):
            raise RuntimeError(f"Invalid exact Gaussian result at x={ratio}")
        f_fd = central_difference_force(DEFAULT_N, L, 1e-4 * rg)
        rel = abs(f_fd - f) / max(abs(f), 1e-300)
        if rel > 1e-7:
            raise RuntimeError(f"Chain-rule self-test failed at x={ratio}: rel={rel:.3e}")
    print("Self-test passed: chain-rule and small-step finite-difference forces agree.")


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Exact Gaussian force finite-difference audit for Supplementary Fig. S2(i).",
    )
    ap.add_argument("--N", type=int, default=DEFAULT_N)
    ap.add_argument("--a", type=float, default=DEFAULT_A)
    ap.add_argument("--kBT", type=float, default=DEFAULT_KBT)
    ap.add_argument("--tether-fraction", type=float, default=DEFAULT_TETHER_FRACTION)
    ap.add_argument("--n-modes", type=int, default=DEFAULT_MODES)
    ap.add_argument("--output-dir", type=Path, default=Path("Gaussian_S2i_FINAL"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.N < 2 or args.a <= 0 or args.kBT <= 0:
        raise ValueError("Require N>=2, a>0, kBT>0")
    if not 0.0 < args.tether_fraction < 1.0:
        raise ValueError("tether fraction must lie strictly between 0 and 1")
    if args.n_modes < 100:
        raise ValueError("Use at least 100 modes")

    ratios = DEFAULT_RATIOS[:9] if args.smoke else DEFAULT_RATIOS
    deltas = (1.0, 2.0) if args.smoke else DEFAULT_DELTAS
    rg = gaussian_rg(args.N, args.a)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ratio in ratios:
        L = float(ratio * rg)
        F, f, S = exact_free_energy_and_force(
            args.N, L, args.a, args.kBT, args.tether_fraction, args.n_modes
        )
        fRg_exact = f * rg
        for delta in deltas:
            f_fd = central_difference_force(
                args.N, L, delta, args.a, args.kBT,
                args.tether_fraction, args.n_modes
            )
            fRg_fd = f_fd * rg if np.isfinite(f_fd) else np.nan
            rel = abs(f_fd - f) / abs(f) if np.isfinite(f_fd) and f != 0 else np.nan
            abs_err_scaled = abs(fRg_fd - fRg_exact) if np.isfinite(fRg_fd) else np.nan
            rows.append({
                "N": args.N,
                "a": args.a,
                "kBT": args.kBT,
                "Rg": rg,
                "L": L,
                "L_over_Rg": ratio,
                "tether_fraction": args.tether_fraction,
                "tether_definition": "z0=tether_fraction*L, co-moving",
                "boundary": "absorbing z=0,L",
                "survival_exact": S,
                "F_exact": F,
                "force_exact": f,
                "force_exact_Rg_over_kBT": fRg_exact,
                "delta_L": delta,
                "deltaL_over_Rg": delta / rg,
                "force_FD": f_fd,
                "force_FD_Rg_over_kBT": fRg_fd,
                "relative_FD_error": rel,
                "absolute_scaled_force_error": abs_err_scaled,
                "production_deltaL": bool(abs(delta - 2.0) < 1e-12),
            })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir / f"FigS2i_Gaussian_Force_Audit_{stamp}"
    csv_path = base.with_name(base.name + "_points.csv")
    json_path = base.with_name(base.name + "_provenance.json")
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    x = np.asarray(ratios, dtype=float)
    FDs = {}
    Erel = {}
    for delta in deltas:
        sel = [r for r in rows if abs(r["delta_L"] - delta) < 1e-12]
        FDs[delta] = np.array([r["force_FD_Rg_over_kBT"] for r in sel])
        Erel[delta] = np.array([r["relative_FD_error"] for r in sel])
    exact = np.array([
        exact_free_energy_and_force(args.N, ratio * rg, args.a, args.kBT,
                                    args.tether_fraction, args.n_modes)[1] * rg
        for ratio in ratios
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    ax = axes[0]
    ax.plot(x, exact, "k-", lw=2.2, label="Exact chain-rule force")
    markers = ["o-", "s-", "^-", "d-"]
    for marker, delta in zip(markers, deltas):
        ax.plot(x, FDs[delta], marker, ms=4.3, lw=1.1,
                label=rf"FD: $\Delta L={delta:g}$" + (" (production)" if delta == 2.0 else ""))
    ax.set_xlabel(r"Reduced separation $L/R_g$")
    ax.set_ylabel(r"$fR_g/(k_BT)$")
    ax.set_title("(a) Exact Gaussian force and finite differences", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    ax = axes[1]
    for marker, delta in zip(markers, deltas):
        ax.semilogy(x, Erel[delta], marker, ms=4.3, lw=1.1,
                    label=rf"$\Delta L={delta:g}$")
    ax.axhline(1e-3, linestyle="--", lw=1.1, label="0.1% reference")
    ax.set_xlabel(r"Reduced separation $L/R_g$")
    ax.set_ylabel(r"Relative error $|f_{FD}-f|/|f|$")
    ax.set_title("(b) Finite-difference step-size audit", fontweight="bold")
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
        "N": args.N,
        "a": args.a,
        "kBT": args.kBT,
        "Rg": rg,
        "geometry": {
            "boundary": "absorbing z=0,L",
            "tether": "z0=L/2, co-moving midpoint",
            "continuous_Gaussian": True,
        },
        "exact_force": "analytic chain-rule derivative of the absorbing-interval eigenmode survival probability",
        "finite_difference": "central difference of the same exact free energy",
        "delta_L_values": list(map(float, deltas)),
        "production_delta_L": 2.0,
        "no_smoothing": True,
        "no_artificial_floor": True,
        "sign_filtering": False,
        "note": "The finite-difference audit is deterministic; weak-confinement relative errors can become large when the exact force approaches zero.",
        "outputs": {"points_csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)},
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    print("Saved:")
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
