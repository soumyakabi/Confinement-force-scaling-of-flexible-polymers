#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 — publication plotting and audit code
==============================================

Purpose
-------
Create the final Figure 2 from already-generated point/block data.  This
script does NOT run Monte Carlo sampling.  It only loads the validated Fig. 2
outputs and the numerical master R_g dataset, performs consistency checks,
fits the strong-confinement SAW force regime, and writes publication-ready
PNG/PDF figures plus audit tables.

Physical setup represented here
--------------------------------
* N = 200
* 3D polymer; confinement along one Cartesian coordinate
* absorbing boundaries at x = 0 and x = L
* midpoint tether: x_0 = L/2
* Gaussian reference: continuous 3D Gaussian chain, R_g = a sqrt(N/6)
* SAW reference: unconfined cubic-lattice R_g taken from Rg_MASTER_FINAL.csv
* Entropic force: f = d ln Z / dL = -d(Delta F)/dL
* SAW force points are signed estimates; no sign-based filtering is performed.

Important plotting choices
--------------------------
Panel (a): Delta F versus L/R_g, with SAW statistical error bars.
Panel (b): f R_g versus L/R_g on log-log axes, with SAW statistical error
bars and the exact Gaussian curve.  A weighted empirical power-law fit is
shown only over L/R_g <= 1.9, which is treated as the strong-confinement
window for this figure.  The fit exponent is printed on the figure and saved
in the audit JSON; it is not presented as an asymptotic proof.

The Gaussian curve is evaluated analytically.  Its strong-confinement force
prefactor is 2*pi^2, not 6*pi^2:

    f R_g ~ 2*pi^2 (L/R_g)^(-3).

The SAW theoretical exponent is not given an arbitrary amplitude in the plot.
Instead, the displayed SAW power law is an empirical weighted fit to the
specified strong-confinement data window.

Input files
-----------
Default names (same directory as this script/current working directory):
    Fig2_SAW_vs_Gaussian_REVISION_FINAL_points.csv
    Fig2_SAW_vs_Gaussian_REVISION_FINAL_blocks.csv
    Rg_MASTER_FINAL.csv

The master file is required to contain an SAW, N=200 row for strict
publication mode.  If a supplied master file is only a partial subset, run
with --allow-point-rg-fallback; in that case the SAW R_g embedded in the Fig. 2
point file is used explicitly and the audit records that fallback.

Outputs
-------
    Fig2_SAW_vs_Gaussian_PLOT_FINAL.png
    Fig2_SAW_vs_Gaussian_PLOT_FINAL.pdf
    Fig2_SAW_vs_Gaussian_PLOT_FINAL_points.csv
    Fig2_SAW_vs_Gaussian_PLOT_FINAL_blocks.csv
    Fig2_SAW_vs_Gaussian_PLOT_FINAL_audit.json

The plotting style intentionally uses Matplotlib's default color cycle and
marker distinctions rather than hard-coding colors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Publication style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Constants and model definitions
# -----------------------------------------------------------------------------
A = 1.0
KBT = 1.0
N_TARGET = 200
NU_SAW = 0.587597
SAW_EXPECTED_FORCE_EXPONENT = -(1.0 / NU_SAW + 1.0)
STRONG_FIT_X_MAX = 1.90


def gaussian_rg(N: int) -> float:
    """Continuous 3D Gaussian-chain radius of gyration."""
    return A * math.sqrt(N / 6.0)


def gaussian_log_partition_and_force_rg(L: np.ndarray, N: int, n_max: int = 401) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact midpoint-tethered absorbing Gaussian result.

    For x0=L/2 and alpha = a^2 N / 6 = R_g^2,

        Z(L) = sum_n [4/(n*pi) sin(n*pi/2)] exp[-n^2*pi^2 R_g^2/L^2].

    The free energy is Delta F = -ln Z, and

        f R_g = R_g * d ln Z / dL.

    Only odd modes contribute, but summing all n is convenient and makes the
    expression transparent. Terms with even n vanish exactly.
    """
    L = np.asarray(L, dtype=float)
    rg = gaussian_rg(N)

    n = np.arange(1, n_max + 1, dtype=float)[:, None]
    LL = L[None, :]

    coeff = 4.0 * np.sin(0.5 * np.pi * n) / (n * np.pi)
    exponent = -(n * np.pi * rg / LL) ** 2
    terms = coeff * np.exp(exponent)
    Z = np.sum(terms, axis=0)

    if np.any(Z <= 0.0) or np.any(~np.isfinite(Z)):
        raise RuntimeError("Exact Gaussian partition function became non-positive/non-finite.")

    deltaF = -np.log(Z)

    # d/dL[-(n*pi*Rg/L)^2] = +2(n*pi)^2 Rg^2/L^3
    dZ_dL = np.sum(
        terms * (2.0 * (n * np.pi) ** 2 * rg ** 2 / LL ** 3),
        axis=0,
    )
    force_rg = rg * dZ_dL / Z
    return deltaF, force_rg


# -----------------------------------------------------------------------------
# Input and consistency handling
# -----------------------------------------------------------------------------
def load_inputs(points_path: Path, blocks_path: Path, master_path: Path,
                allow_point_rg_fallback: bool = False):
    points = pd.read_csv(points_path)
    blocks = pd.read_csv(blocks_path)
    master = pd.read_csv(master_path)

    required_point_cols = {
        "L", "L_minus", "L_plus", "L_over_Rg", "L_over_Rg_err",
        "N", "Rg", "Rg_err", "deltaF", "deltaF_err", "fRg", "fRg_err",
        "model", "signed_force_retained"
    }
    missing = sorted(required_point_cols - set(points.columns))
    if missing:
        raise ValueError(f"Point CSV is missing required columns: {missing}")

    if not ((points["N"] == N_TARGET).all()):
        raise ValueError("Fig. 2 point data contain N != 200.")

    gaussian = points.loc[points["model"].astype(str).str.lower() == "gaussian"].copy()
    saw = points.loc[points["model"].astype(str).str.lower() == "saw"].copy()

    if len(gaussian) == 0 or len(saw) == 0:
        raise ValueError("Both Gaussian and SAW point data are required.")

    gaussian = gaussian.sort_values("L").reset_index(drop=True)
    saw = saw.sort_values("L").reset_index(drop=True)

    if not np.all(np.diff(saw["L"].to_numpy()) > 0):
        raise ValueError("SAW widths are not strictly increasing.")

    # Gaussian numerical entries must agree with the exact curve to the
    # tolerance expected from the point-data audit.
    g_rg = gaussian_rg(N_TARGET)
    if not np.allclose(gaussian["Rg"].to_numpy(), g_rg, rtol=0, atol=1e-10):
        raise ValueError("Gaussian point file does not use the continuous Gaussian R_g=a*sqrt(N/6).")

    # Find the canonical unconfined SAW R_g in the master file.
    master_saw = master.loc[
        (master["model"].astype(str).str.lower() == "saw") &
        (master["N"].astype(int) == N_TARGET)
    ].copy()

    source = "Rg_MASTER_FINAL.csv"
    if len(master_saw) == 1:
        saw_rg = float(master_saw.iloc[0]["Rg"])
        saw_rg_err = float(master_saw.iloc[0]["Rg_err"])
        master_quality = str(master_saw.iloc[0].get("quality_flag", "UNKNOWN"))
    elif len(master_saw) > 1:
        raise ValueError("Rg_MASTER_FINAL.csv contains multiple SAW, N=200 rows; ambiguous canonical R_g.")
    elif allow_point_rg_fallback:
        vals = saw["Rg"].to_numpy(dtype=float)
        errs = saw["Rg_err"].to_numpy(dtype=float)
        if not np.allclose(vals, vals[0], rtol=0, atol=1e-12):
            raise ValueError("SAW point file contains inconsistent R_g values; fallback is unsafe.")
        if not np.allclose(errs, errs[0], rtol=0, atol=1e-12):
            raise ValueError("SAW point file contains inconsistent R_g uncertainties; fallback is unsafe.")
        saw_rg = float(vals[0])
        saw_rg_err = float(errs[0])
        master_quality = "NOT_PRESENT_IN_SUPPLIED_MASTER"
        source = "Fig2_SAW_vs_Gaussian_REVISION_FINAL_points.csv (explicit fallback)"
    else:
        raise ValueError(
            "Strict publication mode: Rg_MASTER_FINAL.csv does not contain the required "
            "SAW, N=200 row. Use the complete master file or explicitly pass --allow-point-rg-fallback."
        )

    # Compare the point-file R_g to the canonical master R_g.
    point_saw_rg = float(saw["Rg"].iloc[0])
    point_saw_rg_err = float(saw["Rg_err"].iloc[0])
    rg_match = math.isclose(point_saw_rg, saw_rg, rel_tol=0, abs_tol=1e-10)
    if not rg_match:
        raise ValueError(
            f"SAW R_g mismatch: point file has {point_saw_rg:.10f}, master has {saw_rg:.10f}."
        )

    # Recalculate L/Rg and compare with the supplied point table.
    expected_x = saw["L"].to_numpy(dtype=float) / saw_rg
    if not np.allclose(expected_x, saw["L_over_Rg"].to_numpy(dtype=float), rtol=0, atol=5e-8):
        raise ValueError("SAW L/R_g values are inconsistent with the canonical R_g.")

    # Confirm the central finite-difference geometry.
    expected_dL = saw["L_plus"].to_numpy(dtype=float) - saw["L_minus"].to_numpy(dtype=float)
    if not np.allclose(expected_dL, 4.0, rtol=0, atol=1e-12):
        raise ValueError("Fig. 2 does not use the expected symmetric delta_L=2 central difference.")

    # Do not silently remove signed estimates.
    if not saw["signed_force_retained"].astype(bool).all():
        raise ValueError("Some SAW force estimates were marked as not retained.")

    return points, blocks, master, gaussian, saw, g_rg, saw_rg, saw_rg_err, master_quality, source


# -----------------------------------------------------------------------------
# Weighted power-law fit
# -----------------------------------------------------------------------------
def weighted_log_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Dict[str, float]:
    """Fit y = A x^p using weighted linear regression in log-log space."""
    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) &
        (x > 0) & (y > 0) & (sy > 0)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("Too few positive SAW points for a log-log fit.")

    lx = np.log(x[mask])
    ly = np.log(y[mask])
    sly = sy[mask] / y[mask]

    X = np.column_stack([np.ones_like(lx), lx])
    W = np.diag(1.0 / sly ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ (X.T @ W @ ly)

    intercept, exponent = beta
    intercept_err, exponent_err = np.sqrt(np.diag(cov))
    residual = ly - X @ beta
    chi2 = float(np.sum((residual / sly) ** 2))
    dof = max(1, len(lx) - 2)

    return {
        "intercept": float(intercept),
        "intercept_err": float(intercept_err),
        "amplitude": float(np.exp(intercept)),
        "exponent": float(exponent),
        "exponent_err": float(exponent_err),
        "chi2": chi2,
        "dof": int(dof),
        "reduced_chi2": chi2 / dof,
        "n_points": int(len(lx)),
        "x_min": float(np.min(x[mask])),
        "x_max": float(np.max(x[mask])),
    }


# -----------------------------------------------------------------------------
# Main plotting routine
# -----------------------------------------------------------------------------
def make_figure(points: pd.DataFrame, blocks: pd.DataFrame,
                gaussian: pd.DataFrame, saw: pd.DataFrame,
                g_rg: float, saw_rg: float, saw_rg_err: float,
                master_quality: str, rg_source: str, output_stem: Path):
    # Evaluate the exact Gaussian result directly in the plotted scaling
    # variable x=L/R_g^(0).  This lets the exact theory extend smoothly into
    # the strong-confinement region rather than starting only at the first
    # simulated lattice width.
    x_dense = np.logspace(np.log10(0.70), np.log10(4.20), 700)
    L_dense = x_dense * g_rg
    gF_dense, gf_dense = gaussian_log_partition_and_force_rg(L_dense, N_TARGET)

    # Strong-confinement SAW fit: x <= 1.9 and positive signed force points.
    x_saw = saw["L_over_Rg"].to_numpy(float)
    y_saw = saw["fRg"].to_numpy(float)
    e_saw = saw["fRg_err"].to_numpy(float)
    fit_mask = (x_saw <= STRONG_FIT_X_MAX) & (y_saw > 0) & (e_saw > 0)
    fit = weighted_log_fit(x_saw[fit_mask], y_saw[fit_mask], e_saw[fit_mask])

    x_fit = np.logspace(np.log10(max(0.65, x_saw[fit_mask].min() * 0.95)),
                        np.log10(min(2.0, STRONG_FIT_X_MAX * 1.03)), 250)
    y_fit = fit["amplitude"] * x_fit ** fit["exponent"]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))

    # ------------------------------------------------------------------
    # Panel (a): free energy
    # ------------------------------------------------------------------
    ax = axes[0]
    xg = gaussian["L_over_Rg"].to_numpy(float)
    yg = gaussian["deltaF"].to_numpy(float)
    xs = saw["L_over_Rg"].to_numpy(float)
    ys = saw["deltaF"].to_numpy(float)
    es = saw["deltaF_err"].to_numpy(float)
    xerr_s = saw["L_over_Rg_err"].to_numpy(float)

    ax.plot(xg, yg, "o", markersize=7.5, label="Gaussian exact", zorder=4)
    ax.plot(x_dense, gF_dense, "-", linewidth=2.2, label="Gaussian exact theory", zorder=2)

    ax.errorbar(
        xs, ys, xerr=xerr_s, yerr=es,
        fmt="s", markersize=7.5, capsize=3.5, capthick=1.2,
        elinewidth=1.15, label="SAW (Rosenbluth)", zorder=5,
    )

    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$\Delta F$ ($k_BT$)")
    ax.set_title("(a) Confinement free energy")
    ax.set_xlim(0.65, 4.2)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)

    # ------------------------------------------------------------------
    # Panel (b): force, log-log
    # ------------------------------------------------------------------
    ax = axes[1]
    # Exact Gaussian curve.
    ax.plot(x_dense, gf_dense, linewidth=2.2, label="Gaussian exact theory", zorder=2)

    # SAW signed estimates: positive values are displayable on log axes;
    # no negative/zero values are deleted from the saved CSV.
    positive = y_saw > 0
    ax.errorbar(
        xs[positive], y_saw[positive],
        xerr=xerr_s[positive], yerr=e_saw[positive],
        fmt="s", markersize=7.0, capsize=3.2, capthick=1.1,
        elinewidth=1.05, label="SAW (signed estimates)", zorder=5,
    )

    # SAW empirical strong-confinement fit.
    ax.plot(
        x_fit, y_fit, "--", linewidth=2.0,
        label=(rf"SAW fit ($L/R_g^{{(0)}}\leq1.9$): $p={fit['exponent']:.3f}\pm{fit['exponent_err']:.3f}$"),
        zorder=3,
    )

    ax.set_xlabel(r"$L/R_g^{(0)}$")
    ax.set_ylabel(r"$fR_g^{(0)}$ ($k_BT$)")
    ax.set_title("(b) Entropic force")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.65, 4.2)
    ax.set_ylim(0.10, 40)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.6)
    ax.legend(
        loc="lower left", ncol=1, frameon=True,
        columnspacing=0.9, handlelength=2.6, borderaxespad=0.4,
    )

    fig.suptitle(
        "Gaussian and self-avoiding chains under absorbing/survival confinement",
        fontsize=17, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, fit, png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-ready Fig. 2 from existing CSV data.")
    parser.add_argument("--points", type=Path,
                        default=Path("Fig2_SAW_vs_Gaussian_REVISION_FINAL_points.csv"))
    parser.add_argument("--blocks", type=Path,
                        default=Path("Fig2_SAW_vs_Gaussian_REVISION_FINAL_blocks.csv"))
    parser.add_argument("--master-rg", type=Path, default=Path("Rg_MASTER_FINAL.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("Fig2_FINAL"))
    parser.add_argument("--allow-point-rg-fallback", action="store_true",
                        help="Explicitly use the SAW R_g stored in the Fig. 2 point CSV when the supplied master CSV is partial.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / "Fig2_SAW_vs_Gaussian_PLOT_FINAL"

    print("=" * 82)
    print("FIGURE 2 — PUBLICATION PLOT / AUDIT")
    print("=" * 82)
    print(f"Points : {args.points}")
    print(f"Blocks : {args.blocks}")
    print(f"Master : {args.master_rg}")

    points, blocks, master, gaussian, saw, g_rg, saw_rg, saw_rg_err, master_quality, rg_source = load_inputs(
        args.points, args.blocks, args.master_rg,
        allow_point_rg_fallback=args.allow_point_rg_fallback,
    )

    # Recompute the Gaussian exact quantities at the tabulated widths and
    # audit against the point file.
    gF_check, gf_check = gaussian_log_partition_and_force_rg(
        gaussian["L"].to_numpy(float), N_TARGET
    )
    max_df_diff = float(np.max(np.abs(gF_check - gaussian["deltaF"].to_numpy(float))))
    max_force_diff = float(np.max(np.abs(gf_check - gaussian["fRg_exact"].to_numpy(float))))
    if max_df_diff > 1e-9 or max_force_diff > 1e-9:
        raise ValueError(
            "Gaussian point data do not reproduce the exact analytical expression "
            f"(max DeltaF diff={max_df_diff:.3e}, max fRg diff={max_force_diff:.3e})."
        )

    # Basic block audit.
    saw_blocks = blocks.loc[blocks["model"].astype(str).str.lower() == "saw"].copy()
    n_saw_block_rows = int(len(saw_blocks))
    if n_saw_block_rows:
        n_unique_blocks = int(saw_blocks.groupby("L")["block"].nunique().min())
        block_counts = saw_blocks.groupby("L").size()
    else:
        n_unique_blocks = 0
        block_counts = pd.Series(dtype=int)

    # Produce the figure.
    fig, fit, png_path, pdf_path = make_figure(
        points, blocks, gaussian, saw,
        g_rg, saw_rg, saw_rg_err,
        master_quality, rg_source, stem,
    )

    # Save compact data copies used by the figure.
    points.to_csv(stem.with_name(stem.name + "_points.csv"), index=False)
    blocks.to_csv(stem.with_name(stem.name + "_blocks.csv"), index=False)

    audit = {
        "figure": "Figure 2",
        "N": N_TARGET,
        "geometry": "3D cubic-lattice SAW; one-coordinate absorbing confinement; x0=L/2",
        "gaussian_model": "continuous 3D Gaussian chain; exact analytic benchmark",
        "gaussian_Rg": g_rg,
        "saw_Rg": saw_rg,
        "saw_Rg_err": saw_rg_err,
        "saw_Rg_source": rg_source,
        "saw_master_quality": master_quality,
        "saw_points": int(len(saw)),
        "saw_block_rows": n_saw_block_rows,
        "minimum_SAW_blocks_per_width": n_unique_blocks,
        "force_estimates_signed_and_retained": bool(saw["signed_force_retained"].astype(bool).all()),
        "central_difference_delta_L": 2,
        "central_difference_total_span": 4,
        "gaussian_exact_validation": {
            "max_abs_deltaF_difference": max_df_diff,
            "max_abs_fRg_difference": max_force_diff,
        },
        "saw_strong_confinement_fit": fit,
        "expected_SAW_exponent": SAW_EXPECTED_FORCE_EXPONENT,
        "fit_window": f"L/Rg <= {STRONG_FIT_X_MAX}",
        "fit_interpretation": "empirical finite-range scaling fit, not an asymptotic proof",
        "gaussian_strong_confinement_reference": "f Rg = 2*pi^2*(L/Rg)^(-3)",
        "figure_files": [png_path.name, pdf_path.name],
        "note": "All signed point-level estimates remain in the saved CSV; log-axis plotting displays only positive values.",
    }
    audit_path = stem.with_name(stem.name + "_audit.json")
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    print("\n" + "=" * 82)
    print("FIGURE 2 COMPLETE")
    print("=" * 82)
    print(f"SAW R_g = {saw_rg:.6f} +/- {saw_rg_err:.6f}")
    print(f"SAW fit exponent = {fit['exponent']:.6f} +/- {fit['exponent_err']:.6f}")
    print(f"SAW expected exponent = {SAW_EXPECTED_FORCE_EXPONENT:.6f}")
    print(f"Fit reduced chi^2 = {fit['reduced_chi2']:.3f}")
    print(f"Gaussian max |DeltaF_exact - CSV| = {max_df_diff:.3e}")
    print(f"Gaussian max |fRg_exact - CSV| = {max_force_diff:.3e}")
    print(f"PNG : {png_path.resolve()}")
    print(f"PDF : {pdf_path.resolve()}")
    print(f"Audit: {audit_path.resolve()}")
    print("=" * 82)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
