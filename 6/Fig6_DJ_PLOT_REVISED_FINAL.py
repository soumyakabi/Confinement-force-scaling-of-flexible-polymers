#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig6_DJ_PLOT_REVISED_FINAL.py
=============================
Plot-only finalization workflow for the already completed Fig. 6 production.

This script does NOT rerun the Monte Carlo/PERM simulations.
It reads the production CSV/audit outputs, regenerates the publication figure,
and writes a corrected reviewer-facing provenance JSON.

Key corrections
---------------
1. Rg provenance is written per w from the canonical Fig. 5 CSV; the old bug
   that repeated the w=0.1 Rg for every w is eliminated.
2. Panel (a) y-limits are determined from the actual data + error bars and are
   padded upward so the largest-force points are fully visible.
3. Panel (a) legend is moved to the lower-right, away from the strong-force data.
4. The "strong-confinement fit window" annotation is shifted downward.
5. Force-axis notation is dimensionally explicit: f Rg / kBT.
6. Panel (c) explicitly states that error bars are 95% bootstrap CIs.
7. Provenance records the exact audit files, Rg values, fit/sensitivity tables,
   finite-difference audit, QC counts, and plotting choices.

Example
-------
python Fig6_DJ_PLOT_REVISED_FINAL.py \
  --production-dir /kaggle/working/Fig6_DJ_FORCE_REVISED \
  --rg-csv /kaggle/input/.../Fig5_DJ_Rg_Crossover_Data.csv \
  --outdir /kaggle/working/Fig6_DJ_PLOT_FINAL \
  --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


# -----------------------------------------------------------------------------
# Constants / publication style
# -----------------------------------------------------------------------------
SCRIPT_VERSION = "Fig6-DJ-PLOT-REV-FINAL-1.1"
A = 1.0
KBT = 1.0
NU_SAW = 0.587597
ALPHA_GAUSSIAN = -3.0
ALPHA_SAW = -(1.0 + 1.0 / NU_SAW)
W_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
FIT_WINDOW = (0.60, 1.80)

COLORS = {
    0.1: "#355C9A",
    0.2: "#1B8A8A",
    0.3: "#4AA564",
    0.4: "#D28A2E",
    0.5: "#B64E4E",
}
MARKERS = {0.1: "o", 0.2: "s", 0.3: "^", 0.4: "D", 0.5: "P"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "savefig.dpi": 600,
})


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def csv_exists(prod: Path, name: str) -> Path | None:
    p = prod / name
    return p if p.is_file() else None


def nice_upper_limit(value: float, minimum: float = 1.0) -> float:
    """Round upward to a visually clean 1/2/5 x 10^n limit."""
    value = max(float(value), minimum)
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    mantissa = value / scale
    if mantissa <= 1.0:
        nice = 1.0
    elif mantissa <= 2.0:
        nice = 2.0
    elif mantissa <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * scale


def load_rg_master(path: Path) -> Dict[float, Dict[str, float]]:
    df = pd.read_csv(path)
    required = {"model", "N", "w", "Rg", "Rg_err"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rg CSV is missing columns: {sorted(missing)}")

    out: Dict[float, Dict[str, float]] = {}
    for w in W_VALUES:
        rows = df[np.isclose(pd.to_numeric(df["w"], errors="coerce"), w)]
        # Prefer the DJ row for this finite w.
        rows = rows[rows["model"].astype(str).str.upper().eq("DJ")]
        if rows.empty:
            raise ValueError(f"No DJ Rg row found for w={w}")
        if len(rows) > 1:
            rows = rows.iloc[[0]]
        row = rows.iloc[0]
        out[w] = {
            "Rg": float(row["Rg"]),
            "Rg_err": float(row["Rg_err"]),
            "N": int(row["N"]),
            "quality_flag": str(row.get("quality_flag", "")),
            "source_model": str(row["model"]),
        }
    return out


def load_csvs(prod: Path) -> Dict[str, pd.DataFrame]:
    names = [
        "Fig6_PointLevel.csv",
        "Fig6_BlockLevel.csv",
        "Fig6_ExponentFits.csv",
        "Fig6_FitWindowSensitivity.csv",
        "Fig6_FiniteDifference_Audit.csv",
        "Fig6_CollapseSpread.csv",
        "Fig6_CollapseResiduals.csv",
        "Fig6_Convergence_Z.csv",
        "Fig6_Convergence_DeltaF_relative.csv",
        "Fig6_Convergence_Force.csv",
    ]
    tables = {}
    for name in names:
        p = prod / name
        if p.is_file():
            tables[name] = pd.read_csv(p)
    required = ["Fig6_PointLevel.csv", "Fig6_ExponentFits.csv"]
    for name in required:
        if name not in tables:
            raise FileNotFoundError(f"Required production file missing: {prod/name}")
    return tables


def validate_rg_consistency(point: pd.DataFrame, rg: Dict[float, Dict[str, float]]) -> Dict[str, Any]:
    checks = []
    max_abs = 0.0
    for w in W_VALUES:
        vals = point.loc[np.isclose(point["w"], w), "Rg"].astype(float).unique()
        if len(vals) != 1:
            raise ValueError(f"Point-level data have inconsistent Rg values for w={w}: {vals}")
        delta = abs(vals[0] - rg[w]["Rg"])
        max_abs = max(max_abs, delta)
        checks.append({
            "w": w,
            "Rg_fig5": rg[w]["Rg"],
            "Rg_point_level": float(vals[0]),
            "absolute_difference": float(delta),
            "pass": bool(delta < 1e-10),
        })
    return {"max_absolute_difference": max_abs, "checks": checks}


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def create_figure(point: pd.DataFrame, fits: pd.DataFrame, outdir: Path) -> Dict[str, str]:
    fig, axes = plt.subplots(
        1, 3, figsize=(17.8, 5.4), gridspec_kw={"wspace": 0.30}
    )
    ax1, ax2, ax3 = axes

    # ------------------------------------------------------------------
    # Panel (a): full signed force data
    # ------------------------------------------------------------------
    for w in W_VALUES:
        rows = point[np.isclose(point["w"], w)].sort_values("L_over_Rg")
        if rows.empty:
            continue
        x = rows["L_over_Rg"].to_numpy(float)
        y = rows["fRg"].to_numpy(float)
        err = rows["fRg_err_bootstrap"].to_numpy(float)
        fallback = rows["fRg_err_sem_propagated"].to_numpy(float)
        err = np.where(np.isfinite(err), err, fallback)
        ax1.errorbar(
            x, y, yerr=err,
            fmt=MARKERS[w], ms=7, capsize=3, elinewidth=1.0,
            lw=0.0, color=COLORS[w], markeredgecolor="black",
            markeredgewidth=0.35, alpha=0.90,
            label=fr"$w={w:.1f}$", zorder=4,
        )

    xall = point["L_over_Rg"].to_numpy(float)
    xlo = max(0.55, np.nanmin(xall) * 0.92)
    xhi = min(4.2, np.nanmax(xall) * 1.05)

    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=0.02, linscale=1.0)
    ax1.set_xlim(xlo, xhi)
    xticks = [t for t in [0.6, 1.0, 1.5, 2.0, 3.0, 4.0] if xlo <= t <= xhi]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{t:g}" for t in xticks])
    ax1.xaxis.set_minor_formatter(NullFormatter())

    # Explicitly pad the upper limit so the strongest force/error bar is visible.
    upper_data = np.nanmax(point["fRg"].to_numpy(float) +
                           point["fRg_err_bootstrap"].fillna(point["fRg_err_sem_propagated"]).to_numpy(float))
    lower_data = np.nanmin(point["fRg"].to_numpy(float) -
                           point["fRg_err_bootstrap"].fillna(point["fRg_err_sem_propagated"]).to_numpy(float))
    y_upper = nice_upper_limit(upper_data * 1.15, minimum=10.0)
    # Keep the zero line visible while retaining signed estimates if they occur.
    y_lower = min(-0.5, lower_data * 1.20)
    ax1.set_ylim(y_lower, y_upper)

    ax1.set_xlabel(r"Slit width $L/R_g(w)$")
    ax1.set_ylabel(r"Signed entropic force $fR_g/k_BT$")
    ax1.set_title("(a) Domb–Joyce force crossover", pad=12)
    ax1.axhline(0.0, color="0.2", lw=0.9, alpha=0.6)
    ax1.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], facecolor="0.75", alpha=0.10, zorder=0)

    # Shift the annotation well downward, below the strong-force data.
    ax1.text(
        0.04, 0.12, "strong-confinement fit window",
        transform=ax1.transAxes, fontsize=9.5, style="italic",
        va="bottom", ha="left",
    )
    ax1.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax1.grid(True, which="minor", alpha=0.12, linewidth=0.35)
    # Bottom-right placement: the data there are already in the weak-force regime.
    ax1.legend(
        loc="lower right", ncol=1, frameon=True, fancybox=False,
        edgecolor="black", framealpha=0.95, borderpad=0.5,
    )

    # ------------------------------------------------------------------
    # Panel (b): strong-confinement scaling
    # ------------------------------------------------------------------
    fit_map = {float(r["w"]): r for _, r in fits.iterrows()}
    for w in W_VALUES:
        rows = point[(np.isclose(point["w"], w)) & (point["fRg"] > 0)].sort_values("L_over_Rg")
        if rows.empty:
            continue
        x = rows["L_over_Rg"].to_numpy(float)
        y = rows["fRg"].to_numpy(float)
        err = rows["fRg_err_bootstrap"].to_numpy(float)
        fallback = rows["fRg_err_sem_propagated"].to_numpy(float)
        err = np.where(np.isfinite(err), err, fallback)
        ax2.errorbar(
            x, y, yerr=err, fmt=MARKERS[w], ms=7, capsize=3,
            elinewidth=1.0, color=COLORS[w], markeredgecolor="black",
            markeredgewidth=0.35, alpha=0.90, zorder=4,
        )

        fit = fit_map.get(w, {})
        if str(fit.get("status", "")) == "RESOLVED" and np.isfinite(float(fit.get("alpha_primary", np.nan))):
            use = rows[(rows["L_over_Rg"] >= FIT_WINDOW[0]) &
                       (rows["L_over_Rg"] <= FIT_WINDOW[1])]
            if len(use) >= 3:
                lx = np.log(use["L_over_Rg"].to_numpy(float))
                ly = np.log(use["fRg"].to_numpy(float))
                slope = float(fit["alpha_primary"])
                intercept = float(np.mean(ly - slope * lx))
                xx = np.geomspace(max(FIT_WINDOW[0], float(np.exp(lx.min()))),
                                  min(FIT_WINDOW[1], float(np.exp(lx.max()))), 100)
                yy = np.exp(intercept) * xx ** slope
                ax2.plot(xx, yy, "-", color=COLORS[w], lw=1.6, alpha=0.70, zorder=2)

    xxg = np.geomspace(FIT_WINDOW[0], FIT_WINDOW[1], 120)
    yg = 2.0 * math.pi**2 * xxg ** ALPHA_GAUSSIAN
    ax2.plot(xxg, yg, "k--", lw=1.5, alpha=0.70, label=r"Gaussian: $\alpha=-3$")

    # SAW is shown only as a slope guide.
    near_one = point[(point["fRg"] > 0) & (np.abs(point["L_over_Rg"] - 1.0) < 0.20)]
    ref_amp = float(np.nanmedian(near_one["fRg"])) if not near_one.empty else 1.0
    ysaw = ref_amp * xxg ** ALPHA_SAW
    ax2.plot(xxg, ysaw, "k:", lw=1.8, alpha=0.70,
             label=fr"SAW slope: $\alpha={ALPHA_SAW:.3f}$")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(FIT_WINDOW[0] * 0.95, FIT_WINDOW[1] * 1.03)
    xticks = [t for t in [0.6, 1.0, 1.5, 1.8] if FIT_WINDOW[0] * 0.95 <= t <= FIT_WINDOW[1] * 1.03]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{t:g}" for t in xticks])
    ax2.xaxis.set_minor_formatter(NullFormatter())

    positives = point[(point["fRg"] > 0) &
                      (point["L_over_Rg"] >= FIT_WINDOW[0] * 0.95) &
                      (point["L_over_Rg"] <= FIT_WINDOW[1] * 1.03)]["fRg"].to_numpy(float)
    if positives.size:
        ax2.set_ylim(max(np.nanmin(positives) * 0.35, 1e-3), np.nanmax(positives) * 3.2)

    ax2.set_xlabel(r"Slit width $L/R_g(w)$")
    ax2.set_ylabel(r"$fR_g/k_BT$")
    ax2.set_title("(b) Strong-confinement scaling", pad=12)
    ax2.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax2.grid(True, which="minor", alpha=0.12, linewidth=0.35)
    ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)

    # Compact exponent labels at lower left.
    for i, w in enumerate(W_VALUES):
        fit = fit_map.get(w, {})
        if str(fit.get("status", "")) == "RESOLVED":
            text = fr"$w={w:.1f}$: $\alpha={float(fit['alpha_primary']):.2f}\pm{float(fit['alpha_primary_err']):.2f}$"
        else:
            text = fr"$w={w:.1f}$: unresolved"
        ax2.text(0.04, 0.035 + i * 0.052, text,
                 transform=ax2.transAxes, color=COLORS[w], fontsize=9.5)

    # ------------------------------------------------------------------
    # Panel (c): exponent crossover
    # ------------------------------------------------------------------
    resolved = []
    for w in W_VALUES:
        fit = fit_map.get(w, {})
        if str(fit.get("status", "")) == "RESOLVED" and np.isfinite(float(fit.get("alpha_primary", np.nan))):
            resolved.append((w, float(fit["alpha_primary"]), float(fit["alpha_primary_err"])))

    if resolved:
        ws = np.array([r[0] for r in resolved])
        aa = np.array([r[1] for r in resolved])
        ee = np.array([r[2] for r in resolved])
        ax3.errorbar(ws, aa, yerr=ee, fmt="o", ms=8, capsize=4,
                     color="black", markerfacecolor="white", markeredgewidth=1.2,
                     linewidth=1.2, zorder=5)
        for w, a, e in resolved:
            ax3.scatter([w], [a], s=50, color=COLORS[w], edgecolor="black", linewidth=0.6, zorder=6)

    ax3.axhline(ALPHA_GAUSSIAN, color="black", ls="--", lw=1.5, alpha=0.75,
                label=r"Gaussian limit: $\alpha=-3$")
    ax3.axhline(ALPHA_SAW, color="black", ls=":", lw=1.8, alpha=0.75,
                label=fr"SAW limit: $\alpha={ALPHA_SAW:.3f}$")
    ax3.set_xlabel(r"Excluded-volume strength $w$")
    ax3.set_ylabel(r"Effective exponent $\alpha$")
    ax3.set_title("(c) Strong-confinement exponent", pad=12)
    ax3.set_xticks(W_VALUES)
    if resolved:
        amin = min(ALPHA_GAUSSIAN, ALPHA_SAW, min(r[1] for r in resolved))
        amax = max(ALPHA_GAUSSIAN, ALPHA_SAW, max(r[1] for r in resolved))
    else:
        amin, amax = ALPHA_SAW, ALPHA_GAUSSIAN
    ax3.set_ylim(amin - 0.15, amax + 0.15)
    ax3.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax3.grid(True, which="minor", alpha=0.12, linewidth=0.35)
    ax3.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    ax3.text(
        0.05, 0.05,
        "finite-range effective exponent\n(95% bootstrap CI; not an asymptotic proof)",
        transform=ax3.transAxes, fontsize=9.5, style="italic", va="bottom",
    )

    fig.suptitle(
        "Domb–Joyce confinement-force crossover, N=200; absorbing/survival ensemble",
        fontsize=18, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.20)

    out_png = outdir / "Fig6_DJ_Force_Crossover_FINAL.png"
    out_pdf = outdir / "Fig6_DJ_Force_Crossover_FINAL.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf), "panel_a_y_upper": float(y_upper)}


# -----------------------------------------------------------------------------
# Corrected provenance
# -----------------------------------------------------------------------------
def build_provenance(prod: Path, rg_path: Path, tables: Dict[str, pd.DataFrame],
                     rg: Dict[float, Dict[str, float]], consistency: Dict[str, Any],
                     figure_paths: Dict[str, str], outdir: Path) -> Dict[str, Any]:
    point = tables["Fig6_PointLevel.csv"]
    fit = tables["Fig6_ExponentFits.csv"]

    qc = {
        "n_force_points": int(len(point)),
        "n_width_rows": int(len(tables.get("Fig6_WidthLevel.csv", []))),
        "n_block_rows": int(len(tables.get("Fig6_BlockLevel.csv", []))),
        "all_sign_selected_false": bool((point["sign_selected"].astype(bool) == False).all()) if "sign_selected" in point else None,
        "n_excluded": int(point["excluded"].astype(bool).sum()) if "excluded" in point else None,
        "force_quality_counts": point["force_quality_flag"].value_counts(dropna=False).to_dict() if "force_quality_flag" in point else {},
        "min_endpoint_ESS": float(point["min_ESS_endpoints"].min()) if "min_ESS_endpoints" in point else None,
        "max_endpoint_weight_fraction": float(point["max_weight_fraction_endpoints"].max()) if "max_weight_fraction_endpoints" in point else None,
    }

    rg_values = {
        str(w): {
            "Rg": rg[w]["Rg"],
            "Rg_err": rg[w]["Rg_err"],
            "N": rg[w]["N"],
            "quality_flag": rg[w]["quality_flag"],
            "source_model": rg[w]["source_model"],
        }
        for w in W_VALUES
    }

    provenance = {
        "script_version": SCRIPT_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Plot-only finalization and provenance correction for completed Fig. 6 production",
        "simulation_rerun": False,
        "N": 200,
        "w_values": W_VALUES,
        "Rg_source": {
            "canonical_fig5_csv": str(rg_path),
            "mapping_method": "matched canonical Fig.5 DJ row by exact finite w value",
            "point_level_consistency": consistency,
            "Rg_values": rg_values,
        },
        "ensemble": "center-tethered Domb–Joyce survival/first-passage ensemble in absorbing slit",
        "geometry": {
            "walls": "z=0 and z=L",
            "accessible_layers": "z=1,...,L-1",
            "tether": "z0=L/2",
            "width": "even integer wall separation",
        },
        "derivative": {
            "definition": "f = kBT d ln Z / dL",
            "primary_delta_L": 2,
            "mechanical_rule": "both walls displaced symmetrically; tether follows midpoint",
            "finite_difference_audit": "delta_L=2 versus delta_L=4 using production endpoint data",
            "finite_difference_note": "delta_L=4 is a coarse sensitivity audit, not a second local-derivative estimator",
        },
        "statistics": {
            "signed_force_policy": True,
            "sign_based_deletion": False,
            "SNR_filtering": False,
            "sigma_clipping": False,
            "primary_exponent_window": list(FIT_WINDOW),
            "exponent_interpretation": "finite-range effective strong-confinement exponent, not an asymptotic proof",
            "bootstrap_CI": "95% bootstrap confidence interval",
        },
        "reference_exponents": {
            "Gaussian": ALPHA_GAUSSIAN,
            "SAW": ALPHA_SAW,
            "SAW_nu": NU_SAW,
        },
        "qc_summary": qc,
        "figure_changes": {
            "panel_a_y_limit": "data-driven upper padding, rounded to a clean 1/2/5 x 10^n value",
            "panel_a_legend": "lower right",
            "panel_a_fit_window_annotation": "shifted downward to avoid data",
            "force_axis_label": "f Rg / kBT",
            "panel_c_uncertainty_label": "95% bootstrap CI",
        },
        "audit_files": {
            name: str(prod / name) for name in tables.keys()
        },
        "outputs": figure_paths,
        "notes": [
            "This JSON corrects the prior provenance bug in which the w=0.1 Rg value was repeated for every w.",
            "No simulation data were changed by this plotting/provenance step.",
        ],
    }

    write_json(outdir / "Fig6_Provenance_FINAL.json", provenance)
    return provenance


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Final plotting and provenance correction for completed Fig. 6")
    ap.add_argument("--production-dir", required=True, help="Directory containing Fig6_* CSV outputs")
    ap.add_argument("--rg-csv", required=True, help="Canonical Fig5_DJ_Rg_Crossover_Data.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    prod = Path(args.production_dir).expanduser().resolve()
    rg_path = Path(args.rg_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not prod.is_dir():
        raise FileNotFoundError(prod)
    if not rg_path.is_file():
        raise FileNotFoundError(rg_path)

    tables = load_csvs(prod)
    point = tables["Fig6_PointLevel.csv"]
    fits = tables["Fig6_ExponentFits.csv"]
    rg = load_rg_master(rg_path)
    consistency = validate_rg_consistency(point, rg)
    if consistency["max_absolute_difference"] >= 1e-10:
        raise RuntimeError(
            "Canonical Fig.5 Rg values do not match Fig6 point-level Rg values. "
            f"Maximum difference = {consistency['max_absolute_difference']:.3e}"
        )

    figure_paths = create_figure(point, fits, outdir)
    provenance = build_provenance(prod, rg_path, tables, rg, consistency, figure_paths, outdir)

    summary = [
        "FIGURE 6 PLOT-ONLY FINALIZATION",
        f"Script version: {SCRIPT_VERSION}",
        f"Production directory: {prod}",
        f"Canonical Fig.5 Rg CSV: {rg_path}",
        f"N: 200",
        f"w values: {W_VALUES}",
        f"Primary fit window: {FIT_WINDOW}",
        f"Panel (a) y-upper limit: {figure_paths['panel_a_y_upper']:.6g}",
        f"Rg consistency max abs difference: {consistency['max_absolute_difference']:.3e}",
        "Simulation rerun: NO",
        "",
        "Generated files:",
    ]
    for p in sorted(outdir.iterdir()):
        summary.append(f"  {p.name}")
    (outdir / "Fig6_Plot_Finalization_Summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FIGURE 6 FINAL PLOTTING / PROVENANCE COMPLETE")
    print("=" * 78)
    print(f"PNG: {figure_paths['png']}")
    print(f"PDF: {figure_paths['pdf']}")
    print(f"JSON: {outdir / 'Fig6_Provenance_FINAL.json'}")
    print(f"Panel (a) y upper limit: {figure_paths['panel_a_y_upper']:.6g}")
    print(f"Rg consistency max difference: {consistency['max_absolute_difference']:.3e}")
    print("=" * 78)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
