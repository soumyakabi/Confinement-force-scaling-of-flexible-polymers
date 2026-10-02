#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S3 — Canonical R_g(w) using the frozen master dataset.

Purpose
-------
This version is a PLOT/AUDIT-ONLY script. It performs NO Monte Carlo sampling,
NO reweighting, NO smoothing, and NO fitting.

The Domb–Joyce R_g(w) values are taken directly from the canonical
Rg_MASTER_FINAL.csv file and are locked to the values already adopted for
the revised main Fig. 5. The strict-SAW endpoint is supplied as a frozen
independent benchmark because the current Rg master CSV contains only the
Domb–Joyce w-sweep rows.

This design avoids the previous failure:
    "No matching master row found for model=SAW, N=200."

Methodological choices
----------------------
1. The master CSV is the sole numerical source for the DJ w-sweep.
2. The published DJ values/errors are checked against the frozen Fig. 5
   values before plotting. A mismatch stops the run rather than silently
   changing the dataset.
3. The strict-SAW endpoint is NOT searched for in the DJ master CSV.
   It is entered explicitly as an independently frozen benchmark.
4. All master diagnostics (ESS, weight fraction, quality flag, block
   information, bootstrap CI) are retained in the audit CSV/provenance.
5. No point is deleted because of a caution flag. Caution points remain
   visible and are marked differently only to make the diagnostic status
   transparent.
6. No crossover fit is performed and no smoothing is applied.

Frozen Fig. 5 DJ values for N=200
---------------------------------
w      R_g          SEM
0.0    5.775079     0.004848
0.1    6.667454     0.007271
0.2    7.159098     0.010553
0.3    7.547955     0.014648
0.4    7.807885     0.021618
0.5    7.998447     0.019620
1.0    8.784617     0.040295
2.0    9.296299     0.066296
4.0    9.624508     0.049471
8.0    9.717929     0.059451

Frozen strict-SAW endpoint used previously for Fig. 5:
    R_g = 9.669318 +/- 0.063693

The SAW value is deliberately kept separate from the DJ master CSV, because
the uploaded master file contains model="DJ" rows only.

Outputs
-------
FigS3_Rg_vs_w_MASTER_REVISED_FINAL.png
FigS3_Rg_vs_w_MASTER_REVISED_FINAL.pdf
FigS3_Rg_vs_w_MASTER_REVISED_FINAL_points.csv
FigS3_Rg_vs_w_MASTER_REVISED_FINAL_provenance.json

Example
-------
python FigS3_Rg_vs_w_MASTER_REVISED_FINAL.py \
    --rg-master /kaggle/input/datasets/soumyajyotikabi/n200-rg/Rg_MASTER_FINAL.csv \
    --output-dir /kaggle/working/FigS3_FINAL
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_NAME = "FigS3_Rg_vs_w_MASTER_REVISED_FINAL.py"
SCRIPT_VERSION = "4.0"

# ---------------------------------------------------------------------------
# Frozen Fig. 5 Domb–Joyce dataset for N=200.
# These are the values adopted as the canonical master values for the w sweep.
# ---------------------------------------------------------------------------
FROZEN_S5_DJ = {
    0.0: (5.775079, 0.004848),
    0.1: (6.667454, 0.007271),
    0.2: (7.159098, 0.010553),
    0.3: (7.547955, 0.014648),
    0.4: (7.807885, 0.021618),
    0.5: (7.998447, 0.019620),
    1.0: (8.784617, 0.040295),
    2.0: (9.296299, 0.066296),
    4.0: (9.624508, 0.049471),
    8.0: (9.717929, 0.059451),
}

# Frozen independent strict-SAW benchmark used previously in Fig. 5.
FROZEN_SAW_RG = 9.669318
FROZEN_SAW_ERR = 0.063693

# Comparison tolerance allows only harmless CSV/decimal representation
# differences. It is intentionally much tighter than the scientific error.
FROZEN_RTOL = 1.0e-7
FROZEN_ATOL = 5.0e-6

DEFAULT_W_VALUES = list(FROZEN_S5_DJ.keys())

# Publication plotting defaults.
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.2,
})


def _normalise_model_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].astype(str).str.strip().str.upper()
    return out


def _select_unique_dj_row(
    df: pd.DataFrame,
    N: int,
    w: float,
) -> pd.Series:
    """Select exactly one DJ row for (N, w)."""
    mask = (
        (df["model"] == "DJ")
        & (df["N"].astype(int) == int(N))
        & np.isclose(df["w"].astype(float), float(w), rtol=0.0, atol=1e-12)
    )
    rows = df.loc[mask]

    if len(rows) == 0:
        raise ValueError(
            f"No matching master row found for model=DJ, N={N}, w={w:g}."
        )
    if len(rows) > 1:
        raise ValueError(
            f"Multiple matching master rows found for model=DJ, N={N}, w={w:g}; "
            "the master dataset is not unique."
        )
    return rows.iloc[0]


def load_master_dataset(
    master_csv: str,
    N: int,
    w_values: list[float],
) -> tuple[pd.DataFrame, dict]:
    """Load and audit the canonical DJ master dataset."""
    df = pd.read_csv(master_csv)
    df = _normalise_model_column(df)

    required = {
        "model",
        "N",
        "w",
        "Rg",
        "Rg_err",
        "Rg2",
        "Rg2_err",
        "Rg_bootstrap_CI95_low",
        "Rg_bootstrap_CI95_high",
        "attempts",
        "blocks",
        "valid_blocks",
        "min_ESS",
        "median_ESS",
        "mean_ESS",
        "max_block_weight_fraction",
        "quality_flag",
        "estimator",
        "single_chain_definition",
        "ensemble_weighting",
        "dead_chain_treatment",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Master CSV is missing required columns: " + ", ".join(missing)
        )

    selected = []
    for w in w_values:
        row = _select_unique_dj_row(df, N, w)

        # Lock the numerical dataset to the already-adopted Fig. 5 values.
        if w not in FROZEN_S5_DJ:
            raise ValueError(
                f"w={w:g} was not part of the frozen Fig. 5 DJ dataset."
            )

        expected_rg, expected_err = FROZEN_S5_DJ[w]
        rg_ok = np.isclose(
            float(row["Rg"]),
            expected_rg,
            rtol=FROZEN_RTOL,
            atol=FROZEN_ATOL,
        )
        err_ok = np.isclose(
            float(row["Rg_err"]),
            expected_err,
            rtol=FROZEN_RTOL,
            atol=FROZEN_ATOL,
        )

        if not (rg_ok and err_ok):
            raise ValueError(
                f"Frozen Fig. 5 mismatch at w={w:g}: "
                f"master Rg={float(row['Rg']):.9f}, expected={expected_rg:.9f}; "
                f"master Rg_err={float(row['Rg_err']):.9f}, "
                f"expected={expected_err:.9f}."
            )

        selected.append(row.to_dict())

    out = pd.DataFrame(selected)

    # Preserve requested w ordering.
    w_order = {float(w): i for i, w in enumerate(w_values)}
    out["_order"] = out["w"].astype(float).map(w_order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    audit = {
        "master_model_rows_total": int(len(df)),
        "master_DJ_rows_for_requested_N": int(
            ((df["model"] == "DJ") & (df["N"].astype(int) == int(N))).sum()
        ),
        "master_SAW_rows_total": int((df["model"] == "SAW").sum()),
        "frozen_S5_match": True,
        "requested_w_values": [float(w) for w in w_values],
    }
    return out, audit


def make_plot(
    dj: pd.DataFrame,
    N: int,
    output_base: str,
) -> None:
    """Create the publication plot."""
    x = dj["w"].to_numpy(dtype=float)
    y = dj["Rg"].to_numpy(dtype=float)
    yerr = dj["Rg_err"].to_numpy(dtype=float)
    flags = dj["quality_flag"].astype(str).str.upper().to_numpy()

    ok = flags == "OK"
    caution = ~ok

    fig, ax = plt.subplots(figsize=(8.0, 6.1))

    # All points remain in the dataset. Diagnostic status is visual only.
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        lw=1.8,
        ms=6.5,
        capsize=3,
        label=r"Domb--Joyce, $N=200$",
    )

    # Caution points are overlaid with open markers; they are not deleted.
    if np.any(caution):
        ax.errorbar(
            x[caution],
            y[caution],
            yerr=yerr[caution],
            fmt="o",
            mfc="white",
            mew=1.5,
            ms=7.5,
            capsize=3,
            linestyle="none",
            label="Master points flagged CAUTION",
        )

    # Strict SAW is a separate endpoint benchmark, not a finite-w DJ point.
    ax.axhline(
        FROZEN_SAW_RG,
        linestyle=":",
        lw=1.8,
        label=r"Strict SAW endpoint ($w\rightarrow\infty$)",
    )
    ax.fill_between(
        x,
        FROZEN_SAW_RG - FROZEN_SAW_ERR,
        FROZEN_SAW_RG + FROZEN_SAW_ERR,
        alpha=0.10,
    )

    ax.set_xlabel(r"Excluded-volume strength $w$")
    ax.set_ylabel(r"$R_g/a$")
    ax.set_title(rf"Unconfined radius of gyration vs. $w$ ($N={N}$)")

    ax.set_xlim(left=-0.03)
    ax.margins(x=0.03)
    ax.grid(alpha=0.20, linewidth=0.8)
    ax.legend(loc="best", frameon=True)

    ax.text(
        0.98,
        0.03,
        "Canonical Fig. 5 DJ dataset\n"
        "block-level uncertainties from master CSV\n"
        "no smoothing; no fit; no point deletion",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.5",
            alpha=0.90,
        ),
    )

    fig.tight_layout()
    fig.savefig(output_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(output_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary Fig. S3: canonical Domb–Joyce Rg(w) "
            "from the frozen master dataset."
        )
    )
    parser.add_argument(
        "--rg-master",
        required=True,
        help="Path to Rg_MASTER_FINAL.csv",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=200,
        help="Chain length; frozen Fig. 5 dataset is N=200.",
    )
    parser.add_argument(
        "--w-values",
        type=float,
        nargs="+",
        default=DEFAULT_W_VALUES,
        help="DJ w values to load from the master CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="FigS3_Rg_vs_w_MASTER_REVISED_FINAL",
        help="Output directory.",
    )
    parser.add_argument(
        "--saw-rg",
        type=float,
        default=FROZEN_SAW_RG,
        help="Frozen strict-SAW endpoint Rg.",
    )
    parser.add_argument(
        "--saw-rg-err",
        type=float,
        default=FROZEN_SAW_ERR,
        help="Frozen strict-SAW endpoint uncertainty.",
    )
    args = parser.parse_args()

    if args.N != 200:
        raise ValueError(
            "This frozen S3/Fig. 5 crossover dataset is defined for N=200."
        )

    # Require the requested SAW benchmark to remain frozen unless the user
    # deliberately edits the command line.
    if not np.isclose(args.saw_rg, FROZEN_SAW_RG, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Frozen strict-SAW Rg is {FROZEN_SAW_RG:.6f}; "
            "do not replace it silently."
        )
    if not np.isclose(args.saw_rg_err, FROZEN_SAW_ERR, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Frozen strict-SAW uncertainty is {FROZEN_SAW_ERR:.6f}; "
            "do not replace it silently."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 84)
    print("SUPPLEMENTARY FIG. S3 — CANONICAL Rg MASTER-DATA REVISION")
    print("=" * 84)
    print(f"Script            : {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"Master CSV        : {args.rg_master}")
    print(f"N                 : {args.N}")
    print(f"DJ w values       : {args.w_values}")
    print("Monte Carlo       : OFF (plot/audit only)")
    print("Smoothing/fitting : OFF")
    print("SAW source        : frozen independent Fig. 5 endpoint")
    print("-" * 84)

    dj, master_audit = load_master_dataset(
        args.rg_master,
        args.N,
        [float(w) for w in args.w_values],
    )

    # Additional audit checks required by the adopted workflow.
    if not (dj["valid_blocks"].astype(int) == dj["blocks"].astype(int)).all():
        print("WARNING: at least one master row has fewer valid blocks than requested.")

    if (dj["Rg_err"].astype(float) <= 0).any():
        raise ValueError("Non-positive Rg uncertainty found in master dataset.")

    # Print a concise audit table.
    display_cols = [
        "w",
        "Rg",
        "Rg_err",
        "blocks",
        "valid_blocks",
        "min_ESS",
        "max_block_weight_fraction",
        "quality_flag",
    ]
    print(dj[display_cols].to_string(index=False))
    print("-" * 84)

    base = os.path.join(
        args.output_dir,
        "FigS3_Rg_vs_w_MASTER_REVISED_FINAL",
    )

    points_csv = base + "_points.csv"

    # Point-level audit table: retain all methodological metadata available
    # in the master CSV, plus frozen-reference status.
    dj_out = dj.copy()
    dj_out["data_source"] = "Rg_MASTER_FINAL.csv"
    dj_out["frozen_fig5_dataset"] = True
    dj_out["frozen_strict_SAW_Rg"] = float(FROZEN_SAW_RG)
    dj_out["frozen_strict_SAW_Rg_err"] = float(FROZEN_SAW_ERR)
    dj_out["MC_rerun"] = False
    dj_out["smoothing"] = False
    dj_out["fitting"] = False

    dj_out.to_csv(points_csv, index=False)

    make_plot(dj, args.N, base)

    provenance = {
        "metadata": {
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "figure": "Supplementary Fig. S3",
            "N": int(args.N),
            "master_csv": os.path.abspath(args.rg_master),
            "data_mode": "plot/audit only",
            "Monte_Carlo_rerun": False,
            "smoothing": False,
            "fitting": False,
            "DJ_data_source": "Rg_MASTER_FINAL.csv",
            "DJ_dataset_locked_to_Fig5": True,
            "strict_SAW_data_source": (
                "frozen independent benchmark adopted for revised Fig. 5; "
                "not present in the uploaded DJ-only master CSV"
            ),
            "strict_SAW_Rg": float(FROZEN_SAW_RG),
            "strict_SAW_Rg_err": float(FROZEN_SAW_ERR),
            "master_SAW_rows_total": master_audit["master_SAW_rows_total"],
            "frozen_fig5_match": bool(master_audit["frozen_S5_match"]),
            "requested_w_values": master_audit["requested_w_values"],
            "interpretation": (
                "Finite-w Domb–Joyce points represent the excluded-volume "
                "crossover; the strict SAW benchmark is a separate w->infinity "
                "endpoint and is not treated as a finite-w Domb–Joyce result."
            ),
        },
        "master_audit": master_audit,
        "DJ_results": dj_out.to_dict(orient="records"),
    }

    provenance_path = base + "_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, default=float)

    print("Saved:")
    print(f"  {base}.png")
    print(f"  {base}.pdf")
    print(f"  {points_csv}")
    print(f"  {provenance_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
