#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_TARGET = 200
W_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.0, 8.0]
A = 1.0
DPI = 600


def norm_model(x):
    return str(x).strip().lower().replace(" ", "")


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def get_dj_rows(df, N=N_TARGET):
    req = {"model", "N", "w", "Rg", "Rg_err"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"DJ master missing columns: {sorted(missing)}")
    d = df[
        (df["model"].map(norm_model) == "dj") &
        (df["N"].astype(int) == int(N))
    ].copy()
    d["w_num"] = d["w"].astype(float)
    rows = []
    for w in W_VALUES:
        hit = d[np.isclose(d["w_num"], w, atol=1e-12, rtol=0)]
        if len(hit) != 1:
            raise ValueError(
                f"Expected exactly one DJ row for N={N}, w={w:g}; found {len(hit)}"
            )
        rows.append(hit.iloc[0])
    out = pd.DataFrame(rows).sort_values("w_num").reset_index(drop=True)
    return out


def get_saw_row(df, N=N_TARGET):
    req = {"model", "N", "Rg", "Rg_err"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"SAW master missing columns: {sorted(missing)}")
    d = df[
        (df["model"].map(norm_model) == "saw") &
        (df["N"].astype(int) == int(N))
    ].copy()
    if len(d) != 1:
        raise ValueError(f"Expected exactly one SAW row for N={N}; found {len(d)}")
    return d.iloc[0]


def ci95(row):
    if "Rg_bootstrap_CI95_low" in row.index and "Rg_bootstrap_CI95_high" in row.index:
        lo = float(row["Rg_bootstrap_CI95_low"])
        hi = float(row["Rg_bootstrap_CI95_high"])
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
            return lo, hi, "bootstrap_CI95"
    rg = float(row["Rg"])
    err = float(row["Rg_err"])
    return max(1e-12, rg - 1.96 * err), rg + 1.96 * err, "Rg_pm_1.96_err"


def lattice_gaussian_rg(N=N_TARGET, a=A):
    return a * math.sqrt((N * N - 1.0) / (6.0 * N))


def z_vs_saw(rg, err, saw_rg, saw_err):
    s = math.sqrt(err * err + saw_err * saw_err)
    return (rg - saw_rg) / s if s > 0 else np.nan


def plot(dj, saw, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 17,
        "axes.labelweight": "bold",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "legend.fontsize": 10.5,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    w = dj["w_num"].to_numpy(float)
    rg = dj["Rg"].to_numpy(float)
    err = dj["Rg_err"].to_numpy(float)

    q = dj["quality_flag"].astype(str).str.upper().to_numpy() if "quality_flag" in dj else np.array(["OK"] * len(dj))
    ok = q == "OK"
    caution = ~ok

    # Finite-w DJ points. Connecting line is a guide to the crossover, not a fit.
    if np.any(ok):
        ax.errorbar(
            w[ok], rg[ok], yerr=err[ok],
            fmt="o-", ms=7, lw=1.8, capsize=3.5,
            label="Finite-$w$ Domb–Joyce", zorder=4
        )
    if np.any(caution):
        ax.errorbar(
            w[caution], rg[caution], yerr=err[caution],
            fmt="o", ms=7, mfc="white", capsize=3.5,
            label="Finite-$w$ DJ (sampling caution)", zorder=5
        )

    # Exact finite-N cubic-lattice Gaussian reference at w=0.
    rg_g = lattice_gaussian_rg()
    ax.axhline(
        rg_g, ls="--", lw=1.8, alpha=0.75,
        label=rf"$w=0$ lattice Gaussian: $R_g={rg_g:.4f}$",
        zorder=1
    )

    saw_rg = float(saw["Rg"])
    saw_err = float(saw["Rg_err"])
    saw_lo, saw_hi, _ = ci95(saw)

    # Strict SAW is the w -> infinity endpoint, not a finite-w point.
    ax.axhspan(
        saw_lo, saw_hi, alpha=0.12,
        label="Strict-SAW 95% CI ($w\\to\\infty$)", zorder=0
    )
    ax.axhline(
        saw_rg, ls=":", lw=2.2,
        label=rf"Strict SAW ($w\to\infty$): $R_g={saw_rg:.3f}\pm{saw_err:.3f}$",
        zorder=2
    )

    x_end = 8.65
    ax.errorbar(
        [x_end], [saw_rg], yerr=[saw_err],
        fmt="D", ms=7.5, capsize=3.5,
        label=r"SAW endpoint, $w\rightarrow\infty$", zorder=6
    )

    idx8 = int(np.argmax(w))
    ax.annotate(
        r"$w=8$ (finite)",
        xy=(w[idx8], rg[idx8]),
        xytext=(7.1, saw_rg - 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=10.5
    )

    ax.set_xlabel(r"Domb–Joyce overlap penalty $w$")
    ax.set_ylabel(r"Unconfined radius of gyration $R_g$ (lattice units)")
    ax.set_title(rf"Domb–Joyce swelling toward the strict self-avoiding limit ($N={N_TARGET}$)")
    ax.set_xlim(-0.12, 9.25)

    yy = np.concatenate([rg - err, rg + err, [rg_g, saw_lo, saw_hi]])
    pad = 0.08 * (yy.max() - yy.min())
    ax.set_ylim(max(0.0, yy.min() - pad), yy.max() + pad)
    ax.grid(True, alpha=0.20, linestyle=":")
    ax.legend(loc="lower right", frameon=True)

    ax.text(
        0.02, 0.97,
        "Finite-$w$ points are crossover states; "
        "$w\\rightarrow\\infty$ denotes the strict-SAW limit.",
        transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92)
    )

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.90)
    png = outdir / "Fig5_DJ_Rg_Crossover_REVISED.png"
    pdf = outdir / "Fig5_DJ_Rg_Crossover_REVISED.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    ap = argparse.ArgumentParser(description="Revised Fig. 5 Rg(w) plot/audit.")
    ap.add_argument("--dj-master", required=True, type=Path)
    ap.add_argument("--saw-master", required=True, type=Path)
    ap.add_argument("--dj-blocks", type=Path, default=None)
    ap.add_argument("--dj-report", type=Path, default=None)
    ap.add_argument("--dj-provenance", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("Fig5_DJ_Rg_REVISED"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    dj_df = load_csv(args.dj_master)
    saw_df = load_csv(args.saw_master)
    dj = get_dj_rows(dj_df)
    saw = get_saw_row(saw_df)

    # Core QA.
    if not np.all(np.diff(dj["w_num"]) > 0):
        raise ValueError("DJ w values are not strictly increasing.")
    if not np.all(np.diff(dj["Rg"].astype(float)) >= 0):
        raise ValueError("Canonical DJ Rg(w) is not monotonic non-decreasing.")

    rg0 = lattice_gaussian_rg()
    rg_w0 = float(dj.loc[np.isclose(dj["w_num"], 0.0), "Rg"].iloc[0])
    rg_w0_err = float(dj.loc[np.isclose(dj["w_num"], 0.0), "Rg_err"].iloc[0])

    saw_rg = float(saw["Rg"])
    saw_err = float(saw["Rg_err"])

    # Publication/audit table.
    rows = []
    for _, r in dj.iterrows():
        lo, hi, ci_source = ci95(r)
        rg = float(r["Rg"])
        err = float(r["Rg_err"])
        rows.append({
            "model": "DJ",
            "N": N_TARGET,
            "w": float(r["w_num"]),
            "Rg": rg,
            "Rg_err": err,
            "CI95_low": lo,
            "CI95_high": hi,
            "quality_flag": str(r["quality_flag"]) if "quality_flag" in r.index else "UNKNOWN",
            "min_ESS": r.get("min_ESS", np.nan),
            "max_block_weight_fraction": r.get("max_block_weight_fraction", np.nan),
            "valid_blocks": r.get("valid_blocks", np.nan),
            "blocks": r.get("blocks", np.nan),
            "delta_vs_strict_SAW": rg - saw_rg,
            "z_vs_strict_SAW": z_vs_saw(rg, err, saw_rg, saw_err),
            "CI_source": ci_source,
        })

    saw_lo, saw_hi, saw_ci_source = ci95(saw)
    rows.append({
        "model": "strict_SAW_endpoint",
        "N": N_TARGET,
        "w": "infinity",
        "Rg": saw_rg,
        "Rg_err": saw_err,
        "CI95_low": saw_lo,
        "CI95_high": saw_hi,
        "quality_flag": str(saw["quality_flag"]) if "quality_flag" in saw.index else "UNKNOWN",
        "min_ESS": saw.get("min_ESS", np.nan),
        "max_block_weight_fraction": saw.get("max_block_weight_fraction", np.nan),
        "valid_blocks": saw.get("valid_blocks", np.nan),
        "blocks": saw.get("blocks", np.nan),
        "delta_vs_strict_SAW": 0.0,
        "z_vs_strict_SAW": 0.0,
        "CI_source": saw_ci_source,
    })

    data = pd.DataFrame(rows)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_csv = outdir / "Fig5_DJ_Rg_Crossover_Data.csv"
    audit_csv = outdir / "Fig5_DJ_Rg_Crossover_Audit.csv"
    data.to_csv(data_csv, index=False)
    data.to_csv(audit_csv, index=False)

    w8 = dj.loc[np.isclose(dj["w_num"], 8.0)].iloc[0]
    w8_z = z_vs_saw(float(w8["Rg"]), float(w8["Rg_err"]), saw_rg, saw_err)

    prov = {
        "script": "Fig5_DJ_Rg_PLOT_REVISED_FINAL.py",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "dj_master": str(args.dj_master.resolve()),
            "saw_master": str(args.saw_master.resolve()),
            "dj_blocks": str(args.dj_blocks.resolve()) if args.dj_blocks else None,
            "dj_report": str(args.dj_report.resolve()) if args.dj_report else None,
            "dj_provenance": str(args.dj_provenance.resolve()) if args.dj_provenance else None,
        },
        "N": N_TARGET,
        "finite_w": W_VALUES,
        "Rg_definition": "canonical master estimator Rg = sqrt(<Rg^2>_W)",
        "w0_reference": {
            "formula": "(N^2-1)/(6N) * a^2 under Rg^2",
            "Rg_exact": rg0,
            "Rg_measured": rg_w0,
            "Rg_measured_err": rg_w0_err,
            "difference": rg_w0 - rg0,
        },
        "strict_SAW": {
            "interpretation": "mathematical w -> infinity limit",
            "Rg": saw_rg,
            "Rg_err": saw_err,
        },
        "reviewer_1d_check": {
            "Rg_w_0_5": float(dj.loc[np.isclose(dj["w_num"], 0.5), "Rg"].iloc[0]),
            "Rg_w_0_5_err": float(dj.loc[np.isclose(dj["w_num"], 0.5), "Rg_err"].iloc[0]),
            "strict_SAW_Rg": saw_rg,
            "w_0_5_below_strict_SAW": bool(
                float(dj.loc[np.isclose(dj["w_num"], 0.5), "Rg"].iloc[0]) < saw_rg
            ),
            "Rg_w_8": float(w8["Rg"]),
            "Rg_w_8_vs_SAW_z": w8_z,
            "interpretation": (
                "Finite-w DJ values are treated as crossover states; "
                "the strict-SAW endpoint is not identified with any finite w."
            ),
        },
        "qa": {
            "dj_rows": int(len(dj)),
            "unique_w": bool(dj["w_num"].is_unique),
            "monotonic_Rg": bool(np.all(np.diff(dj["Rg"].astype(float)) >= 0)),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "command": " ".join(sys.argv),
    }
    prov_path = outdir / "Fig5_DJ_Rg_Crossover_Provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=float), encoding="utf-8")

    png, pdf = plot(dj, saw, outdir)
    print("=" * 90)
    print("REVISED FIGURE 5 COMPLETE")
    print("=" * 90)
    print(f"DJ N=200 master: {args.dj_master.resolve()}")
    print(f"SAW N=200 master: {args.saw_master.resolve()}")
    print()
    for _, r in dj.iterrows():
        print(f"w={float(r['w_num']):g}: Rg={float(r['Rg']):.6f} ± {float(r['Rg_err']):.6f} [{r.get('quality_flag','UNKNOWN')}]")
    print(f"\nFinite-N lattice Gaussian reference: Rg={rg0:.6f}")
    print(f"Measured DJ w=0: Rg={rg_w0:.6f} ± {rg_w0_err:.6f}")
    print(f"Strict SAW (w→∞): Rg={saw_rg:.6f} ± {saw_err:.6f}")
    print(f"w=8 vs strict SAW combined z = {w8_z:.3f}")
    print(f"\nPNG: {png.resolve()}")
    print(f"PDF: {pdf.resolve()}")
    print(f"Data: {data_csv.resolve()}")
    print(f"Provenance: {prov_path.resolve()}")


if __name__ == "__main__":
    main()
