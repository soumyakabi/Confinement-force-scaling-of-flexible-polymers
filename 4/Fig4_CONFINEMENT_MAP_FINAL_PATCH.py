#!/usr/bin/env python3
"""
Fig4_CONFINEMENT_MAP_FINAL_PATCH.py
===================================

Plot-only finalization of Fig. 4 after rebasing the map to the canonical
unconfined SAW R_g^(0) master.

No Monte Carlo simulation is performed.
No Fig. 3 force/free-energy values are altered.
The physical L plan is preserved from the final Fig. 3 production plan.
Only the mapping L/R_g^(0), the uncertainty bands, and same-L diagnostic
are recalculated from the canonical R_g master.

Required inputs
---------------
Rg_MASTER_FINAL_Fig4_SAW_SUBSET.csv
Fig4_Physical_L_Plan.csv

Outputs
-------
Fig4_Confinement_Regime_Map_FINAL.png/.pdf
Fig4_ExploredRanges_FINAL.csv
Fig4_SameL_Diagnostic_FINAL.csv
Fig4_Provenance_FINAL.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_N = [100,150,200,250]
DEFAULT_SAME_L = 16.0

def load_inputs(master_path: Path, lplan_path: Path):
    m = pd.read_csv(master_path)
    required_m = {"N","Rg","Rg_err","CI_low","CI_high","quality_flag"}
    if required_m - set(m.columns):
        raise ValueError(f"Master subset missing columns: {sorted(required_m-set(m.columns))}")
    m["N"] = m["N"].astype(int)
    if set(m["N"]) != set(DEFAULT_N) or len(m) != 4:
        raise ValueError("Expected exactly N=100,150,200,250.")
    if m["N"].duplicated().any():
        raise ValueError("Duplicate N rows in master subset.")

    p = pd.read_csv(lplan_path)
    required_p = {"N","L_min","L_max","n_fig3_points"}
    if required_p - set(p.columns):
        raise ValueError(f"L plan missing columns: {sorted(required_p-set(p.columns))}")
    p["N"] = p["N"].astype(int)
    if set(p["N"]) != set(DEFAULT_N):
        raise ValueError("L plan N set does not match expected N values.")
    return m.sort_values("N").reset_index(drop=True), p.sort_values("N").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--lplan", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("Fig4_FINAL"))
    ap.add_argument("--same-L", type=float, default=DEFAULT_SAME_L)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    m, p = load_inputs(args.master, args.lplan)
    d = p.merge(m, on="N", how="left")
    if d["Rg"].isna().any():
        raise RuntimeError("Missing canonical Rg for one or more N values.")

    d["L_over_Rg_min"] = d["L_min"]/d["Rg"]
    d["L_over_Rg_max"] = d["L_max"]/d["Rg"]
    d.to_csv(args.outdir/"Fig4_ExploredRanges_FINAL.csv", index=False)

    same = d[["N","Rg","Rg_err","quality_flag"]].copy()
    same["same_L"] = args.same_L
    same["same_L_over_Rg"] = args.same_L/same["Rg"]
    same.to_csv(args.outdir/"Fig4_SameL_Diagnostic_FINAL.csv", index=False)

    plt.rcParams.update({
        "font.size":13,"axes.labelsize":18,"axes.labelweight":"bold",
        "axes.titlesize":18,"axes.titleweight":"bold","legend.fontsize":10.2,
        "xtick.labelsize":13,"ytick.labelsize":13,"axes.linewidth":1.6
    })
    fig, ax = plt.subplots(figsize=(11.5,7.5))
    x = np.linspace(0.0, max(35.0, float(d["L_max"].max())*1.15), 500)
    for _, r in d.iterrows():
        rg=float(r["Rg"]); lo=float(r["CI_low"]); hi=float(r["CI_high"]); N=int(r["N"])
        ax.plot(x, x/rg, lw=2.5,
                label=rf"$N={N}$, $R_g^{{(0)}}={rg:.3f}\pm{r['Rg_err']:.3f}$")
        ax.fill_between(x, x/hi, x/lo, alpha=0.10, linewidth=0)
        ax.axvspan(float(r["L_min"]), float(r["L_max"]), alpha=0.025)

    ax.axhline(1.0,color="gray",lw=1.4,ls="--",alpha=0.65)
    ax.axhline(2.0,color="gray",lw=1.4,ls="--",alpha=0.65)
    ax.axvline(args.same_L,color="black",lw=1.5,ls=":",alpha=0.60)

    lines=[rf"$L={args.same_L:g}$"]
    for _, r in same.sort_values("N").iterrows():
        lines.append(rf"$N={int(r['N'])}$: $L/R_g^{{(0)}}={r['same_L_over_Rg']:.2f}$")
    ax.text(0.985,0.965,"\n".join(lines),transform=ax.transAxes,ha="right",va="top",
            fontsize=10.5,bbox=dict(boxstyle="round,pad=0.42",facecolor="white",
            edgecolor="gray",alpha=0.92))
    ax.text(0.02,0.035,"Reference levels: $L/R_g^{(0)}=1$ and $2$ are\n"
            "organizational guides, not phase boundaries.",
            transform=ax.transAxes,ha="left",va="bottom",fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.38",facecolor="white",edgecolor="gray",alpha=0.90))
    ax.set_xlabel(r"Physical slit width $L$ (lattice units)")
    ax.set_ylabel(r"$L/R_g^{(0)}$")
    ax.set_title(r"Physical slit width and the dimensionless confinement scale $L/R_g^{(0)}$")
    ax.set_xlim(0,float(x.max())); ax.set_ylim(0,3.5)
    ax.set_xticks(np.arange(0,36,5)); ax.grid(True,alpha=0.20,linestyle=":")
    ax.legend(loc="lower right",frameon=True)
    fig.text(0.5,0.015,r"$R_g^{(0)}$: unconfined 3D radius of gyration; "
             r"bands show its 95% uncertainty; vertical shading marks the physical $L$ ranges used in Fig. 3.",
             ha="center",va="bottom",fontsize=10.3)
    fig.subplots_adjust(left=0.10,right=0.98,bottom=0.11,top=0.91)
    fig.savefig(args.outdir/"Fig4_Confinement_Regime_Map_FINAL.png",dpi=600,bbox_inches="tight")
    fig.savefig(args.outdir/"Fig4_Confinement_Regime_Map_FINAL.pdf",bbox_inches="tight")
    plt.close(fig)

    prov = {
        "figure":"Fig. 4",
        "script":"Fig4_CONFINEMENT_MAP_FINAL_PATCH.py",
        "purpose":"Plot-only rebase of L/Rg^(0) onto the canonical SAW Rg master.",
        "no_monte_carlo_rerun":True,
        "physical_L_plan_source":"final Fig. 3 production plan",
        "same_L":args.same_L,
        "N_list":DEFAULT_N,
        "reference_levels":[1.0,2.0],
        "reference_level_interpretation":"organizational guides, not phase boundaries",
        "canonical_Rg_rebase":True,
        "Rg_N200_updated_from_old":9.669317863734271,
        "Rg_N200_final":9.744501656839986,
        "changes":"Recomputed all L/Rg values and 95% Rg bands from canonical Rg; physical L plan unchanged."
    }
    (args.outdir/"Fig4_Provenance_FINAL.json").write_text(json.dumps(prov,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
