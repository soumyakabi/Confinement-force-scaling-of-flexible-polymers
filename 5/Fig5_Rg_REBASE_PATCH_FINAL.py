#!/usr/bin/env python3
"""
Fig5_Rg_REBASE_PATCH_FINAL.py

Plot/data-only rebase of the strict-SAW endpoint in Fig. 5 to the current
canonical unconfined R_g master.

No Monte Carlo simulation is rerun.
Finite-w Domb-Joyce measurements are not changed.
Only the strict-SAW endpoint value and all endpoint-relative audit columns
are updated, and the figure is regenerated.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OLD_SAW_RG = 9.669317863734271
OLD_SAW_ERR = 0.0636926796208264
NEW_SAW_RG = 9.744501656839986
NEW_SAW_ERR = 0.0276709510674734
NEW_SAW_LO = 9.690649404460176
NEW_SAW_HI = 9.798977044825792
SAW_X = 8.65

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("Fig5_FINAL"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    mask = df["model"].astype(str).str.lower().eq("strict_saw_endpoint")
    if mask.sum() != 1:
        raise ValueError("Expected exactly one strict-SAW endpoint.")

    df.loc[mask, "Rg"] = NEW_SAW_RG
    df.loc[mask, "Rg_err"] = NEW_SAW_ERR
    df.loc[mask, "CI95_low"] = NEW_SAW_LO
    df.loc[mask, "CI95_high"] = NEW_SAW_HI

    finite = ~mask
    df.loc[finite, "delta_vs_strict_SAW"] = df.loc[finite, "Rg"] - NEW_SAW_RG
    df.loc[finite, "z_vs_strict_SAW"] = (
        (df.loc[finite, "Rg"] - NEW_SAW_RG) /
        np.sqrt(df.loc[finite, "Rg_err"]**2 + NEW_SAW_ERR**2)
    )
    df.loc[mask, "delta_vs_strict_SAW"] = 0.0
    df.loc[mask, "z_vs_strict_SAW"] = 0.0
    df.to_csv(args.outdir/"Fig5_DJ_Rg_Crossover_Data_FINAL.csv", index=False)

    # Recreate the figure from the corrected CSV.
    f = df[finite].sort_values("w")
    saw = df.loc[mask].iloc[0]
    primary = f[f["w"] <= 0.5]
    caution = f[f["w"] >= 1.0]

    plt.rcParams.update({"font.size":13,"axes.labelsize":20,"axes.labelweight":"bold",
                         "axes.titlesize":19,"axes.titleweight":"bold",
                         "legend.fontsize":11,"xtick.labelsize":13,
                         "ytick.labelsize":13,"axes.linewidth":1.5})
    fig, ax = plt.subplots(figsize=(12.5,7.4))
    ax.plot(primary["w"],primary["Rg"],"o-",lw=2.3,ms=7,label="Finite-w Domb–Joyce")
    ax.errorbar(primary["w"],primary["Rg"],yerr=primary["Rg_err"],fmt="none",capsize=4,lw=1.5)
    ax.errorbar(caution["w"],caution["Rg"],yerr=caution["Rg_err"],fmt="o",ms=7,capsize=4,
                mfc="white",lw=1.5,label="Finite-w DJ (sampling caution)")
    w0_exact=float(np.sqrt((200**2-1)/(6*200)))
    ax.axhline(w0_exact,ls="--",lw=2.0,label=rf"$w=0$ lattice Gaussian: $R_g={w0_exact:.4f}$")
    ax.axhspan(saw["CI95_low"],saw["CI95_high"],alpha=0.14,label="Strict-SAW 95% CI ($w\\to\\infty$)")
    ax.axhline(saw["Rg"],ls=":",lw=2.5,label=rf"Strict SAW ($w\\to\\infty$): $R_g={saw['Rg']:.3f}\pm{saw['Rg_err']:.3f}$")
    ax.errorbar([SAW_X],[saw["Rg"]],yerr=[saw["Rg_err"]],fmt="D",ms=11,capsize=5,label="SAW endpoint, $w\\to\\infty$")
    w8=f[np.isclose(f["w"],8.0)].iloc[0]
    ax.annotate(r"$w=8$ (finite)",xy=(8.0,w8["Rg"]),xytext=(7.1,9.12),
                arrowprops=dict(arrowstyle="->",lw=1.2),fontsize=14)
    ax.text(0.02,0.962,"Finite-$w$ points are crossover states; $w\\to\\infty$ denotes the strict-SAW limit.",
            transform=ax.transAxes,ha="left",va="top",fontsize=12,
            bbox=dict(boxstyle="round,pad=0.38",facecolor="white",edgecolor="gray",alpha=0.92))
    ax.set_xlabel(r"Domb–Joyce overlap penalty $w$")
    ax.set_ylabel(r"Unconfined radius of gyration $R_g$ (lattice units)")
    ax.set_title(r"Domb–Joyce swelling toward the strict self-avoiding limit ($N=200$)")
    ax.set_xlim(-0.1,9.25); ax.set_ylim(5.45,10.05)
    ax.set_xticks([0,0.5,1,2,4,6,8,9]); ax.grid(True,alpha=0.20,linestyle=":")
    ax.legend(loc="lower right",frameon=True)
    fig.subplots_adjust(left=0.11,right=0.985,bottom=0.11,top=0.91)
    fig.savefig(args.outdir/"Fig5_DJ_Rg_Crossover_FINAL.png",dpi=600,bbox_inches="tight")
    fig.savefig(args.outdir/"Fig5_DJ_Rg_Crossover_FINAL.pdf",bbox_inches="tight")
    plt.close(fig)

    prov={"figure":"Fig. 5","no_MC_rerun":True,
          "finite_w_values_unchanged":True,
          "old_strict_SAW_Rg":OLD_SAW_RG,
          "new_strict_SAW_Rg":NEW_SAW_RG,
          "new_strict_SAW_Rg_err":NEW_SAW_ERR,
          "new_CI95":[NEW_SAW_LO,NEW_SAW_HI],
          "interpretation":"Corrected strict-SAW endpoint from canonical Rg master; finite-w DJ states are unchanged."}
    (args.outdir/"Fig5_Provenance_FINAL.json").write_text(json.dumps(prov,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
