#!/usr/bin/env python3
"""
Fig. 1 REV9 plotting patch — no Monte Carlo rerun.

Purpose
-------
Replace the force panel of Fig. 1 with a reviewer-transparent comparison:
  * exact chain-rule Gaussian force;
  * exact central finite-difference force with the production delta L=2;
  * Monte Carlo force estimates produced with the same delta L=2;
  * Gaussian strong-confinement reference.

The numerical Monte Carlo dataset is read from the frozen REV9 output files.
No MC data are changed, filtered, smoothed, or re-estimated.

Inputs
------
Fig1_PointLevel.csv
Fig1_ForcePointLevel.csv

Outputs
-------
Fig1_Gaussian_Confinement_REV9_PATCHED.png/.pdf
Fig1_ForcePatch_Audit.csv
Fig1_Patch_Provenance.json

Usage
-----
python Fig1_REV9_PlotPatch_FINAL.py \
  --input-dir /path/to/Fig1_REV9_FINAL \
  --output-dir /path/to/Fig1_REV9_FINAL_PATCHED
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exact_survival(N,L,a=1.0,tether_fraction=0.5,n_modes=6000):
    D=a*a/6.0
    x0=tether_fraction*L
    S=0.0
    for n in range(1,n_modes+1,2):
        k=n*np.pi/L
        term=(4.0/(n*np.pi))*np.sin(k*x0)*np.exp(-k*k*D*N)
        S += term
        if n>31 and abs(term)<1e-16:
            break
    return S

def exact_F(N,L,a=1.0,kBT=1.0,tether_fraction=0.5):
    S=exact_survival(N,L,a,tether_fraction)
    return -kBT*np.log(S)

def exact_force_chain(N,L,a=1.0,kBT=1.0,tether_fraction=0.5,n_modes=6000):
    D=a*a/6.0
    x0=tether_fraction*L
    S=dS=0.0
    for n in range(1,n_modes+1,2):
        k=n*np.pi/L
        decay=np.exp(-k*k*D*N)
        amp=4.0/(n*np.pi)
        s=np.sin(k*x0); c=np.cos(k*x0)
        term=amp*s*decay
        S += term
        dk=-n*np.pi/(L*L)
        dtheta=dk*x0+k*tether_fraction
        dS += amp*(c*dtheta*decay + s*decay*(-2.0*k*D*N*dk))
        if n>31 and abs(term)<1e-16:
            break
    return kBT*dS/S

def exact_force_fd(N,L,delta=2.0,a=1.0,kBT=1.0,tether_fraction=0.5):
    if L-delta <= 0:
        return np.nan
    return -(exact_F(N,L+delta,a,kBT,tether_fraction)
             - exact_F(N,L-delta,a,kBT,tether_fraction))/(2.0*delta)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=None)
    ap.add_argument("--delta-L",type=float,default=2.0)
    ap.add_argument("--exact-min-ratio",type=float,default=0.6)
    ap.add_argument("--exact-max-ratio",type=float,default=5.0)
    args=ap.parse_args()
    outdir=args.output_dir or (args.input_dir/"patched_plot")
    outdir.mkdir(parents=True,exist_ok=True)

    free=pd.read_csv(args.input_dir/"Fig1_PointLevel.csv")
    force=pd.read_csv(args.input_dir/"Fig1_ForcePointLevel.csv")
    N=int(free["N"].iloc[0]); Rg=np.sqrt(N/6.0); kBT=1.0

    x=np.linspace(args.exact_min_ratio,args.exact_max_ratio,700)
    Fex=np.array([exact_F(N,xx*Rg) for xx in x])
    fchain=np.array([exact_force_chain(N,xx*Rg)*Rg/kBT for xx in x])
    centers=force["L"].to_numpy(float)/Rg
    ffd=np.array([exact_force_fd(N,L,args.delta_L)*Rg/kBT for L in force["L"]])

    audit=force[["L","L_over_Rg","delta_L","force_Rg","force_Rg_err"]].copy()
    audit["exact_chain_rule_fRg"]=[exact_force_chain(N,L)*Rg/kBT for L in force["L"]]
    audit["exact_deltaL2_fRg"]=ffd
    audit["MC_minus_exact_deltaL2_percent"]=100*(audit["force_Rg"]-audit["exact_deltaL2_fRg"])/audit["exact_deltaL2_fRg"]
    audit.to_csv(outdir/"Fig1_ForcePatch_Audit.csv",index=False)

    plt.rcParams.update({"font.size":12,"axes.labelsize":15,"axes.labelweight":"bold",
                         "axes.titlesize":15,"legend.fontsize":10,"xtick.labelsize":11,
                         "ytick.labelsize":11,"axes.linewidth":1.3})
    fig,axes=plt.subplots(1,2,figsize=(14.0,5.8))
    ax=axes[0]
    ax.errorbar(free["L_over_Rg"],free["DeltaF_sim"]/kBT,yerr=free["DeltaF_err"]/kBT,
                fmt="o",ms=5.8,capsize=2.5,label="Gaussian MC")
    ax.plot(x,Fex/kBT,linestyle="-",linewidth=2.2,label="Exact eigenmode")
    ax.set_xlabel(r"$L/R_g^{(0)}$"); ax.set_ylabel(r"$\Delta F/k_BT$")
    ax.set_title("(a) Confinement free energy"); ax.set_xlim(args.exact_min_ratio,args.exact_max_ratio)
    ax.set_ylim(bottom=0); ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5])
    ax.grid(True,alpha=0.20); ax.legend(loc="upper right",frameon=True)

    ax=axes[1]
    pos=force["force_Rg"]>0
    ax.errorbar(force.loc[pos,"L_over_Rg"],force.loc[pos,"force_Rg"]/kBT,
                yerr=force.loc[pos,"force_Rg_err"]/kBT,fmt="o",ms=6.0,capsize=2.5,
                label=rf"MC, $\delta L={args.delta_L:g}$")
    ax.plot(x,fchain,linestyle="-",linewidth=2.1,label="Exact chain-rule derivative")
    ax.plot(centers,ffd,linestyle="--",linewidth=2.0,marker="s",ms=3.5,
            label=rf"Exact FD, $\delta L={args.delta_L:g}$")
    gx=np.linspace(0.6,1.55,180); ax.plot(gx,2*np.pi**2*gx**(-3),linestyle=":",
            linewidth=1.4,label=r"Gaussian strong-confinement reference: $2\pi^2(L/R_g^{(0)})^{-3}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L/R_g^{(0)}$"); ax.set_ylabel(r"$fR_g^{(0)}/k_BT$")
    ax.set_title(r"(b) Entropic force and finite-difference validation")
    ax.set_xlim(args.exact_min_ratio,args.exact_max_ratio); ax.set_ylim(0.03,120)
    ax.grid(True,which="both",alpha=0.18); ax.legend(loc="upper right",frameon=True)
    ax.text(0.02,0.035,r"MC uses signed block estimator with $\delta L=2$."+"\n"+
            "Exact FD curve uses the same central difference.",transform=ax.transAxes,
            fontsize=9,va="bottom",ha="left")
    fig.suptitle("Gaussian polymer confinement: exact benchmark and numerical validation",
                 fontsize=19,fontweight="bold",y=0.99)
    fig.subplots_adjust(left=0.08,right=0.985,bottom=0.13,top=0.86,wspace=0.22)
    fig.savefig(outdir/"Fig1_Gaussian_Confinement_REV9_PATCHED.png",dpi=600,bbox_inches="tight")
    fig.savefig(outdir/"Fig1_Gaussian_Confinement_REV9_PATCHED.pdf",bbox_inches="tight")
    plt.close(fig)

    prov={"script":"Fig1_REV9_PlotPatch_FINAL.py","delta_L":args.delta_L,
          "N":N,"Rg":Rg,"input_dir":str(args.input_dir.resolve()),
          "interpretation":"MC force is compared to the exact finite-difference estimator with the same delta_L; chain-rule force is the continuum derivative reference.",
          "no_simulation_rerun":True,"no_data_filtering":True}
    (outdir/"Fig1_Patch_Provenance.json").write_text(json.dumps(prov,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
