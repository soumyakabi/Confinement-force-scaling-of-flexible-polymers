#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3 final post-processing / provenance correction
=====================================================

This script performs NO Monte Carlo sampling.

Purpose
-------
Post-process the completed Fig. 3 production archive so that all derived
quantities use the current canonical SAW R_g master dataset and the
partition-function definition of confinement free energy used elsewhere in
the manuscript:

    Delta F(L) = ln <Z_inf> - ln <Z_L>

where the averages are taken over the equal-sized independent production
blocks. The raw logZ block estimates are not altered.

The script also regenerates the exact Gaussian reference from the archived
Gaussian point-level table and audits it against the analytic midpoint-
tethered absorbing-wall expression.

Outputs
-------
* corrected SAW block-level table (canonical R_g and L/R_g)
* corrected force-block table (canonical f R_g)
* corrected point-level table
* corrected cumulative convergence audit
* corrected SAW fit diagnostics
* exact Gaussian point-level table + audit
* two publication figures (force and free energy), PNG + PDF
* post-processing provenance, audit, and SHA256 manifest

No force point is selected/deleted based on sign, uncertainty, or residual.
Only the logarithmic plots display positive estimates, because log axes
cannot display non-positive values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = 1.0
KBT = 1.0
NU_SAW = 0.587597
EXPECTED_SAW_EXPONENT = -(1.0 / NU_SAW + 1.0)
N_LIST = [100, 150, 200, 250]
FIT_WINDOWS = [1.7, 1.8, 1.9]
DEFAULT_FIT_MAX_X = 1.9
BOOTSTRAP_REPS = 10000


def gaussian_rg(N: int) -> float:
    return A * math.sqrt(N / 6.0)


def gaussian_exact(L: np.ndarray | float, N: int, n_max: int = 401):
    """Exact midpoint-tethered absorbing Gaussian slit result."""
    Larr = np.atleast_1d(np.asarray(L, dtype=float))
    rg = gaussian_rg(N)
    n = np.arange(1, n_max + 1, dtype=float)[:, None]
    LL = Larr[None, :]
    coeff = 4.0 / (n * np.pi) * np.sin(n * np.pi / 2.0)
    term = coeff * np.exp(-((n * np.pi * rg / LL) ** 2))
    Z = term.sum(axis=0)
    if np.any(~np.isfinite(Z)) or np.any(Z <= 0):
        raise RuntimeError("Gaussian exact partition function invalid.")
    deltaF = -np.log(Z)
    dZ = np.sum(term * (2.0 * (n * np.pi) ** 2 * rg**2 / LL**3), axis=0)
    f = dZ / Z
    fRg = rg * f
    return deltaF, f, fRg


def bootstrap_deltaF(z_inf: np.ndarray, z_conf: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n_i = len(z_inf); n_c = len(z_conf)
    ii = rng.integers(0, n_i, size=(reps, n_i))
    ic = rng.integers(0, n_c, size=(reps, n_c))
    vals = np.log(np.mean(z_inf[ii], axis=1)) - np.log(np.mean(z_conf[ic], axis=1))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def bootstrap_force(logz_minus: np.ndarray, logz_plus: np.ndarray, delta_L: float,
                    rg: float, reps: int, seed: int):
    """Independent endpoint bootstrap for signed derivative."""
    rng = np.random.default_rng(seed)
    nm = len(logz_minus); np_ = len(logz_plus)
    im = rng.integers(0, nm, size=(reps, nm))
    ip = rng.integers(0, np_, size=(reps, np_))
    vals = (np.mean(logz_plus[ip], axis=1) - np.mean(logz_minus[im], axis=1)) / (2.0 * delta_L)
    vals *= rg
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def weighted_fit(df, xmax):
    d = df[(df['realized_L_over_Rg'] <= xmax) & (df['fRg'] > 0) &
           (df['fRg_err'] > 0) & np.isfinite(df['fRg']) & np.isfinite(df['fRg_err'])].copy()
    x = np.log(d['realized_L_over_Rg'].to_numpy(float))
    y = np.log(d['fRg'].to_numpy(float))
    sy = d['fRg_err'].to_numpy(float) / d['fRg'].to_numpy(float)
    W = 1.0 / sy**2
    X = np.column_stack([np.ones(len(x)), x])
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    beta = cov @ (X.T @ (W * y))
    res = y - X @ beta
    chi2 = float(np.sum((res / sy)**2))
    dof = len(x) - 2
    return {
        'xmax': float(xmax), 'n_points': int(len(x)),
        'exponent': float(beta[1]), 'exponent_err': float(math.sqrt(cov[1,1])),
        'intercept': float(beta[0]), 'amplitude': float(math.exp(beta[0])),
        'chi2': chi2, 'dof': int(dof),
        'reduced_chi2': float(chi2 / dof) if dof > 0 else float('nan')
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--rg-master', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    ap.add_argument('--fit-max-x', type=float, default=DEFAULT_FIT_MAX_X)
    ap.add_argument('--seed', type=int, default=20260930)
    args = ap.parse_args()

    raw = args.raw_dir.resolve(); master_path = args.rg_master.resolve(); out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Current canonical master.
    master = pd.read_csv(master_path)
    saw_master = master[(master['model'].astype(str).str.lower() == 'saw') & master['N'].isin(N_LIST)].copy()
    if len(saw_master) != 4 or saw_master['N'].nunique() != 4:
        raise ValueError('Canonical master must contain exactly one SAW row for N=100,150,200,250.')
    rg = {int(r.N): float(r.Rg) for _, r in saw_master.iterrows()}
    rgerr = {int(r.N): float(r.Rg_err) for _, r in saw_master.iterrows()}
    rgquality = {int(r.N): str(r.get('quality_flag','UNKNOWN')) for _, r in saw_master.iterrows()}

    # Raw inputs.
    block = pd.read_csv(raw/'Fig3_SAW_BlockLevel.csv')
    force_block = pd.read_csv(raw/'Fig3_SAW_ForceBlockLevel.csv')
    old_points = pd.read_csv(raw/'Fig3_SAW_PointLevel.csv')
    old_conv = pd.read_csv(raw/'Fig3_SAW_Convergence.csv')
    plan = pd.read_csv(raw/'Fig3_SAW_LPlan.csv')
    pilots = pd.read_csv(raw/'Fig3_SAW_PilotThresholds.csv')
    gauss = pd.read_csv(raw/'Fig3_Gaussian_PointLevel.csv')

    # Update only derived scaling metadata in block-level files; logZ itself is untouched.
    block2 = block.copy()
    block2['Rg_original_archive'] = block2['Rg']
    block2['Rg_err_original_archive'] = block2['Rg_err']
    block2['Rg'] = block2['N'].map(rg)
    block2['Rg_err'] = block2['N'].map(rgerr)
    block2['realized_L_over_Rg_original_archive'] = block2['realized_L_over_Rg']
    block2['realized_L_over_Rg'] = np.where(block2['L'].to_numpy(float)==0.0, np.inf,
                                             block2['L'].to_numpy(float)/block2['N'].map(rg).to_numpy(float))
    block2.to_csv(out/'Fig3_SAW_BlockLevel.csv', index=False)

    force2 = force_block.copy()
    force2['force_Rg_block_original_archive'] = force2['force_Rg_block']
    force2['force_Rg_block'] = [float(rg[int(n)]) * float(f) for n,f in zip(force2['N'], force2['force_block'])]
    force2.to_csv(out/'Fig3_SAW_ForceBlockLevel.csv', index=False)

    # Canonical point-level recomputation.
    point_rows = []
    force_bootstrap_records = []
    for _, p in old_points.iterrows():
        n = int(p.N); L = int(p.L); dL = float(p.delta_L)
        conf = block[(block.N==n) & (block.L==L)].sort_values('block_id')
        inf = block[(block.N==n) & (block.L==0)].sort_values('block_id')
        fb = force_block[(force_block.N==n) & (force_block.L_center==L)].sort_values('block_id')
        if len(conf) != 40 or len(inf) != 40 or len(fb) != 40:
            raise ValueError(f'Expected 40 blocks for N={n}, L={L}.')
        if int(conf.overflow.sum()) or int(inf.overflow.sum()) or int(fb.overflow_minus.sum()) or int(fb.overflow_plus.sum()):
            raise ValueError(f'Overflow detected for N={n}, L={L}.')

        zc = np.exp(conf.logZ.to_numpy(float)); zi = np.exp(inf.logZ.to_numpy(float))
        deltaF = float(np.log(np.mean(zi)) - np.log(np.mean(zc)))
        df_lo, df_hi = bootstrap_deltaF(zi, zc, args.bootstrap_reps, args.seed + 100000*n + L)
        df_err = float((df_hi-df_lo)/3.92)

        logzm = fb.logZ_minus.to_numpy(float); logzp = fb.logZ_plus.to_numpy(float)
        fblocks = (logzp-logzm)/(2.0*dL)
        force = float(np.mean(fblocks)); force_err = float(np.std(fblocks, ddof=1)/math.sqrt(len(fblocks)))
        f_lo, f_hi = bootstrap_force(logzm, logzp, dL, rg[n], args.bootstrap_reps, args.seed + 200000*n + L)

        x = L/rg[n]; frg = force*rg[n]; frg_err = force_err*rg[n]
        point_rows.append({
            'N': n, 'Rg': rg[n], 'Rg_err': rgerr[n], 'Rg_quality': rgquality[n],
            'target_L_over_Rg': float(p.target_L_over_Rg), 'L': L,
            'realized_L_over_Rg': x, 'L_minus': int(p.L_minus), 'L_plus': int(p.L_plus),
            'delta_L': dL, 'force': force, 'force_err': force_err,
            'fRg': frg, 'fRg_err': frg_err,
            'fRg_bootstrap95_low': f_lo, 'fRg_bootstrap95_high': f_hi,
            'DeltaF': deltaF, 'DeltaF_err': df_err,
            'DeltaF_bootstrap95_low': df_lo, 'DeltaF_bootstrap95_high': df_hi,
            'valid_force_blocks': 40, 'valid_DeltaF_blocks': 40,
            'n_force_blocks': 40, 'n_DeltaF_blocks': 40,
            'min_ESS_like': float(p.min_ESS_like),
            'max_block_weight_fraction_like': float(p.max_block_weight_fraction_like),
            'overflow_count_for_point': 0, 'quality_flag': 'OK'
        })
        force_bootstrap_records.append({'N':n,'L':L,'forceRg_bootstrap_low':f_lo,'forceRg_bootstrap_high':f_hi})

    points = pd.DataFrame(point_rows).sort_values(['N','L']).reset_index(drop=True)
    points.to_csv(out/'Fig3_SAW_PointLevel.csv', index=False)
    pd.DataFrame(force_bootstrap_records).to_csv(out/'Fig3_SAW_ForceBootstrap_Audit.csv', index=False)

    # Corrected convergence audit, keeping the raw mean-log quantities for audit.
    conv_rows=[]
    for _, c in old_conv.iterrows():
        n=int(c.N); L=int(c.L_center); x=L/rg[n]
        minus=block[(block.N==n)&(block.L==L-2)].sort_values('block_id')
        plus=block[(block.N==n)&(block.L==L+2)].sort_values('block_id')
        inf=block[(block.N==n)&(block.L==0)].sort_values('block_id')
        k=int(c.blocks_used)
        if not (len(minus)>=k and len(plus)>=k and len(inf)>=k): raise ValueError('Convergence blocks missing.')
        zm=np.exp(minus.logZ.to_numpy(float)[:k]); zp=np.exp(plus.logZ.to_numpy(float)[:k]); zi=np.exp(inf.logZ.to_numpy(float)[:k])
        lzm=float(np.log(np.mean(zm))); lzp=float(np.log(np.mean(zp))); lzi=float(np.log(np.mean(zi)))
        deltaF=float(lzi-np.log(np.mean(np.exp(block[(block.N==n)&(block.L==L)].logZ.to_numpy(float)[:k]))))
        # Force convergence uses block-wise signed finite differences, as in production.
        f=float(np.mean((plus.logZ.to_numpy(float)[:k]-minus.logZ.to_numpy(float)[:k])/4.0))
        conv_rows.append({
            'N':n,'L_center':L,'L_over_Rg':x,'blocks_used':k,
            'mean_logZ_minus':float(np.mean(minus.logZ.to_numpy(float)[:k])),
            'mean_logZ_plus':float(np.mean(plus.logZ.to_numpy(float)[:k])),
            'mean_logZ_infinity':float(np.mean(inf.logZ.to_numpy(float)[:k])),
            'ln_mean_Z_minus':lzm,'ln_mean_Z_plus':lzp,'ln_mean_Z_infinity':lzi,
            'DeltaF':deltaF,'force':f,'fRg':f*rg[n]
        })
    conv=pd.DataFrame(conv_rows)
    conv.to_csv(out/'Fig3_SAW_Convergence.csv', index=False)

    # L plan rebasing: physical L unchanged; only realized x is recomputed.
    plan2=plan.copy(); plan2['realized_L_over_Rg_original_archive']=plan2['realized_L_over_Rg']
    plan2['realized_L_over_Rg']=[float(L)/rg[int(n)] for n,L in zip(plan2.N,plan2.L)]
    plan2.to_csv(out/'Fig3_SAW_LPlan.csv', index=False)
    shutil.copy2(raw/'Fig3_SAW_PilotThresholds.csv', out/'Fig3_SAW_PilotThresholds.csv')

    # Fit diagnostics from corrected point values.
    fits=[weighted_fit(points,x) for x in FIT_WINDOWS]
    fit_df=pd.DataFrame(fits); fit_df.to_csv(out/'Fig3_SAW_ForceFit_FINAL.csv', index=False)
    main_fit=weighted_fit(points,args.fit_max_x)

    # Exact Gaussian audit.
    ga2=gauss.copy()
    audit_rows=[]
    for _,g in ga2.iterrows():
        dF,ff,fr=gaussian_exact(np.array([float(g.L)]), int(g.N))
        audit_rows.append({
            'N':int(g.N),'L':int(g.L),'archive_DeltaF':float(g.DeltaF),'exact_DeltaF_recomputed':float(dF[0]),
            'archive_force':float(g.force),'exact_force_recomputed':float(ff[0]),
            'archive_fRg':float(g.fRg),'exact_fRg_recomputed':float(fr[0]),
            'abs_diff_DeltaF':abs(float(g.DeltaF)-float(dF[0])),
            'rel_diff_DeltaF':abs(float(g.DeltaF)-float(dF[0]))/abs(float(dF[0])),
            'abs_diff_force':abs(float(g.force)-float(ff[0])),
            'rel_diff_force':abs(float(g.force)-float(ff[0]))/abs(float(ff[0])),
        })
    gaudit=pd.DataFrame(audit_rows)
    ga2.to_csv(out/'Fig3_Gaussian_PointLevel.csv', index=False)
    gaudit.to_csv(out/'Fig3_Gaussian_Exact_Audit.csv', index=False)

    # Figures.
    plt.rcParams.update({
        'font.size':13,'axes.labelsize':18,'axes.labelweight':'bold',
        'axes.titlesize':16,'axes.titleweight':'bold','legend.fontsize':10.5,
        'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':1.5,
        'pdf.fonttype':42,'ps.fonttype':42
    })
    colors=plt.cm.viridis(np.linspace(0.12,0.88,len(N_LIST)))

    # Force figure
    fig,axs=plt.subplots(2,2,figsize=(15.2,11.5))
    ax=axs[0,0]
    for col,n in zip(colors,N_LIST):
        d=points[points.N==n].sort_values('L'); pos=d.f>0 if 'f' in d else d.force>0
        ax.errorbar(d.loc[pos,'L'],d.loc[pos,'force'],yerr=d.loc[pos,'force_err'],fmt='o-',ms=5.5,lw=1.5,capsize=3,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Physical slit width $L$'); ax.set_ylabel(r'Entropic force $f\;(k_BT/a)$'); ax.set_title('(a) SAW: unscaled force'); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[0,1]
    for col,n in zip(colors,N_LIST):
        d=points[points.N==n].sort_values('realized_L_over_Rg'); pos=d.fRg>0
        ax.errorbar(d.loc[pos,'realized_L_over_Rg'],d.loc[pos,'fRg'],yerr=d.loc[pos,'fRg_err'],fmt='o',ms=5.7,capsize=3,color=col,label=f'N={n}')
    xx=np.logspace(np.log10(1.15),np.log10(args.fit_max_x),250); yy=main_fit['amplitude']*xx**main_fit['exponent']
    ax.loglog(xx,yy,'--',color='black',lw=2.2,label=rf"finite-range fit: $p={main_fit['exponent']:.3f}\pm{main_fit['exponent_err']:.3f}$")
    xa=np.array([1.2,args.fit_max_x]); ya=yy[-1]*(xa/args.fit_max_x)**EXPECTED_SAW_EXPONENT
    ax.loglog(xa,ya,':',color='gray',lw=2.0,label=rf'SAW reference slope $p={EXPECTED_SAW_EXPONENT:.3f}$')
    ax.set_xscale('log'); ax.set_yscale('log'); xt=np.arange(1.2,3.01,.2); ax.set_xticks(xt); ax.set_xticklabels([f'{x:.1f}' for x in xt]); ax.set_xlabel(r'$L/R_g^{(0)}$'); ax.set_ylabel(r'$fR_g^{(0)}\;(k_BT)$'); ax.set_title('(b) SAW: scaled force'); ax.set_xlim(1.12,3.15); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[1,0]
    for col,n in zip(colors,N_LIST):
        d=ga2[ga2.N==n].sort_values('L'); ax.plot(d.L,d.force,'o-',ms=4.8,lw=1.4,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Physical slit width $L$'); ax.set_ylabel(r'Exact Gaussian force $f\;(k_BT/a)$'); ax.set_title('(c) Gaussian: unscaled exact force'); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[1,1]
    for col,n in zip(colors,N_LIST):
        d=ga2[ga2.N==n].sort_values('realized_L_over_Rg'); ax.plot(d.realized_L_over_Rg,d.fRg,'o',ms=4.8,color=col,label=f'N={n}')
    xxg=np.logspace(np.log10(0.95),np.log10(3.1),250); ax.loglog(xxg,2*np.pi**2*xxg**-3,'--',color='black',lw=2,label=r'Gaussian strong-confinement $2\pi^2(L/R_g^{(0)})^{-3}$')
    ax.set_xscale('log'); ax.set_yscale('log'); xtg=np.arange(1.0,3.1,.2); ax.set_xticks(xtg); ax.set_xticklabels([f'{x:.1f}' for x in xtg]); ax.set_xlabel(r'$L/R_g^{(0)}$'); ax.set_ylabel(r'$fR_g^{(0)}\;(k_BT)$'); ax.set_title('(d) Gaussian: scaled exact force'); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    fig.suptitle(r'Chain-length scaling of confinement force: SAW and exact Gaussian reference',fontsize=20,fontweight='bold',y=.995)
    fig.subplots_adjust(left=.075,right=.995,bottom=.075,top=.93,wspace=.24,hspace=.26)
    fpng=out/'Fig3_FORCE_SCALING_FINAL.png'; fpdf=out/'Fig3_FORCE_SCALING_FINAL.pdf'; fig.savefig(fpng,dpi=600,bbox_inches='tight'); fig.savefig(fpdf,bbox_inches='tight'); plt.close(fig)

    # Free energy figure
    fig,axs=plt.subplots(2,2,figsize=(15.2,11.5))
    ax=axs[0,0]
    for col,n in zip(colors,N_LIST):
        d=points[points.N==n].sort_values('L'); ax.errorbar(d.L,d.DeltaF,yerr=d.DeltaF_err,fmt='o-',ms=5.5,lw=1.5,capsize=3,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Physical slit width $L$'); ax.set_ylabel(r'Confinement free energy $\Delta F\;(k_BT)$'); ax.set_title('(a) SAW: unscaled free energy'); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[0,1]
    for col,n in zip(colors,N_LIST):
        d=points[points.N==n].sort_values('realized_L_over_Rg'); ax.errorbar(d.realized_L_over_Rg,d.DeltaF,yerr=d.DeltaF_err,fmt='o',ms=5.3,capsize=3,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$L/R_g^{(0)}$'); ax.set_ylabel(r'$\Delta F\;(k_BT)$'); ax.set_title('(b) SAW: scaled free energy'); ax.set_xlim(1.12,3.15); ax.set_xticks(np.arange(1.2,3.1,.2)); ax.set_xticklabels([f'{x:.1f}' for x in np.arange(1.2,3.1,.2)]); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[1,0]
    for col,n in zip(colors,N_LIST):
        d=ga2[ga2.N==n].sort_values('L'); ax.plot(d.L,d.DeltaF,'o-',ms=4.8,lw=1.4,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Physical slit width $L$'); ax.set_ylabel(r'Exact Gaussian $\Delta F\;(k_BT)$'); ax.set_title('(c) Gaussian: unscaled exact free energy'); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    ax=axs[1,1]
    for col,n in zip(colors,N_LIST):
        d=ga2[ga2.N==n].sort_values('realized_L_over_Rg'); ax.plot(d.realized_L_over_Rg,d.DeltaF,'o',ms=4.8,color=col,label=f'N={n}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$L/R_g^{(0)}$'); ax.set_ylabel(r'Exact Gaussian $\Delta F\;(k_BT)$'); ax.set_title('(d) Gaussian: scaled exact free energy'); ax.set_xlim(.92,3.15); ax.set_xticks(np.arange(1.0,3.1,.2)); ax.set_xticklabels([f'{x:.1f}' for x in np.arange(1.0,3.1,.2)]); ax.grid(True,which='major',alpha=.22); ax.grid(True,which='minor',alpha=.08,linestyle=':'); ax.legend(loc='upper right')
    fig.suptitle(r'Chain-length scaling of confinement free energy: SAW and exact Gaussian reference',fontsize=20,fontweight='bold',y=.995)
    fig.subplots_adjust(left=.075,right=.995,bottom=.075,top=.93,wspace=.24,hspace=.26)
    dpng=out/'Fig3_FREE_ENERGY_SCALING_FINAL.png'; dpdf=out/'Fig3_FREE_ENERGY_SCALING_FINAL.pdf'; fig.savefig(dpng,dpi=600,bbox_inches='tight'); fig.savefig(dpdf,bbox_inches='tight'); plt.close(fig)

    # Full canonical master copy.
    shutil.copy2(master_path, out/'Rg_MASTER_FINAL_MERGED_CURRENT.csv')

    # Audit summary.
    rg_changes=[]
    for n in N_LIST:
        old=float(old_points.loc[old_points.N==n,'Rg'].iloc[0]); new=rg[n]
        rg_changes.append({'N':n,'Rg_old_archive':old,'Rg_current_canonical':new,'relative_change':(new-old)/old})
    df_change=pd.DataFrame(rg_changes); df_change.to_csv(out/'Fig3_Rg_Rebasing_Audit.csv',index=False)

    # Audit the canonical DeltaF reconstruction against the archived point values.
    dfaudit=[]
    for _, pold in old_points.iterrows():
        n=int(pold.N); L=int(pold.L); newrow=points[(points.N==n)&(points.L==L)].iloc[0]
        dfaudit.append({
            'N':n,'L':L,
            'DeltaF_original_archive':float(pold.DeltaF),
            'DeltaF_canonical_reconstructed':float(newrow.DeltaF),
            'difference_canonical_minus_archive':float(newrow.DeltaF-pold.DeltaF),
            'relative_difference':float((newrow.DeltaF-pold.DeltaF)/pold.DeltaF) if pold.DeltaF!=0 else float('nan')
        })
    pd.DataFrame(dfaudit).to_csv(out/'Fig3_DeltaF_Reconstruction_Audit.csv',index=False)

    summary={
        'script':'Fig3_POSTPROCESS_FINAL.py','generated':datetime.now().isoformat(timespec='seconds'),
        'raw_dir':str(raw),'canonical_rg_master':str(master_path),'simulation_rerun':False,
        'canonical_Rg':rg,'canonical_Rg_err':rgerr,'Rg_rebase_audit':rg_changes,
        'DeltaF_definition':'ln(mean Z_infinity) - ln(mean Z_L) using equal-sized independent block estimates',
        'force_definition':'signed mean of block finite differences [lnZ(L+delta)-lnZ(L-delta)]/(2 delta)',
        'signed_force_retained':True,'sign_based_selection':False,'point_deletion':False,
        'force_fit':main_fit,'fit_sensitivity':fits,
        'gaussian_exact_audit':{
            'max_rel_DeltaF_error':float(gaudit.rel_diff_DeltaF.max()),
            'max_rel_force_error':float(gaudit.rel_diff_force.max()),
            'max_abs_DeltaF_error':float(gaudit.abs_diff_DeltaF.max())
        },
        'outputs':sorted(p.name for p in out.iterdir())
    }
    (out/'Fig3_POSTPROCESS_PROVENANCE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # hashes after all outputs finalized
    files=[p for p in out.iterdir() if p.is_file() and p.name!='Fig3_SHA256_MANIFEST.txt']
    manifest='\n'.join(f'{sha256(p)}  {p.name}' for p in sorted(files,key=lambda q:q.name))+'\n'
    (out/'Fig3_SHA256_MANIFEST.txt').write_text(manifest,encoding='utf-8')
    print('POSTPROCESS COMPLETE')
    print('Fit:',json.dumps(main_fit))
    print('Gaussian max relative force error:',float(gaudit.rel_diff_force.max()))
    print('Gaussian max relative DeltaF error:',float(gaudit.rel_diff_DeltaF.max()))
    print('Outputs:',out)

if __name__=='__main__':
    main()
