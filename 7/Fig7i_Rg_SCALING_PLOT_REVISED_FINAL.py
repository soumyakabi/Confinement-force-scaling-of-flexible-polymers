#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig7i_Rg_SCALING_PLOT_REVISED_FINAL.py

Figure 7(i): finite-w Domb–Joyce unconfined R_g(N) scaling at w=0.1.
PLOT/AUDIT ONLY: no Monte Carlo sampling is performed here.

Reads the canonical Rg_MASTER_FINAL.csv and Rg_MASTER_FINAL_BLOCKS.csv.
The canonical estimator is
    R_g = sqrt(<R_g^2>_W),
with block-level R_g estimates averaged for the published SEM.

Main figure defaults to the manuscript's N={100,150,200,250}; an extended
finite-size audit uses N={500,1000,2000} when present in the same master.
The fitted exponent is a finite-N, finite-w effective exponent, NOT the
asymptotic SAW exponent.
"""
from __future__ import annotations

import argparse, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

W_TARGET = 0.1
NU_G = 0.5
NU_SAW = 0.587597
CORE_DEFAULT = [100,150,200,250]
AUDIT_DEFAULT = [500,1000,2000]
BOOT_DEFAULT = 4000
SEED_DEFAULT = 20260927

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':13,'axes.labelsize':17,'axes.labelweight':'bold',
    'axes.titlesize':15,'axes.titleweight':'bold','xtick.labelsize':13,
    'ytick.labelsize':13,'legend.fontsize':10,'axes.linewidth':1.3,
    'savefig.dpi':600,
})


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def model_norm(x):
    return str(x).strip().lower().replace('–','-').replace(' ','').replace('_','')


def safe_sem(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    return float(np.std(a,ddof=1)/np.sqrt(a.size)) if a.size>=2 else np.nan


def ci95(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5))) if a.size else (np.nan,np.nan)


def load_master(path):
    df=pd.read_csv(path)
    req={'model','N','w','Rg','Rg_err','quality_flag'}
    miss=req-set(df.columns)
    if miss: raise ValueError(f'Missing columns: {sorted(miss)}')
    df['_model']=df['model'].map(model_norm)
    df['N']=pd.to_numeric(df['N'],errors='coerce').astype('Int64')
    df['w_num']=pd.to_numeric(df['w'],errors='coerce')
    df['Rg']=pd.to_numeric(df['Rg'],errors='coerce')
    df['Rg_err']=pd.to_numeric(df['Rg_err'],errors='coerce')
    dj=df[(df['_model']=='dj') & np.isclose(df['w_num'].astype(float),W_TARGET,rtol=0,atol=1e-12)].copy()
    if dj.empty: raise ValueError('No DJ rows at w=0.1 in master CSV.')
    counts=dj['N'].value_counts(); dup=counts[counts>1]
    if not dup.empty: raise ValueError(f'Duplicate DJ w=0.1 N rows: {dup.index.tolist()}')
    return dj


def load_blocks(path, Ns):
    df=pd.read_csv(path)
    req={'model','N','w','block','rg_block','ESS','max_weight_fraction'}
    miss=req-set(df.columns)
    if miss: raise ValueError(f'Missing block columns: {sorted(miss)}')
    df['_model']=df['model'].map(model_norm)
    df['N']=pd.to_numeric(df['N'],errors='coerce').astype('Int64')
    df['w_num']=pd.to_numeric(df['w'],errors='coerce')
    for c in ['rg_block','ESS','max_weight_fraction']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df[(df['_model']=='dj') & np.isclose(df['w_num'].astype(float),W_TARGET,rtol=0,atol=1e-12) & df['N'].isin(Ns)].copy()


def wls_logfit(N,Rg,err):
    N=np.asarray(N,float); Rg=np.asarray(Rg,float); err=np.asarray(err,float)
    m=np.isfinite(N)&np.isfinite(Rg)&np.isfinite(err)&(N>0)&(Rg>0)&(err>0)
    if m.sum()<3: return None
    x=np.log(N[m]); y=np.log(Rg[m]); sy=err[m]/Rg[m]; wt=1/sy**2
    S=np.sum(wt); Sx=np.sum(wt*x); Sy=np.sum(wt*y); Sxx=np.sum(wt*x*x); Sxy=np.sum(wt*x*y)
    den=S*Sxx-Sx*Sx
    if den<=0: return None
    slope=(S*Sxy-Sx*Sy)/den; intercept=(Sxx*Sy-Sx*Sxy)/den
    slope_err=np.sqrt(S/den)
    pred=intercept+slope*x; res=y-pred
    chi2=float(np.sum((res/sy)**2)); dof=max(len(x)-2,1)
    sst=float(np.sum((y-y.mean())**2)); ssr=float(np.sum(res**2))
    return {'nu_eff':float(slope),'nu_eff_err_WLS':float(slope_err),
            'amplitude':float(np.exp(intercept)),'log_amplitude':float(intercept),
            'chi2':chi2,'dof':dof,'chi2_red':chi2/dof,
            'R2_log':1-ssr/sst if sst>0 else np.nan,
            'rms_log_residual':float(np.sqrt(np.mean(res**2))),
            'max_abs_log_residual':float(np.max(np.abs(res))),
            'N_min':float(N[m].min()),'N_max':float(N[m].max()),'n_points':int(m.sum())}


def bootstrap_slope(blocks_by_N, err_by_N, reps, seed):
    rng=np.random.default_rng(seed); Ns=sorted(blocks_by_N); vals=[]
    if len(Ns)<3: return np.nan,np.nan,np.nan,0
    for _ in range(reps):
        means=[]; errs=[]; ok=True
        for N in Ns:
            b=np.asarray(blocks_by_N[N],float); b=b[np.isfinite(b)]
            if b.size<3: ok=False; break
            draw=rng.choice(b,size=b.size,replace=True); means.append(draw.mean()); errs.append(err_by_N[N])
        if not ok: continue
        fit=wls_logfit(Ns,means,errs)
        if fit is not None and np.isfinite(fit['nu_eff']): vals.append(fit['nu_eff'])
    if len(vals)<max(100,reps//10): return np.nan,np.nan,np.nan,len(vals)
    a=np.asarray(vals); lo,hi=ci95(a)
    return float(np.median(a)),lo,hi,len(a)


def leave_one_out(Ns,rg,err):
    rows=[]
    for omitted in Ns:
        keep=[n for n in Ns if n!=omitted]; f=wls_logfit(keep,[rg[n] for n in keep],[err[n] for n in keep])
        rows.append({'omitted_N':omitted,'remaining_N':','.join(map(str,keep)),
                     'nu_eff':np.nan if f is None else f['nu_eff'],
                     'nu_eff_err_WLS':np.nan if f is None else f['nu_eff_err_WLS'],
                     'chi2_red':np.nan if f is None else f['chi2_red'],
                     'R2_log':np.nan if f is None else f['R2_log']})
    return rows


def contiguous_fits(Ns,rg,err):
    Ns=sorted(Ns); rows=[]
    for i in range(len(Ns)):
        for j in range(i+3,len(Ns)+1):
            sub=Ns[i:j]; f=wls_logfit(sub,[rg[n] for n in sub],[err[n] for n in sub])
            if f: rows.append({'N_range':f'{sub[0]}-{sub[-1]}','N_values':','.join(map(str,sub)),**f})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rg-master',required=True,type=Path)
    ap.add_argument('--rg-blocks',required=True,type=Path)
    ap.add_argument('--outdir',required=True,type=Path)
    ap.add_argument('--N-list',type=int,nargs='+',default=CORE_DEFAULT)
    ap.add_argument('--audit-N-list',type=int,nargs='+',default=AUDIT_DEFAULT)
    ap.add_argument('--bootstrap-reps',type=int,default=BOOT_DEFAULT)
    ap.add_argument('--seed',type=int,default=SEED_DEFAULT)
    ap.add_argument('--show',action='store_true')
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)

    core=sorted(set(args.N_list)); master=load_master(args.rg_master)
    avail=sorted(int(x) for x in master['N'].dropna().unique())
    miss=[n for n in core if n not in avail]
    if miss: raise ValueError(f'Core N absent from master: {miss}')
    ext=[n for n in sorted(set(args.audit_N_list)) if n in avail]
    Ns_all=sorted(set(core+ext)); blocks=load_blocks(args.rg_blocks,Ns_all)
    rg={}; er={}; qual={}; point=[]; bmap={}
    for _,r in master[master['N'].isin(Ns_all)].sort_values('N').iterrows():
        N=int(r['N']); rg[N]=float(r['Rg']); er[N]=float(r['Rg_err']); qual[N]=str(r['quality_flag'])
        b=blocks.loc[blocks['N'].astype(int)==N,'rg_block'].dropna().to_numpy(float)
        if b.size<3: raise ValueError(f'Fewer than 3 blocks for N={N}')
        bmap[N]=b
        point.append({'model':'DJ','w':W_TARGET,'N':N,'Rg':rg[N],'Rg_err_SEM':er[N],
                      'quality_flag':qual[N],'n_blocks':int(b.size),'block_Rg_mean':float(b.mean()),
                      'block_Rg_SEM':safe_sem(b), 'min_ESS':float(blocks.loc[blocks['N'].astype(int)==N,'ESS'].min()),
                      'max_block_weight_fraction':float(blocks.loc[blocks['N'].astype(int)==N,'max_weight_fraction'].max())})

    fit=wls_logfit(core,[rg[n] for n in core],[er[n] for n in core])
    if fit is None: raise RuntimeError('Core finite-N fit failed.')
    med,lo,hi,nok=bootstrap_slope({n:bmap[n] for n in core},{n:er[n] for n in core},args.bootstrap_reps,args.seed)
    loo=leave_one_out(core,rg,er); extfits=contiguous_fits(Ns_all,rg,er)

    residual=[]; A=fit['amplitude']; nu=fit['nu_eff']
    for n in core:
        pred=A*n**nu; residual.append({'N':n,'Rg_observed':rg[n],'Rg_fit':pred,
                                       'relative_residual':(rg[n]-pred)/pred,
                                       'log_residual':math.log(rg[n]/pred),'quality_flag':qual[n]})

    # ---------- figure ----------
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.6))
    x=np.logspace(np.log10(min(core)*0.85),np.log10(max(core)*1.15),300); pivot=200.0
    pivot_rg=rg[200] if 200 in rg else float(np.exp(np.interp(np.log(pivot),np.log(core),np.log([rg[n] for n in core]))))
    gref=pivot_rg*(x/pivot)**NU_G; sref=pivot_rg*(x/pivot)**NU_SAW

    y=[rg[n] for n in core]; ye=[er[n] for n in core]
    ax1.errorbar(core,y,yerr=ye,fmt='o',markersize=8,capsize=4,capthick=1.4,elinewidth=1.3,label=fr'Domb–Joyce, $w={W_TARGET:g}$',zorder=4)
    ax1.plot(x,gref,':',lw=1.5,label='Gaussian slope reference: ν=1/2')
    ax1.plot(x,sref,'--',lw=1.5,label=f'SAW slope reference: ν={NU_SAW:.4f}')
    ax1.set_xlabel(r'Chain length $N$'); ax1.set_ylabel(r'Unconfined radius of gyration $R_g$')
    ax1.set_title('(a) Size scaling'); ax1.grid(True,alpha=.25); ax1.set_xlim(min(core)*.92,max(core)*1.08)
    ax1.legend(loc='upper left',frameon=True,fontsize=10)

    ax2.errorbar(core,y,yerr=ye,fmt='o',markersize=8,capsize=4,capthick=1.4,elinewidth=1.3,zorder=4)
    ax2.plot(x,fit['amplitude']*x**fit['nu_eff'],lw=2.2,label=f'Finite-N effective fit: ν_eff={med:.3f}\n95% block-bootstrap CI [{lo:.3f}, {hi:.3f}]')
    ax2.plot(x,gref,':',lw=1.4,label='Gaussian slope: ν=1/2'); ax2.plot(x,sref,'--',lw=1.4,label=f'SAW slope: ν={NU_SAW:.4f}')
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_xlabel(r'Chain length $N$'); ax2.set_ylabel(r'$R_g$')
    ax2.set_title(rf'(b) Finite-$N$ scaling at $w={W_TARGET:g}$'); ax2.grid(True,which='both',alpha=.25)
    ax2.text(.04,.05,'Finite-range effective exponent\nnot an asymptotic SAW exponent',transform=ax2.transAxes,fontsize=10,va='bottom',ha='left',
             bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor='black',lw=.7,alpha=.9))
    ax2.legend(loc='upper left',frameon=True,fontsize=9.3)
    fig.suptitle(rf'Figure 7(i): Finite-$w$ Domb–Joyce size scaling, $w={W_TARGET:g}$',fontsize=17,fontweight='bold',y=.995)
    fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.88,wspace=.24)
    png=args.outdir/'Fig7i_Rg_Scaling_REVISED.png'; pdf=args.outdir/'Fig7i_Rg_Scaling_REVISED.pdf'
    fig.savefig(png,dpi=600,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight')
    if args.show: plt.show()
    else: plt.close(fig)

    # ---------- outputs ----------
    pd.DataFrame(point).sort_values('N').to_csv(args.outdir/'Fig7i_PointLevel.csv',index=False)
    blocks[blocks['N'].astype(int).isin(Ns_all)].to_csv(args.outdir/'Fig7i_BlockLevel.csv',index=False)
    pd.DataFrame([{**fit,'analysis':'core_main_figure','w':W_TARGET,'N_values':','.join(map(str,core)),
                    'nu_eff_bootstrap_median':med,'nu_eff_bootstrap_CI95_low':lo,'nu_eff_bootstrap_CI95_high':hi,
                    'bootstrap_successes':nok,'bootstrap_reps_requested':args.bootstrap_reps,
                    'nu_gaussian_reference':NU_G,'nu_saw_reference':NU_SAW}]).to_csv(args.outdir/'Fig7i_FitDiagnostics.csv',index=False)
    pd.DataFrame(loo).to_csv(args.outdir/'Fig7i_LeaveOneOut.csv',index=False)
    pd.DataFrame(extfits).to_csv(args.outdir/'Fig7i_ExtendedRangeFits.csv',index=False)
    pd.DataFrame(residual).to_csv(args.outdir/'Fig7i_Residuals.csv',index=False)

    prov={
      'figure':'Fig. 7(i)','script':'Fig7i_Rg_SCALING_PLOT_REVISED_FINAL.py','version':'1.0',
      'timestamp':pd.Timestamp.now().isoformat(),'command_line':' '.join(sys.argv),
      'model_identity':'Finite-w Domb–Joyce, unconfined 3D cubic lattice, fixed w=0.1; not strict SAW; strict SAW is w->infinity.',
      'Rg_definition':'Rg=sqrt(<Rg^2>_W), Rg^2=(1/N) sum_i |r_i-r_cm|^2, weighted using the canonical Rosenbluth/SIS ensemble.',
      'uncertainty':'SEM across independent canonical block-level Rg estimates; 95% percentile block bootstrap reported as audit/fit uncertainty.',
      'primary_N_values':core,'extended_audit_N_values_available':ext,'extended_audit_N_values_missing':[n for n in sorted(set(args.audit_N_list)) if n not in avail],
      'fit_definition':'ln(Rg)=ln(A)+nu_eff ln(N), weighted by inverse variance in ln(Rg).',
      'core_fit':{**fit,'bootstrap_median':med,'bootstrap_CI95_low':lo,'bootstrap_CI95_high':hi,'bootstrap_successes':nok},
      'leave_one_N_out':loo,'extended_range_fits':extfits,'residuals':residual,
      'quality_policy':'Master quality flags retained; no point deleted after inspection; no finite-w point identified as strict SAW.',
      'reference_slopes':{'Gaussian':NU_G,'SAW':NU_SAW,'normalization':'references are normalized at N=200 for visual comparison only, not finite-w amplitude predictions.'},
      'master_files':{'Rg_MASTER_FINAL.csv':{'path':str(args.rg_master.resolve()),'sha256':sha256_file(args.rg_master)},
                      'Rg_MASTER_FINAL_BLOCKS.csv':{'path':str(args.rg_blocks.resolve()),'sha256':sha256_file(args.rg_blocks)}},
      'outputs':{'png':str(png.resolve()),'pdf':str(pdf.resolve())}
    }
    with open(args.outdir/'Fig7i_Provenance.json','w',encoding='utf-8') as f: json.dump(prov,f,indent=2,default=float)
    summary=(f'Fig. 7(i) revised\nw={W_TARGET}\ncore N={core}\n'
             f'nu_eff WLS={fit["nu_eff"]:.6f} +/- {fit["nu_eff_err_WLS"]:.6f}\n'
             f'bootstrap median={med:.6f}\nbootstrap 95% CI=[{lo:.6f},{hi:.6f}]\n'
             f'chi2/dof={fit["chi2_red"]:.4f}\nR2_log={fit["R2_log"]:.6f}\n')
    (args.outdir/'Fig7i_RunSummary.txt').write_text(summary,encoding='utf-8')
    print(summary); print(f'PNG: {png}'); print(f'PDF: {pdf}'); print(f'Provenance: {args.outdir/"Fig7i_Provenance.json"}')

if __name__=='__main__': main()
