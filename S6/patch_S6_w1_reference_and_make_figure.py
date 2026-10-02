#!/usr/bin/env python3
"""Patch only the w=1, L=infinity S6 reference and regenerate audits/figure."""
from __future__ import annotations
import csv, json, math, os, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/mnt/data/S6_unzip')
NEW_SUMMARY = Path('/mnt/data/S6_w1_unconfined_highstat_summary.json')
NEW_BLOCKS = Path('/mnt/data/S6_w1_unconfined_highstat_blocks.csv')
OUT = Path('/mnt/data/S6_FINAL_PATCHED')
OUT.mkdir(parents=True, exist_ok=True)

# New high-statistics unconfined reference.
sumj = json.loads(NEW_SUMMARY.read_text())
new_logz_inf = float(sumj['logZ_inf'])
new_logz_se = float(sumj['logZ_SE_block'])


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def logmeanexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x-m))))


def bootstrap_log_ratio(conf, inf, n_boot, seed):
    a=np.asarray(conf,dtype=float); b=np.asarray(inf,dtype=float)
    rng=np.random.default_rng(seed)
    vals=np.empty(n_boot)
    for i in range(n_boot):
        ia=rng.integers(0,len(a),len(a)); ib=rng.integers(0,len(b),len(b))
        vals[i]=logmeanexp(b[ib])-logmeanexp(a[ia])
    lo,hi=np.percentile(vals,[2.5,97.5])
    return float(np.std(vals,ddof=1)),float(lo),float(hi)


def monotonicity(rows):
    rows=sorted(rows,key=lambda r:float(r['L_over_Rg']))
    out=[]
    raw=gt=0
    for a,b in zip(rows[:-1],rows[1:]):
        x1=float(a['L_over_Rg']); x2=float(b['L_over_Rg'])
        y1=float(a['DeltaF']); y2=float(b['DeltaF'])
        e1=float(a['DeltaF_bootstrap_SE']); e2=float(b['DeltaF_bootstrap_SE'])
        d=y2-y1; de=math.sqrt(e1*e1+e2*e2)
        rr=d>0; gg=(d>2*de)
        raw += int(rr); gt += int(gg)
        out.append({'w':a['w'],'x_left':x1,'x_right':x2,
                    'delta_F_right_minus_left':d,'combined_1sigma':de,
                    'z_like':d/de if de>0 else math.nan,
                    'raw_increase':rr,'increase_over_2sigma':gg})
    return out,raw,gt


def pairwise(data,w1,w2,x_cut=3.0,tolerance=0.18):
    a=sorted(data[w1],key=lambda r:float(r['L_over_Rg']))
    b=sorted(data[w2],key=lambda r:float(r['L_over_Rg']))
    out=[]
    for ra in a:
        xa=float(ra['L_over_Rg'])
        rb=min(b,key=lambda r:abs(float(r['L_over_Rg'])-xa))
        xb=float(rb['L_over_Rg'])
        if abs(xa-xb)<=tolerance:
            ea=float(ra['DeltaF_bootstrap_SE']); eb=float(rb['DeltaF_bootstrap_SE'])
            diff=float(ra['DeltaF'])-float(rb['DeltaF'])
            comb=math.sqrt(ea*ea+eb*eb)
            xm=0.5*(xa+xb)
            out.append({'w1':w1,'w2':w2,'x1':xa,'x2':xb,'x_mean':xm,
                        'region':'strong' if xm<x_cut else 'weak',
                        'F1_minus_F2':diff,'combined_1sigma':comb,
                        'compatible_within_1sigma':abs(diff)<=comb})
    return out

# Load existing data.
point_old=read_csv(BASE/'FigS6_point_level.csv')
block_old=read_csv(BASE/'FigS6_block_level.csv')
state_old=read_csv(BASE/'FigS6_state_level_production.csv')
conv_old=read_csv(BASE/'FigS6_convergence_audit (1).csv')

# New unconfined w=1 blocks.
with open(NEW_BLOCKS,newline='',encoding='utf-8') as f:
    nb=list(csv.DictReader(f))
new_inf_logs=[float(r['logZ']) for r in nb]
assert len(new_inf_logs)==64

# Current Rg values come from existing S6 point data (same Rg master file).
rg_rows={float(r['w']):r for r in point_old}

# Patch finite point rows only for w=1.
point_new=[]
finite_by_wL={(float(r['w']),int(r['L_realized'])):r for r in point_old}
for r0 in point_old:
    w=float(r0['w']); L=int(r0['L_realized'])
    r=dict(r0)
    if w==1.0:
        logsL=[float(rr['logZ']) for rr in block_old if rr['w']=='1.0' and rr['L']==str(L)]
        assert len(logsL)==40
        F=new_logz_inf-logmeanexp(logsL)
        seed=20260928+700_000_007+int(round(w*1000))*1009+L
        se,lo,hi=bootstrap_log_ratio(logsL,new_inf_logs,5000,seed)
        r['logZ_inf']=f'{new_logz_inf:.14f}'
        r['DeltaF']=f'{F:.14f}'
        r['DeltaF_bootstrap_SE']=f'{se:.14f}'
        r['DeltaF_bootstrap_CI95_low']=f'{lo:.14f}'
        r['DeltaF_bootstrap_CI95_high']=f'{hi:.14f}'
        # block convergence uses first n blocks, as the original code does.
        for n in (12,16,40):
            if n==12 or n==16 or n==40:
                infn=logmeanexp(new_inf_logs[:n])
                cf=logmeanexp(logsL[:n])
                r[f'DeltaF_{n}']=f'{infn-cf:.14f}'
    point_new.append(r)

# Patch state-level only the w=1, L=inf row. Keep all finite states exactly as produced.
state_new=[]
for r0 in state_old:
    r=dict(r0)
    if r['w']=='1.0' and r['L']=='inf':
        ess=[float(x['ESS']) for x in nb]; mf=[float(x['max_weight_fraction']) for x in nb]
        r.update({'logZ_mean':f'{new_logz_inf:.14f}',
                  'logZ_SE_block':f'{new_logz_se:.14f}',
                  'ESS_min':f'{min(ess):.14f}',
                  'ESS_median':f'{np.median(ess):.14f}',
                  'max_weight_fraction_max':f'{max(mf):.14f}',
                  'sampler':'canonical_unconfined_SIS'})
    state_new.append(r)

# Patch block-level: remove the old w=1/L=inf blocks and insert the new 64 blocks.
block_new=[r for r in block_old if not (r['w']=='1.0' and r['L']=='inf')]
for i,rn in enumerate(nb):
    block_new.append({
        'w':'1.0','block':str(i),'logZ':rn['logZ'],'ESS':rn['ESS'],
        'max_weight_fraction':rn['max_weight_fraction'],'valid_chains':rn['valid_chains'],
        'roots':rn['roots'],'sampler':rn['sampler'],'L':'inf','tether_z':'0',
        'population_final':'','population_max_seen':'','status':''
    })
# Stable sort.
def lkey(v): return math.inf if v=='inf' else int(v)
block_new.sort(key=lambda r:(float(r['w']),lkey(r['L']),int(r['block'])))

# Rebuild w-dependent data from patched point data, removing existing audit duplication.
byw={w:[] for w in (0.2,0.5,1.0)}
for r in point_new: byw[float(r['w'])].append(r)
mono_rows=[]
mono_summary={}
for w,rs in byw.items():
    rows,raw,gt=monotonicity(rs)
    mono_rows.extend(rows)
    mono_summary[str(w)]={'n_points':len(rs),'n_raw_violations':raw,'n_gt_2sigma_violations':gt}

pair_rows=[]; pair_summary={}
for w1,w2 in ((0.2,0.5),(0.5,1.0),(0.2,1.0)):
    rows=pairwise(byw,w1,w2)
    pair_rows.extend(rows)
    for region in ('strong','weak'):
        sub=[r for r in rows if r['region']==region]
        pair_summary[f'{w1:g}_vs_{w2:g}__{region}']={
            'n':len(sub),
            'n_compatible_1sigma':sum(r['compatible_within_1sigma'] for r in sub),
            'fraction_compatible_1sigma':(sum(r['compatible_within_1sigma'] for r in sub)/len(sub) if sub else math.nan),
            'mean_difference':(float(np.mean([r['F1_minus_F2'] for r in sub])) if sub else math.nan)
        }

# Rebuild convergence audit from patched block-level data; this fixes w=1 only but keeps structure clean.
conv_new=[]
for w in (0.2,0.5,1.0):
    inflogs=[float(r['logZ']) for r in block_new if float(r['w'])==w and r['L']=='inf']
    for L in (10,12,14,18,20,24,26,30,34):
        logs=[float(r['logZ']) for r in block_new if float(r['w'])==w and r['L']==str(L)]
        for n in (12,16,40):
            conv_new.append({'w':w,'L':L,'blocks_used':n,'DeltaF_blocks':logmeanexp(inflogs[:n])-logmeanexp(logs[:n])})

# Also make a compact publication source-data table.
source_rows=[]
for r in sorted(point_new,key=lambda r:(float(r['w']),float(r['L_over_Rg']))):
    source_rows.append({
        'model':r['model'],'N':r['N'],'w':r['w'],'L':r['L_realized'],
        'L_over_Rg':r['L_over_Rg'],'L_over_Rg_err':r['L_over_Rg_err'],
        'DeltaF_kBT':r['DeltaF'],'DeltaF_SE':r['DeltaF_bootstrap_SE'],
        'DeltaF_CI95_low':r['DeltaF_bootstrap_CI95_low'],
        'DeltaF_CI95_high':r['DeltaF_bootstrap_CI95_high']
    })

# Publication-quality figure: no internal diagnostic label such as "Rg caution".
fig,ax=plt.subplots(figsize=(7.8,6.2))
# Keep the original visual ordering/palette for direct comparability.
colors=plt.cm.plasma(np.linspace(0.18,0.88,3))
for color,w in zip(colors,(0.2,0.5,1.0)):
    rs=sorted(byw[w],key=lambda r:float(r['L_over_Rg']))
    x=np.array([float(r['L_over_Rg']) for r in rs])
    y=np.array([float(r['DeltaF']) for r in rs])
    e=np.array([float(r['DeltaF_bootstrap_SE']) for r in rs])
    ax.errorbar(x,y,yerr=e,fmt='o-',ms=6.5,lw=1.6,capsize=3,color=color,label=rf'$w={w:g}$')
ax.axvline(3.0,color='0.45',ls='--',lw=1.0,alpha=0.8)
y0,y1=ax.get_ylim(); ax.text(3.03,y1-0.04*(y1-y0),r'$L/R_g=3$',fontsize=11,va='top',ha='left')
ax.set_xlabel(r'Scaled slit width $L/R_g(w)$',fontsize=18,fontweight='bold')
ax.set_ylabel(r'Confinement free energy $\Delta F\;(k_BT)$',fontsize=18,fontweight='bold')
ax.set_title(r'Domb--Joyce confinement free energy crossover ($N=200$)',fontsize=17)
ax.tick_params(axis='both',labelsize=14,width=1.2)
for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
ax.grid(alpha=0.24)
ax.legend(frameon=True,loc='best',fontsize=13)
fig.tight_layout()
png=OUT/'FigS6_DJ_FreeEnergy_MASTER_REVISED_PATCHED.png'
pdf=OUT/'FigS6_DJ_FreeEnergy_MASTER_REVISED_PATCHED.pdf'
fig.savefig(png,dpi=600,bbox_inches='tight')
fig.savefig(pdf,bbox_inches='tight')
plt.close(fig)

# Save patched outputs.
write_csv(point_new,OUT/'FigS6_point_level_patched.csv')
write_csv(source_rows,OUT/'FigS6_source_data_for_paper.csv')
write_csv(state_new,OUT/'FigS6_state_level_production_patched.csv')
write_csv(block_new,OUT/'FigS6_block_level_patched.csv')
write_csv(mono_rows,OUT/'FigS6_monotonicity_audit_patched.csv')
write_csv(pair_rows,OUT/'FigS6_w_dependence_region_audit_patched.csv')
write_csv(conv_new,OUT/'FigS6_convergence_audit_patched.csv')

# Preserve raw new unconfined reference.
shutil.copy2(NEW_SUMMARY,OUT/'S6_w1_unconfined_highstat_summary.json')
shutil.copy2(NEW_BLOCKS,OUT/'S6_w1_unconfined_highstat_blocks.csv')
shutil.copy2('/mnt/data/rerun_S6_w1_unconfined_reference.py',OUT/'rerun_S6_w1_unconfined_reference.py')
shutil.copy2('/mnt/data/FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py',OUT/'FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py')

prov={
 'purpose':'Patch only the w=1, L=infinity S6 reference using the independent high-statistics unconfined rerun.',
 'original_S6_dataset':'S6.zip',
 'patched_reference':{'N':200,'w':1.0,'blocks':64,'roots_per_block':8192,'total_chains':524288,
                      'logZ_inf':new_logz_inf,'logZ_SE_block':new_logz_se,
                      'ESS_min':sumj['ESS_min'],'ESS_median':sumj['ESS_median'],
                      'max_weight_fraction_max':sumj['max_weight_fraction_max']},
 'old_reference':{'logZ_inf':323.10579583759295,'logZ_SE_block':0.01611692756437444,
                  'ESS_min':38.57156588736073,'ESS_median':97.25292769497867,
                  'max_weight_fraction_max':0.1408679413902319},
 'delta_logZ_inf':new_logz_inf-323.10579583759295,
 'bootstrap':{'replicates':5000,'w1_seed_formula':'20260928 + 700000007 + int(round(w*1000))*1009 + L'},
 'finite_width_data':'Copied unchanged from original S6 production except w=1 DeltaF values and associated bootstrap/convergence quantities were recomputed against the new reference.',
 'geometry':'center-tethered absorbing slit, even integer L',
 'monotonicity_summary':mono_summary,
 'pairwise_summary':pair_summary,
 'publication_note':'Figure legend does not expose internal Rg quality diagnostic; diagnostic remains in source CSV.',
}
(OUT/'FigS6_patch_provenance.json').write_text(json.dumps(prov,indent=2,default=float))

readme=f'''S6 FINAL PATCHED DATASET\n\nPurpose\n-------\nPatch only the w=1, L=infinity normalization in the S6 production dataset using the independent high-statistics unconfined SIS/Rosenbluth rerun. All finite-width PERM states are retained from the original S6 production run.\n\nNew reference\n-------------\nlogZ_inf(w=1) = {new_logz_inf:.12f} +/- {new_logz_se:.12f} (block SE)\n64 blocks x 8192 roots/block = 524288 chains.\n\nOutputs\n-------\nFigS6_DJ_FreeEnergy_MASTER_REVISED_PATCHED.png / .pdf : final figure\nFigS6_source_data_for_paper.csv : compact 27-row plotted source data\nFigS6_point_level_patched.csv : full point-level audit data\nFigS6_state_level_production_patched.csv : state-level production summary\nFigS6_block_level_patched.csv : complete block-level record, with new 64 w=1,L=inf blocks\nFigS6_monotonicity_audit_patched.csv : monotonicity audit\nFigS6_w_dependence_region_audit_patched.csv : unique pairwise w-dependence audit\nFigS6_convergence_audit_patched.csv : block convergence audit\nS6_w1_unconfined_highstat_summary.json / blocks.csv : raw independent w=1 reference rerun\nFigS6_patch_provenance.json : patch provenance\nFigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py : finite-width S6 production code\nrerun_S6_w1_unconfined_reference.py : independent w=1 reference rerun code\n'''
(OUT/'README_S6_FINAL.txt').write_text(readme)

# Zip the full final package.
zip_path=Path('/mnt/data/S6_FINAL_PATCHED_PACKAGE.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,p.name)

# Print compact results.
print('NEW_LOGZ',new_logz_inf,new_logz_se)
print('DELTA_LOGZ',new_logz_inf-323.10579583759295)
print('W1_POINTS')
for r in sorted(byw[1.0],key=lambda r:float(r['L_over_Rg'])):
    print(float(r['L_realized']),float(r['L_over_Rg']),float(r['DeltaF']),float(r['DeltaF_bootstrap_SE']))
print('MONO',mono_summary)
print('PAIR_SUMMARY')
for k,v in pair_summary.items(): print(k,v)
print('FIG',png,pdf)
print('ZIP',zip_path)
