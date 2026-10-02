#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S13 — LOWO residual consistency diagnostic (REV4).

Purpose
-------
A downstream, reviewer-facing diagnostic of the Domb–Joyce force crossover
(Fig. 6).  For each interaction strength w, the target curve is compared with
an independent leave-one-w-out (LOWO) reference assembled from the *other*
w-curves at the target x=L/Rg(w).

REV4 changes relative to REV3
------------------------------
1. LOWO reference is evaluated directly at each target x; no auxiliary dense
   x-grid followed by a second interpolation is used.  This removes a small
   but unnecessary interpolation layer.
2. Default master-support requirement is raised from 2 to 3 supporting curves.
   With the current five-w dataset this still retains the common overlap while
   making the diagnostic less sensitive to a single comparator curve.
3. Master uncertainty is measurement-aware: for each contributing curve, the
   force is interpolated in log x/log f and its local log-space uncertainty is
   propagated from the two bracketing point uncertainties.  A bootstrap then
   resamples contributing curves and their measurement distributions.
4. Residual uncertainty is propagated in log space and the output includes both
   R=f/f_master and log(R), which is symmetric for multiplicative deviations.
5. Descriptive correction-to-scaling diagnostics are added: weighted linear
   trend of log(R) versus log(x), both globally and per w.  These are explicitly
   diagnostic and are not presented as independent-point hypothesis tests.
6. Sensitivity diagnostics are added for the predefined support thresholds and
   central-x window, plus one-point-deletion sensitivity of the global metric.
   No point is selected or deleted on the basis of these diagnostics.
7. Every excluded point is written to a dedicated exclusions CSV with a reason.
   Signed/non-positive force values are retained in the source data but are not
   used in the logarithmic LOWO residual; this is a mathematical requirement of
   the ratio diagnostic, not force selection.
8. Legacy source flags (excluded/sign_selected) are audited but never silently
   used to delete points.  If legacy flags are present, they are written to the
   provenance and the user is warned.
9. Canonical Rg(master) consistency is checked per w and must agree to the stated
   tolerance.
10. No universal master curve is fitted.  The LOWO reference is a diagnostic
    composite only and cannot establish universality.

Inputs
------
--point-csv   Final point-level force dataset from Fig. 6 production.
--rg-master   Canonical Rg_MASTER_FINAL.csv (or compatible CSV).
--input-json  Optional Fig. 6 provenance JSON for traceability.

Outputs
-------
FigS13_*_points.csv
FigS13_*_exclusions.csv
FigS13_*_sensitivity.csv
FigS13_*_provenance.json
FigS13_*.png / .pdf

Recommended production command
------------------------------
python S13_LOWO_Residuals_REV4_FINAL.py \\
  --point-csv /kaggle/input/.../Fig6_PointLevel.csv \\
  --rg-master /kaggle/input/.../Rg_MASTER_FINAL_MERGED.csv \\
  --input-json /kaggle/input/.../Fig6_Provenance_FINAL.json \\
  --N 200 --x-min 0.8 --x-max 3.0 \\
  --min-curves-for-master 3 --bootstrap 5000 --central-x-min 1.0 \\
  --central-x-max 2.8 --nproc 1 \\
  --outdir /kaggle/working/S13_REV4_FINAL
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_NAME = "S13_LOWO_Residuals_REV4_FINAL.py"
SCRIPT_VERSION = "4.0"

# Publication style: no explicit color values are required by the scientific
# logic; matplotlib's default cycle is used in the figure.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.3,
    "savefig.dpi": 600,
})


def finite_float(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def load_rg_master(path: Path, N: int) -> Dict[float, dict]:
    df = pd.read_csv(path)
    req = {"model", "N", "w", "Rg", "Rg_err"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Rg master missing columns: {sorted(missing)}")
    dj = df[(df["model"].astype(str).str.upper() == "DJ") & (df["N"].astype(int) == N)].copy()
    out: Dict[float, dict] = {}
    for _, r in dj.iterrows():
        w = float(r["w"])
        if w <= 0.0:
            continue
        if w in out:
            raise ValueError(f"Duplicate DJ Rg row for w={w:g}, N={N} in {path}")
        out[w] = {
            "Rg": float(r["Rg"]),
            "Rg_err": float(r["Rg_err"]),
            "quality_flag": str(r.get("quality_flag", "UNKNOWN")),
        }
    if not out:
        raise ValueError(f"No DJ rows with w>0 and N={N} in {path}")
    return out


def load_point_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        "w": ["w", "w_value"],
        "x": ["x", "L_over_Rg", "realized_L_over_Rg", "L_ratio"],
        "force": ["force", "fRg", "fRg_sim"],
        "force_err": ["force_err", "fRg_err_bootstrap", "fRg_err", "fRg_sim_err", "fRg_err_sem_propagated"],
    }
    resolved: Dict[str, str] = {}
    for target, opts in aliases.items():
        for c in opts:
            if c in df.columns:
                resolved[target] = c
                break
        if target not in resolved:
            raise ValueError(f"Point CSV missing '{target}'; tried {opts}")

    out = df.copy()
    for target, src in resolved.items():
        out[target] = out[src]
    # Preserve common audit/provenance fields where present.
    return out


def audit_rg_consistency(df: pd.DataFrame, rg_master: Dict[float, dict], tol: float) -> List[dict]:
    report = []
    if "Rg" not in df.columns:
        print("[S13] Point CSV has no Rg column; canonical master Rg will be used.")
        return report
    for w, sub in df.groupby("w"):
        w = float(w)
        if w not in rg_master:
            raise ValueError(f"Point CSV contains w={w:g} absent from Rg master")
        vals = pd.to_numeric(sub["Rg"], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        rg_p = float(vals[0])
        if not np.allclose(vals, rg_p, rtol=0, atol=tol):
            raise ValueError(f"Multiple inconsistent Rg values in point CSV at w={w:g}")
        rg_m = float(rg_master[w]["Rg"])
        rel = abs(rg_p - rg_m) / rg_m
        report.append({"w": w, "Rg_point": rg_p, "Rg_master": rg_m, "relative_difference": rel,
                       "pass": bool(rel <= tol)})
        if rel > tol:
            raise ValueError(
                f"Rg mismatch w={w:g}: point={rg_p:.9f}, master={rg_m:.9f}, rel={rel:.3e}")
        print(f"[S13] Rg(w={w:g}) OK: {rg_p:.8f} vs {rg_m:.8f}")
    return report


def prepare_datasets(df: pd.DataFrame, x_min: float, x_max: float) -> Tuple[Dict[float, dict], List[dict], dict]:
    datasets: Dict[float, dict] = {}
    exclusions: List[dict] = []
    legacy_flags = {"excluded": False, "sign_selected": False}

    for w, group in df.groupby("w"):
        w = float(w)
        if w <= 0.0:
            continue
        xs, fs, es, rows = [], [], [], []
        for idx, row in group.iterrows():
            x = row.get("x"); f = row.get("force"); e = row.get("force_err")
            # Audit legacy source flags, but never silently apply them as selection rules.
            if "excluded" in row.index and pd.notna(row["excluded"]):
                flag = str(row["excluded"]).strip().lower()
                if flag in {"true", "1", "yes"}:
                    legacy_flags["excluded"] = True
            if "sign_selected" in row.index and pd.notna(row["sign_selected"]):
                flag = str(row["sign_selected"]).strip().lower()
                if flag in {"true", "1", "yes"}:
                    legacy_flags["sign_selected"] = True

            reason = None
            if not (finite_float(x) and finite_float(f) and finite_float(e)):
                reason = "missing_or_nonfinite_x_force_or_error"
            else:
                x = float(x); f = float(f); e = float(e)
                if x < x_min or x > x_max:
                    reason = "outside_declared_x_range"
                elif e <= 0.0:
                    reason = "nonpositive_force_uncertainty"
                elif f <= 0.0:
                    # Keep signed value in the audit, but log-space LOWO residual is undefined.
                    reason = "nonpositive_force_log_residual_undefined"
            if reason is not None:
                exclusions.append({
                    "w": w, "source_row": int(idx), "x": float(x) if finite_float(x) else None,
                    "force": float(f) if finite_float(f) else None,
                    "force_err": float(e) if finite_float(e) else None,
                    "reason": reason,
                })
                continue
            xs.append(x); fs.append(f); es.append(e); rows.append(int(idx))

        if len(xs) < 2:
            if xs:
                exclusions.append({"w": w, "reason": "fewer_than_two_valid_positive_points_after_filter"})
            continue
        order = np.argsort(xs)
        xs = np.asarray(xs, float)[order]
        fs = np.asarray(fs, float)[order]
        es = np.asarray(es, float)[order]
        rows = np.asarray(rows, int)[order]
        # Repeated x values are not allowed for log interpolation.
        if np.any(np.diff(xs) <= 0):
            raise ValueError(f"Duplicate/non-increasing x values for w={w:g}")
        datasets[w] = {"x": xs, "force": fs, "err": es, "source_row": rows}

    if len(datasets) < 3:
        raise RuntimeError(f"Need at least 3 usable w-curves; found {sorted(datasets)}")
    return datasets, exclusions, legacy_flags


def log_interp_with_uncertainty(curve: dict, xq: float) -> Tuple[float, float, bool]:
    """Return interpolated force, 1-sigma log-force uncertainty, and support flag."""
    x = curve["x"]; f = curve["force"]; e = curve["err"]
    if xq < x[0] or xq > x[-1]:
        return math.nan, math.nan, False
    if xq == x[0]:
        return float(f[0]), float(e[0] / f[0]), True
    if xq == x[-1]:
        return float(f[-1]), float(e[-1] / f[-1]), True

    i = int(np.searchsorted(x, xq, side="right") - 1)
    i = max(0, min(i, len(x) - 2))
    lx0, lx1, lq = np.log(x[i]), np.log(x[i + 1]), math.log(xq)
    t = (lq - lx0) / (lx1 - lx0)
    ly = (1.0 - t) * math.log(f[i]) + t * math.log(f[i + 1])
    rel0, rel1 = e[i] / f[i], e[i + 1] / f[i + 1]
    sigma_log = math.sqrt((1.0 - t) ** 2 * rel0 ** 2 + t ** 2 * rel1 ** 2)
    return float(math.exp(ly)), float(sigma_log), True


def bootstrap_lowomaster(values: np.ndarray, sigma_log: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Bootstrap LOWO master in log space.

    Returns: median_force, sd(log master), p2.5 force, p97.5 force.
    """
    if values.size < 1:
        return math.nan, math.nan, math.nan, math.nan
    logv = np.log(values)
    if values.size == 1:
        draws = logv[0] + rng.normal(0.0, max(float(sigma_log[0]), 0.0), size=reps)
    else:
        idx = rng.integers(0, values.size, size=(reps, values.size))
        jitter = rng.normal(0.0, sigma_log[idx], size=(reps, values.size))
        draws = np.median(logv[idx] + jitter, axis=1)
    med = float(np.median(values))
    return med, float(np.std(draws, ddof=1)), float(np.exp(np.quantile(draws, 0.025))), float(np.exp(np.quantile(draws, 0.975)))


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> dict:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0) & (x > 0)
    x = np.asarray(x)[finite]; y = np.asarray(y)[finite]; sy = np.asarray(sy)[finite]
    if x.size < 3:
        return {"n": int(x.size), "slope": math.nan, "slope_err": math.nan, "intercept": math.nan,
                "intercept_err": math.nan, "chi2_red": math.nan}
    X = np.column_stack([np.ones_like(x), np.log(x)])
    w = 1.0 / sy ** 2
    A = X.T @ (w[:, None] * X)
    b = X.T @ (w * y)
    beta = np.linalg.solve(A, b)
    cov = np.linalg.inv(A)
    resid = y - X @ beta
    dof = max(1, x.size - 2)
    chi2 = float(np.sum((resid / sy) ** 2))
    chi2_red = chi2 / dof
    return {
        "n": int(x.size), "slope": float(beta[1]), "slope_err": float(math.sqrt(max(cov[1,1], 0.0))),
        "intercept": float(beta[0]), "intercept_err": float(math.sqrt(max(cov[0,0], 0.0))),
        "chi2_red": float(chi2_red),
    }


def build_lowow_rows(datasets: Dict[float, dict], min_curves: int, bootstrap_reps: int, seed: int) -> Tuple[List[dict], List[dict]]:
    rng = np.random.default_rng(seed)
    residual_rows: List[dict] = []
    exclusions: List[dict] = []
    for w in sorted(datasets):
        target = datasets[w]
        others = [wo for wo in sorted(datasets) if wo != w]
        for xq, f_target, e_target, src in zip(target["x"], target["force"], target["err"], target["source_row"]):
            vals, sigmas, sources = [], [], []
            for wo in others:
                fv, slog, ok = log_interp_with_uncertainty(datasets[wo], float(xq))
                if ok and np.isfinite(fv) and fv > 0:
                    vals.append(fv); sigmas.append(slog); sources.append(wo)
            if len(vals) < min_curves:
                exclusions.append({
                    "w": w, "source_row": int(src), "x": float(xq),
                    "reason": f"LOWO_master_support_{len(vals)}_below_minimum_{min_curves}",
                    "supporting_w": ";".join(f"{v:g}" for v in sources),
                })
                continue
            vals = np.asarray(vals, float); sigmas = np.asarray(sigmas, float)
            master = float(np.median(vals))
            # Seed per point is deterministic and independent of loop order.
            point_seed = int(seed + round(10000*w) + int(src)*1000003)
            prng = np.random.default_rng(point_seed)
            _, master_log_sd, mlo, mhi = bootstrap_lowomaster(vals, sigmas, bootstrap_reps, prng)
            logR = math.log(float(f_target)) - math.log(master)
            sigma_target_log = float(e_target / f_target)
            sigma_logR = math.sqrt(max(sigma_target_log ** 2 + master_log_sd ** 2, 0.0))
            R = float(math.exp(logR))
            Rerr = float(R * sigma_logR)
            residual_rows.append({
                "w": float(w), "x": float(xq), "force": float(f_target), "force_err": float(e_target),
                "master_LOWO": master,
                "master_bootstrap_log_sd": float(master_log_sd),
                "master_bootstrap_CI95_low": mlo,
                "master_bootstrap_CI95_high": mhi,
                "n_curves_in_master": int(len(vals)),
                "supporting_w": ";".join(f"{v:g}" for v in sources),
                "residual": R, "residual_err": Rerr,
                "log_residual": float(logR), "log_residual_err": float(sigma_logR),
                "source_row": int(src),
            })
    return residual_rows, exclusions


def summarize(rows: List[dict], x_min: float, x_max: float) -> dict:
    if not rows:
        return {"n": 0}
    r = np.asarray([a["residual"] for a in rows], float)
    lr = np.asarray([a["log_residual"] for a in rows], float)
    le = np.asarray([a["log_residual_err"] for a in rows], float)
    finite = np.isfinite(r) & np.isfinite(lr) & np.isfinite(le) & (le > 0)
    r = r[finite]; lr = lr[finite]; le = le[finite]
    mad = np.abs(r - 1.0)
    abslog = np.abs(lr)
    wi = 1.0 / le ** 2
    return {
        "n": int(r.size),
        "mean_abs_ratio_deviation": float(np.mean(mad)),
        "median_abs_ratio_deviation": float(np.median(mad)),
        "mean_abs_log_residual": float(np.mean(abslog)),
        "median_abs_log_residual": float(np.median(abslog)),
        "weighted_mean_log_residual": float(np.sum(wi*lr)/np.sum(wi)),
        "weighted_mean_log_residual_err": float(1.0/math.sqrt(np.sum(wi))),
        "fraction_within_10pct_multiplicative": float(np.mean(abslog <= math.log(1.10))),
        "fraction_within_2sigma": float(np.mean(abslog <= 2.0*le)),
        "max_abs_log_residual": float(np.max(abslog)),
        "x_min": float(x_min), "x_max": float(x_max),
    }


def per_w_summary(rows: List[dict]) -> Dict[float, dict]:
    out: Dict[float, dict] = {}
    for w in sorted({r["w"] for r in rows}):
        sub = [r for r in rows if r["w"] == w]
        lr = np.asarray([r["log_residual"] for r in sub], float)
        le = np.asarray([r["log_residual_err"] for r in sub], float)
        ok = np.isfinite(lr) & np.isfinite(le) & (le > 0)
        lr = lr[ok]; le = le[ok]
        wt = 1.0 / le ** 2
        fit = weighted_linear_fit(np.asarray([r["x"] for r in sub], float)[ok], lr, le)
        out[w] = {
            "n": int(lr.size),
            "mean_log_residual": float(np.mean(lr)),
            "median_log_residual": float(np.median(lr)),
            "weighted_mean_log_residual": float(np.sum(wt*lr)/np.sum(wt)),
            "weighted_mean_log_residual_err": float(1.0/math.sqrt(np.sum(wt))),
            "descriptive_logx_slope": fit["slope"],
            "descriptive_logx_slope_err": fit["slope_err"],
        }
    return out


def sensitivity_analysis(datasets: Dict[float, dict], base_seed: int, bootstrap_reps: int,
                         support_thresholds: List[int], central_x: Tuple[float,float]) -> Tuple[List[dict], Dict[int, List[dict]]]:
    """Recompute LOWO residuals under predefined support thresholds and x windows.

    This is a genuine sensitivity analysis: changing the support threshold can
    add/remove only points because of interpolation support, never because of
    force magnitude or sign.
    """
    results=[]
    rows_by_mc={}
    for mc in sorted(set(support_thresholds)):
        rows,_ = build_lowow_rows(datasets, mc, bootstrap_reps, base_seed + 1009*mc)
        rows_by_mc[mc]=rows
        for range_name, lo, hi in [
            ("full", -np.inf, np.inf),
            ("central", central_x[0], central_x[1]),
        ]:
            rr=[r for r in rows if lo <= r["x"] <= hi]
            if not rr:
                results.append({"diagnostic":"support_threshold","min_curves":mc,
                                "range":range_name,"n":0})
                continue
            sm=summarize(rr,
                         min(r["x"] for r in rr) if not np.isfinite(lo) else lo,
                         max(r["x"] for r in rr) if not np.isfinite(hi) else hi)
            results.append({"diagnostic":"support_threshold","min_curves":mc,
                            "range":range_name,**sm})

    # One-point deletion sensitivity on the primary threshold dataset.
    rows=rows_by_mc[max(support_thresholds)] if rows_by_mc else []
    if rows:
        vals=[]
        for i in range(len(rows)):
            rr=rows[:i]+rows[i+1:]
            if rr:
                vals.append(summarize(rr,min(r["x"] for r in rr),max(r["x"] for r in rr))["mean_abs_log_residual"])
        if vals:
            base=summarize(rows,min(r["x"] for r in rows),max(r["x"] for r in rows))
            results.append({"diagnostic":"one_point_deletion","min_curves":max(support_thresholds),
                            "range":"retained","base_mean_abs_log_residual":base["mean_abs_log_residual"],
                            "min_after_deletion":float(min(vals)),"max_after_deletion":float(max(vals)),
                            "range_after_deletion":float(max(vals)-min(vals))})
    return results, rows_by_mc

def make_figure(rows: List[dict], per_w: Dict[float,dict], metrics: dict,
                out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.84, wspace=0.30)
    markers=["o","s","^","D","v","P"]
    ws=sorted(per_w)

    # Panel (a): pointwise LOWO residuals.
    ax=axes[0]
    for i,w in enumerate(ws):
        sub=sorted([r for r in rows if r["w"]==w], key=lambda z:z["x"])
        if not sub:
            continue
        x=np.array([r["x"] for r in sub])
        y=np.array([r["residual"] for r in sub])
        e=np.array([r["residual_err"] for r in sub])
        ax.errorbar(x,y,yerr=e,fmt=markers[i%len(markers)],ms=7,capsize=3,lw=0,label=fr"$w={w:g}$")
    ax.axhline(1.0,ls="--",lw=1.5,label="LOWO agreement")
    ax.axhspan(0.9,1.1,alpha=0.10,label=r"$\pm10\%$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/R_g(w)$")
    ax.set_ylabel(r"LOWO residual  $f_w/f_{\rm master}^{(-w)}$")
    ax.set_title("(a) Leave-one-$w$-out residuals",loc="left")
    ax.grid(alpha=0.25,which="both")
    ax.legend(loc="lower right",framealpha=0.92,fontsize=8.5)
    ax.text(0.025,0.975,
            f"N={metrics['n']}\nmean $|\\ln R|$={metrics['mean_abs_log_residual']:.3f}\n"
            f"within $\\pm10\\%$: {100*metrics['fraction_within_10pct_multiplicative']:.0f}%",
            transform=ax.transAxes,ha="left",va="top",fontsize=9,
            bbox=dict(boxstyle="round",facecolor="white",alpha=0.86,edgecolor="0.5"))

    # Panel (b): per-w systematic offsets in log space; no connecting line is used
    # so the discrete w values are not visually interpreted as an interpolated law.
    ax=axes[1]
    ym=np.array([per_w[w]["weighted_mean_log_residual"] for w in ws])
    ye=np.array([per_w[w]["weighted_mean_log_residual_err"] for w in ws])
    ax.errorbar(ws,ym,yerr=ye,fmt="o",ms=7,capsize=4,lw=0)
    ax.axhline(0.0,ls="--",lw=1.4,label="zero systematic offset")
    ax.axhspan(-math.log(1.10),math.log(1.10),alpha=0.10,label=r"$\pm10\%$ multiplicative band")
    ax.set_xlabel(r"Interaction strength $w$")
    ax.set_ylabel(r"Weighted mean $\ln(f/f_{\rm master})$")
    ax.set_title("(b) $w$-dependent systematic residual",loc="left")
    ax.set_xticks(ws)
    ax.grid(alpha=0.25)
    ax.legend(loc="best",framealpha=0.92,fontsize=8.5)
    for i,w in enumerate(ws):
        ax.annotate(f"n={per_w[w]['n']}",(w,ym[i]),textcoords="offset points",xytext=(5,7),fontsize=8)

    fig.suptitle("Supplementary Fig. S13 — LOWO residual consistency diagnostic",
                 fontsize=15,fontweight="bold",y=0.965)
    fig.savefig(out_png,bbox_inches="tight")
    fig.savefig(out_pdf,bbox_inches="tight")
    plt.close(fig)

def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="S13 LOWO residual diagnostic REV4")
    p.add_argument("--point-csv",required=False,type=Path)
    p.add_argument("--rg-master",required=False,type=Path)
    p.add_argument("--input-json",type=Path,default=None)
    p.add_argument("--N",type=int,default=200)
    p.add_argument("--x-min",type=float,default=0.8)
    p.add_argument("--x-max",type=float,default=3.0)
    p.add_argument("--min-curves-for-master",type=int,default=3)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20260930)
    p.add_argument("--rg-tol",type=float,default=1e-4)
    p.add_argument("--central-x-min",type=float,default=1.0)
    p.add_argument("--central-x-max",type=float,default=2.8)
    p.add_argument("--outdir",type=Path,default=Path("S13_REV4_FINAL"))
    p.add_argument("--self-test",action="store_true")
    return p.parse_args()


def self_test():
    # Synthetic curves with exact, mutually consistent power-law shape plus
    # small measurement errors.  This tests direct LOWO interpolation,
    # support counting, log-space uncertainty propagation, and Rg-independent
    # execution without external files.
    x=np.array([1.0,1.5,2.0,2.5])
    ds={}
    for i,w in enumerate([0.1,0.2,0.3]):
        f=5.0*x**-2.0*(1.0+0.002*i)
        e=np.full_like(f,0.02)
        ds[w]={"x":x.copy(),"force":f.copy(),"err":e.copy(),"source_row":np.arange(4)}
    rows,ex=build_lowow_rows(ds,2,1500,12345)
    assert len(rows)==12 and len(ex)==0
    assert max(abs(r["residual"]-1.0) for r in rows) < 0.01
    sens,_=sensitivity_analysis(ds,12345,1000,[2,3],(1.0,2.5))
    assert any(r.get("diagnostic")=="support_threshold" and r.get("min_curves")==3 for r in sens)
    print("S13 REV4 self-test: PASS")


def main():
    args=parse_args()
    if args.self_test:
        self_test(); return
    if args.point_csv is None or args.rg_master is None:
        raise ValueError("--point-csv and --rg-master are required unless --self-test is used")
    if not (0<args.x_min<args.x_max): raise ValueError("Require 0 < x_min < x_max")
    if not (args.x_min <= args.central_x_min < args.central_x_max <= args.x_max):
        raise ValueError("Central x-range must lie inside declared x-range")
    if args.min_curves_for_master<2: raise ValueError("min-curves-for-master must be >=2")
    if args.bootstrap<1000: raise ValueError("Use at least 1000 bootstrap replicates for final S13")
    if not args.point_csv.is_file(): raise FileNotFoundError(args.point_csv)
    if not args.rg_master.is_file(): raise FileNotFoundError(args.rg_master)
    args.outdir.mkdir(parents=True,exist_ok=True)

    rg=load_rg_master(args.rg_master,args.N)
    df=load_point_csv(args.point_csv)
    rg_report=audit_rg_consistency(df,rg,args.rg_tol)
    datasets, initial_exclusions, legacy_flags=prepare_datasets(df,args.x_min,args.x_max)
    residual_rows, lowo_exclusions=build_lowow_rows(datasets,args.min_curves_for_master,args.bootstrap,args.seed)
    if len(residual_rows)<3: raise RuntimeError("Too few LOWO residuals")
    all_excl=initial_exclusions+lowo_exclusions

    perw=per_w_summary(residual_rows)
    metrics=summarize(residual_rows,args.x_min,args.x_max)
    fit_global=weighted_linear_fit(
        np.array([r["x"] for r in residual_rows],float),
        np.array([r["log_residual"] for r in residual_rows],float),
        np.array([r["log_residual_err"] for r in residual_rows],float),
    )
    sensitivity, rows_by_mc=sensitivity_analysis(datasets,args.seed,args.bootstrap,
                                                  sorted(set([2,args.min_curves_for_master])),
                                                  (args.central_x_min,args.central_x_max))

    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    base=args.outdir/f"FigS13_Residuals_LOWO_{stamp}"
    png=base.with_suffix(".png"); pdf=base.with_suffix(".pdf")
    point_csv=base.with_name(base.name+"_points.csv")
    excl_csv=base.with_name(base.name+"_exclusions.csv")
    sens_csv=base.with_name(base.name+"_sensitivity.csv")
    prov_json=base.with_name(base.name+"_provenance.json")

    pd.DataFrame(residual_rows).to_csv(point_csv,index=False)
    pd.DataFrame(all_excl).to_csv(excl_csv,index=False)
    pd.DataFrame(sensitivity).to_csv(sens_csv,index=False)
    make_figure(residual_rows,perw,metrics,png,pdf)

    prov={
        "script_name":SCRIPT_NAME,"script_version":SCRIPT_VERSION,
        "timestamp":datetime.now().isoformat(),"python":sys.version,"platform":platform.platform(),
        "N":args.N,
        "inputs":{"point_csv":str(args.point_csv.resolve()),"rg_master":str(args.rg_master.resolve()),
                   "input_json":str(args.input_json.resolve()) if args.input_json else None},
        "x_range":[args.x_min,args.x_max],"central_x_range":[args.central_x_min,args.central_x_max],
        "min_curves_for_master":args.min_curves_for_master,"bootstrap_reps":args.bootstrap,"seed":args.seed,
        "rg_tolerance":args.rg_tol,"Rg_consistency":rg_report,
        "usable_w_curves":sorted(float(w) for w in datasets),
        "dataset_ranges":{str(w):[float(d["x"].min()),float(d["x"].max())] for w,d in datasets.items()},
        "metrics":metrics,"global_descriptive_logx_fit":fit_global,
        "per_w_summary":{str(w):v for w,v in perw.items()},
        "sensitivity":sensitivity,"exclusion_count":len(all_excl),"exclusions":all_excl,
        "legacy_source_flags_detected":legacy_flags,
        "statistical_method":{
            "master":"median of directly interpolated log-log force curves from other w values at each target x",
            "master_uncertainty":"bootstrap over contributing curves with local log-force measurement uncertainty propagated from bracketing point errors",
            "residual":"R=f_target/f_master; log residual ln(R)",
            "residual_error":"delta-method log-space combination of target relative force uncertainty and LOWO master bootstrap spread",
            "signed_force_policy":"signed source estimates are retained; non-positive forces are excluded only because a logarithmic residual is undefined and every such exclusion is logged",
            "no_force_selection":"no sign, SNR, sigma-clipping, or outlier rule is used to delete a source force point",
            "no_universal_fit":"no universal/global master curve is fitted",
        },
        "claim_boundary":"LOWO is a cross-dataset consistency diagnostic. It cannot prove universality. Systematic residual trends and sensitivity to support/x-range definitions are reported descriptively.",
        "reviewer_2_mapping":{
            "comment_3":"signed source estimates retained; downstream exclusions are mathematical log-space exclusions and explicitly logged",
            "comment_7":"reports multiplicative/log residuals plus descriptive log-x trend and sensitivity diagnostics",
            "comment_8":"canonical Rg checked per w; all exclusions and support decisions archived",
        },
        "outputs":{"png":str(png.resolve()),"pdf":str(pdf.resolve()),"points_csv":str(point_csv.resolve()),
                   "exclusions_csv":str(excl_csv.resolve()),"sensitivity_csv":str(sens_csv.resolve()),"provenance_json":str(prov_json.resolve())}
    }
    prov_json.write_text(json.dumps(prov,indent=2,allow_nan=False),encoding="utf-8")

    print("\n"+"="*90)
    print(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
    print("="*90)
    print(f"usable w curves: {sorted(datasets)}")
    print(f"retained LOWO residuals: {len(residual_rows)}")
    print(f"exclusions logged: {len(all_excl)}")
    print(f"mean |ln R|: {metrics['mean_abs_log_residual']:.5f}")
    print(f"within +/-10% multiplicative: {100*metrics['fraction_within_10pct_multiplicative']:.1f}%")
    print(f"global descriptive slope d ln R / d ln x: {fit_global['slope']:.6g} +/- {fit_global['slope_err']:.6g}")
    print("Outputs:")
    for p in [png,pdf,point_csv,excl_csv,sens_csv,prov_json]: print(f"  {p.resolve()}")
    print("="*90)

if __name__=="__main__":
    main()
