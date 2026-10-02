#!/usr/bin/env python3
"""High-statistics rerun of the unconfined w=1 Domb--Joyce reference used by S6.

This script imports the validated S6 Numba implementation and reruns ONLY the
unconfined SIS/Rosenbluth reference Z(infinity), not the finite-slit PERM states.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("s6_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import S6 code: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s6-code", required=True,
                   help="Path to validated FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=64)
    p.add_argument("--roots-per-block", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s6 = load_module(args.s6_code)
    table_size = s6.table_size_for(args.N)

    print("=" * 88)
    print("S6 HIGH-STATISTICS UNCONFINED REFERENCE RERUN")
    print("=" * 88)
    print(f"S6 code            : {args.s6_code}")
    print(f"N                  : {args.N}")
    print(f"w                  : {args.w}")
    print(f"blocks             : {args.blocks}")
    print(f"roots/block        : {args.roots_per_block}")
    print(f"total chains       : {args.blocks * args.roots_per_block:,}")
    print(f"seed               : {args.seed}")
    print(f"hash table size    : {table_size}")
    print("sampler            : canonical unconfined Boltzmann-biased SIS/Rosenbluth")
    print("=" * 88)

    rows = []
    logz = []
    for b in range(args.blocks):
        seed = args.seed + 300_000_007 + int(round(args.w * 1000)) * 1009 + 10_000_019 * b
        val, ess, maxfrac, valid = s6.unconfined_sis_block_numba(
            args.N, float(args.w), int(args.roots_per_block), int(seed), table_size
        )
        if not np.isfinite(val):
            raise RuntimeError(f"Non-finite logZ at block {b}")
        logz.append(float(val))
        rows.append({
            "w": float(args.w),
            "block": int(b),
            "logZ": float(val),
            "ESS": float(ess),
            "max_weight_fraction": float(maxfrac),
            "valid_chains": int(valid),
            "roots": int(args.roots_per_block),
            "seed": int(seed),
            "sampler": "canonical_unconfined_SIS",
        })
        print(f"block {b+1:02d}/{args.blocks}: logZ={val:.9f}  ESS={ess:.2f}  maxfrac={maxfrac:.5f}")

    logz = np.asarray(logz, dtype=float)
    # Same estimator convention as the production S6 code:
    # log(mean Z_b), where each block supplies an unbiased Z estimate.
    m = float(np.max(logz))
    logz_inf = float(m + math.log(np.mean(np.exp(logz - m))))
    logz_se_block = float(np.std(logz, ddof=1) / math.sqrt(args.blocks))
    ess = np.asarray([r["ESS"] for r in rows])
    frac = np.asarray([r["max_weight_fraction"] for r in rows])

    summary = {
        "N": int(args.N),
        "w": float(args.w),
        "blocks": int(args.blocks),
        "roots_per_block": int(args.roots_per_block),
        "total_chains": int(args.blocks * args.roots_per_block),
        "seed": int(args.seed),
        "logZ_inf": logz_inf,
        "logZ_SE_block": logz_se_block,
        "ESS_min": float(np.min(ess)),
        "ESS_median": float(np.median(ess)),
        "max_weight_fraction_max": float(np.max(frac)),
        "max_weight_fraction_median": float(np.median(frac)),
        "all_valid": bool(np.all([r["valid_chains"] == r["roots"] for r in rows])),
        "sampler": "canonical_unconfined_SIS",
    }

    pd.DataFrame(rows).to_csv(outdir / "S6_w1_unconfined_highstat_blocks.csv", index=False)
    with open(outdir / "S6_w1_unconfined_highstat_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"logZ_inf                  = {logz_inf:.10f}")
    print(f"block SE(logZ_inf)        = {logz_se_block:.10f}")
    print(f"ESS min / median          = {summary['ESS_min']:.2f} / {summary['ESS_median']:.2f}")
    print(f"max weight fraction       = {summary['max_weight_fraction_max']:.6f}")
    print(f"all chains valid          = {summary['all_valid']}")
    print(f"output directory          = {outdir.resolve()}")


if __name__ == "__main__":
    main()
