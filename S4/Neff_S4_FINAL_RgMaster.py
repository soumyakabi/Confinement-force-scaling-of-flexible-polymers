#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Fig. S4 — Effective sample size of the Rosenbluth estimator
for confined Domb–Joyce chains.

FINAL Rg-MASTER / even-L implementation
---------------------------------------

Purpose
-------
Quantify the effective sample size (Neff) of the blockwise sequential-importance
Rosenbluth estimator for confined finite-w Domb–Joyce chains. The figure is a
diagnostic of sampling quality, not a force or free-energy calculation.

Methodological conventions
---------------------------
1. Slit widths L are even integers.
2. The chain starts at the co-moving integer midpoint z0=L/2.
3. Absorbing walls: proposed z<=0 or z>=L kills the chain.
4. Reflecting walls: a proposed wall-crossing step is folded back into the
   open slit (0,L); distinct lattice directions remain distinct proposals.
5. Domb–Joyce proposal probability is proportional to
       exp[-w * DeltaV]
   where DeltaV is the integer nonbonded occupancy at the proposed site.
6. The sequential importance weight is accumulated in log space as
       log W += log(sum_allowed exp[-w * DeltaV]).
   No extra factor of 6 is inserted: the proposal is already biased by the
   Domb–Joyce Boltzmann factor.
7. Neff is evaluated within each independent block as
       (sum W)^2 / sum(W^2).
8. L/Rg is normalized with the canonical Rg master file. The selected row must
   be unique for the exact key (model, N, w).
9. Rg and its uncertainty are retained in the output/provenance.
10. Absorbing and reflecting ensembles are never pooled.

Canonical Rg source
-------------------
The master CSV must contain at least:
    model, N, w, Rg, Rg_err, quality_flag

For this S4 calculation, rows are selected by:
    model == DJ
    N == requested N
    w  == requested w

Exactly one matching row is required for every requested w.

Example
-------
python Neff_S4_FINAL_RgMaster.py \
    --rg-master Rg_MASTER_FINAL.csv \
    --N 200 \
    --w-values 0.0 0.3 0.5 \
    --L-values 10 12 14 16 20 24 28 32 \
    --attempts 20000 \
    --blocks 20 \
    --nproc 4 \
    --production \
    --output-dir output_S4
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_NAME = "Neff_S4_FINAL_RgMaster.py"
SCRIPT_VERSION = "3.0"

STEPS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.int32)

DEFAULT_W_VALUES = [0.0, 0.3, 0.5]
DEFAULT_L_VALUES = [10, 12, 14, 16, 20, 24, 28, 32]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logsumexp(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    m = np.max(finite)
    return float(m + np.log(np.sum(np.exp(finite - m))))


def validate_even_width(L):
    L = int(L)
    if L < 2 or L % 2 != 0:
        raise ValueError(f"L must be an even integer >=2; received L={L}.")
    return L


def map_reflecting_z(z, L):
    """Fold a single wall-crossing step back into the open slit (0,L)."""
    if z <= 0:
        z = 2 - z
    elif z >= L:
        z = 2 * (L - 1) - z
    return int(z)


def normalize_model(value):
    return str(value).strip().upper()


def load_rg_master(path, N, w_values):
    """Load exactly one canonical DJ Rg row for every requested w."""
    required = {"model", "N", "w", "Rg", "Rg_err", "quality_flag"}

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path}: no rows found.")

    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {sorted(missing)}"
        )

    selected = {}
    duplicates = []

    requested = [float(w) for w in w_values]
    for w_req in requested:
        matches = []
        for row in rows:
            try:
                model = normalize_model(row["model"])
                n = int(float(row["N"]))
                w = float(row["w"])
            except (TypeError, ValueError):
                continue

            if (
                model == "DJ"
                and n == int(N)
                and np.isclose(w, w_req, rtol=0.0, atol=1e-12)
            ):
                matches.append(row)

        if len(matches) == 0:
            raise ValueError(
                f"{path}: no unique DJ master row for N={N}, w={w_req:g}."
            )
        if len(matches) > 1:
            duplicates.append((w_req, len(matches)))
            continue

        row = matches[0]
        Rg = float(row["Rg"])
        Rg_err = float(row["Rg_err"])
        if not (np.isfinite(Rg) and Rg > 0):
            raise ValueError(f"{path}: invalid Rg for w={w_req:g}: {Rg}")
        if not (np.isfinite(Rg_err) and Rg_err >= 0):
            raise ValueError(f"{path}: invalid Rg_err for w={w_req:g}: {Rg_err}")

        selected[w_req] = {
            "Rg": Rg,
            "Rg_err": Rg_err,
            "quality_flag": str(row["quality_flag"]),
            "model": row["model"],
            "N": int(float(row["N"])),
            "w": float(row["w"]),
            "attempts": int(float(row["attempts"])) if row.get("attempts") not in (None, "") else None,
            "blocks": int(float(row["blocks"])) if row.get("blocks") not in (None, "") else None,
            "estimator": row.get("estimator", ""),
            "stream": row.get("stream", ""),
        }

    if duplicates:
        raise ValueError(
            "Rg master contains duplicate canonical rows: "
            + ", ".join(f"N={N}, w={w:g} ({count} rows)" for w, count in duplicates)
        )

    return selected


def grow_confined_log(N, w, L, boundary, rng):
    """Generate one confined DJ Rosenbluth chain and its log importance weight."""
    L = validate_even_width(L)
    if boundary not in {"absorbing", "reflecting"}:
        raise ValueError("boundary must be absorbing or reflecting")

    z0 = L // 2
    position = np.array([0, 0, z0], dtype=np.int32)
    trajectory = np.empty((N, 3), dtype=np.int32)
    trajectory[0] = position
    visited = {(0, 0, z0): 1}
    total_overlaps = 0

    log_weight = 0.0

    for i in range(1, N):
        current = position
        candidates = []
        overlap_increments = []

        for step in STEPS:
            proposed = current + step
            z = int(proposed[2])

            if boundary == "absorbing":
                if z <= 0 or z >= L:
                    continue
            else:
                proposed = proposed.copy()
                proposed[2] = map_reflecting_z(z, L)
                if proposed[2] <= 0 or proposed[2] >= L:
                    continue

            key = tuple(int(v) for v in proposed)
            candidates.append(proposed)
            overlap_increments.append(visited.get(key, 0))

        if not candidates:
            return -np.inf, None, total_overlaps

        local_log_weights = -float(w) * np.asarray(
            overlap_increments, dtype=float
        )
        local_log_norm = logsumexp(local_log_weights)

        probabilities = np.exp(local_log_weights - local_log_norm)
        choice = int(rng.choice(len(candidates), p=probabilities))
        selected = candidates[choice]
        selected_overlap = int(overlap_increments[choice])

        # Proposal:
        #   q_k = exp(-w*DeltaV_k) / sum_j exp(-w*DeltaV_j)
        # Hence importance increment = sum_j exp(-w*DeltaV_j).
        log_weight += local_log_norm
        total_overlaps += selected_overlap

        position = selected.copy()
        trajectory[i] = position
        key = tuple(int(v) for v in position)
        visited[key] = visited.get(key, 0) + 1

    return float(log_weight), trajectory, int(total_overlaps)


def compute_neff(log_weights):
    """Return Neff and the largest normalized weight fraction."""
    arr = np.asarray(log_weights, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0

    m = float(np.max(arr))
    normalized = np.exp(arr - m)
    total = float(np.sum(normalized))
    if total <= 0.0:
        return 0.0, 0.0

    normalized /= total
    neff = 1.0 / float(np.sum(normalized ** 2))
    max_fraction = float(np.max(normalized))
    return float(neff), max_fraction


def simulate_block(N, w, L, boundary, attempts, seed):
    """Worker: generate one independent block."""
    rng = np.random.default_rng(int(seed))

    log_weights = []
    overlap_values = []
    successes = 0

    for _ in range(int(attempts)):
        log_weight, trajectory, overlaps = grow_confined_log(
            N, w, L, boundary, rng
        )
        if np.isfinite(log_weight) and trajectory is not None:
            log_weights.append(log_weight)
            overlap_values.append(overlaps)
            successes += 1

    neff, max_fraction = compute_neff(log_weights)

    return {
        "attempts": int(attempts),
        "successes": int(successes),
        "success_fraction": float(successes / attempts),
        "Neff": float(neff),
        "max_weight_fraction": float(max_fraction),
        "mean_overlap_successful": (
            float(np.mean(overlap_values))
            if overlap_values else np.nan
        ),
    }


def quality_flag(block_neff, max_fraction, success_fraction):
    block_neff = np.asarray(block_neff, dtype=float)

    if block_neff.size < 3:
        return "FAIL", "Fewer than three valid blocks"

    if np.min(block_neff) < 5 or max_fraction > 0.50:
        return "CAUTION", "Severe weight degeneracy in at least one block"

    if np.min(block_neff) < 20 or max_fraction > 0.20:
        return "LIMITED", "Moderate weight degeneracy"

    if success_fraction < 0.90:
        return "LIMITED", "Non-negligible chain mortality"

    return "OK", ""


def run_point_parallel(
    N, w, L, boundary, Rg, Rg_err, attempts, blocks, seed, nproc=1
):
    attempts_per_block = attempts // blocks
    if attempts_per_block < 1:
        raise ValueError("attempts must provide at least one attempt per block.")

    args_list = [
        (
            N, w, L, boundary, attempts_per_block,
            int(seed) + 1000 * block_idx
        )
        for block_idx in range(blocks)
    ]

    block_records = []

    if nproc > 1:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = [
                executor.submit(simulate_block, *args)
                for args in args_list
            ]
            for future in as_completed(futures):
                block_records.append(future.result())
    else:
        for args in args_list:
            block_records.append(simulate_block(*args))

    block_neff = np.asarray(
        [b["Neff"] for b in block_records], dtype=float
    )
    block_max_fraction = np.asarray(
        [b["max_weight_fraction"] for b in block_records], dtype=float
    )
    block_success = np.asarray(
        [b["success_fraction"] for b in block_records], dtype=float
    )
    block_overlap = np.asarray(
        [b["mean_overlap_successful"] for b in block_records], dtype=float
    )

    flag, note = quality_flag(
        block_neff,
        float(np.max(block_max_fraction)),
        float(np.mean(block_success)),
    )

    return {
        "N": int(N),
        "w": float(w),
        "L": int(L),
        "boundary": boundary,
        "Rg": float(Rg),
        "Rg_err": float(Rg_err),
        "L_over_Rg": float(L / Rg),
        "L_over_Rg_err": float(abs(L * Rg_err / Rg**2)),
        "attempts": int(attempts_per_block * blocks),
        "attempts_per_block": int(attempts_per_block),
        "n_blocks": int(blocks),
        "Neff_mean": float(np.mean(block_neff)),
        "Neff_median": float(np.median(block_neff)),
        "Neff_min": float(np.min(block_neff)),
        "Neff_max": float(np.max(block_neff)),
        "max_weight_fraction": float(np.max(block_max_fraction)),
        "success_fraction_mean": float(np.mean(block_success)),
        "success_fraction_min": float(np.min(block_success)),
        "mean_overlap_successful": float(np.nanmean(block_overlap)),
        "quality_flag": flag,
        "quality_note": note,
        "block_Neff": block_neff.tolist(),
        "block_max_weight_fraction": block_max_fraction.tolist(),
        "block_success_fraction": block_success.tolist(),
        "block_mean_overlap_successful": block_overlap.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supplementary Fig. S4: confined DJ Rosenbluth N_eff "
                    "using the canonical Rg master."
    )
    parser.add_argument("--rg-master", required=True,
                        help="Canonical Rg master CSV.")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--a", type=float, default=1.0,
                        help="Retained for compatibility; Rg comes from master.")
    parser.add_argument("--attempts", type=int, default=20000)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--w-values", type=float, nargs="+",
                        default=DEFAULT_W_VALUES)
    parser.add_argument("--L-values", type=int, nargs="+",
                        default=DEFAULT_L_VALUES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--output-dir", default="output_S4")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.N < 2 or args.a <= 0 or args.blocks < 3:
        raise ValueError("Require N>=2, a>0, and at least three blocks.")
    if args.nproc < 1:
        raise ValueError("nproc must be at least 1.")
    if any(int(L) < 2 or int(L) % 2 != 0 for L in args.L_values):
        raise ValueError("All L values must be even integers >=2.")

    if args.smoke:
        attempts = min(2000, args.attempts)
        blocks = min(6, args.blocks)
        w_values = list(args.w_values)
        L_values = list(args.L_values[:3])
    else:
        attempts = args.attempts
        blocks = args.blocks
        w_values = list(args.w_values)
        L_values = list(args.L_values)

    attempts -= attempts % blocks
    if attempts <= 0:
        raise ValueError("attempts must exceed blocks and be divisible by blocks.")

    # Load and validate every canonical Rg before any simulation starts.
    rg_master = load_rg_master(args.rg_master, args.N, w_values)
    master_hash = sha256_file(args.rg_master)

    for w in w_values:
        q = rg_master[w]["quality_flag"]
        if str(q).upper() not in {"OK"}:
            print(
                f"[warning] Rg master quality for N={args.N}, w={w:g} "
                f"is {q}; value will still be used.",
                file=sys.stderr,
            )

    os.makedirs(args.output_dir, exist_ok=True)

    boundaries = ["absorbing", "reflecting"]

    print("=" * 88)
    print("SUPPLEMENTARY FIG. S4 — CONFINED DJ ROSENBLUTH N_eff")
    print("=" * 88)
    print(f"Rg master: {args.rg_master}")
    print(f"Rg master SHA256: {master_hash}")
    print(f"N={args.N}; w={w_values}; L={L_values}")
    print(f"boundaries={boundaries}; attempts={attempts}; blocks={blocks}")
    print(f"nproc={args.nproc}")
    print("L values are even integers; tether z0=L/2.")
    print("L/Rg uses canonical DJ Rg from the master file.")
    print("w=0: all surviving chains have equal weight; reflecting "
          "Neff equals attempts/block.")

    if args.smoke:
        print("SMOKE MODE: output is not suitable for quantitative inference.")

    records = []
    point_index = 0

    for boundary in boundaries:
        for w in w_values:
            Rg = rg_master[w]["Rg"]
            Rg_err = rg_master[w]["Rg_err"]
            master_quality = rg_master[w]["quality_flag"]

            print(
                f"\nBoundary={boundary}; w={w:g}; "
                f"Rg={Rg:.8f} +/- {Rg_err:.8f}; "
                f"master_quality={master_quality}"
            )

            for L in L_values:
                print(f"  Running L={L} ...")

                record = run_point_parallel(
                    args.N,
                    w,
                    int(L),
                    boundary,
                    Rg,
                    Rg_err,
                    attempts,
                    blocks,
                    args.seed + point_index,
                    args.nproc,
                )
                record["Rg_master_quality"] = master_quality
                records.append(record)
                point_index += 1

                print(
                    f"    L/Rg={record['L_over_Rg']:.4f}; "
                    f"median Neff={record['Neff_median']:.2f}; "
                    f"min Neff={record['Neff_min']:.2f}; "
                    f"quality={record['quality_flag']}"
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(
        args.output_dir, f"FigS4_Neff_ConfinedDJ_{stamp}"
    )

    csv_path = base + "_points.csv"
    json_path = base + "_provenance.json"
    png_path = base + ".png"
    pdf_path = base + ".pdf"

    csv_fields = [
        "N", "w", "L", "boundary", "Rg", "Rg_err",
        "L_over_Rg", "L_over_Rg_err",
        "attempts", "attempts_per_block", "n_blocks",
        "Neff_mean", "Neff_median", "Neff_min", "Neff_max",
        "max_weight_fraction",
        "success_fraction_mean", "success_fraction_min",
        "mean_overlap_successful",
        "quality_flag", "quality_note", "Rg_master_quality",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: record.get(field, "") for field in csv_fields}
            )

    # Plot median block Neff against the canonical L/Rg.
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.1), sharey=True
    )

    markers = {
        "OK": "o",
        "LIMITED": "s",
        "CAUTION": "D",
        "FAIL": "x",
    }

    for ax, boundary in zip(axes, boundaries):
        ax.set_title(f"{boundary.capitalize()} boundaries")
        ax.set_xlabel(r"$L/R_g$")
        ax.set_ylabel(r"$N_{\mathrm{eff}}$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.axhline(
            50.0, linestyle="--", lw=1.2,
            label=r"$N_{\mathrm{eff}}=50$"
        )

        for w in w_values:
            subset = [
                r for r in records
                if r["boundary"] == boundary and np.isclose(
                    r["w"], w, rtol=0.0, atol=1e-12
                )
            ]
            subset.sort(key=lambda r: r["L_over_Rg"])

            x = np.array([r["L_over_Rg"] for r in subset], dtype=float)
            y = np.array([r["Neff_median"] for r in subset], dtype=float)
            xerr = np.array(
                [r["L_over_Rg_err"] for r in subset], dtype=float
            )

            ax.plot(x, y, lw=1.2, alpha=0.7)

            for j, record in enumerate(subset):
                ax.errorbar(
                    record["L_over_Rg"],
                    record["Neff_median"],
                    xerr=record["L_over_Rg_err"],
                    fmt=markers.get(record["quality_flag"], "o"),
                    ms=6,
                    capsize=2,
                    label=(
                        fr"$w={w:g}$"
                        if j == 0 else None
                    ),
                )

        ax.legend(loc="best", frameon=True)

    fig.suptitle(
        "Rosenbluth effective sample size for confined Domb–Joyce chains",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "metadata": {
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command_line": " ".join(shlex.quote(v) for v in sys.argv),
            "figure": "Supplementary Fig. S4",
            "N": int(args.N),
            "a": float(args.a),
            "w_values": [float(w) for w in w_values],
            "L_values": [int(L) for L in L_values],
            "boundaries": boundaries,
            "tether": "z0=L/2, integer midpoint",
            "Rg_source": os.path.basename(args.rg_master),
            "Rg_master_sha256": master_hash,
            "Rg_selection_key": ["model=DJ", f"N={args.N}", "w=requested"],
            "Rg_definition": "canonical master value sqrt(<Rg^2>_W)",
            "width_definition": "geometric wall separation",
            "allowed_layers": "z=1,...,L-1",
            "overlap_definition": (
                "incremental nonbonded monomer occupancy at proposed site"
            ),
            "proposal_definition": "q(step) proportional to exp(-w*DeltaV)",
            "weight_definition": (
                "log W += log(sum_allowed exp(-w*DeltaV)); "
                "no extra factor of 6"
            ),
            "Neff_definition": "(sum W)^2/sum(W^2), per independent block",
            "plot_value": "median block Neff",
            "quality_diagnostics": [
                "mean, median, minimum and maximum block Neff",
                "maximum normalized weight fraction",
                "mean and minimum successful-chain fraction",
                "block quality flag",
            ],
            "quality_flag_policy": {
                "CAUTION": "min block Neff < 5 or max normalized weight > 0.50",
                "LIMITED": (
                    "min block Neff < 20 or max normalized weight > 0.20 "
                    "or mean survival fraction < 0.90"
                ),
            },
            "boundary_warning": (
                "absorbing and reflecting ensembles are shown separately "
                "and are not pooled"
            ),
            "even_L_enforced": True,
            "master_Rg_required": True,
            "smoke_mode": bool(args.smoke),
            "points_csv": os.path.basename(csv_path),
        },
        "rg_master_rows_used": {
            str(w): {
                "Rg": rg_master[w]["Rg"],
                "Rg_err": rg_master[w]["Rg_err"],
                "quality_flag": rg_master[w]["quality_flag"],
                "stream": rg_master[w]["stream"],
            }
            for w in w_values
        },
        "records": records,
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, default=float)

    print("\nSaved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
