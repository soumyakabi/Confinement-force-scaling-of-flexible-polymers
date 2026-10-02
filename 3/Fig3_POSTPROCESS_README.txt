FIGURE 3 FINAL POST-PROCESSED PACKAGE
===================================

Purpose
-------
This directory contains the final reviewer-facing Fig. 3 package derived
from the completed SAW PERM production run. No Monte Carlo simulation was
rerun during post-processing.

Corrections applied
--------------------
1. All SAW-derived L/Rg and f Rg quantities use the current canonical
   Rg master (including the high-statistics N=200 SAW value).
2. Delta F is reconstructed from the block partition-function estimates as
      Delta F = ln(mean Z_inf) - ln(mean Z_L)
   using the equal-sized production blocks.
3. Signed block forces are retained; no sign-based point selection is used.
4. Gaussian points are retained as the exact continuous-Gaussian reference
   and audited against the analytic midpoint-tethered absorbing-wall formula.
5. The force fit is restricted to L/Rg <= 1.9 and is interpreted as a
   finite-range effective exponent, not an asymptotic proof.
6. Explicit numerical x ticks are used in scaled panels.

Recommended paper/repository files
-----------------------------------
Fig3_FORCE_SCALING_FINAL.pdf/png
Fig3_FREE_ENERGY_SCALING_FINAL.pdf/png
Fig3_SAW_PointLevel.csv
Fig3_SAW_BlockLevel.csv
Fig3_SAW_ForceBlockLevel.csv
Fig3_SAW_ForceBootstrap_Audit.csv
Fig3_SAW_Convergence.csv
Fig3_SAW_ForceFit_FINAL.csv
Fig3_SAW_LPlan.csv
Fig3_SAW_PilotThresholds.csv
Fig3_Gaussian_PointLevel.csv
Fig3_Gaussian_Exact_Audit.csv
Rg_MASTER_FINAL_MERGED_CURRENT.csv
Fig3_POSTPROCESS_PROVENANCE.json
Fig3_Rg_Rebasing_Audit.csv
Fig3_DeltaF_Reconstruction_Audit.csv
Fig3_SHA256_MANIFEST.txt
Fig3_POSTPROCESS_FINAL.py

The original pre-correction values are not used for manuscript plotting;
they are represented only through the audit files needed for provenance.
