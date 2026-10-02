FIGURE 1 REV9 FINAL — PRODUCTION README
========================================

Purpose
-------
Final Gaussian absorbing-slit benchmark for the revised manuscript.

This version was created to correct the REV8 force-extraction inconsistency.
The REV8 plotted force used the spacing of the surrounding grid points, whereas
its validation used a different derivative step. REV9 uses the production
finite-difference convention literally:

    f(L) = [ln Z(L+delta_L) - ln Z(L-delta_L)]/(2 delta_L)

with delta_L = 2 by default, so the total endpoint separation is 4.

The production estimator uses the exact conditional Brownian-bridge survival
probability from the absorbing interval kernel (two-wall image representation).
The image-series cutoff is audited against a larger cutoff and against the
independent interval-kernel eigenfunction representation.

Geometry
--------
N = 200 Gaussian contour steps
step variance = a^2/3 per confined coordinate
absorbing walls: z=0 and z=L
co-moving midpoint tether: z0=L/2
R_g^(0) = a sqrt(N/6)

Sampling
--------
Independent blocks are used for uncertainty. Signed block-force estimates are
retained. There is no sign filtering, smoothing, post-hoc point deletion, or
artificial error floor.

Recommended production settings
-------------------------------
40 blocks
2,000,000 walkers per state for L/Rg < 1.20
1,000,000 walkers per state for 1.20 <= L/Rg < 1.60
300,000 walkers per state for L/Rg >= 1.60
exact interval image kernel, images = +/-3 (with convergence audit)
bootstrap = 5000
seed = 20260930

The direct MC grid begins at L/Rg=1.0. The exact analytic curve may be shown
from 0.6 to 5.0. Direct survival sampling below about L/Rg=1 is exponentially
rare for N=200 and is not used as a forced production target.

Self-test
---------
In Kaggle:
  !python /kaggle/working/Fig1_Gaussian_Confinement_REV9_FINAL.py --self-test

Smoke test
----------
  !python /kaggle/working/Fig1_Gaussian_Confinement_REV9_FINAL.py --smoke \
      --outdir /kaggle/working/Fig1_REV9_SMOKE --workers 2

Production
----------
  !python /kaggle/working/Fig1_Gaussian_Confinement_REV9_FINAL.py \
    --outdir /kaggle/working/Fig1_REV9_FINAL \
    --N 200 \
    --free-energy-ratios 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 \
    --force-ratios 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 \
    --delta-L 2 \
    --blocks 40 \
    --block-counts 8 16 24 32 40 \
    --walks-strong 2000000 \
    --walks-moderate 1000000 \
    --walks-weak 300000 \
    --strong-cut 1.20 \
    --moderate-cut 1.60 \
    --images 3 \
    --bootstrap-reps 5000 \
    --seed 20260930 \
    --workers 4

Outputs
-------
Fig1_Gaussian_Confinement_REV9.png/.pdf
Fig1_PointLevel.csv                  central free-energy states
Fig1_ForcePointLevel.csv             force states and block-derived uncertainty
Fig1_BlockLevel.csv                  signed block force data
Fig1_Convergence.csv                 cumulative block convergence
Fig1_KernelAudit.csv                 image/eigen kernel and exact-FD audits
Fig1_ValidationSummary.json          compact QA summary
Fig1_Provenance.json                 full reproducibility metadata

Validation philosophy
---------------------
The exact Gaussian chain-rule force is the reference. The Monte Carlo estimator
is independently audited through the exact interval Brownian-bridge kernel and
through convergence with the SAME delta_L=2 estimator used in production.

CVT theorem
-----------
No Contact Value Theorem calculation is included. The final validation chain
uses the exact Gaussian benchmark, Brownian-bridge kernel audit, and block-level
finite-difference convergence.

Do not use smoke-test numbers for inference or publication.
