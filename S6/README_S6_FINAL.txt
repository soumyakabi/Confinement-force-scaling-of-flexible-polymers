S6 FINAL PATCHED DATASET

Purpose
-------
Patch only the w=1, L=infinity normalization in the S6 production dataset using the independent high-statistics unconfined SIS/Rosenbluth rerun. All finite-width PERM states are retained from the original S6 production run.

New reference
-------------
logZ_inf(w=1) = 323.076956090391 +/- 0.009405613948 (block SE)
64 blocks x 8192 roots/block = 524288 chains.

Outputs
-------
FigS6_DJ_FreeEnergy_MASTER_REVISED_PATCHED.png / .pdf : final figure
FigS6_source_data_for_paper.csv : compact 27-row plotted source data
FigS6_point_level_patched.csv : full point-level audit data
FigS6_state_level_production_patched.csv : state-level production summary
FigS6_block_level_patched.csv : complete block-level record, with new 64 w=1,L=inf blocks
FigS6_monotonicity_audit_patched.csv : monotonicity audit
FigS6_w_dependence_region_audit_patched.csv : unique pairwise w-dependence audit
FigS6_convergence_audit_patched.csv : block convergence audit
S6_w1_unconfined_highstat_summary.json / blocks.csv : raw independent w=1 reference rerun
FigS6_patch_provenance.json : patch provenance
FigS6_FreeEnergy_DJ_MASTER_REVISED_FINAL_numba_v1.py : finite-width S6 production code
rerun_S6_w1_unconfined_reference.py : independent w=1 reference rerun code
