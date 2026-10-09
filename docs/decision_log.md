# Working scope and evaluation lock

Recorded 2026-09-30, before any test-cohort result of the batch-128 runs was looked at. Title and method may still
change; the evaluation protocol below may not.

## Scope (working)

- **Working title:** MorphoMix: Mask-Guided Morphological Augmentation for Cross-Laboratory Acute Lymphoblastic Leukemia Classification in Single-Cell and Whole-Field Blood Smear Images.

- **Task:** binary classification, Normal vs ALL (B-ALL).
- **Candidate claim:** mask-guided augmentation (MorphoMix) improves B-ALL classification across laboratories and
  across image formats (single-cell crops to multi-cell fields), without target data in training or selection.
  (Superseded: C2', the Aria crop scale and the C3 erythrocyte density were later set from unlabelled Aria
  measurements, and the tests guided later choices; see the entries of 2026-10-05 and 2026-10-06.)
- **Four-class subtyping (Aria Benign / Early / Pre / Pro) is not a claim.** Its evidence is weak: an ImageNet-only
  probe reaches ~0.99 macro-F1 and six colour statistics alone ~0.83, and Aria has no patient IDs. At most a
  supplementary probe analysis.

## Locked (do not change after this date)

Superseded: the lock below was broken on 2026-09-30, when the test predictions of the development run were read
(see the unblinding record), and the cross-laboratory protocol replaced its cohorts; it is kept as the record of the
original plan. Every later decision is logged below with the evidence it used.

1. **Test cohorts:** C-NMC fold_2 (cross-patient), ALL-IDB2 (cross-laboratory, single cell) and the Aria val and
   test splits as a binary cohort (Benign = Normal, Early/Pre/Pro = ALL; cross-laboratory, multi-cell).
2. **Selection:** checkpoints and every method or hyperparameter choice use C-NMC val only (`--screen`,
   `stress --selection-only`). No choice uses fold_2, ALL-IDB2 or Aria. A selection on Aria val may be reported
   only as a labelled oracle analysis, never as the main result.
3. **Statistics:** paired bootstrap per test cohort, Holm-adjusted p < 0.05 within each (cohort, seed) family.
4. **Blinding:** while the method is still changing, test tables (`phase1_pretrain_results`,
   `statistical_tests`, held-out stress) are not opened; progress reports quote val numbers only. The test cohorts
   are read once, for the final method.

## Free to change (decided on C-NMC val only)

Title, framing, MorphoMix components (including a multi-cell field synthesis, C8), their probabilities and
weights, and training hyperparameters.

## Decision tree (deferred until the method is final)

- MorphoMix significantly better than the baselines on ALL-IDB2 (and supported by stress): full scope above,
  including C8 and the Aria multi-cell evaluation.
- Not better: a rigorous benchmark of augmentations for B-ALL plus the dataset leakage audit (C-NMC twins across
  patient IDs, repeated Aria fields, colour shortcut) as the main contribution; Aria binary stays as a second
  external cohort for every arm through crop-and-aggregate inference.

## Status on 2026-09-30

- The batch-128 run (base_lr 4e-4, 30 epochs, seeds 42-44, every arm) is a development baseline. Its val ROC-AUC:
  basic 0.941, mixup 0.937, cutmix 0.940, saliencymix 0.937, morpho_mix 0.920 (mean of the best epoch over three
  seeds). Its test tables stay unopened under rule 4.
- Not built yet: C8 field synthesis, the Aria binary crop-and-aggregate evaluation.

## Revised method (decided 2026-09-30, after the unblinding below)

One Azure-B cell mask M drives four components, numbered in execution order:

1. **C1 appearance transfer:** a mask-guided CycleGAN (C-NMC to the MLL23 style), trained once beforehand and
   frozen; only cell pixels change, blended with a random strength alpha: `x' = M*[(1-a)x + a*G(x)] + (1-M)*x`.
2. **C2 cell scale** (the former C5).
3. **C3 background randomisation** (the former C2): everything outside M becomes a synthetic smear.
4. **C4 multi-cell field synthesis** (new): same-class cells shrunk to Aria's cell size and composed on a C3
   background.

Dropped: the former C1 transplant, C3 mask alignment, C4 HED jitter and C7 Layer-CAM loss (C6 was dropped
earlier). Layer-CAM is kept for XAI evaluation only. Each component is justified by a leave-one-out ablation on the
LeukemiaAttri val cohort; one extra arm replaces M in C1 by a learned attention mask (as in Baydilli, BSPC 2025).
Data roles: train C-NMC (all folds), val LeukemiaAttri, CycleGAN target MLL23 (unlabelled), test ALL-IDB2 and Aria
binary.

## Unblinding record

- **2026-09-30:** at the author's request, the Phase 1 test predictions of the development baseline (C-NMC fold_2
  and ALL-IDB2, all five arms, seeds 42-44) were read, before `stats` and the ablation had finished. Summary: on
  ALL-IDB2, MorphoMix had the highest mean ROC-AUC (0.790 vs 0.667-0.762) and the only non-degenerate positive rate
  (0.60 vs 0.95-0.99), with unadjusted per-seed bootstrap differences significant in some seeds only; on fold_2 it
  was lowest (0.759 vs 0.798-0.848), significantly in every seed. Any method change after this date was made with
  these numbers known and is reported as such; the Aria multi-cell cohort has not been evaluated.
- **2026-10-03 (cross-lab protocol, first full run, the initial run):** with the method fixed a priori
  (arms basic, hed_jitter, randstainna, stain_mixup, morpho_mix; seeds 42-44; base_lr 4e-4, 30 epochs), the test
  predictions on ALL-IDB2 and Aria were read at the author's request after the 15 main runs, before the ablation and
  `stats` finished. Initial results (ROC-AUC, mean of three seeds): ALL-IDB2 basic 0.608, hed_jitter 0.698,
  randstainna 0.792, stain_mixup 0.665, morpho_mix 0.753; Aria 0.395, 0.362, 0.393, 0.430, 0.351 (all below chance;
  AUPRC below the 0.845 prevalence). The author then chose to change the method and training recipe guided by these
  test results, with full disclosure: every later result is exploratory (hypothesis generating), is reported next
  to these initial results, and is not evidence of generalisation.
- **2026-10-03, test-guided change 1:** the component ablation of the first cross-lab run (3 seeds) showed that
  removing C2 (cell scale) improved every cohort (val ROC-AUC 0.742 vs 0.719, ALL-IDB2 0.772 vs 0.753, Aria 0.474
  vs 0.351), while removing C1 made ALL-IDB2 unstable (0.633 +/- 0.225) and removing C3 lowered val and ALL-IDB2.
  C2 is therefore dropped and replaced by components aimed at the observed failure (Aria whole fields: about 27 px
  cells, many cells per image), chosen with these test results known.
- **2026-10-03, test-guided change 2 (inference):** Aria is scored per detected cell (`inference.field: cells`;
  crop scale 3.6 fixed a priori from the cell-size match of about 26 px to 95 px; top-3 aggregation and the
  val-Youden decision threshold chosen with the test results known). On the existing checkpoints the mean Aria
  ROC-AUC rose from 0.386 to 0.633 and AUPRC from 0.788 to 0.896; ALL-IDB2 macro-F1 rose from 0.492 to 0.563 with
  the val-Youden threshold. Exploratory. `--set inference.field=resize inference.decision_threshold=0.5` reproduces
  the first run.
- **2026-10-03, scope decision for submission (time-limited):** MorphoMix is fixed as C1 (per-region stain
  transfer) + C3 (background randomisation); the cell-scale component (C2) is dropped on the ablation evidence above,
  and the field-scale replacements (small-cell rendering, multi-cell field synthesis) were built but not adopted,
  for lack of time to validate them. All five datasets stay; Aria is scored per detected cell. The final run goes to
  a new Drive folder (the second run); the first run is kept unchanged as the initial, pre-change
  result. Everything in the final run is exploratory in the sense of the unblinding record above.
- **2026-10-04 to 2026-10-05, test-guided change 3 (final composition and inference scale):** the earlier
  submission decision (C1 + C3) was revisited after a local proxy grid (`main.py tune --proxy`: 30 % stratified
  C-NMC subset, 15 epochs, batch 32 x 4 accumulation, laptop GPU, 42 runs), whose val, ALL-IDB2 and Aria results
  were read for every candidate. Four seeds (42-45), ROC-AUC mean +/- sd, Aria per cell (top-3) at crop scale 3.6 /
  4.5:

  | Candidate | val | ALL-IDB2 | Aria 3.6 | Aria 4.5 |
  | --- | --- | --- | --- | --- |
  | C1 RandStainNA form + C2' + C3 (hyb-c2p-rsnonly) | 0.738 +/- 0.022 | 0.798 +/- 0.056 | 0.631 +/- 0.070 | 0.625 +/- 0.103 |
  | C1 MLL23 form + C2' + C3 (ref-c2p) | 0.743 +/- 0.011 | 0.693 +/- 0.060 | 0.658 +/- 0.119 | 0.700 +/- 0.136 |
  | randstainna arm (ref-rsn) | 0.725 +/- 0.011 | 0.743 +/- 0.110 | 0.622 +/- 0.106 | 0.648 +/- 0.168 |
  | C1 MLL23 form + C3 with LP-FT (lpft-ref-noc2) | 0.719 +/- 0.012 | 0.736 +/- 0.034 | 0.633 +/- 0.085 | 0.656 +/- 0.126 |

  On two seeds, adding C4 to C2' lowered val (0.707 vs 0.742) and Aria (0.606 vs 0.680) and raised ALL-IDB2 (0.774
  vs 0.723); both C1 forms together (hyb-c2p) gave ALL-IDB2 0.511; the EMA 0.999 recipe collapsed at the proxy's
  step count; MixStyle raised val but lowered the tests. Post-hoc methods on cached predictions of the first-run and
  proxy checkpoints: Reinhard test-time normalisation lowered Aria (randstainna, three seeds: 0.703 to 0.525), TENT
  raised Aria but lowered val and ALL-IDB2, WiSE-FT moved ALL-IDB2 by at most +0.02 with val flat; none is adopted.
  Crop scale: on the first-run checkpoints with both scales scored (basic, morpho_mix, randstainna; nine runs) mean
  Aria ROC-AUC was 0.649 at 3.6, 0.681 at 4.5 and 0.665 averaged over both; on the four proxy candidates 0.636,
  0.657 and 0.651. Decision: MorphoMix = C1 in its RandStainNA form (`rsn_prob` 0.5, `appearance_prob` 0) + C2'
  (0.25) + C3, C4 off; Aria crops at 224/4.5 field px. The MLL23 form had the better val and Aria but the weakest
  ALL-IDB2; the RandStainNA form had the best mean over the three cohorts. Ablation arms: b no in-cell stain, c no
  C2', d no C3, e the MLL23 form instead of RandStainNA. The proxy differs from the final recipe (full data, 30
  epochs, batch 128), so the final numbers may differ. Everything in the second run is exploratory.
- **2026-10-05, test-guided change 4 (field inference):** a failure analysis of the Aria fields (cached crop scores) showed that the top-3 aggregate mixed two errors: ALL fields (Pro especially) hold 1-3
  white cells, so their top 3 included border-cut fragments and erythrocytes (crop p(ALL) 0.35-0.48); Benign fields
  hold many lymphoid cells (median 9 detections vs 6-7 for Early / Pro), so the top 3 picked the most blast-like of
  many. Two label-free rules now drop a detection when less than 0.8 of its 1.5 r disk lies inside the field or
  less than 0.2 of its disk is Azure-B nucleus inside the cell mask (15 % of the 40,533 detections were cut by the
  border; 22 of 3,256 fields keep no cell and fall back to the whole-field score), and the field score is the mean
  over the remaining cells. Mean Aria ROC-AUC at crop scale 4.5, top-3 vs filter + mean: final candidate (proxy,
  four seeds) 0.625 vs 0.759; MLL23-C1 candidate (proxy) 0.700 vs 0.776; first-run randstainna 0.783 vs 0.794,
  morpho_mix 0.579 vs 0.598, basic 0.681 vs 0.750. A nucleus threshold of 0.4 gave the same result (within 0.003).
  With the filter, crop scales above 4.5 changed little (final candidate 0.759 / 0.764 / 0.765 / 0.756 at 4.5 / 5.0
  / 5.5 / 6.0; randstainna 0.794 / 0.810 / 0.816 / 0.816), so 4.5 is kept rather than tuned further.
- **2026-10-05, auxiliary MLL23 cells, rejected:** MLL23 myeloblasts (as ALL) and lymphocytes (as Normal), 2,500 each,
  cut onto black in the C-NMC format (`aux_train`), were added to the training set of the final candidate in the
  local proxy (four seeds, scored alike: 1 view, whole-cell filter, mean, scale 4.5). ROC-AUC without / with: val
  0.738 / 0.757, ALL-IDB2 0.798 / 0.709, Aria 0.759 / 0.732. A data audit found label shortcuts in the auxiliary set
  (erythrocyte fragments attached to the masks, more often to myeloblasts: artefact-only features reach AUC 0.85;
  a class ratio that differs from C-NMC; myeloblasts larger than lymphocytes), and myeloblasts are not lymphoblasts.
  `aux_train` stays off; MLL23 remains the stain-style bank only.
- **2026-10-05, test-guided change 5 (auxiliary training cells, adopted):** cells of a fourth laboratory, Bodzas et
  al., Sci Data 2023 (Ostrava; peripheral blood, MGG, 100x oil; CC0), chosen after a literature check because it has
  real lymphoblasts and normal lymphoid cells from one laboratory and shares no source with any cohort: 2,300
  lymphoblasts (ALL) and 1,072 lymphocytes and monocytes (Normal), C-NMC's class ratio, so the source carries no
  label prior. Cells are cut onto black in the C-NMC format with a GrabCut mask grown from the nucleus that drops
  erythrocyte-coloured pixels beyond a 7 px rim (`src/datasets/cells.py:smear_cell`); a mask audit found one
  component per cell and artefact features near chance (strongest ROC-AUC 0.68-0.70, mostly cell size and monocyte
  shape). Local proxy, four seeds, scored alike (1 view, whole-cell filter, mean, scale 4.5), ROC-AUC without /
  with: val 0.738 / 0.755, ALL-IDB2 0.798 / 0.873, Aria 0.759 / 0.826. Adopted for every arm (`aux_train.enabled`);
  ablation arm f trains MorphoMix without it. The test cohorts drove this choice, so the gain is exploratory.
- **2026-10-05, after the proxy (code review fixes, not re-evaluated on the tests):** (a) the auxiliary-cell border
  check never fired (`smear_cell` keeps a 3 px background frame), so 53 lymphoblasts and 2 Normal cells cut by the
  crop border were kept; the check now uses a 4 px margin, and the rebuilt set has 2,250 lymphoblasts and 1,049
  lymphocytes and monocytes (C-NMC's ratio), against 2,300 / 1,072 in the proxy run above. (b) The whole-cell filter
  of change 4 emptied 22 Aria fields (21 Benign), which then fell back to the whole-field resize score on another
  scale; such a field now keeps all its detected cells. Both changes follow from the code, not from new test
  results; the final run uses them.
- **2026-10-05, decision rule for macro-F1 (during the final run, from its first 13 predictions):** under
  `val_youden` every arm predicted 84-100 % of the test images as ALL (Aria: almost all), so macro-F1 fell to
  0.36-0.62 on ALL-IDB2 and 0.46-0.53 on Aria while ROC-AUC stayed 0.80-0.96. Fixed 0.5 (the a-priori rule of the
  first run; the class-weighted loss makes it the neutral point) gave 0.66-0.74 and 0.46-0.71; the label-free
  prior-shift rule (EM on the unlabelled test scores) did not help. Macro-F1 is now reported at 0.5 (primary) and
  at val_youden (secondary); both are recomputed from the stored probabilities, no run was retrained. ROC-AUC and
  AUPRC remain the primary metrics. The change was made with these test results known.
- **2026-10-05, after the final run (CBM-style review by four reviewer agents, reports kept
  locally):** no image leakage between roles (exact and perceptual hashes over 8 dihedral views)
  and every table number reproduced from the predictions; one metric bug fixed (sensitivity at 90 % specificity
  dropped the ROC point at exactly FPR 0.1; `analysis` rerun). Findings that change the analysis, not the method:
  (a) per-seed bootstrap families test checkpoints, not methods (Aria MorphoMix - hed_jitter is significant in all
  three seeds with opposite signs); (b) Aria's units are field groups, not its 89 patients (no patient IDs), so
  its intervals are optimistic; (c) the Bodzas lymphoblasts and normal cells differ in slide colour (background
  colour alone AUC 0.98, cell colour 0.996), a patient / slide cue the proxy audit had not tested; (d) the C2' cell
  size, the Aria crop scale and the C3 erythrocyte density were set from (unlabelled) Aria measurements, so no
  claim of "no target data" is made; (e) the background-swap reading of section 3 of `shortcut_analysis.md` was
  wrong (arm d drops less than MorphoMix) and needs a same-class control. Added before writing, with the method
  unchanged: seeds 45 and 46 for every main arm; a seed-level hierarchical bootstrap (`analysis`:
  `method_level_tests`); control arms g (auxiliary cells colour-matched to the C-NMC template; per-feature colour
  AUC between their classes drops from up to 0.81 to 0.50-0.56) and h-k (the four baselines without auxiliary
  cells); frozen DinoBloom-S / -B probes (`foundation`); the shortcut analyses moved into the repo (`shortcut`)
  with a same-class swap control (3 donors) and background removal at test time. The paper framing is decided
  after these results.
- **2026-10-06, test-guided change 6 (final composition, author's decision before the deadline):** MorphoMix becomes
  the former ablation arm e, C1 in its MLL23 form (`appearance_prob` 0.5, `rsn_prob` 0) + C2' + C3, because in the
  second-run ablation (three seeds) it had ROC-AUC 0.888 on ALL-IDB2 and 0.886 on Aria against 0.810 / 0.804
  for the RandStainNA form, and above every baseline on Aria; on val it was lower (0.754 against 0.778), so a val-only
  rule would have kept the RandStainNA form. Arm d (no C3, Aria 0.901) was not adopted because its combination with
  the MLL23 form was never measured. The choice is made on the test cohorts; the claims are revised accordingly.
  Ablation arm e is now the RandStainNA form. Every run, baselines included, is retrained from scratch into
  the final run (`results/`) with seeds 42-46 (ablation and control arms 42-44); the second run is kept unchanged.
- **2026-10-06, test-guided change 7 (MorphoMix-only tuning on the test cohorts, the author's decision; recorded
  before any run of it):** against the reviewers' and Claude's advice, MorphoMix alone is tuned on ALL-IDB2 and Aria
  before the final run; the baselines are not tuned, so any MorphoMix lead on these cohorts is not a fair comparison
  and is not evidence of generalisation; the paper must say so. Grid (`tune --grid final`, Colab A100, 30 epochs, batch
  128, seeds 42 and 43, the C1-MLL23 composition pinned): e-ref (no change), e-noc3 (C3 off), e-noc2 (C2' off),
  e-app08 (`appearance_prob` 0.8), e-lr2 / e-lr8 (`phase1.base_lr` 2e-4 / 8e-4). Pre-declared rule (`--adopt`): the
  candidate with the highest seed-mean of (ALL-IDB2 ROC-AUC + Aria ROC-AUC) / 2 is written into the config only when
  it beats e-ref by at least 0.01; `tuning/adopted.json` records the scores and the choice. The final run follows in
  the same Colab job. Loader workers were raised (8 -> 11, eval 4 -> 8, prefetch 2 -> 4) for speed; batch size stays
  128.
- **2026-10-06, amendment to change 7 (before any final-run training):** the two learning-rate candidates were replaced by
  e-c3low (`background_prob` 0.15), because `phase1.base_lr` is shared by every arm: adopting it would have retrained
  the baselines at a value picked on MorphoMix's test scores. The grid is e-ref, e-noc3, e-noc2, e-app08, e-c3low.
  If e-noc3 or e-noc2 is adopted, ablation arm d or c equals arm a and is reported as such.
- **2026-10-06, second amendment to change 7 (code review, before any final-run training):** the final-grid candidates train with seeds
  101 and 102, outside the reported seeds 42-46, so no reported run is one the choice was made on (deterministic
  runs would otherwise reproduce the selected runs exactly). `all` refuses to run when the config differs from
  `tuning/adopted.json`, and an ablation arm whose config equals the adopted MorphoMix is not trained.
- **2026-10-07, after the final run (manuscript review by five reviewer agents, reports kept
  locally):** the background-only reference and the residual AUCs included the Laplacian
  variance of the whole image, which also sees the cell. `shortcut.py` now fits them on the eight background
  features only (`BACKGROUND`) and writes the nine-feature fit as `shortcut_image_stats_auc.json`. Recomputed on the
  laptop CPU from the stored features and predictions (`main.py shortcut --skip-swap`, scikit-learn 1.9.0; the Colab
  outputs are kept in `results/tables/colab_original/`): background-only ROC-AUC ALL-IDB2 0.874, Aria 0.955,
  val 0.407 (nine features: 0.915 / 0.967 / 0.408 locally, 0.920 / 0.967 / 0.414 on Colab, probably a different
  grouped fold assignment in Colab's scikit-learn version). The val reference is reported as 0.407 (below chance), no longer inverted.
  No model was retrained and no method choice follows from this change.
- **2026-10-07, pre-registered confirmatory analysis (written before L_100X_C1 was downloaded or any of its images
  or labels were seen; the author's decision):** the LeukemiaAttri subset `L_100X_C1` (same laboratory as the
  selection cohort, low-cost microscope, camera 1, 100x) is scored once as a confirmatory cohort. Nothing about the
  method, the recipe, the checkpoints, the thresholds or the inference may change after this entry.
  - **Cohort:** both splits, lymphoblast (ALL) vs lymphocyte (Normal) boxes, built by
    `src/datasets/leukemiaattri.py:build_val_cohort` with every rule unchanged (fixed window 2.18 x this subset's
    median unsqueezed box side, CLL-slide lymphocytes, edge-cut, sliver, conflicting, duplicate and
    opposite-class-neighbour boxes dropped, ALL capped at 100 per slide). `x_scale` 0.60 if the fields are 640 x 640
    like H_100X_C1, else 1.0. `MISPLACED_BOXES` and the same-cell table belong to H_100X_C1 and match nothing here.
    It likely images the same slides / patients as the selection cohort with another microscope, so it is an
    acquisition-shift replicate of the selection cohort, not a new population; this is stated wherever it is
    reported. If fewer than 50 crops per class result, that is reported and nothing else is done.
  - **Models:** every stored checkpoint of the final run (five main arms x seeds 42-46, arms b-k x seeds
    42-44) and the two frozen DinoBloom probes, each scored once with the same 8-view TTA as the selection cohort
    (crops scored directly), all on the same local device (it is not mixed with A100 predictions in any comparison).
  - **Primary analysis:** MorphoMix minus each of the six comparators (four augmentations, two probes), seed-mean
    ROC-AUC, hierarchical bootstrap (2,000 draws) over slides within class and seeds, two-sided, Holm over the six.
    **Secondary (descriptive):** AUPRC; the background-only reference (eight background features, grouped CV by
    slide); residual AUC; decision metrics at p(ALL) >= 0.5; arms b-k against arm a. Every result is reported as
    obtained, in the paper's main text, whatever its direction.
- **2026-10-07, result of the pre-registered confirmatory analysis (reported as obtained):** `L_100X_C1` was
  downloaded by the author through the browser (the automated download was blocked by Drive) and built with the
  unchanged rules (x_scale 0.60, 640 x 640 fields): 1,092 crops (951 ALL, 141 Normal), 40 slides, 39 of them also in
  the selection cohort. Every checkpoint (55) and both refitted probes were scored once on the laptop RTX 3050
  (`main.py confirm`). Primary analysis: MorphoMix 0.752 +/- 0.023 against basic 0.709, hed_jitter 0.690,
  randstainna 0.728, stain_mixup 0.757, DinoBloom-S 0.734, DinoBloom-B 0.745; no difference significant (p_holm >=
  0.59). Background-only ROC-AUC 0.397. Arms h-k (no auxiliary cells) 0.62-0.68, every arm with them 0.69-0.78.
  Tables `results/tables/confirm_l100x_*`. Nothing was changed after this result.
