# P5. A Controlled Study of Preprocessing and Input Modalities for ISLES'24

*Rank 5 of 5 · guide methods M5 and M6 (main), M3 (optional) · lowest risk · shared base: [00_selection.md](00_selection.md)*

## 1. Summary

**Research question.** With the model held fixed (nnU-Net ResEnc L) and identical folds, how much do skull stripping, intensity windowing and the choice of input modalities each contribute to final infarct prediction on ISLES'24, overall and across the two centres?

**Hypotheses.**
- **H1:** custom intensity windowing contributes more to Dice than any single architectural change reported at ISLES'24.
- **H2:** learned skull stripping (SynthStrip) beats threshold-and-morphology skull stripping.
- **H3:** adding NCCT reduces the performance gap between the two centres.
- **H4:** 4D CTP information is redundant once the perfusion maps are present.

**Why it is included despite ranking fifth.** It has the lowest novelty but the strongest evidence and almost no execution risk. Its outputs (preprocessing code, folds, baselines, metrics) are exactly the shared base every other proposal needs. It is the natural fallback for P1–P4, and it can run in parallel with any of them from week 1.

## 2. Motivation and evidence

**Preprocessing drove the winning result.**
- The winning team credits its result to skull stripping and custom intensity windowing ([Ren 2025 #1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1)).
- On one fold, Dice rose from 21.8% with standard normalisation to 31.0% with custom windowing, and to 31.8% with windowing plus histogram equalisation ([Ren 2025 #3](../../literature_review/report/literature_review.md#ex-ren2025isles24-3)).
- Their windows started from clinical guidelines where available and were then tuned for performance ([Ren 2025 #10](../../literature_review/report/literature_review.md#ex-ren2025isles24-10)).

These are single-fold, single-team numbers. A controlled comparison notes that leaderboard results confound architecture with preprocessing, training data and validation strategy ([Martínez Cegarra 2026 #1](../../literature_review/report/literature_review.md#ex-cegarra2026isles-1)).

**Different teams preprocessed very differently.**
- The winner used narrow windows: CTA 0–90 HU, CBF 0–35, CBV 0–10, MTT 0–20 s, Tmax 0–7 s ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)).
- Another team clipped much wider: NCCT 0–100 HU, CTA 0–200 HU, CBF and CBV 0–400, MTT and Tmax 0–20 s ([de la Rosa 2024 #28](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-28)).

Nobody has compared these choices under one protocol.

**Modality findings point in different directions.**
- The top two teams combined all modalities except 4D CTP ([de la Rosa 2024 #7](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-7)).
- NCCT may stabilise models across centres ([de la Rosa 2024 #19](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-19)), where CTP resolution differs ([de la Rosa 2024 #24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)).
- Yet NCCT alone reached a median Dice of only 0.20 against the post-treatment infarct ([Britt 2026 #1](../../literature_review/report/literature_review.md#ex-britt2026ncct-1)), and nearly failed at this time point in another study ([Shah 2026 #2](../../literature_review/report/literature_review.md#ex-shah2026asymfex-2)).
- Baseline masks from NCCT and CTA help only when CTP is missing ([Mojtahedi 2025 #1](../../literature_review/report/literature_review.md#ex-mojtahedi2025switch-1)).

## 3. Method

### 3.1 Fixed model and training

All configurations use nnU-Net ResEnc L ([Ren 2025 #2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2)), with nnU-Net's own self-configuration ([Isensee 2021 #1](../../literature_review/report/literature_review.md#ex-isensee2021nnunet-1)), the same 5 folds, and the same training budget. Only the inputs and preprocessing change. Every configuration is a full 5-fold run, so each comparison is paired over all 149 cases.

### 3.2 Factors

**Skull stripping (3 levels).**
- None.
- Threshold-and-morphology on NCCT, as in the controlled replication ([Martínez Cegarra 2026 #10](../../literature_review/report/literature_review.md#ex-cegarra2026isles-10)).
- SynthStrip on NCCT with the mask applied to all scans ([Ren 2025 #8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8); [Ren 2025 #9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)).

**Intensity handling (4 levels).**
- nnU-Net default normalisation.
- Wide clipping ([de la Rosa 2024 #28](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-28)).
- Narrow custom windows ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)).
- Narrow windows plus histogram equalisation ([Ren 2025 #3](../../literature_review/report/literature_review.md#ex-ren2025isles24-3)).

**Input modalities (6 sets).**
- 4 maps
- CTA + 4 maps (the winning configuration)
- NCCT + CTA + 4 maps
- NCCT only
- NCCT + CTA
- 4 maps + 16 peak-centred CTP frames as extra channels

### 3.3 Design

A full factorial design (3 × 4 × 6 = 72 configurations × 5 folds) is too expensive. We use a **one-factor-at-a-time design around the winning configuration** (SynthStrip, narrow windows, CTA + 4 maps), plus targeted interactions:

1. Skull stripping: 3 runs, with the others at the winning setting.
2. Intensity: 4 runs.
3. Modalities: 6 runs.
4. Interactions:
   - windowing × skull stripping, 2 extra runs
   - NCCT inclusion × windowing, 2 extra runs

   These test whether NCCT's benefit depends on preprocessing.

That is about 15 unique configurations after removing duplicates, or 75 nnU-Net trainings.

### 3.4 Optional extension: cascade

If time allows, test the second-place cascade ([de la Rosa 2024 #5](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-5)) with and without CTP. A controlled study found cascades gave only marginal gains over a single network ([Martínez Cegarra 2026 #4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4)).

## 4. Experimental design

### 4.1 Outcomes

**Primary:** per-case Dice. **Secondary:** absolute volume difference, lesion-wise F1 and absolute lesion count difference ([de la Rosa 2024 #3](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-3)). All are reported overall, by centre and for infarcts under 5 mL.

### 4.2 Analysis

- **Effect sizes:** for each factor, the mean paired Dice difference against the winning configuration, with bootstrap 95% confidence intervals.
- **Significance:** paired Wilcoxon tests with Holm correction within each factor.
- **Centre gap (H3):** a mixed-effects model of per-case Dice with configuration, centre and their interaction as fixed effects and patient as a random effect. A significant configuration × centre interaction for NCCT supports H3.
- **Ranking stability:** the ranking of configurations is checked across folds, to see whether conclusions hold beyond one split. This is a direct response to the single-fold ablation in the winning paper.

### 4.3 Reproducibility deliverables

- a preprocessing package that applies any configuration to raw ISLES'24 data
- fixed fold files
- an evaluation script wrapping the official metrics
- a results table for all configurations

These deliverables serve the other proposals directly.

## 5. Expected outcomes and interpretation

| Outcome | Interpretation |
|---|---|
| Windowing effect large and stable across folds | Confirms the winning team's single-fold finding under a controlled design |
| Windowing effect shrinks or flips across folds | The reported 21.8 → 31.0 → 31.8% gains were partly fold-specific |
| NCCT narrows the centre gap | Supports the organisers' hypothesis that NCCT helps generalisation |
| 4D CTP channels add nothing to maps | Confirms that the maps are sufficient for CNNs ([de la Rosa 2024 #11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11)). Clarifies what P1 must overcome. |
| SynthStrip ≈ morphology | The simpler method is adequate, which is useful for practitioners |

**Success criterion.** This study succeeds whatever direction the effects take, provided the confidence intervals are tight enough to rank the factors. **Our assessment:** this is its main appeal as a fallback.

### 5.1 Threats to validity

- **Internal.** All configurations share nnU-Net's automatic configuration, which itself adapts patch size and normalisation to the input. Part of an observed effect may therefore be nnU-Net reacting to the input rather than the preprocessing alone. We record the configuration nnU-Net selects for every run and report when it changes.
- **Statistical.** With about 15 configurations, some differences will look significant by chance. We correct within each factor and treat the interaction runs as exploratory.
- **External.** Two centres and 149 cases limit generality. Centre-level conclusions rest on about 100 Munich and 50 Zurich training cases ([de la Rosa 2024 #32](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-32)). Any centre effect is a hypothesis to confirm on other data, not a finding about scanners in general.
- **Construct.** Dice on cases with tiny or absent infarcts is unstable. Infarcts under 5 mL are where all methods struggled ([de la Rosa 2024 #17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17)). We report the subgroup separately rather than letting it dominate the mean.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Limited novelty | Frame it as a controlled re-examination of challenge findings, with a centre-gap analysis no prior work reports. Pair it with P3 for a novel component. |
| Compute (about 75 nnU-Net runs) | Reduce to 3 folds for the screening stage, then 5 folds for the top configurations. Use the Newton cluster in parallel. |
| Implementation differences from the winning pipeline | Use the published windows ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)) and SynthStrip as described. Report any deviations. |
| Small per-factor effects | Paired design over 149 cases. Report effect sizes, not just p-values. |

## 7. Compute and timeline

**Compute.** **Our assessment:** a full-resolution nnU-Net ResEnc L fold takes about one GPU-day. So about 75 runs need roughly 75 GPU-days, which is feasible only in parallel on the cluster. Hence the 3-fold screening stage.

| Weeks | Milestone | Lead |
|---|---|---|
| 1–2 | Preprocessing package, folds, metrics | Data & evaluation |
| 2–3 | Winning configuration (5 folds), sanity checks | Baseline |
| 3–6 | Skull-stripping and intensity factors | Baseline + Model |
| 5–8 | Modality factors and interactions | Model |
| 8–9 | Mixed-effects analysis, ranking stability | Analysis |
| 9–10 | Report and released code | All |

## 8. What this means for the course proposal

As the main project, P5 gives:
- **§4:** the fixed nnU-Net pipeline and the factor list
- **§5:** the design, the analysis plan and the centre-gap hypothesis
- **§6 Novelty:** the first controlled, multi-fold attribution of ISLES'24 performance to preprocessing and modality choices, including a test of the NCCT generalisation hypothesis

**Our assessment:** most teams should treat P5 as the foundation and §7 fallback rather than the headline, and pair it with P1, P2 or P3 for novelty.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **Britt 2026.** Linus Britt, M. Nielsen, S. Klapproth, et al. (2026). Ischemic Stroke Segmentation and Net Water Uptake Quantification on Multicenter Non-Contrast CT Using Supervised Target-Domain Adaptation. *arXiv preprint arXiv:2609.20151*. <https://arxiv.org/abs/2609.20151>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-britt2026ncct-1)
- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#3](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-3), [#5](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-5), [#7](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-7), [#11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11), [#17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17), [#19](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-19), [#24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24), [#28](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-28), [#32](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-32)
- **Isensee 2021.** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211. doi:10.1038/s41592-020-01008-z. <https://doi.org/10.1038/s41592-020-01008-z>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-isensee2021nnunet-1)
- **Martínez Cegarra 2026.** Pablo Martínez Cegarra, Juan Francisco Zapata Pérez, Juan Martínez-Alajarín (2026). Automatic Segmentation of Ischaemic Stroke Lesions Using Transformers and Convolutional Neural Networks Applied to Multimodal Neuroimaging. *Sensors*, 26(16), 5021. doi:10.3390/s26165021. <https://doi.org/10.3390/s26165021>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-cegarra2026isles-1), [#4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4), [#10](../../literature_review/report/literature_review.md#ex-cegarra2026isles-10)
- **Mojtahedi 2025.** Mahsa Mojtahedi, Laura van Poppel, Lucas de Vries, et al. (2025). Utilizing Baseline Infarct and Penumbra Masks to Improve NCCT and CTA Based Final Infarct Prediction. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 89-94. doi:10.1007/978-3-031-81101-2_10. <https://doi.org/10.1007/978-3-031-81101-2_10>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-mojtahedi2025switch-1)
- **Ren 2025.** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. <https://arxiv.org/abs/2505.18424>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1), [#2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2), [#3](../../literature_review/report/literature_review.md#ex-ren2025isles24-3), [#7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7), [#8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8), [#9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9), [#10](../../literature_review/report/literature_review.md#ex-ren2025isles24-10)
- **Shah 2026.** M. Shah, V. Sundaresan (2026). AsymFeX: A Symmetry-Driven Framework for Ischemic Stroke Segmentation Across Imaging Modalities and Stroke Stages. *arXiv preprint arXiv:2608.19769*. <https://arxiv.org/abs/2608.19769>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-shah2026asymfex-2)
