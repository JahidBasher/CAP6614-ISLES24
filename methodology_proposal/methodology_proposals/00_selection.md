# Top-5 Methodology Proposals: Selection and Shared Base

*CAP6614 project · 30 September 2026 · builds on [methodology_guide.md](../methodology_guide.md)*

This folder contains five full methodology proposals, each about 1,500 words, chosen from the 26 options in the methodology guide. Each can become sections 4 and 5 of the course proposal. Evidence links such as ([Ren 2025 #3](../../literature_review/report/literature_review.md#ex-ren2025isles24-3)) open the verbatim quote in the literature review. Text marked **Our assessment** is the team's judgment. Every design detail without a citation is our own plan.

## How the five were chosen

We scored each candidate from 1 (weak) to 3 (strong) on four criteria. All scores are **Our assessment**:

- **Novelty:** goes beyond reproducing published ISLES'24 work.
- **Evidence:** support in the literature that the idea can work.
- **Feasibility:** one semester, four people, about 48 GB of GPU memory.
- **Course fit:** uses pretrained models, adaptation methods or agentic/VLM components, as the assignment asks.

| Rank | Proposal | Guide IDs | Novelty | Evidence | Feasibility | Course fit | Total |
|---|---|---|---|---|---|---|---|
| 1 | [P1 Native 4D CTP as video](P1_native_ctp_video.md) | M13 (+M4, M15) | 3 | 2 | 2 | 3 | **10** |
| 2 | [P2 Perfusion maps as video](P2_perfusion_maps_video.md) | M12 (+M15, M16) | 3 | 1 | 3 | 3 | **10** |
| 3 | [P3 Text-conditioned clinical fusion](P3_text_conditioned_clinical_fusion.md) | M21 (+M20, M23) | 2 | 2 | 3 | 3 | **10** |
| 4 | [P4 Slices as video: MedSAM2 refinement](P4_medsam2_slice_video_refinement.md) | M14 + M18 | 2 | 2 | 2 | 3 | **9** |
| 5 | [P5 Controlled preprocessing and modality study](P5_preprocessing_modality_study.md) | M5 + M6 | 1 | 3 | 3 | 1 | **8** |

Ties are broken by evidence strength and by how well the proposal answers an open question named by the ISLES'24 organisers.

- **P1 ranks first.** The organisers explicitly ask for better spatio-temporal models ([de la Rosa 2024 #14](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-14)), and single-cohort evidence favours native CTP ([Amador 2022 #1](../../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1)).
- **P2 ranks second.** It is the current proposal (`overleaf/sections/method.tex`) and the most novel option, but no paper supports treating the maps as frames.

### Why the other candidates did not make the top five

- **Spatio-temporal CNNs (M8, M9, M11)** are folded into P1 as comparison points. P1 tests the same question with a pretrained encoder, which fits the course better.
- **Frozen DINOv3 features (M17)** appear in P2 as the image-pretraining control.
- **3D transformers (M19)** trailed CNNs on ISLES'24 ([Martínez Cegarra 2026 #4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4)).
- **Health-record supervision (M22)** needs data we don't have.
- **The two reformulated tasks (M25, M26)** change the task.
- **The agentic quality-control idea (M24)** is an optional add-on to P3 or P4.

## Shared base used by every proposal

All five proposals reuse the same foundation, so the team can start on it before choosing.

**Data.** The 149 public ISLES'24 training cases ([Riedel 2026 #1](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-1)).
- Follow-up MRI is acquired 2–9 days after treatment ([Riedel 2026 #2](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-2)).
- Only patients with successful recanalisation are included ([de la Rosa 2024 #20](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-20)).
- The hidden test set is not used.

**Splits.** 5-fold cross-validation at patient level, stratified by centre and by lesion volume (under vs. at least 5 mL).
- The folds are identical for every method.
- Model selection happens on a validation split inside each training fold, never on the test fold. The winning entry picked its best fold ([Ren 2025 #4](../../literature_review/report/literature_review.md#ex-ren2025isles24-4)), which a controlled comparison must avoid.

**Baselines.**
- **B0:** the rCBF < 30% core threshold ([de la Rosa 2024 #31](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-31)), which matches deep learning on volume but not on voxel overlap ([de la Rosa 2024 #8](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-8)).
- **B1:** nnU-Net ResEnc L ([Ren 2025 #2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2)) with the winning preprocessing ([Ren 2025 #1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1)). That preprocessing is SynthStrip skull stripping on NCCT, with the mask transferred to the other scans ([Ren 2025 #8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8); [Ren 2025 #9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)).

**Metrics.** Dice, absolute volume difference, lesion-wise F1 and absolute lesion count difference ([de la Rosa 2024 #3](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-3)), computed with the official challenge code.
- Results are also reported for infarcts under 5 mL ([de la Rosa 2024 #17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17)) and for each centre ([de la Rosa 2024 #24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)).

**Statistics.** Paired Wilcoxon signed-rank tests on per-case Dice over all 149 out-of-fold predictions.
- Holm correction when one proposal makes several comparisons.
- Bootstrap 95% confidence intervals for mean Dice.

**Context.** The best hidden-test Dice of 0.285 ([de la Rosa 2024 #2](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-2)) is quoted as context only, never as a comparison.

**Team roles** (a suggestion, to adjust to each proposal):
- **Data and evaluation lead:** folds, preprocessing and the metrics pipeline.
- **Baseline lead:** nnU-Net and the threshold baseline.
- **Model lead:** the proposed model.
- **Analysis and writing lead:** ablations, statistics, figures and the report.

## Combining proposals

The proposals are not mutually exclusive.
- **P5 is the shared base** done thoroughly, so it can run alongside any other proposal.
- **P3 is a module.** Its conditioning can be added to the B1 baseline or to the P1 or P2 backbone.
- **P1 and P2 share one codebase.** They differ only in what fills the time axis, so a team that builds one can run the other as a secondary experiment.

**Our assessment:** P1 or P2 as the main study, with P3 as one ablation, is the strongest single-semester package.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **Amador 2022.** Kimberly Amador, Matthias Wilms, Anthony Winder, et al. (2022). Predicting treatment-specific lesion outcomes in acute ischemic stroke from 4D CT perfusion imaging using spatio-temporal convolutional neural networks. *Medical Image Analysis*, 82, 102610. doi:10.1016/j.media.2022.102610. <https://doi.org/10.1016/j.media.2022.102610>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1)
- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-2), [#3](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-3), [#8](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-8), [#14](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-14), [#17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17), [#20](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-20), [#24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24), [#31](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-31)
- **Martínez Cegarra 2026.** Pablo Martínez Cegarra, Juan Francisco Zapata Pérez, Juan Martínez-Alajarín (2026). Automatic Segmentation of Ischaemic Stroke Lesions Using Transformers and Convolutional Neural Networks Applied to Multimodal Neuroimaging. *Sensors*, 26(16), 5021. doi:10.3390/s26165021. <https://doi.org/10.3390/s26165021>  
  Quotes: [#4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4)
- **Ren 2025.** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. <https://arxiv.org/abs/2505.18424>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1), [#2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2), [#3](../../literature_review/report/literature_review.md#ex-ren2025isles24-3), [#4](../../literature_review/report/literature_review.md#ex-ren2025isles24-4), [#8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8), [#9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)
- **Riedel 2026.** Evamaria Olga Riedel, Ezequiel de la Rosa, The Anh Baran, et al. (2026). The Ischemic Stroke Lesion Segmentation Challenge (ISLES)’24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. *Radiology: Artificial Intelligence*, 8(3), e250603. doi:10.1148/ryai.250603. <https://doi.org/10.1148/ryai.250603>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-1), [#2](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-2)
