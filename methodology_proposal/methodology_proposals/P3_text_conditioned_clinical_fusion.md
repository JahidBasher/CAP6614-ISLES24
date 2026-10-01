# P3. Text-Conditioned Clinical Fusion: Can Language Embeddings Make Clinical Data Useful?

*Rank 3 of 5 · guide methods M21 (main), M20 and M23 (controls), optional M24 · shared base: [00_selection.md](00_selection.md)*

## 1. Summary

**Research question.** ISLES'24 provides clinical variables, but adding them by concatenation brought no measurable gain. Does conditioning the segmentation network on a text embedding of those variables improve final infarct prediction?

**Hypotheses.**
- **H1:** text-conditioned fusion beats the image-only baseline on per-case Dice.
- **H2:** text conditioning beats plain concatenation of the same variables.
- **H3:** any gain depends on the patient's *own* variables. Shuffling the variables between patients removes it.
- **H4:** the text encoder matters. A text embedding beats a learned embedding of the raw numbers given the same conditioning mechanism.

**Why it ranks third.**
- **It answers an open question the organisers named.** The challenge report calls the use of clinical data a research gap and lists CLIP-style fusion as untested ([de la Rosa 2024 #10](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-10)).
- **It is cheap and course-relevant.** It adds a vision-language component to a strong baseline without replacing it.
- **It is a module.** It can sit on top of nnU-Net (B1) or on the P1 or P2 backbones.

## 2. Motivation and evidence

**The data exist but are unused.**
- ISLES'24 provides demographics, patient history, admission scores such as NIHSS, and clinical outcomes such as the 3-month mRS ([de la Rosa 2024 #25](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-25)).
- The dataset paper describes data on risk factors, medication, type of referral and intervention times, up to the 3-month outcome ([Riedel 2026 #4](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-4)).

**Concatenation failed.** The one ISLES'24 team that used clinical variables saw no tangible gain ([de la Rosa 2024 #9](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-9)). The organisers judged plain concatenation insufficient for capturing image-clinical relationships ([de la Rosa 2024 #10](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-10)).

**Conditioning works elsewhere.**
- Robben et al. predicted the final infarct from native CTP together with time and treatment metadata ([Robben 2020 #1](../../literature_review/report/literature_review.md#ex-robben2020nativectp-1)).
- The CLIP-Driven Universal Model conditions segmentation on CLIP text embeddings ([Liu 2023 #1](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-1)). A small network turns each text embedding plus a global image feature into parameters for the segmentation head ([Liu 2023 #2](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-2)). This is the mechanism we adapt.
- A review of stroke outcome prediction notes a lack of suitable data for applying language models ([Samak 2025 #2](../../literature_review/report/literature_review.md#ex-samak2024review-2)). **Our assessment:** that is a reason to use a frozen, pretrained text encoder rather than train one.

**Caveats in the data.**
- The lab values have ±5% random noise added for de-identification ([de la Rosa 2024 #29](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-29)).
- Some variables are recorded *after* treatment and must be excluded (see 3.1).

## 3. Method

### 3.1 Variable selection and leakage control

Only information available **at admission, before treatment** may be used:
- **Allowed:** demographics, pre-existing risk factors and medication, admission scores such as NIHSS, and pre-imaging times if recorded. Lab values are allowed, with the noise caveat above. The exact field names will be confirmed from the dataset's clinical table.
- **Excluded:**
  - the 3-month mRS and any discharge or follow-up scores
  - the recanalisation grade (TICI)
  - procedure times measured after imaging

  These encode the outcome or the treatment result, and using them would leak the answer.

**Our assessment:** before modelling, the data lead writes a variable sheet listing each field, its time point, and whether it is allowed. The team reviews it once, and it is frozen.

### 3.2 From variables to text

Each patient's allowed variables are written into a fixed sentence template, for example (illustrative values): *"72-year-old woman, admission NIHSS 14, onset to imaging 95 minutes, history of atrial fibrillation and hypertension."*

- **Missing values:** the field is omitted, rather than imputed. A text encoder handles absent phrases naturally, unlike a fixed-length vector.
- **Numbers:** written with units, and also bucketed into words ("severe stroke" for NIHSS ≥ 16) in a second template variant. This tests whether the encoder uses numbers or words better.

### 3.3 Text encoder

A frozen pretrained text encoder produces one embedding per patient. Candidates:
- the CLIP text encoder, used by the CLIP-Driven model ([Liu 2023 #1](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-1))
- a biomedical or clinical BERT variant

Keeping the encoder frozen means only the conditioning layers are trained, which suits 149 cases.

### 3.4 Conditioning mechanisms

All arms share the B1 nnU-Net ResEnc L backbone and preprocessing. Only the fusion changes:

- **F0, image only:** B1 itself.
- **F1, concatenation (negative control):** the variables, normalised with missing-value indicators, are tiled and concatenated at the bottleneck. This mirrors the ISLES'24 approach that showed no gain ([de la Rosa 2024 #9](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-9)).
- **F2, FiLM conditioning:** the text embedding passes through a small network that predicts a scale and shift for the feature maps at each decoder stage.
- **F3, text-driven dynamic head:** following the CLIP-Driven Universal Model, the text embedding is concatenated with a pooled image feature, and a small network generates the parameters of the final segmentation convolutions ([Liu 2023 #2](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-2)).
- **F4, tabular-embedding control (H4):** the same mechanism as F2, but fed by a learned embedding of the raw variables instead of the text embedding.

For F2–F4, the conditioning layers start at zero, so training begins from the baseline's behaviour and only moves away if the clinical signal helps.

**Implementation.** A custom nnU-Net trainer that passes a per-case embedding to the network. The embeddings are precomputed once, because the text encoder is frozen.

## 4. Experimental design

### 4.1 Arms and controls

| Arm | Fusion | Tests |
|---|---|---|
| F0 | none | Baseline |
| F1 | concatenation | H2 negative control |
| F2 | FiLM with text embedding | H1 |
| F3 | text-driven dynamic head | H1, alternative mechanism |
| F4 | FiLM with tabular embedding | H4 |
| F2-shuffled | F2 with variables permuted across patients | **H3 control** |

The shuffled control is essential. With 149 cases, the extra layers alone could change results. Shuffling keeps the layers and the distribution of variables, but breaks the link to each patient.

### 4.2 Ablations

- **A1:** numeric vs. bucketed-word templates.
- **A2:** CLIP vs. clinical-domain text encoder.
- **A3:** variable groups: NIHSS only; NIHSS + timing; all allowed variables.
- **A4, optional:** the same conditioning on the P1 or P2 backbone, if one of those is run.

### 4.3 Evaluation

The shared base protocol applies. One extra analysis: the gain from F2 over F0 is plotted against admission NIHSS and lesion volume, to see whether clinical information helps most in particular subgroups. The infarct-growth phenotypes depend on collateral circulation ([de la Rosa 2024 #15](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-15)), which the imaging may not fully capture.

## 5. Expected outcomes and interpretation

| Outcome | Interpretation |
|---|---|
| F2 or F3 > F0 and > F2-shuffled; F1 ≈ F0 | Text conditioning makes clinical data useful where concatenation failed. Strongest result. |
| F2 ≈ F4, both > F0 | Conditioning helps, but the text encoder adds nothing over a learned embedding. Report as conditioning, not "VLM". |
| F2 ≈ F2-shuffled | Gains are capacity or regularisation, not clinical information. |
| All ≈ F0 | Clinical variables carry little signal beyond imaging in this cohort. Consistent with the challenge finding ([de la Rosa 2024 #9](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-9)). |

**Success criterion:** F2 or F3 significantly beats both F0 and F2-shuffled on per-case Dice. **Our assessment:** a small or null effect is plausible. The cohort includes only patients with successful recanalisation ([de la Rosa 2024 #20](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-20)), which removes the treatment-outcome variation that clinical variables explain best. This is worth stating as a limitation up front.

## 6. Optional agentic extension

A brain-CT language model, of the kind BrainGPT shows is possible ([Li 2025 #2](../../literature_review/report/literature_review.md#ex-li2025braingpt-2)), could produce a short text summary of each prediction for quality control. Reports would combine the predicted volume, the location and the patient sentence. 3D medical language models are still limited by scarce data and weak evaluation metrics ([Li 2025 #1](../../literature_review/report/literature_review.md#ex-li2025braingpt-1)). Include this only if the course rewards an agentic component, and evaluate it qualitatively.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Leakage through outcome or treatment variables | Frozen variable sheet (3.1). Reviewers check the template. |
| Missing variables | Omit them from the text. Missing-value indicators for F1 and F4. |
| Noisy lab values ([de la Rosa 2024 #29](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-29)) | A3 tests a no-labs variant |
| Small effect sizes on 149 cases | Paired tests, bootstrap confidence intervals, the shuffled control. Pre-registered success criterion. |
| nnU-Net customisation effort | Implement as a trainer subclass. The embeddings are fixed vectors per case. |

## 8. Compute and timeline

**Compute.** Each arm is a standard nnU-Net 5-fold run plus a negligible conditioning network. About 6 arms plus about 6 ablation variants come to 12 nnU-Net trainings. The text embeddings take seconds to compute.

| Weeks | Milestone | Lead |
|---|---|---|
| 1–2 | Folds, preprocessing, metrics, variable sheet | Data & evaluation |
| 2–4 | F0 baseline, conditioning trainer | Baseline + Model |
| 4–7 | F1–F4, shuffled control | Model |
| 7–8 | Ablations A1–A3 | Model + Analysis |
| 8–9 | Subgroup analysis, statistics | Analysis |
| 9–10 | Report | All |

## 9. What this means for the course proposal

As a standalone project, P3 gives:
- **§4:** the conditioning mechanisms
- **§5:** the arms table and the shuffled control
- **§6 Novelty:** the first test of text-conditioned fusion on ISLES'24, answering the organisers' open question

Combined with P1 or P2, it becomes one ablation line in §5, and it strengthens the course link through its vision-language component.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#9](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-9), [#10](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-10), [#15](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-15), [#20](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-20), [#25](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-25), [#29](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-29)
- **Li 2025.** Cheng-Yi Li, Kao-Jung Chang, Cheng-Fu Yang, et al. (2025). Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation. *Nature Communications*, 16(1), 2258. doi:10.1038/s41467-025-57426-0 · arXiv:2407.02235. <https://doi.org/10.1038/s41467-025-57426-0>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-li2025braingpt-1), [#2](../../literature_review/report/literature_review.md#ex-li2025braingpt-2)
- **Liu 2023.** Jie Liu, Yixiao Zhang, Jie-Neng Chen, et al. (2023). CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection. *ICCV 2023*, 21095-21107. doi:10.1109/ICCV51070.2023.01934 · arXiv:2301.00785. <https://doi.org/10.1109/ICCV51070.2023.01934>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-1), [#2](../../literature_review/report/literature_review.md#ex-liu2023clipdriven-2)
- **Riedel 2026.** Evamaria Olga Riedel, Ezequiel de la Rosa, The Anh Baran, et al. (2026). The Ischemic Stroke Lesion Segmentation Challenge (ISLES)’24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. *Radiology: Artificial Intelligence*, 8(3), e250603. doi:10.1148/ryai.250603. <https://doi.org/10.1148/ryai.250603>  
  Quotes: [#4](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-4)
- **Robben 2020.** David Robben, Anna M.M. Boers, Henk A. Marquering, et al. (2020). Prediction of final infarct volume from native CT perfusion and treatment parameters using deep learning. *Medical Image Analysis*, 59, 101589. doi:10.1016/j.media.2019.101589 · arXiv:1812.02496. <https://doi.org/10.1016/j.media.2019.101589>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-robben2020nativectp-1)
- **Samak 2025.** Zeynel A. Samak, Philip Clatworthy, Majid Mirmehdi (2025). Automatic prediction of stroke treatment outcomes: latest advances and perspectives. *Biomedical Engineering Letters*, 15(3), 467-488. doi:10.1007/s13534-025-00462-y · arXiv:2412.04812. <https://doi.org/10.1007/s13534-025-00462-y>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-samak2024review-2)
