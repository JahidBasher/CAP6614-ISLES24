# P1. Native 4D CTP as Video: a Pretrained Video Encoder for Final Infarct Prediction

*Rank 1 of 5 · guide methods M13 (main), M4 and M15 (controls) · shared base: [00_selection.md](00_selection.md)*

## 1. Summary

**Research question.** On ISLES'24, does a video encoder pretrained on natural video predict the final infarct better than nnU-Net, when both receive the same native CT perfusion (CTP) time points as input?

**Hypotheses.**
- **H1:** the video model beats an nnU-Net that receives the same time points as stacked channels, on per-case Dice.
- **H2:** randomly reordering the time points hurts the video model, which would show it uses the temporal dynamics.
- **H3:** video pretraining beats random initialisation of the same encoder.

**Why it ranks first.** The organisers of ISLES'24 call for better spatio-temporal models ([de la Rosa 2024 #14](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-14)). Yet at the challenge itself, models built to capture contrast dynamics consistently underperformed ([de la Rosa 2024 #13](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-13)). A pretrained video encoder is a different way to model the same time axis, and nobody has tested one on this data.

## 2. Motivation and evidence

**Native CTP carries information that the perfusion maps may lose.**
- The standard pipeline deconvolves the CTP signal into maps and then thresholds them ([Riedel 2026 #3](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-3)).
- Deconvolution is ill-posed given CTP's low signal-to-noise ratio ([Robben 2020 #3](../../literature_review/report/literature_review.md#ex-robben2020nativectp-3)), and an ablation found native CTP outperformed deconvolved inputs ([Robben 2020 #2](../../literature_review/report/literature_review.md#ex-robben2020nativectp-2)).
- A 3D+time network reached Dice 0.45, against 0.38 for a perfusion-map method, in 10-fold cross-validation on 147 patients ([Amador 2022 #1](../../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1)).
- Vendors' perfusion software disagree with one another ([de Vries 2023 #1](../../literature_review/report/literature_review.md#ex-devries2023perfunet-1)). Using native CTP sidesteps that.

**But ISLES'24 is a warning.**
- Models using perfusion maps consistently beat models trained on raw 4D CTP ([de la Rosa 2024 #11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11)).
- Spatial models beat ones that modelled the full spatio-temporal structure ([de la Rosa 2024 #12](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-12)).
- A CNN + Transformer submission on ISLES'24 data reached only Dice 0.20 ([Amador 2025 #2](../../literature_review/report/literature_review.md#ex-amador2025switch-2)).
- The best CTP-only entry simply stacked 20 time points as channels of an nnU-Net ([de la Rosa 2024 #22](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-22)) and finished third ([de la Rosa 2024 #6](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-6)).

**Why a pretrained video encoder could change this.**
- VideoMAE learns from very small video datasets ([Tong 2022 #1](../../literature_review/report/literature_review.md#ex-tong2022videomae-1)) and exploits redundancy across frames ([Tong 2022 #2](../../literature_review/report/literature_review.md#ex-tong2022videomae-2)). Both properties match 149 cases with smoothly changing contrast curves.
- Natural-video pretraining improved 3D models on chest CT ([Ke 2023 #1](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-1)), and Kinetics pretraining beat in-domain CT pretraining ([Ke 2023 #2](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-2)).

**Our assessment:** the earlier ISLES'24 temporal models were trained from scratch. Pretraining is the main untested ingredient.

## 3. Method

### 3.1 Input construction

The organisers motion-corrected the 4D CTP series and resampled it to one frame per second ([de la Rosa 2024 #26](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-26)). For each case we will:

1. **Skull-strip.** Apply the NCCT brain mask to the CTP series, following the winning pipeline ([Ren 2025 #8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8); [Ren 2025 #9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)).
2. **Find the contrast peak.** Average the CTP signal inside the brain mask to get a global time-attenuation curve, then take its peak time *t*.
3. **Select 16 frames, *t*−8 to *t*+7.** VideoMAE's default is a 16-frame ViT-B ([Tong 2022 #4](../../literature_review/report/literature_review.md#ex-tong2022videomae-4)). The 3rd-place team similarly chose a window of time points around peak contrast ([de la Rosa 2024 #22](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-22)).
4. **Subtract the baseline.** Subtract the mean of the pre-contrast frames, so frames show contrast enhancement rather than anatomy. Then clip and scale to [0, 1].
5. **Form clips.** For every axial slice *z*, a clip has 16 frames. Each frame's three colour channels hold slices *z*−1, *z*, *z*+1 at that time point. Frames are resized to 224 × 224.

VideoMAE embeds 2 × 16 × 16 space-time cubes ([Tong 2022 #3](../../literature_review/report/literature_review.md#ex-tong2022videomae-3)), so a 16-frame clip yields 8 temporal × 196 spatial tokens.

### 3.2 Network

- **Encoder:** ViT-B VideoMAE, initialised from the Kinetics-400 checkpoint.
- **Temporal fusion:** a learned attention-pooling layer reduces the 8 temporal tokens at each spatial position to one. A learned query attends over time, which lets the model weight early arrival, the peak or the wash-out differently in each region.
- **Decoder:** four upsampling stages from 1/16 resolution to full resolution. Skip features come from a light CNN stem applied to the peak frame at full resolution. Every ISLES'24 method struggled with small infarcts ([de la Rosa 2024 #17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17)), and a 1/16-resolution token grid would erase them.
- **Output:** a per-slice infarct probability. Slices are stacked into a volume, and connected components under 10 voxels are removed.

### 3.3 Adaptation

We will compare three settings:
- **Frozen encoder:** train only the pooling and decoder. Frozen video features can work well ([Bardes 2024 #1](../../literature_review/report/literature_review.md#ex-bardes2024vjepa-1)).
- **LoRA:** rank 8 on the query and value projections, plus the decoder.
- **Full fine-tuning:** with layer-wise learning-rate decay of 0.75.

The loss is Dice plus cross-entropy. Lesion-containing slices will be oversampled 2:1 because infarcts occupy few slices. Training uses AdamW for 100 epochs with cosine decay. Learning rate and LoRA rank are tuned only on the inner validation split.

## 4. Experimental design

### 4.1 Comparisons

| Arm | Description | Purpose |
|---|---|---|
| B0 | rCBF < 30% threshold | Clinical reference |
| B1 | nnU-Net ResEnc L on CTA + 4 maps, winning preprocessing | Strongest map-based baseline |
| **B2** | nnU-Net on the same 16 CTP frames stacked as channels (as the 3rd-place team did) | **Main comparator for H1** |
| P1-LoRA | Video model, LoRA adaptation | Main proposed arm |
| P1-frozen / P1-full | Other adaptation settings | Cost vs. accuracy |

The decisive comparison is B2 vs. P1, which share inputs and preprocessing. B1 vs. P1 answers the practical question of whether native CTP can beat the maps.

### 4.2 Ablations

- **A1, temporal shuffle (H2):** randomly permute the 16 frames at training and test time. If Dice does not drop, the model is not using dynamics.
- **A2, single frame:** only the peak frame, repeated. This tests whether time adds anything over a static snapshot.
- **A3, initialisation (H3):** Kinetics-400 weights vs. random initialisation vs. ImageNet weights inflated to video.
- **A4, number of frames:** 8 vs. 16 vs. 32. For 32, two clips are encoded and their pooled features averaged.
- **A5, late fusion:** combine the video model's output with B1's by averaging probabilities. This tests whether native CTP and the maps are complementary.

### 4.3 Evaluation

The shared base protocol applies: 5-fold cross-validation, the four challenge metrics, and results split by size and centre. CTP resolution differs between the two centres ([de la Rosa 2024 #24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)), and CTP-only methods may be more sensitive to the acquisition protocol ([de la Rosa 2024 #18](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-18)). The per-centre analysis is therefore part of the hypothesis test, not an afterthought. The statistics are paired Wilcoxon tests with Holm correction across H1–H3.

## 5. Expected outcomes and interpretation

| Result | Interpretation | What we write |
|---|---|---|
| P1 > B2, and A1 hurts | Pretrained temporal modelling helps | Main positive finding. Compare with B1. |
| P1 > B2, A1 doesn't hurt | Gain comes from pretrained features, not dynamics | Report as a pretraining effect; P2-style follow-up |
| P1 ≈ B2 | No advantage over channel stacking | A rigorous negative result that confirms the ISLES'24 finding with a pretrained model |
| P1 < B2 | Video features don't transfer to CTP | Analyse by centre and size, then fall back to P5 |

**Success criterion:** a statistically significant gain in per-case Dice over B2, with no loss in lesion-wise F1. **Our assessment:** a clean negative result still answers the organisers' open question and is a valid outcome for a course project.

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Temporal models underperform, as at ISLES'24 ([de la Rosa 2024 #13](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-13)) | Medium–high | A1 and A2 tell us why. Keep B1/B2 results. P5 as fallback. |
| Peak detection fails on noisy or motion-affected scans | Medium | Inspect the time-attenuation curves. Fall back to a fixed window after bolus arrival. |
| CTP resolution differs between centres ([de la Rosa 2024 #24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)) | Certain | Stratified folds, per-centre reporting, intensity normalisation per case |
| Small infarcts lost at 1/16 resolution | High | Full-resolution CNN skip connection. Lesion-wise F1 tracked separately. |
| GPU memory | Low | ViT-B with 1,568 tokens per clip, batch 8–16 clips with gradient checkpointing. The 3rd-place team trained with batch size 2 on two A6000 GPUs ([de la Rosa 2024 #27](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-27)), which is comparable to our budget. |
| Different temporal sampling between centres | Medium | Use the organisers' resampled series. Peak-centred windows align the bolus. |

## 7. Compute and timeline

**Compute.**
- B1 and B2 are standard nnU-Net 5-fold runs.
- P1 needs about 3 adaptation settings × 5 folds, plus about 5 ablations × 5 folds, roughly 40 video-model trainings. With LoRA and 224 × 224 frames this fits one 48 GB GPU.
- **Our assessment:** budget about 1–2 GPU-days per configuration across all folds on the Newton cluster.

**Ten-week plan.**

| Weeks | Milestone | Lead |
|---|---|---|
| 1–2 | Folds, preprocessing, CTP peak detection, metric pipeline. B0 results. | Data & evaluation |
| 2–4 | B1 and B2 nnU-Net runs | Baseline |
| 3–6 | P1 implementation, LoRA, first fold | Model |
| 6–8 | All folds, ablations A1–A5 | Model + Analysis |
| 8–9 | Statistics, stratified analysis, figures | Analysis |
| 9–10 | Report and presentation | All |

## 8. What this means for the course proposal

- **§4 Proposed method:** sections 3.1–3.3 above, plus a figure: CTP clip → VideoMAE → temporal pooling → decoder.
- **§5 Dataset and experiments:** the table in 4.1, ablations A1–A3, and the evaluation in 4.3.
- **§6 Novelty:** the first pretrained video encoder on ISLES'24 native CTP, tested directly against channel stacking.
- **§7 Fallback:** P5 and the B1/B2 comparison.

This differs from the current Novelty section, which is about the perfusion maps. If the team adopts P1, it should be updated, and P2 can become a secondary experiment that reuses the same code.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **Amador 2022.** Kimberly Amador, Matthias Wilms, Anthony Winder, et al. (2022). Predicting treatment-specific lesion outcomes in acute ischemic stroke from 4D CT perfusion imaging using spatio-temporal convolutional neural networks. *Medical Image Analysis*, 82, 102610. doi:10.1016/j.media.2022.102610. <https://doi.org/10.1016/j.media.2022.102610>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1)
- **Amador 2025.** Kimberly Amador, Anthony J. Winder, Nils D. Forkert (2025). Spatio-Temporal Deep Learning for Final Infarct Prediction Using Acute Stroke CT Perfusion Data. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 83-88. doi:10.1007/978-3-031-81101-2_9. <https://doi.org/10.1007/978-3-031-81101-2_9>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-amador2025switch-2)
- **Bardes 2024.** Adrien Bardes, Q. Garrido, Jean Ponce, et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *Transactions on Machine Learning Research (TMLR) 2024*. arXiv:2404.08471. <https://arxiv.org/abs/2404.08471>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-bardes2024vjepa-1)
- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#6](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-6), [#11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11), [#12](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-12), [#13](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-13), [#14](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-14), [#17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17), [#18](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-18), [#22](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-22), [#24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24), [#26](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-26), [#27](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-27)
- **de Vries 2023.** Lucas de Vries, Bart J. Emmer, Charles B.L.M. Majoie, et al. (2023). PerfU-Net: Baseline infarct estimation from CT perfusion source data for acute ischemic stroke. *Medical Image Analysis*, 85, 102749. doi:10.1016/j.media.2023.102749. <https://doi.org/10.1016/j.media.2023.102749>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-devries2023perfunet-1)
- **Ke 2023.** Alexander Ke, Shih-Cheng Huang, Chloe O'Connell, et al. (2023). Video Pretraining Advances 3D Deep Learning on Chest CT Tasks. *MIDL 2023 (PMLR)*. arXiv:2304.00546. <https://arxiv.org/abs/2304.00546>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-1), [#2](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-2)
- **Ren 2025.** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. <https://arxiv.org/abs/2505.18424>  
  Quotes: [#8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8), [#9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)
- **Riedel 2026.** Evamaria Olga Riedel, Ezequiel de la Rosa, The Anh Baran, et al. (2026). The Ischemic Stroke Lesion Segmentation Challenge (ISLES)’24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. *Radiology: Artificial Intelligence*, 8(3), e250603. doi:10.1148/ryai.250603. <https://doi.org/10.1148/ryai.250603>  
  Quotes: [#3](../../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-3)
- **Robben 2020.** David Robben, Anna M.M. Boers, Henk A. Marquering, et al. (2020). Prediction of final infarct volume from native CT perfusion and treatment parameters using deep learning. *Medical Image Analysis*, 59, 101589. doi:10.1016/j.media.2019.101589 · arXiv:1812.02496. <https://doi.org/10.1016/j.media.2019.101589>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-robben2020nativectp-2), [#3](../../literature_review/report/literature_review.md#ex-robben2020nativectp-3)
- **Tong 2022.** Zhan Tong, Yi-Bing Song, Jue Wang, et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS 2022*. arXiv:2203.12602. <https://arxiv.org/abs/2203.12602>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-tong2022videomae-1), [#2](../../literature_review/report/literature_review.md#ex-tong2022videomae-2), [#3](../../literature_review/report/literature_review.md#ex-tong2022videomae-3), [#4](../../literature_review/report/literature_review.md#ex-tong2022videomae-4)
