# Methodology Options for Final Infarct Prediction on ISLES'24

*CAP6614 project · 30 September 2026 · derived from the [literature review](../literature_review/report/literature_review.md)*

This guide lists every methodology the literature review supports for our project. For each one it gives what the method is, what evidence exists, how well it fits our setting, and the role it could play: baseline, main method, ablation or fallback.

**How to read it.**
- Links such as [de la Rosa 2024 #13](../literature_review/report/literature_review.md#ex-delarosa2024isles24-13) open the verbatim quote in the literature review's evidence appendix. Every one of these links was checked against `data/excerpts.json`.
- Text marked **Our assessment** is the team's judgment, not a finding from a paper. This includes all effort and risk ratings.
- **Effort:** S is about one week for one person, M is two to three weeks, L is more than three weeks.
- **Risk:** the chance that the method fails to beat the baseline or fails to run within our compute (one 48 GB GPU and the Newton cluster).

---

## 0. All methods at a glance

| ID | Method | Family | Tested on ISLES'24? | Effort | Risk | Suggested role |
|---|---|---|---|---|---|---|
| M1 | rCBF < 30% threshold | Clinical baseline | Yes (challenge baseline) | S | none | Baseline |
| M2 | nnU-Net ResEnc L + winning preprocessing | CNN | Yes (1st place) | S–M | low | **Main baseline** |
| M3 | Cascaded nnU-Net with baseline masks | CNN | Yes (2nd place) | M | low | Optional baseline |
| M4 | nnU-Net on 4D CTP time points as channels | CNN | Yes (3rd place) | S–M | low | Baseline for any 4D CTP method |
| M5 | Preprocessing study (skull strip, windowing) | Data-centric | Yes | S | low | Ablation / fallback |
| M6 | Modality selection (NCCT, CTA, maps, CTP) | Data-centric | Yes | S–M | low | Ablation / fallback |
| M7 | Hemispheric symmetry priors | Architecture prior | Partly (different labels) | M | medium | Optional add-on |
| M8 | U-Net + temporal convolutions on 4D CTP | Spatio-temporal | No (other cohort) | M–L | high | Alternative main |
| M9 | CNN encoder + Transformer over time | Spatio-temporal | Yes (Dice 0.20) | M–L | high | Comparison only |
| M10 | Deconvolution-free prediction with metadata | Spatio-temporal | No (MR CLEAN) | M | medium | Ablation idea |
| M11 | Spatio-temporal encoder, static decoder | Spatio-temporal | No (ISLES 2018) | M–L | high | Alternative main |
| M12 | **Video encoder, perfusion maps as frames** | Video FM | No | M | high | **Main (proposal V1)** |
| M13 | Video encoder, CTP time points as frames | Video FM | No | M–L | high | Main (V2) / fallback |
| M14 | Video encoder, axial slices as frames | Video FM | No | M | medium | Alternative main |
| M15 | Frozen video features + light decoder | Video FM | No | S–M | medium | Ablation / low-compute route |
| M16 | Factorised space-time attention variants | Video FM | No | M | medium | Ablation |
| M17 | Frozen 2D image foundation features (DINOv3) | Image FM | Partly (slice protocol) | S–M | medium | Comparison |
| M18 | Promptable SAM-family fine-tuning | Segmentation FM | No | M | high | Not recommended as main |
| M19 | 3D transformers (Swin UNETR, SegFormer3D) | Transformer | Yes (weaker) | S–M | high | Comparison only |
| M20 | Concatenated clinical variables | Fusion | Yes (no gain) | S | high | Negative control |
| M21 | Text-conditioned clinical fusion (CLIP-style) | VLM / fusion | No | M | medium | Ablation (open question) |
| M22 | Health-record supervision (Merlin-style) | VLM / fusion | No | L | high | Out of scope |
| M23 | Time and treatment metadata conditioning | Fusion | No (MR CLEAN) | S–M | medium | Ablation |
| M24 | Brain-CT language model for quality control | VLM / agentic | No | M | low (does not affect Dice) | Optional agentic add-on |
| M25 | Reperfusion-specific models | Task reformulation | Not applicable | — | — | Not possible on ISLES'24 |
| M26 | Re-annotated pre-treatment targets | Task reformulation | Changes the task | L | — | Not recommended |
| E1–E6 | Evaluation methodology | Cross-cutting | — | S | — | **Required for all** |

---

## 1. Evaluation methodology, required for every option

These choices decide whether any comparison is credible, so they come before the methods.

**E1. Use one fixed protocol for everything.** Leaderboard comparisons confound architecture with preprocessing, training data and validation strategy [Martínez Cegarra 2026 #1](../literature_review/report/literature_review.md#ex-cegarra2026isles-1). Many claimed gains in 3D segmentation also disappear under proper baselines and validation [Isensee 2024 #2](../literature_review/report/literature_review.md#ex-isensee2024revisited-2). *Our assessment:* use 5-fold cross-validation on the 149 public cases [Riedel 2026 #1](../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-1), with identical folds for every method.

**E2. Select models on validation data, never on the test fold.** The winning entry submitted the best of its 10 folds [Ren 2025 #4](../literature_review/report/literature_review.md#ex-ren2025isles24-4). That is fine for a challenge entry, but it inflates a controlled comparison.

**E3. Split by patient, never by slice.** The DINOv3 study split labelled 2D slices at random [Zhang 2025 #1](../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-1) and reported a Dice of 0.5678 [Zhang 2025 #2](../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-2). That number cannot be compared with a patient-level result.

**E4. Report all four challenge metrics** [de la Rosa 2024 #3](../literature_review/report/literature_review.md#ex-delarosa2024isles24-3). rCBF thresholds already match deep learning on volume, and the networks win only voxel by voxel [de la Rosa 2024 #8](../literature_review/report/literature_review.md#ex-delarosa2024isles24-8).

**E5. Stratify the results.**
- **By infarct size, at 5 mL:** every challenge method showed poor volume agreement on small infarcts [de la Rosa 2024 #17](../literature_review/report/literature_review.md#ex-delarosa2024isles24-17).
- **By centre:** CTP resolution differs between the two sites [de la Rosa 2024 #24](../literature_review/report/literature_review.md#ex-delarosa2024isles24-24), and CTP-only methods may be more sensitive to the acquisition protocol [de la Rosa 2024 #18](../literature_review/report/literature_review.md#ex-delarosa2024isles24-18).

**E6. Treat the leaderboard as context only.** The best hidden-test Dice was 0.285 ± 0.213 [de la Rosa 2024 #2](../literature_review/report/literature_review.md#ex-delarosa2024isles24-2). Even a careful replication evaluated on only 15 held-out patients [Martínez Cegarra 2026 #5](../literature_review/report/literature_review.md#ex-cegarra2026isles-5), so small test sets give noisy estimates. Use a paired test, such as Wilcoxon on per-case Dice, for our own comparisons.

---

## 2. Baselines

### M1. Clinical threshold (rCBF < 30%)
- **What:** label voxels with relative CBF below 30% as infarct. This is the clinical standard of deconvolution followed by thresholds [Riedel 2026 #3](../literature_review/report/literature_review.md#ex-riedel2026isles24dataset-3).
- **Evidence:** in ISLES 2018 it reached a Dice of 0.3, against 0.51 for the best learned method [Hakim 2021 #2](../literature_review/report/literature_review.md#ex-hakim2021isles18-2). At ISLES'24 it matched deep learning on volume but not on voxel overlap [de la Rosa 2024 #8](../literature_review/report/literature_review.md#ex-delarosa2024isles24-8). In another cohort it scored 15.73% Dice against 35.36% for nnU-Net in recanalized patients [Yang 2026 #1](../literature_review/report/literature_review.md#ex-ajnr2026recanalization-1).
- **Our assessment:** costs almost nothing. It shows whether learning adds anything beyond the clinical rule. **Include.**

### M2. nnU-Net ResEnc L with the winning preprocessing
- **What:** the self-configuring nnU-Net [Isensee 2021 #1](../literature_review/report/literature_review.md#ex-isensee2021nnunet-1) in its large residual-encoder preset [Ren 2025 #2](../literature_review/report/literature_review.md#ex-ren2025isles24-2). Inputs are CTA plus the four maps [Ren 2025 #7](../literature_review/report/literature_review.md#ex-ren2025isles24-7). Preprocessing is skull stripping plus a custom window for each modality [Ren 2025 #1](../literature_review/report/literature_review.md#ex-ren2025isles24-1).
- **Evidence:** won ISLES'24 [de la Rosa 2024 #2](../literature_review/report/literature_review.md#ex-delarosa2024isles24-2). All three top teams used nnU-Net [de la Rosa 2024 #4](../literature_review/report/literature_review.md#ex-delarosa2024isles24-4). Its authors find that CNN U-Nets inside nnU-Net remain the recipe for state-of-the-art results [Isensee 2024 #1](../literature_review/report/literature_review.md#ex-isensee2024revisited-1).
- **Our assessment:** this is the baseline every proposed method must beat. Run it twice: on the same inputs as our model, and with CTA added. **Required.**

### M3. Cascaded nnU-Net with baseline masks (AMC-Axolotls)
- **What:** a first stage predicts baseline infarct and penumbra masks from NCCT and CTA. These masks join the perfusion maps as input to a second stage [de la Rosa 2024 #21](../literature_review/report/literature_review.md#ex-delarosa2024isles24-21) [de la Rosa 2024 #5](../literature_review/report/literature_review.md#ex-delarosa2024isles24-5).
- **Evidence:** finished 2nd [Ren 2025 #5](../literature_review/report/literature_review.md#ex-ren2025isles24-5). The masks raise Dice by 16% without CTP but add nothing once CTP maps are present [Mojtahedi 2025 #1](../literature_review/report/literature_review.md#ex-mojtahedi2025switch-1). In a controlled replication, the cascade scored highest at Dice 0.224 [Martínez Cegarra 2026 #2](../literature_review/report/literature_review.md#ex-cegarra2026isles-2), but its gains over a single network were marginal [Martínez Cegarra 2026 #4](../literature_review/report/literature_review.md#ex-cegarra2026isles-4).
- **Our assessment:** useful only if we study the case where CTP is missing. **Optional.**

### M4. nnU-Net on 4D CTP time points as channels (Ninjas)
- **What:** 20 time points around peak contrast become 20 input channels [de la Rosa 2024 #22](../literature_review/report/literature_review.md#ex-delarosa2024isles24-22) of an nnU-Net [de la Rosa 2024 #4](../literature_review/report/literature_review.md#ex-delarosa2024isles24-4).
- **Evidence:** finished 3rd using CTP only [de la Rosa 2024 #6](../literature_review/report/literature_review.md#ex-delarosa2024isles24-6), with a Dice of 25.46% [Ren 2025 #5](../literature_review/report/literature_review.md#ex-ren2025isles24-5).
- **Our assessment:** this is the correct baseline for any method that treats CTP time as a sequence (M8, M9, M11, M13). It isolates "sequence modelling" from "using CTP at all". **Required if we pursue M13.**

---

## 3. Data-centric and input methods

### M5. Preprocessing study
- **What:** compare standard nnU-Net normalisation, custom windowing, and windowing plus histogram equalisation.
- **Evidence:** on one fold, Dice rose from 21.8% to 31.0% to 31.8% across those three [Ren 2025 #3](../literature_review/report/literature_review.md#ex-ren2025isles24-3). Leaderboard gaps can't be attributed to architecture alone, because preprocessing also differs between entries [Martínez Cegarra 2026 #1](../literature_review/report/literature_review.md#ex-cegarra2026isles-1).
- **Our assessment:** cheap and almost guaranteed to produce a result. **A good fallback contribution.**

### M6. Modality selection
- **What:** train the same model on different input sets: NCCT, CTA, the perfusion maps, or 4D CTP.
- **Evidence:**
  - The top two teams combined all modalities except 4D CTP [de la Rosa 2024 #7](../literature_review/report/literature_review.md#ex-delarosa2024isles24-7).
  - NCCT may stabilise the model across centres [de la Rosa 2024 #19](../literature_review/report/literature_review.md#ex-delarosa2024isles24-19).
  - On its own, NCCT reached a median Dice of only 0.20 against the post-treatment infarct [Britt 2026 #1](../literature_review/report/literature_review.md#ex-britt2026ncct-1), and nearly failed at this time point in another study [Shah 2026 #2](../literature_review/report/literature_review.md#ex-shah2026asymfex-2).
- **Our assessment:** a natural ablation for any main method. **Include as A5-style ablations.**

### M7. Hemispheric symmetry priors
- **What:** compare each voxel with its mirror-image counterpart across the brain's midline. AsymFeX does this inside nnU-Net [Shah 2026 #3](../literature_review/report/literature_review.md#ex-shah2026asymfex-3), and PerfU-Net is symmetry-aware [de Vries 2023 #2](../literature_review/report/literature_review.md#ex-devries2023perfunet-2).
- **Evidence:** AsymFeX was evaluated on re-drawn *pre-treatment* labels, not the challenge target [Shah 2026 #1](../literature_review/report/literature_review.md#ex-shah2026asymfex-1).
- **Our assessment:** a plausible add-on for any backbone, but there's no evidence for the final-infarct target. **Optional.**

---

## 4. Spatio-temporal models of native 4D CTP

The evidence conflicts:
- **For:** single-cohort studies favour native 4D CTP. Robben found native CTP better than deconvolved maps [Robben 2020 #2](../literature_review/report/literature_review.md#ex-robben2020nativectp-2), and Amador reached Dice 0.45 against 0.38 for perfusion maps [Amador 2022 #1](../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1).
- **Against:** at ISLES'24 the perfusion maps beat raw CTP [de la Rosa 2024 #11](../literature_review/report/literature_review.md#ex-delarosa2024isles24-11), and spatial models beat spatio-temporal ones [de la Rosa 2024 #12](../literature_review/report/literature_review.md#ex-delarosa2024isles24-12). Explicitly temporal models consistently underperformed [de la Rosa 2024 #13](../literature_review/report/literature_review.md#ex-delarosa2024isles24-13).

### M8. U-Net + temporal convolutional networks
- **What:** a U-Net-like spatial network merged with temporal convolutions over the time series [Amador 2022 #2](../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-2).
- **Evidence:** 3D+time Dice 0.45, against 0.38 for perfusion maps, in 10-fold cross-validation on 147 patients from a different cohort [Amador 2022 #1](../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1).
- **Our assessment:** the strongest evidence for temporal modelling, but not from ISLES'24. **Alternative main method.**

### M9. CNN encoder + Transformer over time
- **What:** a shared CNN encoder applied to each time point, a Transformer across time, and a CNN decoder [Amador 2025 #1](../literature_review/report/literature_review.md#ex-amador2025switch-1).
- **Evidence:** on ISLES'24, Dice 0.20 and lesion-wise F1 0.02 [Amador 2025 #2](../literature_review/report/literature_review.md#ex-amador2025switch-2).
- **Our assessment:** a documented weak result on our data. Cite it as a comparison; don't reimplement it.

### M10. Deconvolution-free prediction with metadata
- **What:** predict directly from native CTP plus time and treatment metadata [Robben 2020 #1](../literature_review/report/literature_review.md#ex-robben2020nativectp-1). The motivation is that deconvolution is ill-posed at CTP's low signal-to-noise ratio [Robben 2020 #3](../literature_review/report/literature_review.md#ex-robben2020nativectp-3).
- **Evidence:** in an ablation, native CTP beat deconvolved inputs [Robben 2020 #2](../literature_review/report/literature_review.md#ex-robben2020nativectp-2), on 188 subjects [Robben 2020 #4](../literature_review/report/literature_review.md#ex-robben2020nativectp-4).
- **Our assessment:** the metadata part is reusable (see M23). **Ablation idea.**

### M11. Spatio-temporal encoder, static decoder (PerfU-Net)
- **What:** encode the perfusion time series, decode a static mask [de Vries 2023 #2](../literature_review/report/literature_review.md#ex-devries2023perfunet-2). Motivated by differences between vendors' perfusion software [de Vries 2023 #1](../literature_review/report/literature_review.md#ex-devries2023perfunet-1).
- **Evidence:** Dice 0.46 for infarct *core*, close to methods using vendor maps [de Vries 2023 #3](../literature_review/report/literature_review.md#ex-devries2023perfunet-3).
- **Our assessment:** architecturally the closest to a video encoder plus segmentation decoder. **Alternative main method.**

---

## 5. Video foundation models (our novelty)

Shared evidence:
- **VideoMAE:** learns from very small video datasets [Tong 2022 #1](../literature_review/report/literature_review.md#ex-tong2022videomae-1) and exploits redundancy across frames [Tong 2022 #2](../literature_review/report/literature_review.md#ex-tong2022videomae-2).
- **Video pretraining on CT:** improved 3D models on chest CT [Ke 2023 #1](../literature_review/report/literature_review.md#ex-ke2023videopretraining-1), and Kinetics pretraining beat in-domain CT pretraining [Ke 2023 #2](../literature_review/report/literature_review.md#ex-ke2023videopretraining-2).

Nobody has yet applied a pretrained video encoder to ISLES'24.

### M12. Perfusion maps as frames (current proposal, version 1)
- **What:** CBF, CBV, MTT and Tmax become the four frames of a clip, fused by attention pooling and decoded to a mask. See `overleaf/sections/method.tex`.
- **Evidence:** none direct. The four maps have no natural time order. Medical SAM 2 handles frames without a temporal order by using a memory that ignores order [Zhu 2024 #2](../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-2).
- **Our assessment:** the most novel option, and the least supported. The map-order shuffle and maps-as-channels ablations are essential to show whether any gain comes from sequence modelling or only from pretraining. **Main method (V1).**

### M13. CTP time points as frames (version 2)
- **What:** about 16 CTP time points around peak contrast form the clip. Each frame's colour channels hold three neighbouring slices.
- **Evidence:** this is a real time axis, matching the native-CTP literature [Amador 2022 #1](../literature_review/report/literature_review.md#ex-amador2022spatiotemporal-1). It must also overcome the ISLES'24 finding that temporal models underperform [de la Rosa 2024 #13](../literature_review/report/literature_review.md#ex-delarosa2024isles24-13), and the challenge organisers explicitly call for better spatio-temporal models [de la Rosa 2024 #14](../literature_review/report/literature_review.md#ex-delarosa2024isles24-14).
- **Our assessment:** better grounded than M12, but higher risk on this dataset. Requires the M4 baseline. **Main method (V2), or Raiyaan's fallback.**

### M14. Axial slices as frames
- **What:** treat the volume's slices as a video, as Medical SAM 2 does [Zhu 2024 #1](../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-1). MedSAM2 fine-tuned SAM 2 on more than 455,000 3D image-mask pairs [Ma 2025 #1](../literature_review/report/literature_review.md#ex-ma2025medsam2-1).
- **Evidence:** established for 3D medical segmentation in general, but not for perfusion inputs.
- **Our assessment:** the perfusion maps would go in the channels, which pretrained RGB encoders limit to three. **Alternative.**

### M15. Frozen video features + light decoder
- **What:** keep a pretrained video encoder frozen and train only a decoder. V-JEPA features work well with a frozen backbone [Bardes 2024 #1](../literature_review/report/literature_review.md#ex-bardes2024vjepa-1).
- **Our assessment:** the cheapest way to test whether video features carry any signal. Already one arm of the adaptation ablation in `method.tex`.

### M16. Space-time attention variants
- **What:** Video Swin's local space-time windows [Liu 2022 #1](../literature_review/report/literature_review.md#ex-liu2022videoswin-1), or TimeSformer's divided attention, which applies time and space attention separately [Bertasius 2021 #1](../literature_review/report/literature_review.md#ex-bertasius2021timesformer-1).
- **Our assessment:** encoder alternatives for M12 or M13 if VideoMAE underperforms. **Ablation.**

---

## 6. Other foundation models and transformers

### M17. Frozen 2D image foundation features (DINOv3)
- **What:** DINOv3, pretrained on more than 1.6 billion images [Zhang 2025 #3](../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-3), used frozen with a decoder [Zhang 2025 #4](../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-4).
- **Evidence:** its ISLES'24 number comes from a slice-level random split [Zhang 2025 #1](../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-1). See E3.
- **Our assessment:** a useful *image* (not video) pretraining comparison for M12. Re-evaluate it at patient level. **Comparison.**

### M18. Promptable SAM-family models
- **What:** MedSAM [Ma 2024 #2](../literature_review/report/literature_review.md#ex-ma2024medsam-2) and SAM-Med3D [Wang 2023 #1](../literature_review/report/literature_review.md#ex-wang2023sammed3d-1) segment from point or box prompts.
- **Evidence:** SAM struggles with weak boundaries and low contrast [Ma 2024 #1](../literature_review/report/literature_review.md#ex-ma2024medsam-1), which describes perfusion CT. Models transferred from natural images often work slice by slice [Zhao 2023 #1](../literature_review/report/literature_review.md#ex-zhao2023sat-1).
- **Our assessment:** ISLES'24 needs fully automatic prediction, so prompts would have to come from another model, such as M2's output. **Not recommended as main.**

### M19. 3D transformers
- **What:** Swin UNETR [Hatamizadeh 2022 #1](../literature_review/report/literature_review.md#ex-hatamizadeh2022swinunetr-1) and SegFormer3D [Perera 2024 #1](../literature_review/report/literature_review.md#ex-perera2024segformer3d-1).
- **Evidence:**
  - SegFormer3D scored lowest in the controlled ISLES'24 study (Dice 0.163) [Martínez Cegarra 2026 #3](../literature_review/report/literature_review.md#ex-cegarra2026isles-3), where convolutional inductive bias kept the advantage [Martínez Cegarra 2026 #4](../literature_review/report/literature_review.md#ex-cegarra2026isles-4).
  - The only pure-transformer challenge entry scored 0.068 [Martínez Cegarra 2026 #7](../literature_review/report/literature_review.md#ex-cegarra2026isles-7).
- **Our assessment:** a cautionary comparison, not a main method. **Comparison only.**

---

## 7. Clinical data fusion and vision-language models

### M20. Concatenated clinical variables
- **Evidence:** the one ISLES'24 team that used clinical variables saw no tangible gain [de la Rosa 2024 #9](../literature_review/report/literature_review.md#ex-delarosa2024isles24-9). The organisers judged concatenation insufficient [de la Rosa 2024 #10](../literature_review/report/literature_review.md#ex-delarosa2024isles24-10).
- **Our assessment:** include only as the negative control for M21.

### M21. Text-conditioned clinical fusion (CLIP-style)
- **What:** write each patient's clinical variables as a sentence, embed it with a frozen text encoder, and condition the decoder on it. The CLIP-Driven Universal Model shows text embeddings conditioning segmentation [Liu 2023 #1](../literature_review/report/literature_review.md#ex-liu2023clipdriven-1).
- **Evidence:** the challenge report names CLIP-style fusion as an open question [de la Rosa 2024 #10](../literature_review/report/literature_review.md#ex-delarosa2024isles24-10).
- **Our assessment:** a clean ablation against M20, and a direct link to the VLM part of the course. **Ablation.**

### M22. Health-record supervision (Merlin-style)
- **What:** Merlin uses health records and radiology reports as supervision for 3D CT [Blankemeier 2024 #1](../literature_review/report/literature_review.md#ex-blankemeier2024merlin-1).
- **Our assessment:** ISLES'24 has no reports, and its tabular data is small. **Out of scope.**

### M23. Time and treatment metadata conditioning
- **What:** feed timing and treatment variables alongside the images, as Robben et al. did [Robben 2020 #1](../literature_review/report/literature_review.md#ex-robben2020nativectp-1).
- **Our assessment:** overlaps with M21. A cheaper non-text version of the same idea. **Ablation.**

### M24. Brain-CT language model for quality control (agentic)
- **What:** BrainGPT generates reports from 3D brain CT, trained on 18,885 scans paired with reports [Li 2025 #2](../literature_review/report/literature_review.md#ex-li2025braingpt-2). A model like it could describe or flag our predictions.
- **Evidence:** 3D medical language models remain limited by scarce data and weak metrics [Li 2025 #1](../literature_review/report/literature_review.md#ex-li2025braingpt-1). Evaluations of 3D VLMs are limited in general [Blankemeier 2024 #2](../literature_review/report/literature_review.md#ex-blankemeier2024merlin-2).
- **Our assessment:** doesn't change the segmentation metrics. Add it only if the course rewards an agentic component.

---

## 8. Task reformulations (not recommended for our protocol)

### M25. Reperfusion-specific models
- **What:** train separate models for patients with and without recanalization [Yang 2026 #2](../literature_review/report/literature_review.md#ex-ajnr2026recanalization-2).
- **Why not:** ISLES'24 includes only successfully recanalized patients [de la Rosa 2024 #20](../literature_review/report/literature_review.md#ex-delarosa2024isles24-20), so there is no "no recanalization" group to model.

### M26. Re-annotated pre-treatment targets
- **What:** have an expert redraw core and penumbra before treatment [Shah 2026 #1](../literature_review/report/literature_review.md#ex-shah2026asymfex-1).
- **Why not:** this changes the task away from final infarct prediction and needs expert annotators.

---

## 9. Putting it together: three study designs

**Design A: proposal version 1 (current `method.tex`)**
- **Main:** M12, perfusion maps as frames.
- **Baselines:** M1 and M2.
- **Ablations:** maps as channels, map order shuffled, M15 (frozen vs. LoRA vs. full fine-tuning), Kinetics vs. random initialisation, M6 (adding NCCT and CTA).
- **Evaluation:** E1–E6.
- **Fallback:** M5 and M6 as a data-centric study, or switch to Design B.

**Design B: native CTP as video (version 2)**
- **Main:** M13, CTP time points as frames.
- **Baselines:** M1, M2 and M4.
- **Ablations:** number of time points, M15, M16, M12 as a secondary experiment, M21 against M20.
- **Evaluation:** E1–E6.
- **Why:** best grounded in the native-CTP literature, and the ISLES'24 organisers ask for exactly this kind of model [de la Rosa 2024 #14](../literature_review/report/literature_review.md#ex-delarosa2024isles24-14).

**Design C: low-risk study**
- **Main:** M2 reproduced under E1–E6, plus M5 and M6 as a systematic study of preprocessing and modalities.
- **Add-on:** M21 for the open question about clinical fusion.
- **Why:** almost certain to produce clean, publishable comparisons, but weaker on novelty.

**Our assessment:** Design A is what the proposal currently describes. Design B is the stronger research bet given the literature. Design C is the fallback. Since all three share M1, M2 and E1–E6, the team can start on that common core immediately.

---

## 10. What the literature cannot tell us yet

- Whether pretrained **video** encoders transfer to perfusion CT at all. The closest evidence is detection tasks on chest CT [Ke 2023 #1](../literature_review/report/literature_review.md#ex-ke2023videopretraining-1).
- Whether the **order** of perfusion maps carries information (M12).
- Whether better clinical fusion beats concatenation (M21). The organisers leave this open [de la Rosa 2024 #10](../literature_review/report/literature_review.md#ex-delarosa2024isles24-10).
- How any method performs on the hidden test set, which is no longer open for routine submissions. All our numbers will be cross-validation estimates on 149 cases.
