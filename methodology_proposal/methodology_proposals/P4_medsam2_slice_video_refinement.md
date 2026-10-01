# P4. Slices as Video: Refining nnU-Net Predictions with a Medical Video Foundation Model

*Rank 4 of 5 · guide methods M14 and M18 combined · shared base: [00_selection.md](00_selection.md)*

## 1. Summary

**Research question.** Can a medical segmentation foundation model, which treats a 3D scan's slices as video frames, refine nnU-Net's final-infarct predictions on ISLES'24? We focus on small and scattered lesions.

**Hypotheses.**
- **H1:** prompting MedSAM2 with nnU-Net's predicted masks and propagating them through the volume improves lesion-wise F1 over nnU-Net alone.
- **H2:** fine-tuning MedSAM2 on ISLES'24 beats using it zero-shot.
- **H3:** mask prompts beat box or point prompts derived from the same nnU-Net output.

**Why it ranks fourth.**
- **Course fit:** it uses the most mature medical *video* foundation model available and tests a realistic adaptation recipe.
- **Precedent:** treating slices as frames is established for 3D medical images.
- **Two drawbacks:**
  - The model needs prompts, so it has to be paired with nnU-Net rather than replace it.
  - Its natural-image ancestor struggles with the low-contrast targets typical of perfusion CT.

## 2. Motivation and evidence

**Slices as frames is proven for 3D medical segmentation.**
- Medical SAM 2 recasts 2D and 3D medical segmentation as video object tracking ([Zhu 2024 #1](../../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-1)).
- MedSAM2 fine-tuned SAM 2 on more than 455,000 3D image-mask pairs ([Ma 2025 #1](../../literature_review/report/literature_review.md#ex-ma2025medsam2-1)). Its user study included 5,000 annotated CT lesions ([Ma 2025 #3](../../literature_review/report/literature_review.md#ex-ma2025medsam2-3)).
- MedSAM2 keeps SAM 2's architecture: image encoder, memory attention, prompt encoder and mask decoder ([Ma 2025 #2](../../literature_review/report/literature_review.md#ex-ma2025medsam2-2)). Prompts can be clicks, boxes or masks, and a streaming memory carries information between frames ([Ravi 2025 #3](../../literature_review/report/literature_review.md#ex-ravi2024sam2-3)).

**The challenge.**
- SAM struggles with weak boundaries and low contrast ([Ma 2024 #1](../../literature_review/report/literature_review.md#ex-ma2024medsam-1)), and needs points or boxes to say what to segment ([Ma 2024 #2](../../literature_review/report/literature_review.md#ex-ma2024medsam-2)).
- ISLES'24 requires fully automatic prediction. Its targets are subtle, because acute CT is less sensitive than DWI and suffers from artifacts and low signal-to-noise ([de la Rosa 2024 #30](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-30)).
- Every challenge method struggled with infarcts under 5 mL ([de la Rosa 2024 #17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17)).

**The idea.** Let nnU-Net, the strongest automatic model ([de la Rosa 2024 #4](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-4)), find candidate lesions. Then let the foundation model refine their shape and track them across slices using its memory. **Our assessment:** the refinement step is where a foundation model's pretrained shape priors could help most, especially for lesion-wise F1 and the lesion count difference. This is plausible but untested.

## 3. Method

### 3.1 Stage 1: nnU-Net candidates

B1, nnU-Net ResEnc L with the winning preprocessing ([Ren 2025 #1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1)), is trained within each fold. Its out-of-fold probability maps are thresholded at 0.5, and each connected component is a **candidate lesion**. A lower threshold of 0.3 gives a high-recall candidate set as an ablation, so that MedSAM2 can also recover lesions nnU-Net nearly missed.

### 3.2 Stage 2: MedSAM2 propagation

**Image formation.** MedSAM2 expects 3-channel 2D frames. Each axial slice is turned into an RGB image whose channels are three windowed perfusion maps: Tmax, CBF and CBV, using the winning windows ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)). The volume's slices form the "video" in cranio-caudal order.

**Prompting.** For each candidate lesion:
- find the slice where it is largest
- give its nnU-Net mask as a **mask prompt** on that slice
- let MedSAM2 propagate the mask up and down through the volume with its memory

The union of propagated masks over all candidates is the refined prediction. Cases with no candidate stay empty, so MedSAM2 never adds lesions where nnU-Net found none. This is a deliberate precision safeguard.

**Fusion.** Optionally, the final prediction is the voxel-wise average of nnU-Net's probability and MedSAM2's, re-thresholded. The averaging weight is tuned on the inner validation split.

### 3.3 Adaptation

- **Zero-shot:** the released MedSAM2 weights.
- **Fine-tuned:** LoRA on the image encoder's attention layers, with the memory attention and mask decoder fully trained.
  - Training data are ISLES'24 training-fold volumes, with prompts simulated from the ground-truth lesions by eroding, dilating or shifting them. Simulating imperfect prompts teaches the model to correct nnU-Net-like errors.
  - Loss: Dice plus cross-entropy per frame.

### 3.4 Implementation details

- **Resolution and orientation.** Volumes are kept in the organisers' NCCT space. Each axial slice is resized to MedSAM2's input size with the brain cropped to its bounding box first, so that small lesions occupy more pixels. Predictions are resized back and pasted into the original grid.
- **Several lesions per case.** Each candidate is propagated separately, then the masks are merged by union. Where two propagated masks overlap, the voxel keeps the higher MedSAM2 score. This keeps scattered lesions apart, which matters for the lesion count difference.
- **Propagation direction.** From the prompt slice, propagation runs upward and downward independently, and stops after two consecutive empty predictions.
- **Post-processing.** Refined voxels outside the skull-stripped brain mask ([Ren 2025 #9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)) are removed. Components smaller than the smallest lesion seen in the training folds are dropped.

### 3.5 Cases without infarct and false positives

Some ISLES'24 cases may have very small or no final infarct. Lesion-wise metrics punish false positives and false negatives alike, so the refinement must not invent lesions.

The design guarantees this in two ways:
- MedSAM2 is only ever prompted from an nnU-Net candidate.
- Propagation is confined to the brain mask.

We will report separately how many cases nnU-Net leaves empty and how many false-positive candidates the refinement removes or keeps. If refinement mostly enlarges false positives, the fusion arm with a tuned weight is the fallback.

## 4. Experimental design

### 4.1 Comparisons

| Arm | Description | Purpose |
|---|---|---|
| B0 | rCBF < 30% | Clinical reference |
| B1 | nnU-Net ResEnc L | Baseline and candidate source |
| R-zs | B1 + zero-shot MedSAM2 refinement | Does an off-the-shelf foundation model help? |
| **R-ft** | B1 + fine-tuned MedSAM2 refinement | **Main arm (H1, H2)** |
| R-ft + fusion | R-ft averaged with B1 probabilities | Upper bound of a simple ensemble |
| B1-ens | Two nnU-Net models averaged | **Control:** is any gain just from ensembling? |

The B1-ens control matters. Refinement plus fusion is a form of ensembling, and ensembles of nnU-Net often help on their own.

### 4.2 Ablations

- **A1, prompt type (H3):** mask vs. bounding box vs. a single point at the lesion centre.
- **A2, candidate threshold:** 0.5 vs. 0.3.
- **A3, channel mapping:** Tmax/CBF/CBV vs. CTA/Tmax/CBF vs. one map repeated three times.
- **A4, propagation:** from the largest slice only vs. prompting every slice in the candidate (no propagation). This separates the benefit of the memory.
- **A5, 3D alternative:** SAM-Med3D, which segments volumes from 3D point prompts ([Wang 2025 #1](../../literature_review/report/literature_review.md#ex-wang2023sammed3d-1)).

### 4.3 Evaluation

The shared base protocol applies. The primary metric for H1 is **lesion-wise F1**, with Dice and absolute lesion count difference as secondary metrics, because refinement targets lesion detection and shape. Results for infarcts under 5 mL are the key secondary analysis. Failure modes will be reviewed visually:
- propagation leaking into healthy tissue
- missed scattered lesions
- prompts on the wrong slice

## 5. Expected outcomes and interpretation

| Outcome | Interpretation |
|---|---|
| R-ft > B1 and > B1-ens on lesion-wise F1 | Foundation-model refinement adds value beyond ensembling |
| R-ft ≈ B1-ens > B1 | The gain is ensembling. Report as such. |
| R-zs ≈ B1 < R-ft | Domain adaptation is required (supports H2) |
| R-ft < B1 | Low contrast defeats the foundation model's priors, as MedSAM's authors observed for SAM ([Ma 2024 #1](../../literature_review/report/literature_review.md#ex-ma2024medsam-1)) |
| A4: propagation ≈ per-slice prompting | The memory adds nothing here. Refinement is purely 2D. |

**Success criterion:** a significant improvement in lesion-wise F1 over both B1 and B1-ens, with no significant Dice loss.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Low-contrast targets ([Ma 2024 #1](../../literature_review/report/literature_review.md#ex-ma2024medsam-1)) | Perfusion maps as channels make lesions more visible than raw CT. Windowing from the winning pipeline. |
| Propagation drifts into healthy tissue | Stop when the predicted area falls below a minimum or the mask leaves the brain mask. Tune on validation. |
| Dependence on nnU-Net recall | The 0.3-threshold candidate set (A2). Report the recall ceiling explicitly. |
| MedSAM2 code and weights integration | Released weights and an architecture identical to SAM 2 ([Ma 2025 #2](../../literature_review/report/literature_review.md#ex-ma2025medsam2-2)). Budget one week for integration. |
| Ensembling confound | The B1-ens control arm |
| Memory and time | 2D frames at 512 or 1024 resolution, one volume at a time. Inference is light. LoRA fine-tuning fits 48 GB. |

## 7. Compute and timeline

**Compute.**
- nnU-Net: 2 models × 5 folds, one of them for the ensembling control.
- MedSAM2 fine-tuning: 5 folds plus about 4 ablation variants, about 25 runs, each lighter than full video training.
- Inference: about one forward pass per slice.

**Our assessment:** the second-cheapest proposal after P5.

| Weeks | Milestone | Lead |
|---|---|---|
| 1–2 | Folds, preprocessing, metrics. B0 results. | Data & evaluation |
| 2–4 | B1 and B1-ens, out-of-fold candidates | Baseline |
| 3–5 | MedSAM2 integration, zero-shot refinement (R-zs) | Model |
| 5–7 | Prompt simulation, fine-tuning (R-ft), fusion | Model |
| 7–8 | Ablations A1–A5 | Model + Analysis |
| 8–9 | Lesion-level analysis, failure cases | Analysis |
| 9–10 | Report | All |

## 8. What this means for the course proposal

- **§4:** the two-stage pipeline: nnU-Net candidates, then MedSAM2 mask-prompt propagation.
- **§5:** the arms table with the ensembling control, and lesion-wise F1 as the primary metric.
- **§6 Novelty:** the first use of a medical video foundation model's memory to refine final-infarct predictions.
- **§7 Fallback:** zero-shot results and B1 remain reportable even if fine-tuning fails.

The main weakness is that it builds on nnU-Net rather than proposing a new predictor of tissue fate. Presented that way, it is a clear and honest scope.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#4](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-4), [#17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17), [#30](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-30)
- **Ma 2024.** Jun Ma, Yuting He, Feifei Li, et al. (2024). Segment anything in medical images. *Nature Communications*, 15(1), 654. doi:10.1038/s41467-024-44824-z · arXiv:2304.12306. <https://doi.org/10.1038/s41467-024-44824-z>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ma2024medsam-1), [#2](../../literature_review/report/literature_review.md#ex-ma2024medsam-2)
- **Ma 2025.** Jun Ma, Zongxin Yang, Sumin Kim, et al. (2025). MedSAM2: Segment Anything in 3D Medical Images and Videos. *arXiv preprint arXiv:2504.03600*. <https://arxiv.org/abs/2504.03600>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ma2025medsam2-1), [#2](../../literature_review/report/literature_review.md#ex-ma2025medsam2-2), [#3](../../literature_review/report/literature_review.md#ex-ma2025medsam2-3)
- **Ravi 2025.** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. (2025). SAM 2: Segment Anything in Images and Videos. *ICLR 2025*. arXiv:2408.00714. <https://arxiv.org/abs/2408.00714>  
  Quotes: [#3](../../literature_review/report/literature_review.md#ex-ravi2024sam2-3)
- **Ren 2025.** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. <https://arxiv.org/abs/2505.18424>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ren2025isles24-1), [#7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7), [#9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)
- **Wang 2025.** Haoyu Wang, Sizheng Guo, Jin Ye, et al. (2025). SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images. *ECCV Workshops, Lecture Notes in Computer Science*, 51-67. doi:10.1007/978-3-031-91721-9_4 · arXiv:2310.15161. <https://doi.org/10.1007/978-3-031-91721-9_4>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-wang2023sammed3d-1)
- **Zhu 2024.** Jia-Yuan Zhu, Yunli Qi, Jun-De Wu (2024). Medical SAM 2: Segment medical images as video via Segment Anything Model 2. *arXiv preprint arXiv:2408.00874*. <https://arxiv.org/abs/2408.00874>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-1)
