# P2. Perfusion Maps as Video: Modelling Relations Between CBF, CBV, MTT and Tmax

*Rank 2 of 5 · guide methods M12 (main), M15, M16 and M17 (controls) · current proposal version, see `overleaf/sections/method.tex` · shared base: [00_selection.md](00_selection.md)*

## 1. Summary

**Research question.** Does presenting the four perfusion maps to a pretrained video encoder, as the frames of a short clip, improve final infarct prediction over nnU-Net with the same maps stacked as channels?

**Hypotheses.**
- **H1:** the video model beats nnU-Net on per-case Dice when both receive CBF, CBV, MTT and Tmax.
- **H2:** treating the maps as separate frames beats fusing them as channels inside the *same* backbone.
- **H3:** the order of the maps matters, so a fixed order beats a shuffled or order-free (set) arrangement.
- **H4:** video pretraining beats image pretraining and random initialisation.

**Why it ranks second.**
- **Highest novelty:** no published work treats the perfusion maps as a sequence.
- **Cheaper than P1:** only four inputs per slice, against sixteen CTP frames in P1.
- **Uses the inputs that won:** the perfusion maps are what the best ISLES'24 models used. Maps consistently beat raw 4D CTP at the challenge ([de la Rosa 2024 #11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11)).
- **The weakness:** no paper directly supports the idea that a sequence model helps for maps. The ablations below are designed to expose that.

## 2. Motivation and evidence

**The maps are the strongest input.**
- The top two ISLES'24 teams combined all modalities except 4D CTP ([de la Rosa 2024 #7](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-7)).
- The winning entry fed CTA plus the four maps ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)) to nnU-Net ResEnc L ([Ren 2025 #2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2)).
- The second-place cascade also relied on the perfusion maps ([de la Rosa 2024 #21](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-21)). Adding baseline masks stopped helping once those maps were present ([Mojtahedi 2025 #1](../../literature_review/report/literature_review.md#ex-mojtahedi2025switch-1)).

**Stacking channels discards structure.** nnU-Net mixes the maps in the first convolution, which treats them as interchangeable intensity channels. **Our assessment:** the four maps are related measurements of one contrast bolus, so a model that compares them explicitly might extract more. One map's abnormality only means something relative to the others.

**Video encoders model relations across frames.**
- Video Swin adds a locality bias across space and time ([Liu 2022 #1](../../literature_review/report/literature_review.md#ex-liu2022videoswin-1)).
- TimeSformer found that attending over time and space separately works best ([Bertasius 2021 #1](../../literature_review/report/literature_review.md#ex-bertasius2021timesformer-1)).
- VideoMAE learns from very small video datasets ([Tong 2022 #1](../../literature_review/report/literature_review.md#ex-tong2022videomae-1)).
- Video pretraining has improved 3D CT models ([Ke 2023 #1](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-1)), though only for detection tasks on chest CT ([Ke 2023 #3](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-3)).

**The order problem.** The maps have no natural time order. Medical SAM 2 handles frames without a temporal order by using a memory that ignores order ([Zhu 2024 #2](../../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-2)). That suggests an order-free variant as a principled alternative (H3).

## 3. Method

### 3.1 Preprocessing and input construction

1. **Shared preprocessing.** All arms use the winning team's pipeline:
   - SynthStrip skull stripping of NCCT, with the mask applied to the other scans ([Ren 2025 #8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8); [Ren 2025 #9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)).
   - Fixed windows for CTA (0–90 HU), CBF (0–35 mL/100 g/min), CBV (0–10 mL/100 g), MTT (0–20 s) and Tmax (0–7 s) ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)).
2. **Frame order.** CBF, CBV, MTT, Tmax: volume and flow first, then the timing maps.
3. **Channels.** Each frame's three colour channels hold slices *z*−1, *z*, *z*+1 of that map, giving 2.5D context in the pretrained input format.
4. **Cube alignment.** VideoMAE embeds 2 × 16 × 16 space-time cubes ([Tong 2022 #3](../../literature_review/report/literature_review.md#ex-tong2022videomae-3)). Each map is repeated twice, so the clip has 8 frames and every cube covers exactly one map. This reuses the pretrained embedding unchanged.
5. **Size.** Frames are resized to 224 × 224, giving 4 map tokens × 196 spatial positions.

### 3.2 Network

- **Encoder:** ViT-B VideoMAE, initialised from Kinetics-400 ([Tong 2022 #4](../../literature_review/report/literature_review.md#ex-tong2022videomae-4)). Temporal position embeddings are kept for the ordered variant and removed for the set variant.
- **Map fusion:** a learned attention-pooling layer over the four map tokens at each spatial position. Its attention weights show which map drives each region's prediction, and we will report them as an interpretability by-product.
- **Decoder:** four upsampling stages. Skip features come from a light CNN stem applied to the full-resolution input, to protect small infarcts ([de la Rosa 2024 #17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17)).
- **Output:** per-slice probabilities stacked into a volume, with small connected components removed.

### 3.3 Adaptation

The three settings are compared directly:
- **Frozen encoder.** Frozen video features can work well ([Bardes 2024 #1](../../literature_review/report/literature_review.md#ex-bardes2024vjepa-1)).
- **LoRA,** rank 8, on the query and value projections.
- **Full fine-tuning** with layer-wise learning-rate decay.

**Training:**
- Loss: Dice plus cross-entropy.
- Lesion slices oversampled 2:1.
- Optimiser: AdamW, 100 epochs.
- Hyperparameters chosen on the inner validation split only.

## 4. Experimental design

### 4.1 Comparisons

| Arm | Inputs | Model | Purpose |
|---|---|---|---|
| B0 | CBF | rCBF < 30% threshold | Clinical reference |
| B1a | 4 maps | nnU-Net ResEnc L | **Main comparator (H1)**, same inputs |
| B1b | CTA + 4 maps | nnU-Net ResEnc L | Winning configuration ([Ren 2025 #7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7)) |
| P2 | 4 maps | VideoMAE, ordered clip, LoRA | Proposed |

### 4.2 Ablations

- **A1, channels vs. frames (H2):** the same VideoMAE backbone given a single frame whose input layer takes all 12 channels (4 maps × 3 slices). This holds pretraining and capacity fixed and changes only the frame structure.
- **A2, order (H3):**
  - a fixed order
  - a random order for every sample during training, then fixed at test
  - an order-free set with no temporal position embedding

  If all three match, the model treats the maps as a set, and the "sequence" framing should be dropped from the claims.
- **A3, initialisation (H4):** Kinetics-400 VideoMAE vs. ImageNet weights inflated to video vs. random. This separates "video pretraining" from "any pretraining".
- **A4, image foundation control:** frozen DINOv3 features computed per map and fused by the same attention pooling. DINOv3 offers strong dense features from large-scale pretraining ([Zhang 2025 #3](../../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-3)), but its ISLES'24 result used a slice-level random split ([Zhang 2025 #1](../../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-1)), so it is re-evaluated here at patient level.
- **A5, encoder family:** Video Swin instead of VideoMAE ([Liu 2022 #1](../../literature_review/report/literature_review.md#ex-liu2022videoswin-1)).
- **A6, extra frames:** CTA and NCCT added as fifth and sixth frames. NCCT may improve robustness across centres ([de la Rosa 2024 #19](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-19)).

### 4.3 Evaluation

The shared base protocol applies:
- 5-fold cross-validation with identical folds
- the four challenge metrics
- results split by infarct size and centre
- paired Wilcoxon tests with Holm correction across H1–H4

The attention-pooling weights are summarised per centre and per lesion size as an exploratory analysis.

## 5. Expected outcomes and interpretation

| Outcome | Interpretation |
|---|---|
| P2 > B1a and P2 > A1 | Frame structure helps. Supports the core novelty claim. |
| P2 > B1a but P2 ≈ A1 | Gains come from the pretrained ViT, not the sequence framing. Claim a pretraining result instead. |
| P2 ≈ A2 (set) | Order is irrelevant. Recast the method as set-based fusion of maps. |
| A3: Kinetics ≈ ImageNet | The "video" in video pretraining isn't what helps. Report honestly. |
| P2 < B1a | The CNN inductive bias wins on 149 cases, consistent with the controlled ISLES'24 comparison ([Martínez Cegarra 2026 #4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4)). Fall back to P5. |

**Success criterion:** a significant per-case Dice gain over B1a and over A1, without a drop in lesion-wise F1. **Our assessment:** the probability of beating B1a is moderate at best. A transformer trailed CNNs on this dataset ([Martínez Cegarra 2026 #3](../../literature_review/report/literature_review.md#ex-cegarra2026isles-3)), and nnU-Net's authors warn that claimed gains often vanish under proper validation ([Isensee 2024 #2](../../literature_review/report/literature_review.md#ex-isensee2024revisited-2)). The ablation design is therefore what makes the project worthwhile whichever way the results go.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No support in the literature for maps as frames | A1 and A2 test the framing directly. The set variant is a ready alternative. |
| ViT at 1/16 resolution misses small lesions | Full-resolution CNN skip connection. Results split at 5 mL. |
| Different intensity ranges per map | Fixed windows from the winning pipeline. Per-map normalisation after windowing. |
| 2D slice-wise model lacks 3D context | 2.5D channels. Optional 3D post-processing (connected components). |
| Overfitting on 149 cases | LoRA or frozen encoder. Strong augmentation (flips, intensity jitter, small rotations). |
| Centre effects ([de la Rosa 2024 #24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)) | Stratified folds and per-centre reporting |

## 7. Compute and timeline

**Compute.** Four tokens per position make this the cheapest of the video options. Per slice there are 784 tokens at 224 × 224. Batches of 32 slices fit easily in 48 GB with LoRA. About 3 adaptation settings and 6 ablations × 5 folds come to roughly 45 runs. **Our assessment:** under a GPU-day each.

| Weeks | Milestone | Lead |
|---|---|---|
| 1–2 | Folds, preprocessing, metrics. B0 results. | Data & evaluation |
| 2–4 | B1a and B1b nnU-Net runs | Baseline |
| 3–5 | P2 implementation and first fold. A1 implemented in the same code. | Model |
| 5–8 | All folds, ablations A2–A6 | Model + Analysis |
| 8–9 | Statistics, attention-weight analysis, figures | Analysis |
| 9–10 | Report and presentation | All |

## 8. What this means for the course proposal

P2 is already written in `overleaf/sections/method.tex`, as version 1. Compared with that draft, this full proposal adds three things:
- the order-free set variant (A2)
- the DINOv3 image-foundation control (A4)
- an explicit interpretation table

If space allows, the set variant is the most valuable addition, because it turns the "maps have no natural order" weakness into a tested hypothesis. P1's native CTP can be described in §7 as the fallback, matching the current Feasibility section.

---

## References

Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.

- **Bardes 2024.** Adrien Bardes, Q. Garrido, Jean Ponce, et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *Transactions on Machine Learning Research (TMLR) 2024*. arXiv:2404.08471. <https://arxiv.org/abs/2404.08471>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-bardes2024vjepa-1)
- **Bertasius 2021.** Gedas Bertasius, Heng Wang, L. Torresani (2021). Is Space-Time Attention All You Need for Video Understanding? *ICML 2021*. arXiv:2102.05095. <https://arxiv.org/abs/2102.05095>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-bertasius2021timesformer-1)
- **de la Rosa 2024.** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand? *arXiv preprint arXiv:2408.10966*. <https://arxiv.org/abs/2408.10966>  
  Quotes: [#7](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-7), [#11](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-11), [#17](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-17), [#19](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-19), [#21](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-21), [#24](../../literature_review/report/literature_review.md#ex-delarosa2024isles24-24)
- **Isensee 2024.** Fabian Isensee, Tassilo Wald, Constantin Ulrich, et al. (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. *MICCAI 2024 (LNCS)*. arXiv:2404.09556. <https://arxiv.org/abs/2404.09556>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-isensee2024revisited-2)
- **Ke 2023.** Alexander Ke, Shih-Cheng Huang, Chloe O'Connell, et al. (2023). Video Pretraining Advances 3D Deep Learning on Chest CT Tasks. *MIDL 2023 (PMLR)*. arXiv:2304.00546. <https://arxiv.org/abs/2304.00546>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-1), [#3](../../literature_review/report/literature_review.md#ex-ke2023videopretraining-3)
- **Liu 2022.** Ze Liu, Jia Ning, Yue Cao, et al. (2022). Video Swin Transformer. *CVPR 2022*, 3192-3201. doi:10.1109/CVPR52688.2022.00320 · arXiv:2106.13230. <https://doi.org/10.1109/CVPR52688.2022.00320>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-liu2022videoswin-1)
- **Martínez Cegarra 2026.** Pablo Martínez Cegarra, Juan Francisco Zapata Pérez, Juan Martínez-Alajarín (2026). Automatic Segmentation of Ischaemic Stroke Lesions Using Transformers and Convolutional Neural Networks Applied to Multimodal Neuroimaging. *Sensors*, 26(16), 5021. doi:10.3390/s26165021. <https://doi.org/10.3390/s26165021>  
  Quotes: [#3](../../literature_review/report/literature_review.md#ex-cegarra2026isles-3), [#4](../../literature_review/report/literature_review.md#ex-cegarra2026isles-4)
- **Mojtahedi 2025.** Mahsa Mojtahedi, Laura van Poppel, Lucas de Vries, et al. (2025). Utilizing Baseline Infarct and Penumbra Masks to Improve NCCT and CTA Based Final Infarct Prediction. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 89-94. doi:10.1007/978-3-031-81101-2_10. <https://doi.org/10.1007/978-3-031-81101-2_10>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-mojtahedi2025switch-1)
- **Ren 2025.** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. <https://arxiv.org/abs/2505.18424>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-ren2025isles24-2), [#7](../../literature_review/report/literature_review.md#ex-ren2025isles24-7), [#8](../../literature_review/report/literature_review.md#ex-ren2025isles24-8), [#9](../../literature_review/report/literature_review.md#ex-ren2025isles24-9)
- **Tong 2022.** Zhan Tong, Yi-Bing Song, Jue Wang, et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS 2022*. arXiv:2203.12602. <https://arxiv.org/abs/2203.12602>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-tong2022videomae-1), [#3](../../literature_review/report/literature_review.md#ex-tong2022videomae-3), [#4](../../literature_review/report/literature_review.md#ex-tong2022videomae-4)
- **Zhang 2025.** Dong-Hao Zhang, Yimin Chen, K. Duarte, et al. (2025). Benchmarking DINOv3 for Multi-Task Stroke Analysis on Non-Contrast CT. *arXiv preprint arXiv:2509.23132*. <https://arxiv.org/abs/2509.23132>  
  Quotes: [#1](../../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-1), [#3](../../literature_review/report/literature_review.md#ex-zhang2025dinov3stroke-3)
- **Zhu 2024.** Jia-Yuan Zhu, Yunli Qi, Jun-De Wu (2024). Medical SAM 2: Segment medical images as video via Segment Anything Model 2. *arXiv preprint arXiv:2408.00874*. <https://arxiv.org/abs/2408.00874>  
  Quotes: [#2](../../literature_review/report/literature_review.md#ex-zhu2024medicalsam2-2)
