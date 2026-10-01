# Predicting the Final Infarct on ISLES'24: Literature Review

*CAP6614 project · 30 September 2026*

What the literature says about predicting post-treatment infarcts from acute CT, whether perfusion time series and video encoders help, and where vision-language models fit.

**41** papers included (of 51 screened) · **39** quoted · **128** verbatim excerpts · **0** unverified · main body **2,496** words

**How to read this document.** A citation such as <sup>[3]</sup> links to the verbatim quote in [Appendix B](#evidence). In most Markdown viewers, hovering over the citation shows the quote. Paragraphs in block quotes that begin with **▸ Our reading:** are our interpretation, not a claim from any source.

**Contents:** [1 Scope](#scope) · [2 Lineage](#lineage) · [3 State of the art](#sota) · [4 Post-challenge work](#recent) · [5 Temporal perfusion](#temporal) · [6 Backbones](#backbones) · [7 Video models](#video) · [8 Foundation models & VLMs](#vlm) · [9 Gaps](#gaps) · [A Paper list](#papers) · [B Evidence](#evidence) · [C Screening](#screening) · [D Verification](#verification)

---

<a id="scope"></a>

## 1. Scope and method

This review supports a CAP6614 project on final infarct prediction with the ISLES'24 dataset. It answers four questions. What is the current state of the art? Does the temporal information in CT perfusion (CTP) help? Are video architectures a plausible backbone? Can vision-language models (VLMs) contribute?

We screened 51 candidate papers, found through forward citation of the ISLES'24 challenge and dataset papers, the reference lists of included papers, and keyword searches on arXiv, Semantic Scholar and MICCAI proceedings. We included 41: 39 are quoted, and 2 ISLES'24 workshop chapters are listed but could not be read. Every numbered marker in the text links to verbatim excerpts. Each excerpt was string-matched against the text extracted from the downloaded paper ([method](#verification)). Paragraphs labelled **▸ Our reading:** are our interpretation and are not attributed to any paper.

<a id="lineage"></a>

## 2. From visible lesions to future tissue

The ISLES challenges have moved between two kinds of task. ISLES 2016 and 2017 already targeted lesion outcome prediction, using MRI <sup>[[1](#ex-winzeck2018isles-1 "ISLES 2016 and 2017 focused on lesion outcome prediction after ischemic stroke")]</sup>. The top entries were mostly CNNs, yet the organisers concluded that outcome prediction remained hard <sup>[[1](#ex-winzeck2018isles-2 "Top ranked teams almost always employed deep learning tools, which were predominately convolutional neural networks (CNNs). Despite the great efforts, lesion outcome prediction persists challenging.")]</sup>. ISLES 2018 switched to CT and asked teams to predict the infarct core from CTP, using DWI acquired shortly after as the reference <sup>[[2](#ex-hakim2021isles18-1 "the aim of the ISLES 2018 was to predict the infarction core based on CTP imaging from acute ischemic stroke patients, using DWI-based manual segmentation from magnetic resonance imaging (MRI) acquired shortly after.")]</sup>. Its best entry reached a Dice of 0.51, against 0.3 for a conventional threshold method <sup>[[2](#ex-hakim2021isles18-2 "Based on the ranking criteria, the top team’s algorithm demonstrated for average Dice score coefficient and average absolute volume difference 0.51 and 10.2 mL, respectively, outperforming the conventional threshold-based method (dice score coefficient, 0.3; volume difference, 15.3).")]</sup>, and top entries used the raw perfusion data <sup>[[2](#ex-hakim2021isles18-3 "Diverse algorithms were used, almost all based on deep learning, with top-ranked approaches making use of the raw perfusion data as well as methods to synthetically generate complementary information to boost prediction performance.")]</sup>. ISLES 2022 went back to segmenting lesions that are visible on MRI, with 400 cases <sup>[[3](#ex-hernandez2022isles22-1 "This dataset comprises 400 multi-vendor MRI cases with high variability in stroke lesion size, quantity and location.")]</sup>. Its top submissions became DeepISLES, an ensemble model <sup>[[4](#ex-delarosa2025deepisles-1 "We present DeepISLES, a robust ensemble algorithm developed from top submissions to the 2022 Ischemic Stroke Lesion Segmentation challenge we organized.")]</sup> whose segmentations neuroradiologists preferred over manual ones <sup>[[4](#ex-delarosa2025deepisles-2 "Neuroradiologists prefer DeepISLES’ segmentations over manual annotations in a Turing-like test.")]</sup>.

ISLES'24 combines both threads. It asks for the final infarct, predicted from imaging and clinical data acquired before treatment <sup>[[5](#ex-delarosa2024isles24-1 "This work introduces the Ischemic Stroke Lesion Segmentation (ISLES) 2024 challenge, which focuses on the prediction of final infarct volumes from pre-interventional acute stroke imaging and clinical data.")]</sup>. The dataset has 245 cases: 149 public training cases and a hidden test set of 96 <sup>[[6](#ex-riedel2026isles24dataset-1 "Data preparation, which is briefly illustrated in Figure 1 and described in detail in the methods section, led to a dataset of 245 cases divided into a publicly available train set (n = 149) and - to ensure fair and continuous evaluation of machine learning algorithms - a hidden test set (n = 96).")]</sup>. CT is acquired before therapy and follow-up MRI 2–9 days after reperfusion therapy <sup>[[6](#ex-riedel2026isles24dataset-2 "by providing longitudinal imaging at two time points: (sub-)acute CT imaging before therapy and follow-up MRI 2-9 days after intracranial reperfusion therapy.")]</sup>. The label masks were derived from that MRI with DeepISLES <sup>[[5](#ex-delarosa2024isles24-23 "Lesion masks are derived from the follow-up MRI using DeepISLES [14].")]</sup>. Only patients with successful recanalization (TICI 2b or 3) are included <sup>[[5](#ex-delarosa2024isles24-20 "Furthermore, the inclusion criteria of the ISLES’24 dataset were restricted to patients with favorable recanalization outcomes (i.e., thrombolysis in cerebral infarction -TICI- scores 2B and 3).")]</sup>. The challenge scores Dice together with volume and lesion-wise metrics <sup>[[5](#ex-delarosa2024isles24-3 "The four following segmentation and task-specific, clinically relevant metrics are used: i) Dice Similarity Coefficient")]</sup>. The clinical reference standard is to deconvolve the perfusion signal into maps and then threshold those maps <sup>[[6](#ex-riedel2026isles24dataset-3 "The current clinical standard for estimating perfusion volumes involves deconvolution analysis, which entails generating perfusion maps through perfusion CT and MR deconvolution and applying thresholds to these maps 10.")]</sup>.

> **▸ Our reading:** The ISLES'24 label describes tissue days after treatment. A model can only infer it, never observe it, in the input. Dice scores from ISLES 2018 or 2022 are therefore not a fair reference point for ISLES'24, because those tasks segment tissue that is already abnormal on the reference scan. Because the cohort is restricted to successful reperfusion, the task is in effect "predict the infarct, given that the vessel was reopened."

<a id="sota"></a>

## 3. The state of the art at ISLES'24

All three top-ranked ISLES'24 methods are based on nnU-Net <sup>[[5](#ex-delarosa2024isles24-4 "All three top-performing methods are based on nnU-Net [75].")]</sup>. The winning team, Kurtlab, used the large residual-encoder preset, ResEnc L <sup>[[7](#ex-ren2025isles24-2 "For our submission, we used a 3D large residual encoder nnUNet, referred to as “nnU-Net ResEnc L” in the nnU-Net documentation.")]</sup>. They credit their result to preprocessing: deep-learning skull stripping plus custom intensity windows for each modality <sup>[[7](#ex-ren2025isles24-1 "Our winning solution shows that a carefully designed preprocessing pipeline including deep-learning-based skull stripping and custom intensity windowing is beneficial for accurate segmentation.")]</sup>. On their best cross-validation fold, standard normalization gave a Dice of 21.8%, custom windowing 31.0%, and their final pipeline 31.8% <sup>[[7](#ex-ren2025isles24-3 "On this fold, the standard pipeline with only Z-score normalization and [1,99] percentile windowing achieved a Dice score of 21.8%, custom windowing alone improved performance to 31.0%, and our submission combining custom windowing with Z-score normalization further increased the validation Dice score to 31.8%.")]</sup>. On the hidden test set the winning model reached a Dice of 0.285 ± 0.213 and an absolute volume difference of 21.2 ± 37.2 mL <sup>[[5](#ex-delarosa2024isles24-2 "On the hidden test set (N = 98), the top-performing model, a multimodal nnU-Net-based architecture, achieved a Dice score of 0.285 ± 0.213 and an absolute volume difference of 21.2 ± 37.2 mL, underlining the significant challenges posed by this task and the need for further advances in multimodal learning.")]</sup>. The top three teams were separated by about 3 Dice points <sup>[[7](#ex-ren2025isles24-5 "Kurtlab (Ours) 28.50 (21.27) 21.23 (37.22) 14.39 (21.19) 7.18 (7.67) AMC-Axolotls 26.27 (24.73) 21.31 (35.23) 14.94 (25.12) 7.66 (7.94) Ninjas 25.46 (19.08) 26.29 (39.73) 9.92 (13.46) 5.98 (6.46)")]</sup>.

The runner-up, AMC-Axolotls, used a cascaded nnU-Net instead of a single model <sup>[[5](#ex-delarosa2024isles24-5 "Unlike the winning team Kurtlab, the second-ranked team AMC-Axolots employed a cascaded nnU-Net approach rather than a single-model strategy.")]</sup>. It added baseline infarct and penumbra masks derived from NCCT and CTA to the perfusion maps <sup>[[5](#ex-delarosa2024isles24-21 "The solution aimed to predict the final infarct by leveraging baseline infarct and penumbra masks derived from NCCT and CTA scans, as well as CTP perfusion maps (CBF, CBV, MTT, and TMAX).")]</sup>. A SWITCH 2024 paper from Amsterdam UMC on the same approach found that these masks improve Dice by 16% when CTP is missing, but no longer help once the perfusion maps are present <sup>[[8](#ex-mojtahedi2025switch-1 "Using the nnU-Net framework, we observe a 16% improvement in Dice when baseline lesion masks are added in the absence of CTP. However, when CTP maps are included, the addition of baseline lesion masks is no longer beneficial.")]</sup>. The third team, Ninjas, used only the 4D CTP series <sup>[[5](#ex-delarosa2024isles24-6 "Notably, the third-ranked team, Ninjas, relied exclusively on the 4D CTP perfusion modality and achieved strong individual-metric performance for Dice and absolute lesion count difference, ranking second and first, respectively.")]</sup>. They took 20 time points around peak contrast and fed them as 20 input channels <sup>[[5](#ex-delarosa2024isles24-22 "A sequence of 20 time points centered around the peak contrast concentration was extracted to form a 20-channel input.")]</sup>.

The challenge report draws several lessons. The first- and second-ranked teams combined all imaging modalities except 4D CTP, and they beat single-modality models <sup>[[5](#ex-delarosa2024isles24-7 "Notably, the first- and second-ranked teams integrated all available imaging modalities—excluding 4D CTP—and outperformed nnU-Net-based models relying solely on single modalities, such as teams utilizing only CTP or CTA data.")]</sup>. Simple rCBF thresholds estimated infarct volume about as well as the best networks, but deep learning localized the infarct better voxel by voxel <sup>[[5](#ex-delarosa2024isles24-8 "Notably, the perfusion-based baseline models using rCBF achieved infarct volume estimates comparable to those of the best-performing deep learning methods. However, deep learning approaches outperformed in voxel-wise metrics, demonstrating superior spatial localization and identification of infarction.")]</sup>. One team added clinical variables and saw no tangible gain <sup>[[5](#ex-delarosa2024isles24-9 "Although one team incorporated clinically relevant variables -such as stroke severity scores at admission, patient demographics, clinical history, and critical time metrics- there was no demonstrated tangible improvement in segmentation performance through the integration of these features.")]</sup>. The organisers attribute this to plain concatenation and list CLIP-style or attention-based fusion as an open question <sup>[[5](#ex-delarosa2024isles24-10 "A standard concatenation-based fusion strategy was used, which proved insufficient to capture the complex relationships between imaging and clinical variables. Whether more sophisticated feature fusion strategies, such as attention-based mechanisms, cross- and multi-modal techniques like CLIP [93], or advanced tabular data architectures such as TabFPN [94], can enhance segmentation performance, remains an open question.")]</sup>.

The report names several causes for the low scores. Infarcts grow at different rates depending on collateral circulation <sup>[[5](#ex-delarosa2024isles24-15 "Distinct infarct growth phenotypes—ranging from slow to fast progressors—have been identified, influenced by patient-specific pathophysiological variables, particularly the status of leptomeningeal collateral circulation, which sustains residual perfusion and limits infarct expansion [15, 16, 91].")]</sup>. Registration between acute CT and follow-up MRI introduces errors <sup>[[5](#ex-delarosa2024isles24-16 "Moreover, technical challenges related to data preprocessing—such as image registration and interpolation across differing image spaces, resolutions, and modalities (e.g., acute CT and follow-up MRI)—further hinder accurate lesion segmentation.")]</sup>. For infarcts under 5 mL, every method showed poor volumetric agreement <sup>[[5](#ex-delarosa2024isles24-17 "When focusing on cases with small infarcts (infarct< 5 mL,N = 25), all methods show limited volumetric agreement (Pearsonr< 0.4).")]</sup>. Models that rely only on CTP may be more sensitive to acquisition protocol <sup>[[5](#ex-delarosa2024isles24-18 "Methods based solely on CTP may be more sensitive to differences in acquisition protocols, resulting in decreased generalizability.")]</sup>, while NCCT may provide a more stable input across centres <sup>[[5](#ex-delarosa2024isles24-19 "As a more standardized and higher-resolution modality, NCCT offers a stable input source that may reduce model sensitivity to domain shifts in CTP data.")]</sup>.

*Table: Methods evaluated on ISLES'24 data, grouped by how they were evaluated. Results from different protocols cannot be ranked against each other.*

| Work | Inputs | Model | Evaluation protocol | Reported result |
|---|---|---|---|---|
| **Official hidden test set (N = 98 in the challenge report)** | | | | |
| Kurtlab, 1st | CTA, CBF, CBV, MTT, Tmax <sup>[[7](#ex-ren2025isles24-6 "Table 1: Windowing ranges for each CT modality used as input to the nnU-Net segmentation model for the final submission to the ISLES’24 challenge."), [7′](#ex-ren2025isles24-7 "Modality CTA (HU) CBF (mL/100g/min) CBV (mL/100g) MTT (s) Tmax (s) Window Range (0, 90) (0, 35) (0, 10) (0, 20) (0, 7)")]</sup> | nnU-Net ResEnc L <sup>[[7](#ex-ren2025isles24-2 "For our submission, we used a 3D large residual encoder nnUNet, referred to as “nnU-Net ResEnc L” in the nnU-Net documentation.")]</sup> | Hidden test; submitted the best of 10 folds <sup>[[7](#ex-ren2025isles24-4 "Results 10-fold cross validation was performed, and the fold with the highest validation Dice score was selected for submission.")]</sup> | Dice 0.285 <sup>[[5](#ex-delarosa2024isles24-2 "On the hidden test set (N = 98), the top-performing model, a multimodal nnU-Net-based architecture, achieved a Dice score of 0.285 ± 0.213 and an absolute volume difference of 21.2 ± 37.2 mL, underlining the significant challenges posed by this task and the need for further advances in multimodal learning.")]</sup> |
| AMC-Axolotls, 2nd | NCCT, CTA, perfusion maps + baseline masks <sup>[[5](#ex-delarosa2024isles24-21 "The solution aimed to predict the final infarct by leveraging baseline infarct and penumbra masks derived from NCCT and CTA scans, as well as CTP perfusion maps (CBF, CBV, MTT, and TMAX).")]</sup> | Cascaded nnU-Net <sup>[[5](#ex-delarosa2024isles24-5 "Unlike the winning team Kurtlab, the second-ranked team AMC-Axolots employed a cascaded nnU-Net approach rather than a single-model strategy.")]</sup> | Hidden test | Dice 26.27% <sup>[[7](#ex-ren2025isles24-5 "Kurtlab (Ours) 28.50 (21.27) 21.23 (37.22) 14.39 (21.19) 7.18 (7.67) AMC-Axolotls 26.27 (24.73) 21.31 (35.23) 14.94 (25.12) 7.66 (7.94) Ninjas 25.46 (19.08) 26.29 (39.73) 9.92 (13.46) 5.98 (6.46)")]</sup> |
| Ninjas, 3rd | 4D CTP, 20 time points as channels <sup>[[5](#ex-delarosa2024isles24-22 "A sequence of 20 time points centered around the peak contrast concentration was extracted to form a 20-channel input.")]</sup> | nnU-Net <sup>[[5](#ex-delarosa2024isles24-4 "All three top-performing methods are based on nnU-Net [75].")]</sup> | Hidden test | Dice 25.46% <sup>[[7](#ex-ren2025isles24-5 "Kurtlab (Ours) 28.50 (21.27) 21.23 (37.22) 14.39 (21.19) 7.18 (7.67) AMC-Axolotls 26.27 (24.73) 21.31 (35.23) 14.94 (25.12) 7.66 (7.94) Ninjas 25.46 (19.08) 26.29 (39.73) 9.92 (13.46) 5.98 (6.46)")]</sup> |
| **Studies with their own data splits** | | | | |
| Amador et al. (SWITCH 2024) | 4D CTP | CNN encoder + Transformer + CNN decoder <sup>[[9](#ex-amador2025switch-1 "The model integrates a shared-weight convolutional neural network (CNN) encoder, a Transformer encoder, and a CNN decoder to capture both spatial and temporal dependencies within the data. We evaluated this approach on a multicenter dataset of 143 patients from the ISLES 2024 challenge.")]</sup> | 143 ISLES'24 patients; split not stated in the abstract | Dice 0.20, F1 0.02 <sup>[[9](#ex-amador2025switch-2 "The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.")]</sup> |
| Martínez Cegarra et al. 2026 | CTA + four perfusion maps; cascade uses 2 then 7 channels <sup>[[10](#ex-cegarra2026isles-9 "(i) a single-stage nnU-Net approach replicating the preprocessing pipeline of the challenge-winning team under controlled experimental conditions, using 5 input channels (CTA, CBF, CBV, MTT, Tmax)"), [10′](#ex-cegarra2026isles-8 "The first of these is a single-stage 5-channel nnU-Net, followed by a two-stage cascaded nnU-Net (2 and 7 channels) and a lightweight Transformer (SegFormer3D).")]</sup> | nnU-Net, cascade, SegFormer3D <sup>[[10](#ex-cegarra2026isles-8 "The first of these is a single-stage 5-channel nnU-Net, followed by a two-stage cascaded nnU-Net (2 and 7 channels) and a lightweight Transformer (SegFormer3D).")]</sup> | Fixed 15-patient test set <sup>[[10](#ex-cegarra2026isles-5 "This section presents the performance results of each of the trained models when evaluated on the held-out test set of 15 patients.")]</sup> | Dice 0.224 cascade, 0.163 SegFormer3D <sup>[[10](#ex-cegarra2026isles-2 "The cascade model achieved the highest Dice Similarity Coefficient (0.224)."), [10′](#ex-cegarra2026isles-3 "On the other hand, SegFormer3D returned the lowest overall metrics (DSC 0.163, AVD 27.28 mL, F1 2.30%).")]</sup> |
| **Different target or unit of evaluation** | | | | |
| Britt et al. 2026 | NCCT only | nnU-Net with target-domain fine-tuning <sup>[[11](#ex-britt2026ncct-2 "To adapt to new domains, the model was fine-tuned on target-domain subsets from Boston (n=11) and ISLES (n=75), with evaluation on held-out cases not used for fine-tuning.")]</sup> | Held-out ISLES cases scored against the post-treatment infarct <sup>[[11](#ex-britt2026ncct-1 "Including smaller lesions, which predominated in ISLES, median Dice was 0.54 (interquartile range [IQR] 0.30 – 0.70) for acute lesion segmentation (Boston dataset) and 0.20 (IQR 0.03–0.41) for NCCT lesion segmentations when compared to post-treatment infarct (primary target of the ISLES challenge).")]</sup> | Median Dice 0.20 <sup>[[11](#ex-britt2026ncct-1 "Including smaller lesions, which predominated in ISLES, median Dice was 0.54 (interquartile range [IQR] 0.30 – 0.70) for acute lesion segmentation (Boston dataset) and 0.20 (IQR 0.03–0.41) for NCCT lesion segmentations when compared to post-treatment infarct (primary target of the ISLES challenge).")]</sup> |
| Zhang et al. 2025 (DINOv3) | NCCT slices | Frozen DINOv3 + decoder <sup>[[12](#ex-zhang2025dinov3stroke-4 "This work presents a unified, reproducible evaluation of a frozen self-supervised transformer backbone (DINOv3) on NCCT-based stroke analysis")]</sup> | Labelled 2D slices, random slice split <sup>[[12](#ex-zhang2025dinov3stroke-1 "We also extracted the labeled slices from this dataset to form our dataset. Finally, we compiled 3982 slices, randomly dividing them into 3185 for training and 797 for testing.")]</sup> | Dice 0.5678 <sup>[[12](#ex-zhang2025dinov3stroke-2 "On ISLES24, FPN+Refine with layers [2,5,8,11] yields the highest Dice of 0.5678")]</sup> |
| Shah & Sundaresan 2026 (AsymFeX) | NCCT + perfusion maps <sup>[[13](#ex-shah2026asymfex-4 "we incorporate CT perfusion parameter maps, including Cerebral Blood Volume (CBV), Cerebral Blood Flow (CBF)")]</sup> | nnU-Net + asymmetry module <sup>[[13](#ex-shah2026asymfex-3 "A 2-stage 3D segmentation pipeline that integrates with nnUNet and incorporates stroke-specific prior knowledge via an explicit geometric tiltcorrection step")]</sup> | Re-annotated pre-treatment core and penumbra <sup>[[13](#ex-shah2026asymfex-1 "ISLES’24 targets prediction of postinterventional infarct from pre-interventional CT, but since infarcts evolve after the initial scan, the released masks do not reflect the pre-interventional state. An expert neuroradiologist therefore manually re-delineated pre-interventional ischemic core and penumbra on the CT perfusion scans using ITK-SNAP [31].")]</sup> | Different labels; not comparable |

<a id="recent"></a>

## 4. Work after the challenge, and why its numbers don't compare

Martínez Cegarra et al. point out that leaderboard comparisons mix architecture with differences in preprocessing, training data and validation strategy <sup>[[10](#ex-cegarra2026isles-1 "The main limitation of drawing conclusions from comparisons derived from the challenge leaderboard is that the results of the competing strategies are subject to confounding variables such as differences in preprocessing protocols, training data volume, and cross-validation strategy, making it difficult to attribute performance gaps to architectural choice alone.")]</sup>. They retrained a single-stage nnU-Net, a cascade and SegFormer3D with shared preprocessing and splits. The cascade scored highest (Dice 0.224) and SegFormer3D lowest (Dice 0.163) <sup>[[10](#ex-cegarra2026isles-2 "The cascade model achieved the highest Dice Similarity Coefficient (0.224)."), [10′](#ex-cegarra2026isles-3 "On the other hand, SegFormer3D returned the lowest overall metrics (DSC 0.163, AVD 27.28 mL, F1 2.30%).")]</sup>. They conclude that convolutional inductive bias still helps on a dataset this size and that cascades add little over a well-tuned single network <sup>[[10](#ex-cegarra2026isles-4 "Within these limits, the local inductive bias of the convolutional models retained an empirical advantage over the single Transformer evaluated when processing this moderately sized neuroimaging dataset, and complex cascade topologies offered only marginal gains compared with a well-calibrated single-stage network.")]</sup>. Their test set has only 15 patients <sup>[[10](#ex-cegarra2026isles-5 "This section presents the performance results of each of the trained models when evaluated on the held-out test set of 15 patients.")]</sup>, and the two frameworks did not share all training settings <sup>[[10](#ex-cegarra2026isles-6 "It must be stated explicitly that the two frameworks used (nnU-Net for M1 and M2, MONAI/SegFormer3D for M3) do not share every training-time setting")]</sup>.

Other recent papers report much higher numbers on ISLES'24 data, but each one changes the task:

- **Zhang et al. (DINOv3):** randomly split the labelled 2D slices <sup>[[12](#ex-zhang2025dinov3stroke-1 "We also extracted the labeled slices from this dataset to form our dataset. Finally, we compiled 3982 slices, randomly dividing them into 3185 for training and 797 for testing.")]</sup> and reached a Dice of 0.5678 <sup>[[12](#ex-zhang2025dinov3stroke-2 "On ISLES24, FPN+Refine with layers [2,5,8,11] yields the highest Dice of 0.5678")]</sup>.
- **AsymFeX:** had a neuroradiologist redraw pre-treatment core and penumbra, because the released masks show the post-treatment state <sup>[[13](#ex-shah2026asymfex-1 "ISLES’24 targets prediction of postinterventional infarct from pre-interventional CT, but since infarcts evolve after the initial scan, the released masks do not reflect the pre-interventional state. An expert neuroradiologist therefore manually re-delineated pre-interventional ischemic core and penumbra on the CT perfusion scans using ITK-SNAP [31].")]</sup>. Its authors also found that NCCT alone nearly fails at this time point <sup>[[13](#ex-shah2026asymfex-2 "NCCT alone collapses to near-failing performance at this timepoint regardless of architecture, underscoring that imaging modality, not model design, is the binding constraint at early stroke stages.")]</sup>.
- **Britt et al.:** evaluated an NCCT model against the actual post-treatment target and obtained a median Dice of 0.20 <sup>[[11](#ex-britt2026ncct-1 "Including smaller lesions, which predominated in ISLES, median Dice was 0.54 (interquartile range [IQR] 0.30 – 0.70) for acute lesion segmentation (Boston dataset) and 0.20 (IQR 0.03–0.41) for NCCT lesion segmentations when compared to post-treatment infarct (primary target of the ISLES challenge).")]</sup>.
- **Amador et al.'s 4D CTP submission:** reported a Dice of 0.20 and a lesion-wise F1 of 0.02 <sup>[[9](#ex-amador2025switch-2 "The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.")]</sup>.

A 2026 multicentre study trained separate models for patients with and without recanalization. Its nnU-Net reached Dice 35.36% versus 15.73% for thresholds in the recanalized group <sup>[[14](#ex-ajnr2026recanalization-1 "The multimodal 3D nnU-Net model demonstrated superior performance, achieving mean Dice scores of 35.36% in patients with CR and 50.22% in those with NR. This model substantially outperformed the current clinically used method, providing more accurate outcome estimates than the conventional single-technique threshold-based measures, which yielded Dice scores of 15.73% and 39.71% for CR and NR groups, respectively.")]</sup>, on a cohort of 350 recanalized and 138 non-recanalized patients <sup>[[14](#ex-ajnr2026recanalization-2 "evaluated them separately for patients with complete recanalization (successful reperfusion [CR], n = 350) and no recanalization (unsuccessful reperfusion [NR], n = 138) after treatment.")]</sup>. The authors of nnU-Net Revisited warn more generally that many claimed gains in 3D segmentation disappear under proper baselines and validation <sup>[[15](#ex-isensee2024revisited-2 "In this study, we demonstrate that many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources.")]</sup>.

> **▸ Our reading:** Only the challenge's hidden-test numbers measure the same thing. The hidden test set is no longer open for routine submissions, so a fair comparison has to train and evaluate every model on the 149 public cases, using identical cross-validation folds and preprocessing. Any comparison to the leaderboard should be labelled as approximate.

<a id="temporal"></a>

## 5. Does temporal perfusion information help?

Single-cohort studies suggest it does. Robben et al. predicted the final infarct directly from native CTP plus time and treatment metadata, without deconvolution <sup>[[16](#ex-robben2020nativectp-1 "We pursue a data-driven and deconvolution-free approach, where a deep neural network learns to predict the final infarct volume directly from the native CTP images and metadata such as the time parameters and treatment.")]</sup>. Their ablation found that native CTP outperformed deconvolved inputs <sup>[[16](#ex-robben2020nativectp-2 "An ablation study shows that using the native CTP measurements instead of the deconvolved measurements improves the prediction.")]</sup>, and they note that deconvolution is ill-posed given CTP's low signal-to-noise ratio <sup>[[16](#ex-robben2020nativectp-3 "Additionally, deconvolution is a mathematically ill-posed problem and that is problematic given the low signal to noise ratio of perfusion images (Fieselmann et al., 2011).")]</sup>. The study used 188 subjects <sup>[[16](#ex-robben2020nativectp-4 "This results in 188 included subjects.")]</sup>. Amador et al. combined a U-Net with temporal convolutions over the 4D series <sup>[[17](#ex-amador2022spatiotemporal-2 "By merging a U-Net-like architecture with temporal convolutional networks, we efficiently process the spatio-temporal information available in CTP datasets to make a tissue outcome prediction.")]</sup>. In 10-fold cross-validation on 147 patients, their 3D+time model reached a Dice of 0.45, against 0.38 for a perfusion-map method <sup>[[17](#ex-amador2022spatiotemporal-1 "The proposed method was evaluated on 147 patients using a 10-fold cross validation, which demonstrated that the proposed 3D+time model (mean Dice=0.45) significantly outperforms both a 2D+time variant of our approach (mean Dice=0.43) and a state-of-the-art method that uses perfusion maps (mean Dice=0.38).")]</sup>. PerfU-Net is motivated by large differences between vendors' perfusion software <sup>[[18](#ex-devries2023perfunet-1 "The outcomes and perfusion maps of CT perfusion software, however, show many discrepancies between vendors.")]</sup>. It encodes the perfusion time series but decodes a static mask <sup>[[18](#ex-devries2023perfunet-2 "To this end, we present a symmetry-aware spatio-temporal segmentation model that encodes the micro-perfusion dynamics in the brain, while decoding a static segmentation map for infarct core assessment.")]</sup>, and it came close to methods that use vendor maps <sup>[[18](#ex-devries2023perfunet-3 "We achieve state-of-the-art results compared to methods that only use CT perfusion source imaging with a Dice score of 0.46. We are almost on par with methods that use perfusion maps from third party software")]</sup>.

ISLES'24 points the other way. Models using standard perfusion maps consistently outperformed models trained on raw 4D CTP <sup>[[5](#ex-delarosa2024isles24-11 "However, our challenge benchmark reveals two key findings: ( i) models utilizing standard deconvolution-derived perfusion maps consistently outperform those trained directly on raw 4D CTP series, and ( ii) models that extract spatial features from")]</sup>. Models that extracted spatial features from 4D CTP outperformed those that modelled its full spatio-temporal structure <sup>[[5](#ex-delarosa2024isles24-12 "the 4D CTP data achieve superior performance compared to those attempting to leverage the full spatiotemporal nature of the modality.")]</sup>. The best raw-CTP entry picked a subset of time points and used a spatial nnU-Net, while models built to capture contrast dynamics consistently underperformed <sup>[[5](#ex-delarosa2024isles24-13 "Specifically, the top-performing model utilizing raw 4D CTP data (team Ninjas) selected a subset of meaningful CTP timepoints and applied a spatial convolutional nnU-Net architecture, whereas models designed to explicitly capture temporally-encoded contrast propagation dynamics consistently underperformed.")]</sup>. Amador's spatio-temporal ISLES'24 submission reached a Dice of 0.20 <sup>[[9](#ex-amador2025switch-2 "The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.")]</sup>. The organisers still call for better spatio-temporal models <sup>[[5](#ex-delarosa2024isles24-14 "We thus advocate for the development of advanced spatiotemporal models")]</sup>.

> **▸ Our reading:** The evidence conflicts. The positive results come from single cohorts with their own splits. The negative result comes from a two-centre benchmark in which CTP resolution differed between the centres <sup>[[5](#ex-delarosa2024isles24-24 "While Center #1 provides higher resolution NCCT images, its CTP acquisitions exhibit consistently a lower resolution compared to Center #2.")]</sup>. Whether temporal modelling helps on ISLES'24 is therefore unresolved, which makes it a legitimate research question. It also means a negative result is a realistic outcome, and the project plan should allow for it.

<a id="backbones"></a>

## 6. Segmentation backbones

nnU-Net configures preprocessing, architecture and training automatically for each new dataset <sup>[[19](#ex-isensee2021nnunet-1 "We developed nnU-Net, a deep learning-based segmentation method that automatically configures itself, including preprocessing, network architecture, training and post-processing for any new task.")]</sup>. Its authors' later benchmark found that state-of-the-art 3D segmentation still comes from CNN U-Nets inside the nnU-Net framework, scaled to modern hardware <sup>[[15](#ex-isensee2024revisited-1 "In contrast to current beliefs, we find that the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models, including ResNet and ConvNeXt variants, 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources.")]</sup>. Transformer alternatives include Swin UNETR, a shifted-window transformer encoder with a CNN decoder <sup>[[20](#ex-hatamizadeh2022swinunetr-1 "The swin transformer encoder extracts features at five different resolutions by utilizing shifted windows for computing self-attention and is connected to an FCNN-based decoder at each resolution via skip connections.")]</sup>, and SegFormer3D, which uses far fewer parameters than earlier models <sup>[[21](#ex-perera2024segformer3d-1 "SegFormer3D democratizes deep learning for 3D medical image segmentation by offering a model with 33× less parameters and a 13× reduction in GFLOPS compared to the current state-of-theart (SOTA).")]</sup>. On ISLES'24, transformers have trailed: SegFormer3D trailed nnU-Net in a controlled comparison <sup>[[10](#ex-cegarra2026isles-4 "Within these limits, the local inductive bias of the convolutional models retained an empirical advantage over the single Transformer evaluated when processing this moderately sized neuroimaging dataset, and complex cascade topologies offered only marginal gains compared with a well-calibrated single-stage network.")]</sup>, and the only pure-transformer challenge entry scored 0.068 <sup>[[10](#ex-cegarra2026isles-7 "the only pure Transformer entry (QuiiL, using the MoReT architecture) obtained 0.068 [7].")]</sup>.

> **▸ Our reading:** The baseline to beat is nnU-Net ResEnc with the Kurtlab preprocessing. Any new backbone should get the same preprocessing, so that it isn't penalised for a weaker pipeline.

<a id="video"></a>

## 7. Video models and volumetric data

Several video encoders have properties suited to a small medical dataset:

- **VideoMAE:** works on very small datasets of about 3–4k videos without extra data <sup>[[22](#ex-tong2022videomae-1 "(2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data.")]</sup>, helped by the redundancy across frames <sup>[[22](#ex-tong2022videomae-2 "The temporally redundant video content enables higher masking ratio than that of images.")]</sup>.
- **Video Swin:** adds an inductive bias of locality in space and time <sup>[[23](#ex-liu2022videoswin-1 "In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization."), [23′](#ex-liu2022videoswin-2 "spatiotemporal locality of videos, in which pixels that are closer to each other in spatiotemporal distance are more likely to be correlated.")]</sup>.
- **TimeSformer:** found that attending over time and space separately works best <sup>[[24](#ex-bertasius2021timesformer-1 "Our experimental study compares different self-attention schemes and suggests that “divided attention,” where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered.")]</sup>.
- **V-JEPA:** features perform well with a frozen backbone <sup>[[25](#ex-bardes2024vjepa-1 "Our results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters; e.g., using a frozen backbone.")]</sup>.

Video models have already been applied to 3D medical volumes. SAM 2 uses a streaming memory over frames <sup>[[26](#ex-ravi2024sam2-1 "Our model is a simple transformer architecture with streaming memory for real-time video processing.")]</sup> and treats a single image as a one-frame video <sup>[[26](#ex-ravi2024sam2-2 "For images, the SA task can be considered a subset of PVS with the video reduced to a single frame.")]</sup>. Medical SAM 2 recasts all 2D and 3D medical segmentation as video object tracking <sup>[[27](#ex-zhu2024medicalsam2-1 "The core concept is to leverage the Segment Anything Model 2 (SAM2) pipeline to treat all 2D and 3D medical segmentation tasks as a video object tracking problem.")]</sup>. Its memory bank selects frames by confidence and dissimilarity, regardless of temporal order <sup>[[27](#ex-zhu2024medicalsam2-2 "To put it into practice, we propose a novel self-sorting memory bank mechanism that dynamically selects informative embeddings based on confidence and dissimilarity, regardless of temporal order.")]</sup>. MedSAM2 fine-tuned SAM 2 on more than 455,000 3D image-mask pairs <sup>[[28](#ex-ma2025medsam2-1 "The model is developed by fine-tuning the Segment Anything Model 2 on a large medical dataset with over 455,000 3D image-mask pairs and 76,000 frames, outperforming previous models across a wide range of organs, lesions, and imaging modalities.")]</sup>. Ke et al. found that natural-video pretraining improved the average performance of seven 3D models on two chest CT datasets <sup>[[29](#ex-ke2023videopretraining-1 "We demonstrate video pretraining improves the average performance of seven 3D models on two chest CT datasets, regardless of finetuning dataset size, and that video pretraining allows 3D models to outperform 2D baselines.")]</sup>, and that Kinetics pretraining helped more than in-domain CT pretraining <sup>[[29](#ex-ke2023videopretraining-2 "Lastly, we observe that pretraining on the large-scale out-of-domain Kinetics dataset improves performance more than pretraining on a typically-sized in-domain CT dataset.")]</sup>.

> **▸ Our reading:** There are three ways to map perfusion data onto a video model:

> 1. **Axial slices as frames.** This has direct precedent in Medical SAM 2, MedSAM2 and Ke et al.
> 2. **CTP time points as frames.** This is a true time axis and matches the native-CTP literature.
> 3. **The four perfusion maps as frames.** We found no study that supports this. The maps have no natural order, so an order-invariant design like Medical SAM 2's memory would be more defensible than one that assumes a sequence.

> The SAM family needs prompts, while ISLES'24 needs fully automatic prediction, so only the encoder and memory would carry over. Ke et al.'s evidence comes from chest CT, not brain perfusion.

<a id="vlm"></a>

## 8. Foundation models and VLMs

**Segmentation foundation models.** MedSAM reports that SAM struggles with weak boundaries and low contrast <sup>[[30](#ex-ma2024medsam-1 "However, the model exhibited substantial limitations in segmenting typical medical targets with weak boundaries or low contrast.")]</sup>, and SAM-style models need points or boxes as prompts <sup>[[30](#ex-ma2024medsam-2 "Essentially, SAM is a promptable segmentation method that requires points or bounding boxes to specify the segmentation targets.")]</sup>. SAM-Med3D also works from 3D prompt points <sup>[[31](#ex-wang2023sammed3d-1 "Given only a few 3D prompt points, SAM-Med3D can accurately segment diverse anatomical structures and lesions across various modalities.")]</sup>. SAT notes that models transferred from natural images often segment 2D slices and fuse them afterwards <sup>[[32](#ex-zhao2023sat-1 "However, while transferring to medical scenarios, these models trained on natural images suffer from fundamental limitations: (i) models typically perform 2D slice segmentation, which is later fused into 3D volumes through post-processing.")]</sup>. Its text-prompted SAT-Pro model matches 72 dataset-specific nnU-Nets <sup>[[32](#ex-zhao2023sat-2 "SAT-Pro achieves comparable performance to 72 nnU-Nets—the strongest specialist models trained on each dataset (over 2.2B parameters combined)—over 497 categories.")]</sup>. DINOv3 offers strong dense features from more than 1.6 billion images <sup>[[12](#ex-zhang2025dinov3stroke-3 "Among them, DINOv3 [17] represents a state-of-the-art vision transformer, pretrained on more than 1.6 billion images, and capable of extracting high-fidelity, fine-grained dense features.")]</sup>, but its ISLES'24 result used a slice-level protocol <sup>[[12](#ex-zhang2025dinov3stroke-1 "We also extracted the labeled slices from this dataset to form our dataset. Finally, we compiled 3982 slices, randomly dividing them into 3185 for training and 797 for testing.")]</sup>.

**3D vision-language models.** The available 3D CT VLMs cover other body regions or other tasks:

- **CT-CLIP:** built on 25,692 non-contrast chest CT scans <sup>[[33](#ex-hamamci2024ctclip-1 "CT-RATE comprises 25,692 non-contrast 3D chest CT scans from 21,304 unique patients.")]</sup>.
- **Merlin:** targets abdominal CT and learns from health records and radiology reports <sup>[[34](#ex-blankemeier2024merlin-1 "To overcome these shortcomings for abdominal CT interpretation, we introduce Merlin - a 3D VLM that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining without requiring additional manual annotations.")]</sup>. Its authors note that 3D VLM evaluations elsewhere remain limited <sup>[[34](#ex-blankemeier2024merlin-2 "Recent concurrent research has started to investigate 3D vision language models for radiology, particularly focusing on chest and head CT scans.79, 67 However, the extent of experiments and 3D evaluations are limited.")]</sup>.
- **M3D:** its authors say 3D images are under-explored compared with 2D <sup>[[35](#ex-bai2024m3d-1 "However, previous research has primarily focused on 2D medical images, leaving 3D images under-explored, despite their richer spatial information.")]</sup>.
- **RadFM:** its authors report that its reasoning stays general rather than specific to the case <sup>[[36](#ex-wu2023radfm-1 "However, the limitation can also be observed that the reasoning results are still general and more like background medical knowledge, yet not specific to the input case.")]</sup>.
- **BrainGPT:** the closest match. It was trained on 18,885 brain CT scans paired with reports, and it generates reports <sup>[[37](#ex-li2025braingpt-2 "In this regard, this study collected a 3D-BrainCT dataset (18,885 text-scan pairs) and applied clinical visual instruction tuning (CVIT) to train volumetric anatomy-sensible BrainGPT models to generate radiology-adherent 3D brain CT reports.")]</sup>. Its authors list scarce 3D data, generic base models and weak evaluation metrics as the constraints <sup>[[37](#ex-li2025braingpt-1 "Toward deploying MLLM for more applicable diagnostic context, we noticed that the (1) scarcity of 3D image training dataset, (2) the direct use of undifferentiated foundation MLLMs, and (3) the lack of pertinent caption evaluation metrics were independent domain-specific constraints that integratively hobbles the iteration of next-generation medical MLLM research.")]</sup>.

A recent review of stroke outcome prediction notes a lack of suitable data for applying language models to this problem <sup>[[38](#ex-samak2024review-2 "Hence, there is also a lack of such data for researchers to apply today’s enhanced language models [11, 12] for outcome predictions.")]</sup>.

> **▸ Our reading:** VLMs are a poor fit for the core task. ISLES'24 has no radiology reports, the target is a dense voxel-level prediction of future tissue, and we found no VLM trained on perfusion CT. Two uses remain defensible:

> 1. **Text-conditioned clinical fusion.** The challenge report leaves CLIP-style fusion of clinical data as an open question <sup>[[5](#ex-delarosa2024isles24-10 "A standard concatenation-based fusion strategy was used, which proved insufficient to capture the complex relationships between imaging and clinical variables. Whether more sophisticated feature fusion strategies, such as attention-based mechanisms, cross- and multi-modal techniques like CLIP [93], or advanced tabular data architectures such as TabFPN [94], can enhance segmentation performance, remains an open question.")]</sup>. The CLIP-Driven Universal Model shows that text embeddings can condition a segmentation network <sup>[[39](#ex-liu2023clipdriven-1 "To address these issues, we propose the CLIP-Driven Universal Model, which incorporates text embedding learned from Contrastive Language-Image Pre-training (CLIP) to segmentation models.")]</sup>, and Merlin shows health-record data used as supervision <sup>[[34](#ex-blankemeier2024merlin-1 "To overcome these shortcomings for abdominal CT interpretation, we introduce Merlin - a 3D VLM that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining without requiring additional manual annotations.")]</sup>. We could write each patient's clinical variables (admission NIHSS, timings, demographics) as a sentence, embed it with a frozen text encoder, and inject it into the decoder. The key ablation is against plain concatenation. This directly targets the clinical-data gap.
> 2. **Agentic quality control.** A brain-CT language model of the BrainGPT kind <sup>[[37](#ex-li2025braingpt-2 "In this regard, this study collected a 3D-BrainCT dataset (18,885 text-scan pairs) and applied clinical visual instruction tuning (CVIT) to train volumetric anatomy-sensible BrainGPT models to generate radiology-adherent 3D brain CT reports.")]</sup> could describe or flag predictions for review. This would not raise Dice, so it is worth adding only if the course wants an agentic component.

<a id="gaps"></a>

## 9. Synthesis: gaps this project can address

> **▸ Our reading:** The literature leaves four gaps that one semester can address.

1. **Controlled comparison.** Architecture comparisons on ISLES'24 are confounded by protocol <sup>[[10](#ex-cegarra2026isles-1 "The main limitation of drawing conclusions from comparisons derived from the challenge leaderboard is that the results of the competing strategies are subject to confounding variables such as differences in preprocessing protocols, training data volume, and cross-validation strategy, making it difficult to attribute performance gaps to architectural choice alone.")]</sup>, a general problem in 3D segmentation <sup>[[15](#ex-isensee2024revisited-2 "In this study, we demonstrate that many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources.")]</sup>. Fixed folds and shared preprocessing are a contribution in themselves.
2. **Temporal modelling.** Single-cohort studies favour 4D CTP <sup>[[17](#ex-amador2022spatiotemporal-1 "The proposed method was evaluated on 147 patients using a 10-fold cross validation, which demonstrated that the proposed 3D+time model (mean Dice=0.45) significantly outperforms both a 2D+time variant of our approach (mean Dice=0.43) and a state-of-the-art method that uses perfusion maps (mean Dice=0.38).")]</sup>, while ISLES'24 found that explicit temporal models underperform <sup>[[5](#ex-delarosa2024isles24-13 "Specifically, the top-performing model utilizing raw 4D CTP data (team Ninjas) selected a subset of meaningful CTP timepoints and applied a spatial convolutional nnU-Net architecture, whereas models designed to explicitly capture temporally-encoded contrast propagation dynamics consistently underperformed.")]</sup>. A pretrained video encoder on native CTP time points tests this directly.
3. **Input representation.** Current entries stack inputs as channels, including 20 CTP time points <sup>[[5](#ex-delarosa2024isles24-22 "A sequence of 20 time points centered around the peak contrast concentration was extracted to form a 20-channel input.")]</sup>. We found no study that models the perfusion maps as a set or sequence.
4. **Clinical fusion.** Concatenating clinical variables did not help, and better fusion remains open <sup>[[5](#ex-delarosa2024isles24-9 "Although one team incorporated clinically relevant variables -such as stroke severity scores at admission, patient demographics, clinical history, and critical time metrics- there was no demonstrated tangible improvement in segmentation performance through the integration of these features."), [5′](#ex-delarosa2024isles24-10 "A standard concatenation-based fusion strategy was used, which proved insufficient to capture the complex relationships between imaging and clinical variables. Whether more sophisticated feature fusion strategies, such as attention-based mechanisms, cross- and multi-modal techniques like CLIP [93], or advanced tabular data architectures such as TabFPN [94], can enhance segmentation performance, remains an open question.")]</sup>.

> This suggests making native 4D CTP the primary input for a video encoder, with the perfusion maps treated as an ordered sequence as a secondary experiment. Text-conditioned clinical fusion can be added as an ablation. Everything should be measured against a reproduced nnU-Net ResEnc baseline under identical folds.

> **▸ Our reading:** The literature also shapes how results should be reported. Use all four challenge metrics <sup>[[5](#ex-delarosa2024isles24-3 "The four following segmentation and task-specific, clinically relevant metrics are used: i) Dice Similarity Coefficient")]</sup>, because rCBF thresholds already match deep learning on volume and the networks' advantage is voxel-wise <sup>[[5](#ex-delarosa2024isles24-8 "Notably, the perfusion-based baseline models using rCBF achieved infarct volume estimates comparable to those of the best-performing deep learning methods. However, deep learning approaches outperformed in voxel-wise metrics, demonstrating superior spatial localization and identification of infarction.")]</sup>. Stratify by infarct size at 5 mL, where every challenge method struggled <sup>[[5](#ex-delarosa2024isles24-17 "When focusing on cases with small infarcts (infarct< 5 mL,N = 25), all methods show limited volumetric agreement (Pearsonr< 0.4).")]</sup>, and by acquisition centre, since CTP resolution differs between the two sites <sup>[[5](#ex-delarosa2024isles24-24 "While Center #1 provides higher resolution NCCT images, its CTP acquisitions exhibit consistently a lower resolution compared to Center #2.")]</sup>. Select checkpoints and windowing on validation folds only. The winning pipeline's preprocessing ablation was reported on its best fold <sup>[[7](#ex-ren2025isles24-3 "On this fold, the standard pipeline with only Z-score normalization and [1,99] percentile windowing achieved a Dice score of 21.8%, custom windowing alone improved performance to 31.0%, and our submission combining custom windowing with Z-score normalization further increased the validation Dice score to 31.8%."), [7′](#ex-ren2025isles24-4 "Results 10-fold cross validation was performed, and the fold with the highest validation Dice score was selected for submission.")]</sup>, which is fine for a challenge entry but would inflate a controlled comparison. For broader background on deep-learning stroke outcome prediction, including functional outcomes, see the survey by Samak et al. <sup>[[38](#ex-samak2024review-1 "This article provides a broad review of recent advances and applications of deep learning in the prediction of stroke outcomes, including (i) the data and models used, (ii) the prediction tasks and measures of success, (iii) the current challenges and limitations, and (iv) future directions and potential benefits.")]</sup>.

---

<a id="papers"></a>

## A. Paper list

Numbered in order of first citation. Venue, volume and pages come from Crossref when the paper has a DOI, otherwise from Semantic Scholar or the arXiv listing. "Cited by" counts come from the Semantic Scholar API and were retrieved on 30 September 2026. Very recent papers will show low counts.

| # | Short reference | Published venue | Year | Cited by | Topic |
|---|---|---|---|---|---|
| 1 | Stefan Winzeck, et al. | Frontiers in Neurology | 2018 | 172 | Benchmarks |
| 2 | Arsany Hakim, et al. | Stroke | 2021 | 70 | Benchmarks |
| 3 | Moritz R. Hernandez Petzsche, et al. | Scientific Data | 2022 | 275 | Benchmarks |
| 4 | Ezequiel de la Rosa, et al. | Nature Communications | 2025 | 33 | Benchmarks |
| 5 | Ezequiel de la Rosa, et al. | arXiv preprint arXiv:2408.10966 | 2024 | 7 | Benchmarks |
| 6 | Evamaria Olga Riedel, et al. | Radiology: Artificial Intelligence | 2026 | 3 | Benchmarks |
| 7 | Tian-Yi Ren, et al. | arXiv preprint arXiv:2505.18424 (states PMLR 2025) | 2025 | 6 | ISLES'24 methods & recent work |
| 8 | Mahsa Mojtahedi, et al. | Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science | 2025 | 0 | ISLES'24 methods & recent work |
| 9 | Kimberly Amador, et al. | Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science | 2025 | 0 | ISLES'24 methods & recent work |
| 10 | Pablo Martínez Cegarra, et al. | Sensors | 2026 | 0 | ISLES'24 methods & recent work |
| 11 | Linus Britt, et al. | arXiv preprint arXiv:2609.20151 | 2026 | 0 | ISLES'24 methods & recent work |
| 12 | Dong-Hao Zhang, et al. | arXiv preprint arXiv:2509.23132 | 2025 | 3 | ISLES'24 methods & recent work |
| 13 | M. Shah, et al. | arXiv preprint arXiv:2608.19769 | 2026 | 0 | ISLES'24 methods & recent work |
| 14 | Hongxi Yang, et al. | American Journal of Neuroradiology | 2026 | 2 | ISLES'24 methods & recent work |
| 15 | Fabian Isensee, et al. | MICCAI 2024 (LNCS) | 2024 | 512 | Segmentation backbones |
| 16 | David Robben, et al. | Medical Image Analysis | 2020 | 85 | CTP outcome prediction |
| 17 | Kimberly Amador, et al. | Medical Image Analysis | 2022 | 30 | CTP outcome prediction |
| 18 | Lucas de Vries, et al. | Medical Image Analysis | 2023 | 41 | CTP outcome prediction |
| 19 | Fabian Isensee, et al. | Nature Methods | 2021 | 10,370 | Segmentation backbones |
| 20 | Ali Hatamizadeh, et al. | BrainLes 2021 (MICCAI workshop), Lecture Notes in Computer Science | 2022 | 2,622 | Segmentation backbones |
| 21 | Shehan Perera, et al. | CVPR Workshops (CVPRW) 2024 | 2024 | 127 | Segmentation backbones |
| 22 | Zhan Tong, et al. | NeurIPS 2022 | 2022 | 2,308 | Video models |
| 23 | Ze Liu, et al. | CVPR 2022 | 2022 | 2,296 | Video models |
| 24 | Gedas Bertasius, et al. | ICML 2021 | 2021 | 3,370 | Video models |
| 25 | Adrien Bardes, et al. | Transactions on Machine Learning Research (TMLR) 2024 | 2024 | 563 | Video models |
| 26 | Nikhila Ravi, et al. | ICLR 2025 | 2025 | 4,349 | Video models |
| 27 | Jia-Yuan Zhu, et al. | arXiv preprint arXiv:2408.00874 | 2024 | 286 | Video models |
| 28 | Jun Ma, et al. | arXiv preprint arXiv:2504.03600 | 2025 | 156 | Video models |
| 29 | Alexander Ke, et al. | MIDL 2023 (PMLR) | 2023 | 11 | Video models |
| 30 | Jun Ma, et al. | Nature Communications | 2024 | 1,675 | Foundation models & VLMs |
| 31 | Haoyu Wang, et al. | ECCV Workshops, Lecture Notes in Computer Science | 2025 | 204 | Foundation models & VLMs |
| 32 | Ziheng Zhao, et al. | npj Digital Medicine | 2025 | 101 | Foundation models & VLMs |
| 33 | Ibrahim Ethem Hamamci, et al. | Nature Biomedical Engineering | 2026 | 258 | Foundation models & VLMs |
| 34 | Louis Blankemeier, et al. | Nature | 2026 | 175 | Foundation models & VLMs |
| 35 | Fan Bai, et al. | arXiv preprint arXiv:2404.00578 | 2024 | 195 | Foundation models & VLMs |
| 36 | Chaoyi Wu, et al. | arXiv preprint arXiv:2308.02463 | 2023 | 236 | Foundation models & VLMs |
| 37 | Cheng-Yi Li, et al. | Nature Communications | 2025 | 97 | Foundation models & VLMs |
| 38 | Zeynel A. Samak, et al. | Biomedical Engineering Letters | 2025 | 16 | ISLES'24 methods & recent work |
| 39 | Jie Liu, et al. | ICCV 2023 | 2023 | 387 | Foundation models & VLMs |
| 40 | Luan Matheus Trindade Dalmazo, et al. | Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science | 2025 | 0 | ISLES'24 methods & recent work |
| 41 | Eneko Uruñuela, et al. | Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science | 2025 | 0 | ISLES'24 methods & recent work |

### Full citations

<a id="ref-1"></a>**[1]** Stefan Winzeck, Arsany Hakim, Richard McKinley, et al. (2018). ISLES 2016 and 2017-Benchmarking Ischemic Stroke Lesion Outcome Prediction Based on Multispectral MRI. *Frontiers in Neurology*, 9, 679. doi:10.3389/fneur.2018.00679. <https://doi.org/10.3389/fneur.2018.00679>  
Cited by 172 (Semantic Scholar). Access: full text (publisher PDF). Inclusion: R1/R2: ISLES 2016-17 outcome-prediction benchmark.

<a id="ref-2"></a>**[2]** Arsany Hakim, Søren Christensen, Stefan Winzeck, et al. (2021). Predicting Infarct Core From Computed Tomography Perfusion in Acute Ischemia With Machine Learning: Lessons From the ISLES Challenge. *Stroke*, 52(7), 2328-2337. doi:10.1161/STROKEAHA.120.030696. <https://doi.org/10.1161/STROKEAHA.120.030696>  
Cited by 70 (Semantic Scholar). Access: full text (PMC XML). Inclusion: R2: ISLES 2018 CTP core benchmark.

<a id="ref-3"></a>**[3]** Moritz R. Hernandez Petzsche, Ezequiel de la Rosa, Uta Hanning, et al. (2022). ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. *Scientific Data*, 9(1), 762. doi:10.1038/s41597-022-01875-5 · arXiv:2206.06694. <https://doi.org/10.1038/s41597-022-01875-5>  
Cited by 275 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1 (context): ISLES 2022 MRI segmentation dataset.

<a id="ref-4"></a>**[4]** Ezequiel de la Rosa, Mauricio Reyes, Sook-Lei Liew, et al. (2025). DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge. *Nature Communications*, 16(1), 7357. doi:10.1038/s41467-025-62373-x · arXiv:2403.19425. <https://doi.org/10.1038/s41467-025-62373-x>  
Cited by 33 (Semantic Scholar). Access: full text (publisher PDF). Inclusion: R1: DeepISLES generated the ISLES'24 label masks.

<a id="ref-5"></a>**[5]** Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. (2024). ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand?. *arXiv preprint arXiv:2408.10966*. arXiv:2408.10966. <https://arxiv.org/abs/2408.10966>  
Cited by 7 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1: official ISLES'24 challenge report.

<a id="ref-6"></a>**[6]** Evamaria Olga Riedel, Ezequiel de la Rosa, The Anh Baran, et al. (2026). The Ischemic Stroke Lesion Segmentation Challenge (ISLES)’24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. *Radiology: Artificial Intelligence*, 8(3), e250603. doi:10.1148/ryai.250603. <https://doi.org/10.1148/ryai.250603>  
Cited by 3 (Semantic Scholar). Access: full text of arXiv preprint (2408.11142); published version paywalled. Inclusion: R1: official ISLES'24 dataset descriptor.

<a id="ref-7"></a>**[7]** Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. (2025). How We Won the ISLES'24 Challenge by Preprocessing. *arXiv preprint arXiv:2505.18424 (states PMLR 2025)*. arXiv:2505.18424. <https://arxiv.org/abs/2505.18424>  
Cited by 6 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1: winning ISLES'24 method (Kurtlab).

<a id="ref-8"></a>**[8]** Mahsa Mojtahedi, Laura van Poppel, Lucas de Vries, et al. (2025). Utilizing Baseline Infarct and Penumbra Masks to Improve NCCT and CTA Based Final Infarct Prediction. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 89-94. doi:10.1007/978-3-031-81101-2_10. <https://doi.org/10.1007/978-3-031-81101-2_10>  
Cited by 0 (Semantic Scholar). Access: abstract only (Amsterdam UMC repository page; PDF blocked). Inclusion: R1: AMC-Axolotls (2nd place) method paper.

<a id="ref-9"></a>**[9]** Kimberly Amador, Anthony J. Winder, Nils D. Forkert (2025). Spatio-Temporal Deep Learning for Final Infarct Prediction Using Acute Stroke CT Perfusion Data. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 83-88. doi:10.1007/978-3-031-81101-2_9. <https://doi.org/10.1007/978-3-031-81101-2_9>  
Cited by 0 (Semantic Scholar). Access: abstract only (authors' GitHub README; chapter paywalled). Inclusion: R1: ISLES'24 spatio-temporal 4D CTP submission.

<a id="ref-10"></a>**[10]** Pablo Martínez Cegarra, Juan Francisco Zapata Pérez, Juan Martínez-Alajarín (2026). Automatic Segmentation of Ischaemic Stroke Lesions Using Transformers and Convolutional Neural Networks Applied to Multimodal Neuroimaging. *Sensors*, 26(16), 5021. doi:10.3390/s26165021. <https://doi.org/10.3390/s26165021>  
Cited by 0 (Semantic Scholar). Access: full text (PMC XML). Inclusion: R1: controlled CNN vs Transformer comparison on ISLES'24.

<a id="ref-11"></a>**[11]** Linus Britt, M. Nielsen, S. Klapproth, et al. (2026). Ischemic Stroke Segmentation and Net Water Uptake Quantification on Multicenter Non-Contrast CT Using Supervised Target-Domain Adaptation. *arXiv preprint arXiv:2609.20151*. arXiv:2609.20151. <https://arxiv.org/abs/2609.20151>  
Cited by 0 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1: NCCT model fine-tuned on ISLES'24.

<a id="ref-12"></a>**[12]** Dong-Hao Zhang, Yimin Chen, K. Duarte, et al. (2025). Benchmarking DINOv3 for Multi-Task Stroke Analysis on Non-Contrast CT. *arXiv preprint arXiv:2509.23132*. arXiv:2509.23132. <https://arxiv.org/abs/2509.23132>  
Cited by 3 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1/R3: frozen foundation model (DINOv3) evaluated on ISLES'24 NCCT.

<a id="ref-13"></a>**[13]** M. Shah, V. Sundaresan (2026). AsymFeX: A Symmetry-Driven Framework for Ischemic Stroke Segmentation Across Imaging Modalities and Stroke Stages. *arXiv preprint arXiv:2608.19769*. arXiv:2608.19769. <https://arxiv.org/abs/2608.19769>  
Cited by 0 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R1: uses ISLES'24 with re-annotated pre-treatment labels.

<a id="ref-14"></a>**[14]** Hongxi Yang, Yasmeen George, Deval Mehta, et al. (2026). Multimodal CT Perfusion–Based Deep Learning for Predicting Stroke Lesion Outcomes in Complete and No Recanalization Scenarios. *American Journal of Neuroradiology*, 47(4), 920-928. doi:10.3174/ajnr.A9016. <https://doi.org/10.3174/ajnr.A9016>  
Cited by 2 (Semantic Scholar). Access: abstract only (PubMed). Inclusion: R2: recent CTP-map nnU-Net outcome model, reperfusion-specific.

<a id="ref-15"></a>**[15]** Fabian Isensee, Tassilo Wald, Constantin Ulrich, et al. (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. *MICCAI 2024 (LNCS)*. arXiv:2404.09556. <https://arxiv.org/abs/2404.09556>  
Cited by 512 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: ResEnc presets used by the winner; validation pitfalls.

<a id="ref-16"></a>**[16]** David Robben, Anna M.M. Boers, Henk A. Marquering, et al. (2020). Prediction of final infarct volume from native CT perfusion and treatment parameters using deep learning. *Medical Image Analysis*, 59, 101589. doi:10.1016/j.media.2019.101589 · arXiv:1812.02496. <https://doi.org/10.1016/j.media.2019.101589>  
Cited by 85 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R2: native-CTP final infarct prediction.

<a id="ref-17"></a>**[17]** Kimberly Amador, Matthias Wilms, Anthony Winder, et al. (2022). Predicting treatment-specific lesion outcomes in acute ischemic stroke from 4D CT perfusion imaging using spatio-temporal convolutional neural networks. *Medical Image Analysis*, 82, 102610. doi:10.1016/j.media.2022.102610. <https://doi.org/10.1016/j.media.2022.102610>  
Cited by 30 (Semantic Scholar). Access: abstract only (PubMed). Inclusion: R2: 4D CTP spatio-temporal outcome prediction.

<a id="ref-18"></a>**[18]** Lucas de Vries, Bart J. Emmer, Charles B.L.M. Majoie, et al. (2023). PerfU-Net: Baseline infarct estimation from CT perfusion source data for acute ischemic stroke. *Medical Image Analysis*, 85, 102749. doi:10.1016/j.media.2023.102749. <https://doi.org/10.1016/j.media.2023.102749>  
Cited by 41 (Semantic Scholar). Access: abstract only (PubMed). Inclusion: R2: spatio-temporal core segmentation from CTP source data.

<a id="ref-19"></a>**[19]** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211. doi:10.1038/s41592-020-01008-z. <https://doi.org/10.1038/s41592-020-01008-z>  
Cited by 10,370 (Semantic Scholar). Access: abstract only (PubMed). Inclusion: R3: baseline framework.

<a id="ref-20"></a>**[20]** Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, et al. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *BrainLes 2021 (MICCAI workshop), Lecture Notes in Computer Science*, 272-284. doi:10.1007/978-3-031-08999-2_22 · arXiv:2201.01266. <https://doi.org/10.1007/978-3-031-08999-2_22>  
Cited by 2,622 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: transformer 3D segmentation backbone.

<a id="ref-21"></a>**[21]** Shehan Perera, Pouyan Navard, Alper Yilmaz (2024). SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. *CVPR Workshops (CVPRW) 2024*, 4981-4988. doi:10.1109/CVPRW63382.2024.00503 · arXiv:2404.10156. <https://doi.org/10.1109/CVPRW63382.2024.00503>  
Cited by 127 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: transformer evaluated on ISLES'24 by Cegarra et al..

<a id="ref-22"></a>**[22]** Zhan Tong, Yi-Bing Song, Jue Wang, et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS 2022*. arXiv:2203.12602. <https://arxiv.org/abs/2203.12602>  
Cited by 2,308 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: candidate pretrained video encoder.

<a id="ref-23"></a>**[23]** Ze Liu, Jia Ning, Yue Cao, et al. (2022). Video Swin Transformer. *CVPR 2022*, 3192-3201. doi:10.1109/CVPR52688.2022.00320 · arXiv:2106.13230. <https://doi.org/10.1109/CVPR52688.2022.00320>  
Cited by 2,296 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: candidate pretrained video encoder.

<a id="ref-24"></a>**[24]** Gedas Bertasius, Heng Wang, L. Torresani (2021). Is Space-Time Attention All You Need for Video Understanding?. *ICML 2021*. arXiv:2102.05095. <https://arxiv.org/abs/2102.05095>  
Cited by 3,370 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: space-time attention design.

<a id="ref-25"></a>**[25]** Adrien Bardes, Q. Garrido, Jean Ponce, et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *Transactions on Machine Learning Research (TMLR) 2024*. arXiv:2404.08471. <https://arxiv.org/abs/2404.08471>  
Cited by 563 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: candidate pretrained video encoder (frozen use).

<a id="ref-26"></a>**[26]** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. (2025). SAM 2: Segment Anything in Images and Videos. *ICLR 2025*. arXiv:2408.00714. <https://arxiv.org/abs/2408.00714>  
Cited by 4,349 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3/R4: video segmentation foundation model.

<a id="ref-27"></a>**[27]** Jia-Yuan Zhu, Yunli Qi, Jun-De Wu (2024). Medical SAM 2: Segment medical images as video via Segment Anything Model 2. *arXiv preprint arXiv:2408.00874*. arXiv:2408.00874. <https://arxiv.org/abs/2408.00874>  
Cited by 286 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R4: 3D medical volumes treated as video.

<a id="ref-28"></a>**[28]** Jun Ma, Zongxin Yang, Sumin Kim, et al. (2025). MedSAM2: Segment Anything in 3D Medical Images and Videos. *arXiv preprint arXiv:2504.03600*. arXiv:2504.03600. <https://arxiv.org/abs/2504.03600>  
Cited by 156 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R4: video foundation model fine-tuned for 3D medical images.

<a id="ref-29"></a>**[29]** Alexander Ke, Shih-Cheng Huang, Chloe O'Connell, et al. (2023). Video Pretraining Advances 3D Deep Learning on Chest CT Tasks. *MIDL 2023 (PMLR)*. arXiv:2304.00546. <https://arxiv.org/abs/2304.00546>  
Cited by 11 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R4: natural-video pretraining transferred to 3D CT.

<a id="ref-30"></a>**[30]** Jun Ma, Yuting He, Feifei Li, et al. (2024). Segment anything in medical images. *Nature Communications*, 15(1), 654. doi:10.1038/s41467-024-44824-z · arXiv:2304.12306. <https://doi.org/10.1038/s41467-024-44824-z>  
Cited by 1,675 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: medical segmentation foundation model.

<a id="ref-31"></a>**[31]** Haoyu Wang, Sizheng Guo, Jin Ye, et al. (2025). SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images. *ECCV Workshops, Lecture Notes in Computer Science*, 51-67. doi:10.1007/978-3-031-91721-9_4 · arXiv:2310.15161. <https://doi.org/10.1007/978-3-031-91721-9_4>  
Cited by 204 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: volumetric promptable foundation model.

<a id="ref-32"></a>**[32]** Ziheng Zhao, Yao Zhang, Chaoyi Wu, et al. (2025). Large-vocabulary segmentation for medical images with text prompts. *npj Digital Medicine*, 8(1), 566. doi:10.1038/s41746-025-01964-w · arXiv:2312.17183. <https://doi.org/10.1038/s41746-025-01964-w>  
Cited by 101 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3/R4: text-prompted 3D segmentation.

<a id="ref-33"></a>**[33]** Ibrahim Ethem Hamamci, Sezgin Er, Chenyu Wang, et al. (2026). Generalist foundation models from a multimodal dataset for 3D computed tomography. *Nature Biomedical Engineering*, 10(8), 1610-1628. doi:10.1038/s41551-025-01599-y · arXiv:2403.17834. <https://doi.org/10.1038/s41551-025-01599-y>  
Cited by 258 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: 3D CT vision-language model.

<a id="ref-34"></a>**[34]** Louis Blankemeier, Ashwin Kumar, Joseph Paul Cohen, et al. (2026). Merlin: a computed tomography vision–language foundation model and dataset. *Nature*, 652(8112), 1318-1328. doi:10.1038/s41586-026-10181-8 · arXiv:2406.06512. <https://doi.org/10.1038/s41586-026-10181-8>  
Cited by 175 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: 3D CT VLM using EHR + reports.

<a id="ref-35"></a>**[35]** Fan Bai, Yu-Xin Du, Tiejun Huang, et al. (2024). M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models. *arXiv preprint arXiv:2404.00578*. arXiv:2404.00578. <https://arxiv.org/abs/2404.00578>  
Cited by 195 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: 3D multimodal LLM.

<a id="ref-36"></a>**[36]** Chaoyi Wu, Xiaoman Zhang, Ya Zhang, et al. (2023). Towards Generalist Foundation Model for Radiology. *arXiv preprint arXiv:2308.02463*. arXiv:2308.02463. <https://arxiv.org/abs/2308.02463>  
Cited by 236 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3: generalist radiology foundation model.

<a id="ref-37"></a>**[37]** Cheng-Yi Li, Kao-Jung Chang, Cheng-Fu Yang, et al. (2025). Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation. *Nature Communications*, 16(1), 2258. doi:10.1038/s41467-025-57426-0 · arXiv:2407.02235. <https://doi.org/10.1038/s41467-025-57426-0>  
Cited by 97 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3/R4: 3D brain CT multimodal LLM.

<a id="ref-38"></a>**[38]** Zeynel A. Samak, Philip Clatworthy, Majid Mirmehdi (2025). Automatic prediction of stroke treatment outcomes: latest advances and perspectives. *Biomedical Engineering Letters*, 15(3), 467-488. doi:10.1007/s13534-025-00462-y · arXiv:2412.04812. <https://doi.org/10.1007/s13534-025-00462-y>  
Cited by 16 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R2: survey of DL stroke outcome prediction.

<a id="ref-39"></a>**[39]** Jie Liu, Yixiao Zhang, Jie-Neng Chen, et al. (2023). CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection. *ICCV 2023*, 21095-21107. doi:10.1109/ICCV51070.2023.01934 · arXiv:2301.00785. <https://doi.org/10.1109/ICCV51070.2023.01934>  
Cited by 387 (Semantic Scholar). Access: full text (arXiv PDF). Inclusion: R3/R4: text embeddings as segmentation conditioning.

<a id="ref-40"></a>**[40]** Luan Matheus Trindade Dalmazo, Kumaradevan Punithakumar (2025). Final Stroke Infarct Segmentation Using Deep Neural Networks. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 95-99. doi:10.1007/978-3-031-81101-2_11. <https://doi.org/10.1007/978-3-031-81101-2_11>  
Cited by 0 (Semantic Scholar). Access: not accessible (paywalled; no abstract in Semantic Scholar/OpenAlex; publisher blocks automated access) — listed only, not cited for claims. Inclusion: R1: ISLES'24 submission chapter.

<a id="ref-41"></a>**[41]** Eneko Uruñuela, Annabella Bregazzi, Matthias Wilms (2025). A Multi-modal Deep Learning Framework for Final Infarct Prediction in Acute Ischemic Stroke: Combining CTA, NCCT, and Clinical Data. *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science*, 100-105. doi:10.1007/978-3-031-81101-2_12. <https://doi.org/10.1007/978-3-031-81101-2_12>  
Cited by 0 (Semantic Scholar). Access: not accessible (paywalled; no abstract in Semantic Scholar/OpenAlex; publisher blocks automated access) — listed only, not cited for claims. Inclusion: R1: ISLES'24 submission chapter.

<a id="evidence"></a>

## B. Verbatim evidence

Every excerpt cited above, grouped by paper. Excerpt IDs match `excerpts.json`. Page numbers refer to the downloaded PDF. For PubMed Central XML and abstract-only sources, the location is a section name.

### [1] ISLES 2016 and 2017-Benchmarking Ischemic Stroke Lesion Outcome Prediction Based on Multispectral MRI

Stefan Winzeck, Arsany Hakim, Richard McKinley, et al. · *Frontiers in Neurology* · 2018 · <https://doi.org/10.3389/fneur.2018.00679>  
Access: full text (publisher PDF)

<a id="ex-winzeck2018isles-1"></a>
> ISLES 2016 and 2017 focused on lesion outcome prediction after ischemic stroke
>
> — `winzeck2018isles#1` · p1 · verbatim, exact match

<a id="ex-winzeck2018isles-2"></a>
> Top ranked teams almost always employed deep learning tools, which were predominately convolutional neural networks (CNNs). Despite the great efforts, lesion outcome prediction persists challenging.
>
> — `winzeck2018isles#2` · p1 · verbatim, PDF spacing normalised

### [2] Predicting Infarct Core From Computed Tomography Perfusion in Acute Ischemia With Machine Learning: Lessons From the ISLES Challenge

Arsany Hakim, Søren Christensen, Stefan Winzeck, et al. · *Stroke* · 2021 · <https://doi.org/10.1161/STROKEAHA.120.030696>  
Access: full text (PMC XML)

<a id="ex-hakim2021isles18-1"></a>
> the aim of the ISLES 2018 was to predict the infarction core based on CTP imaging from acute ischemic stroke patients, using DWI-based manual segmentation from magnetic resonance imaging (MRI) acquired shortly after.
>
> — `hakim2021isles18#1` · Abstract · verbatim, exact match

<a id="ex-hakim2021isles18-2"></a>
> Based on the ranking criteria, the top team’s algorithm demonstrated for average Dice score coefficient and average absolute volume difference 0.51 and 10.2 mL, respectively, outperforming the conventional threshold-based method (dice score coefficient, 0.3; volume difference, 15.3).
>
> — `hakim2021isles18#2` · Abstract · verbatim, exact match

<a id="ex-hakim2021isles18-3"></a>
> Diverse algorithms were used, almost all based on deep learning, with top-ranked approaches making use of the raw perfusion data as well as methods to synthetically generate complementary information to boost prediction performance.
>
> — `hakim2021isles18#3` · Abstract · verbatim, exact match

### [3] ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset

Moritz R. Hernandez Petzsche, Ezequiel de la Rosa, Uta Hanning, et al. · *Scientific Data* · 2022 · <https://doi.org/10.1038/s41597-022-01875-5>  
Access: full text (arXiv PDF)

<a id="ex-hernandez2022isles22-1"></a>
> This dataset comprises 400 multi-vendor MRI cases with high variability in stroke lesion size, quantity and location.
>
> — `hernandez2022isles22#1` · p1 · verbatim, exact match

### [4] DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge

Ezequiel de la Rosa, Mauricio Reyes, Sook-Lei Liew, et al. · *Nature Communications* · 2025 · <https://doi.org/10.1038/s41467-025-62373-x>  
Access: full text (publisher PDF)

<a id="ex-delarosa2025deepisles-1"></a>
> We present DeepISLES, a robust ensemble algorithm developed from top submissions to the 2022 Ischemic Stroke Lesion Segmentation challenge we organized.
>
> — `delarosa2025deepisles#1` · p1 · verbatim, PDF spacing normalised

<a id="ex-delarosa2025deepisles-2"></a>
> Neuroradiologists prefer DeepISLES’ segmentations over manual annotations in a Turing-like test.
>
> — `delarosa2025deepisles#2` · p1 · verbatim, PDF spacing normalised

### [5] ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand?

Ezequiel de la Rosa, Ruisheng Su, Mauricio Reyes, et al. · *arXiv preprint arXiv:2408.10966* · 2024 · <https://arxiv.org/abs/2408.10966>  
Access: full text (arXiv PDF)

<a id="ex-delarosa2024isles24-1"></a>
> This work introduces the Ischemic Stroke Lesion Segmentation (ISLES) 2024 challenge, which focuses on the prediction of final infarct volumes from pre-interventional acute stroke imaging and clinical data.
>
> — `delarosa2024isles24#1` · p1 · verbatim, exact match

<a id="ex-delarosa2024isles24-2"></a>
> On the hidden test set (N = 98), the top-performing model, a multimodal nnU-Net-based architecture, achieved a Dice score of 0.285 ± 0.213 and an absolute volume difference of 21.2 ± 37.2 mL, underlining the significant challenges posed by this task and the need for further advances in multimodal learning.
>
> — `delarosa2024isles24#2` · p2 · verbatim, PDF spacing normalised

<a id="ex-delarosa2024isles24-3"></a>
> The four following segmentation and task-specific, clinically relevant metrics are used: i) Dice Similarity Coefficient
>
> — `delarosa2024isles24#3` · p6 · verbatim, exact match

<a id="ex-delarosa2024isles24-4"></a>
> All three top-performing methods are based on nnU-Net [75].
>
> — `delarosa2024isles24#4` · p7 · verbatim, exact match

<a id="ex-delarosa2024isles24-5"></a>
> Unlike the winning team Kurtlab, the second-ranked team AMC-Axolots employed a cascaded nnU-Net approach rather than a single-model strategy.
>
> — `delarosa2024isles24#5` · p7 · verbatim, exact match

<a id="ex-delarosa2024isles24-6"></a>
> Notably, the third-ranked team, Ninjas, relied exclusively on the 4D CTP perfusion modality and achieved strong individual-metric performance for Dice and absolute lesion count difference, ranking second and first, respectively.
>
> — `delarosa2024isles24#6` · p7 · verbatim, exact match

<a id="ex-delarosa2024isles24-7"></a>
> Notably, the first- and second-ranked teams integrated all available imaging modalities—excluding 4D CTP—and outperformed nnU-Net-based models relying solely on single modalities, such as teams utilizing only CTP or CTA data.
>
> — `delarosa2024isles24#7` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-8"></a>
> Notably, the perfusion-based baseline models using rCBF achieved infarct volume estimates comparable to those of the best-performing deep learning methods. However, deep learning approaches outperformed in voxel-wise metrics, demonstrating superior spatial localization and identification of infarction.
>
> — `delarosa2024isles24#8` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-9"></a>
> Although one team incorporated clinically relevant variables -such as stroke severity scores at admission, patient demographics, clinical history, and critical time metrics- there was no demonstrated tangible improvement in segmentation performance through the integration of these features.
>
> — `delarosa2024isles24#9` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-10"></a>
> A standard concatenation-based fusion strategy was used, which proved insufficient to capture the complex relationships between imaging and clinical variables. Whether more sophisticated feature fusion strategies, such as attention-based mechanisms, cross- and multi-modal techniques like CLIP [93], or advanced tabular data architectures such as TabFPN [94], can enhance segmentation performance, remains an open question.
>
> — `delarosa2024isles24#10` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-11"></a>
> However, our challenge benchmark reveals two key findings: ( i) models utilizing standard deconvolution-derived perfusion maps consistently outperform those trained directly on raw 4D CTP series, and ( ii) models that extract spatial features from
>
> — `delarosa2024isles24#11` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-12"></a>
> the 4D CTP data achieve superior performance compared to those attempting to leverage the full spatiotemporal nature of the modality.
>
> — `delarosa2024isles24#12` · p18 · verbatim, exact match

<a id="ex-delarosa2024isles24-13"></a>
> Specifically, the top-performing model utilizing raw 4D CTP data (team Ninjas) selected a subset of meaningful CTP timepoints and applied a spatial convolutional nnU-Net architecture, whereas models designed to explicitly capture temporally-encoded contrast propagation dynamics consistently underperformed.
>
> — `delarosa2024isles24#13` · p18 · verbatim, exact match

<a id="ex-delarosa2024isles24-14"></a>
> We thus advocate for the development of advanced spatiotemporal models
>
> — `delarosa2024isles24#14` · p18 · verbatim, exact match

<a id="ex-delarosa2024isles24-15"></a>
> Distinct infarct growth phenotypes—ranging from slow to fast progressors—have been identified, influenced by patient-specific pathophysiological variables, particularly the status of leptomeningeal collateral circulation, which sustains residual perfusion and limits infarct expansion [15, 16, 91].
>
> — `delarosa2024isles24#15` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-16"></a>
> Moreover, technical challenges related to data preprocessing—such as image registration and interpolation across differing image spaces, resolutions, and modalities (e.g., acute CT and follow-up MRI)—further hinder accurate lesion segmentation.
>
> — `delarosa2024isles24#16` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-17"></a>
> When focusing on cases with small infarcts (infarct< 5 mL,N = 25), all methods show limited volumetric agreement (Pearsonr< 0.4).
>
> — `delarosa2024isles24#17` · p13 · verbatim, exact match

<a id="ex-delarosa2024isles24-18"></a>
> Methods based solely on CTP may be more sensitive to differences in acquisition protocols, resulting in decreased generalizability.
>
> — `delarosa2024isles24#18` · p13 · verbatim, exact match

<a id="ex-delarosa2024isles24-19"></a>
> As a more standardized and higher-resolution modality, NCCT offers a stable input source that may reduce model sensitivity to domain shifts in CTP data.
>
> — `delarosa2024isles24#19` · p18 · verbatim, exact match

<a id="ex-delarosa2024isles24-20"></a>
> Furthermore, the inclusion criteria of the ISLES’24 dataset were restricted to patients with favorable recanalization outcomes (i.e., thrombolysis in cerebral infarction -TICI- scores 2B and 3).
>
> — `delarosa2024isles24#20` · p18 · verbatim, exact match

<a id="ex-delarosa2024isles24-21"></a>
> The solution aimed to predict the final infarct by leveraging baseline infarct and penumbra masks derived from NCCT and CTA scans, as well as CTP perfusion maps (CBF, CBV, MTT, and TMAX).
>
> — `delarosa2024isles24#21` · p9 · verbatim, exact match

<a id="ex-delarosa2024isles24-22"></a>
> A sequence of 20 time points centered around the peak contrast concentration was extracted to form a 20-channel input.
>
> — `delarosa2024isles24#22` · p9 · verbatim, exact match

<a id="ex-delarosa2024isles24-23"></a>
> Lesion masks are derived from the follow-up MRI using DeepISLES [14].
>
> — `delarosa2024isles24#23` · p5 · verbatim, exact match

<a id="ex-delarosa2024isles24-24"></a>
> While Center #1 provides higher resolution NCCT images, its CTP acquisitions exhibit consistently a lower resolution compared to Center #2.
>
> — `delarosa2024isles24#24` · p13 · verbatim, exact match

<a id="ex-delarosa2024isles24-25"></a>
> Moreover, the clinical data contains demographics (age, sex, etc.), patient history, admission scores (e.g., admission NIHSS), and clinical outcomes (e.g., 3month mRS).
>
> — `delarosa2024isles24#25` · p4 · verbatim, exact match

<a id="ex-delarosa2024isles24-26"></a>
> The 4D CTP series are motion-corrected through image co-registration and temporally resampled at 1 frame/second.
>
> — `delarosa2024isles24#26` · p5 · verbatim, exact match

<a id="ex-delarosa2024isles24-27"></a>
> Training was performed with a batch size of 2 and a patch size of 28 × 256 × 192 on two NVIDIA A6000 GPUs.
>
> — `delarosa2024isles24#27` · p9 · verbatim, exact match

<a id="ex-delarosa2024isles24-28"></a>
> All scans underwent preprocessing, including voxel value clipping to specific ranges (NCCT [0, 100] Hounsfield units, CTA [0, 200] Hounsfield units, CBF [0, 400] mL/100mL/min, CBV [0, 400] mL/100mL, MTT [0, 20] s, TMAX [0, 20] s), background value setting to zero, and skull stripping.
>
> — `delarosa2024isles24#28` · p9 · verbatim, exact match

<a id="ex-delarosa2024isles24-29"></a>
> For anonymization purposes, all scans are defaced based on brain and face masks obtained with TotalSegmentator [56], and the clinical tabular data is de-identified by adding ± 5% random noise to the laboratory variables.
>
> — `delarosa2024isles24#29` · p5 · verbatim, exact match

<a id="ex-delarosa2024isles24-30"></a>
> While acute CT imaging is widely accessible and routinely used in clinical practice, it is inherently less sensitive than the gold-standard DWI and is often affected by artifacts (e.g., streak artifacts) and a low signal-to-noise ratio.
>
> — `delarosa2024isles24#30` · p17 · verbatim, exact match

<a id="ex-delarosa2024isles24-31"></a>
> (i) core tissue, defined as regions with rCBF< 30%; (ii) hypoperfused tissue, defined as regions with Tmax> 6 s
>
> — `delarosa2024isles24#31` · p13 · verbatim, exact match

<a id="ex-delarosa2024isles24-32"></a>
> The train (test) set contains N = 100 (N = 50) scans from the University Hospital of Munich and N = 50 (N = 48) scans from the University Hospital of Zurich.
>
> — `delarosa2024isles24#32` · p5 · verbatim, exact match

### [6] The Ischemic Stroke Lesion Segmentation Challenge (ISLES)’24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes

Evamaria Olga Riedel, Ezequiel de la Rosa, The Anh Baran, et al. · *Radiology: Artificial Intelligence* · 2026 · <https://doi.org/10.1148/ryai.250603>  
Access: full text of arXiv preprint (2408.11142); published version paywalled

<a id="ex-riedel2026isles24dataset-1"></a>
> Data preparation, which is briefly illustrated in Figure 1 and described in detail in the methods section, led to a dataset of 245 cases divided into a publicly available train set (n = 149) and - to ensure fair and continuous evaluation of machine learning algorithms - a hidden test set (n = 96).
>
> — `riedel2026isles24dataset#1` · p4 (arXiv v2) · verbatim, exact match

<a id="ex-riedel2026isles24dataset-2"></a>
> by providing longitudinal imaging at two time points: (sub-)acute CT imaging before therapy and follow-up MRI 2-9 days after intracranial reperfusion therapy.
>
> — `riedel2026isles24dataset#2` · p4 (arXiv v2) · verbatim, exact match

<a id="ex-riedel2026isles24dataset-3"></a>
> The current clinical standard for estimating perfusion volumes involves deconvolution analysis, which entails generating perfusion maps through perfusion CT and MR deconvolution and applying thresholds to these maps 10.
>
> — `riedel2026isles24dataset#3` · p3 (arXiv v2) · verbatim, exact match

<a id="ex-riedel2026isles24dataset-4"></a>
> Complemented is longitudinal clinical data up to a three-month outcome, which enables taking into account patient-specific characteristics like demographics, pre-existing risk factors, medication, and the type of referral or intervention times.
>
> — `riedel2026isles24dataset#4` · p4 (arXiv v2) · verbatim, exact match

### [7] How We Won the ISLES'24 Challenge by Preprocessing

Tian-Yi Ren, Juampablo Heras Rivera, Hitender K. Oswal, et al. · *arXiv preprint arXiv:2505.18424 (states PMLR 2025)* · 2025 · <https://arxiv.org/abs/2505.18424>  
Access: full text (arXiv PDF)

<a id="ex-ren2025isles24-1"></a>
> Our winning solution shows that a carefully designed preprocessing pipeline including deep-learning-based skull stripping and custom intensity windowing is beneficial for accurate segmentation.
>
> — `ren2025isles24#1` · p1 · verbatim, exact match

<a id="ex-ren2025isles24-2"></a>
> For our submission, we used a 3D large residual encoder nnUNet, referred to as “nnU-Net ResEnc L” in the nnU-Net documentation.
>
> — `ren2025isles24#2` · p2 · verbatim, exact match

<a id="ex-ren2025isles24-3"></a>
> On this fold, the standard pipeline with only Z-score normalization and [1,99] percentile windowing achieved a Dice score of 21.8%, custom windowing alone improved performance to 31.0%, and our submission combining custom windowing with Z-score normalization further increased the validation Dice score to 31.8%.
>
> — `ren2025isles24#3` · p3 · verbatim, exact match

<a id="ex-ren2025isles24-4"></a>
> Results 10-fold cross validation was performed, and the fold with the highest validation Dice score was selected for submission.
>
> — `ren2025isles24#4` · p3 · verbatim, exact match

<a id="ex-ren2025isles24-5"></a>
> Kurtlab (Ours) 28.50 (21.27) 21.23 (37.22) 14.39 (21.19) 7.18 (7.67) AMC-Axolotls 26.27 (24.73) 21.31 (35.23) 14.94 (25.12) 7.66 (7.94) Ninjas 25.46 (19.08) 26.29 (39.73) 9.92 (13.46) 5.98 (6.46)
>
> — `ren2025isles24#5` · p4 · verbatim, exact match

<a id="ex-ren2025isles24-6"></a>
> Table 1: Windowing ranges for each CT modality used as input to the nnU-Net segmentation model for the final submission to the ISLES’24 challenge.
>
> — `ren2025isles24#6` · p3 · verbatim, exact match

<a id="ex-ren2025isles24-7"></a>
> Modality CTA (HU) CBF (mL/100g/min) CBV (mL/100g) MTT (s) Tmax (s) Window Range (0, 90) (0, 35) (0, 10) (0, 20) (0, 7)
>
> — `ren2025isles24#7` · p3 · verbatim, exact match

<a id="ex-ren2025isles24-8"></a>
> First, we applied SynthStrip on the non-contrast CT (NCCT) scans to obtain a brain masks.
>
> — `ren2025isles24#8` · p2 · verbatim, exact match

<a id="ex-ren2025isles24-9"></a>
> Then, we applied this brain mask to the other co-registered scans (CTP, CTA, etc.) to obtain skull-stripped versions of the data.
>
> — `ren2025isles24#9` · p2 · verbatim, exact match

<a id="ex-ren2025isles24-10"></a>
> When clinical windowing guidelines were available in the literature, they informed the initial settings, with empirical adjustments made to optimize model performance.
>
> — `ren2025isles24#10` · p3 · verbatim, exact match

### [8] Utilizing Baseline Infarct and Penumbra Masks to Improve NCCT and CTA Based Final Infarct Prediction

Mahsa Mojtahedi, Laura van Poppel, Lucas de Vries, et al. · *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science* · 2025 · <https://doi.org/10.1007/978-3-031-81101-2_10>  
Access: abstract only (Amsterdam UMC repository page; PDF blocked)

<a id="ex-mojtahedi2025switch-1"></a>
> Using the nnU-Net framework, we observe a 16% improvement in Dice when baseline lesion masks are added in the absence of CTP. However, when CTP maps are included, the addition of baseline lesion masks is no longer beneficial.
>
> — `mojtahedi2025switch#1` · Abstract · verbatim, exact match

### [9] Spatio-Temporal Deep Learning for Final Infarct Prediction Using Acute Stroke CT Perfusion Data

Kimberly Amador, Anthony J. Winder, Nils D. Forkert · *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science* · 2025 · <https://doi.org/10.1007/978-3-031-81101-2_9>  
Access: abstract only (authors' GitHub README; chapter paywalled)

<a id="ex-amador2025switch-1"></a>
> The model integrates a shared-weight convolutional neural network (CNN) encoder, a Transformer encoder, and a CNN decoder to capture both spatial and temporal dependencies within the data. We evaluated this approach on a multicenter dataset of 143 patients from the ISLES 2024 challenge.
>
> — `amador2025switch#1` · Abstract · verbatim, exact match

<a id="ex-amador2025switch-2"></a>
> The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.
>
> — `amador2025switch#2` · Abstract · verbatim, exact match

### [10] Automatic Segmentation of Ischaemic Stroke Lesions Using Transformers and Convolutional Neural Networks Applied to Multimodal Neuroimaging

Pablo Martínez Cegarra, Juan Francisco Zapata Pérez, Juan Martínez-Alajarín · *Sensors* · 2026 · <https://doi.org/10.3390/s26165021>  
Access: full text (PMC XML)

<a id="ex-cegarra2026isles-1"></a>
> The main limitation of drawing conclusions from comparisons derived from the challenge leaderboard is that the results of the competing strategies are subject to confounding variables such as differences in preprocessing protocols, training data volume, and cross-validation strategy, making it difficult to attribute performance gaps to architectural choice alone.
>
> — `cegarra2026isles#1` · Introduction · verbatim, exact match

<a id="ex-cegarra2026isles-2"></a>
> The cascade model achieved the highest Dice Similarity Coefficient (0.224).
>
> — `cegarra2026isles#2` · Abstract · verbatim, exact match

<a id="ex-cegarra2026isles-3"></a>
> On the other hand, SegFormer3D returned the lowest overall metrics (DSC 0.163, AVD 27.28 mL, F1 2.30%).
>
> — `cegarra2026isles#3` · Abstract · verbatim, exact match

<a id="ex-cegarra2026isles-4"></a>
> Within these limits, the local inductive bias of the convolutional models retained an empirical advantage over the single Transformer evaluated when processing this moderately sized neuroimaging dataset, and complex cascade topologies offered only marginal gains compared with a well-calibrated single-stage network.
>
> — `cegarra2026isles#4` · Abstract · verbatim, exact match

<a id="ex-cegarra2026isles-5"></a>
> This section presents the performance results of each of the trained models when evaluated on the held-out test set of 15 patients.
>
> — `cegarra2026isles#5` · Results · verbatim, exact match

<a id="ex-cegarra2026isles-6"></a>
> It must be stated explicitly that the two frameworks used (nnU-Net for M1 and M2, MONAI/SegFormer3D for M3) do not share every training-time setting
>
> — `cegarra2026isles#6` · Methods · verbatim, exact match

<a id="ex-cegarra2026isles-7"></a>
> the only pure Transformer entry (QuiiL, using the MoReT architecture) obtained 0.068 [7].
>
> — `cegarra2026isles#7` · Related work · verbatim, exact match

<a id="ex-cegarra2026isles-8"></a>
> The first of these is a single-stage 5-channel nnU-Net, followed by a two-stage cascaded nnU-Net (2 and 7 channels) and a lightweight Transformer (SegFormer3D).
>
> — `cegarra2026isles#8` · Abstract · verbatim, exact match

<a id="ex-cegarra2026isles-9"></a>
> (i) a single-stage nnU-Net approach replicating the preprocessing pipeline of the challenge-winning team under controlled experimental conditions, using 5 input channels (CTA, CBF, CBV, MTT, Tmax)
>
> — `cegarra2026isles#9` · Introduction · verbatim, exact match

<a id="ex-cegarra2026isles-10"></a>
> First, morphological skull-stripping is applied to the NCCT of each patient by thresholding to (0,120) HU, binary erosion (3 iterations), selection of the largest 3D connected component with the cc3d library [42], dilation (5 iterations) and filling of internal holes.
>
> — `cegarra2026isles#10` · Methods · verbatim, exact match

### [11] Ischemic Stroke Segmentation and Net Water Uptake Quantification on Multicenter Non-Contrast CT Using Supervised Target-Domain Adaptation

Linus Britt, M. Nielsen, S. Klapproth, et al. · *arXiv preprint arXiv:2609.20151* · 2026 · <https://arxiv.org/abs/2609.20151>  
Access: full text (arXiv PDF)

<a id="ex-britt2026ncct-1"></a>
> Including smaller lesions, which predominated in ISLES, median Dice was 0.54 (interquartile range [IQR] 0.30 – 0.70) for acute lesion segmentation (Boston dataset) and 0.20 (IQR 0.03–0.41) for NCCT lesion segmentations when compared to post-treatment infarct (primary target of the ISLES challenge).
>
> — `britt2026ncct#1` · p2 · verbatim, exact match

<a id="ex-britt2026ncct-2"></a>
> To adapt to new domains, the model was fine-tuned on target-domain subsets from Boston (n=11) and ISLES (n=75), with evaluation on held-out cases not used for fine-tuning.
>
> — `britt2026ncct#2` · p2 · verbatim, PDF spacing normalised

### [12] Benchmarking DINOv3 for Multi-Task Stroke Analysis on Non-Contrast CT

Dong-Hao Zhang, Yimin Chen, K. Duarte, et al. · *arXiv preprint arXiv:2509.23132* · 2025 · <https://arxiv.org/abs/2509.23132>  
Access: full text (arXiv PDF)

<a id="ex-zhang2025dinov3stroke-1"></a>
> We also extracted the labeled slices from this dataset to form our dataset. Finally, we compiled 3982 slices, randomly dividing them into 3185 for training and 797 for testing.
>
> — `zhang2025dinov3stroke#1` · p6 · verbatim, exact match

<a id="ex-zhang2025dinov3stroke-2"></a>
> On ISLES24, FPN+Refine with layers [2,5,8,11] yields the highest Dice of 0.5678
>
> — `zhang2025dinov3stroke#2` · p10 · verbatim, exact match

<a id="ex-zhang2025dinov3stroke-3"></a>
> Among them, DINOv3 [17] represents a state-of-the-art vision transformer, pretrained on more than 1.6 billion images, and capable of extracting high-fidelity, fine-grained dense features.
>
> — `zhang2025dinov3stroke#3` · p2 · verbatim, exact match

<a id="ex-zhang2025dinov3stroke-4"></a>
> This work presents a unified, reproducible evaluation of a frozen self-supervised transformer backbone (DINOv3) on NCCT-based stroke analysis
>
> — `zhang2025dinov3stroke#4` · p11 · verbatim, PDF spacing normalised

### [13] AsymFeX: A Symmetry-Driven Framework for Ischemic Stroke Segmentation Across Imaging Modalities and Stroke Stages

M. Shah, V. Sundaresan · *arXiv preprint arXiv:2608.19769* · 2026 · <https://arxiv.org/abs/2608.19769>  
Access: full text (arXiv PDF)

<a id="ex-shah2026asymfex-1"></a>
> ISLES’24 targets prediction of postinterventional infarct from pre-interventional CT, but since infarcts evolve after the initial scan, the released masks do not reflect the pre-interventional state. An expert neuroradiologist therefore manually re-delineated pre-interventional ischemic core and penumbra on the CT perfusion scans using ITK-SNAP [31].
>
> — `shah2026asymfex#1` · p8 · verbatim, exact match

<a id="ex-shah2026asymfex-2"></a>
> NCCT alone collapses to near-failing performance at this timepoint regardless of architecture, underscoring that imaging modality, not model design, is the binding constraint at early stroke stages.
>
> — `shah2026asymfex#2` · p14 · verbatim, exact match

<a id="ex-shah2026asymfex-3"></a>
> A 2-stage 3D segmentation pipeline that integrates with nnUNet and incorporates stroke-specific prior knowledge via an explicit geometric tiltcorrection step
>
> — `shah2026asymfex#3` · p2 · verbatim, exact match

<a id="ex-shah2026asymfex-4"></a>
> we incorporate CT perfusion parameter maps, including Cerebral Blood Volume (CBV), Cerebral Blood Flow (CBF)
>
> — `shah2026asymfex#4` · p8 · verbatim, exact match

### [14] Multimodal CT Perfusion–Based Deep Learning for Predicting Stroke Lesion Outcomes in Complete and No Recanalization Scenarios

Hongxi Yang, Yasmeen George, Deval Mehta, et al. · *American Journal of Neuroradiology* · 2026 · <https://doi.org/10.3174/ajnr.A9016>  
Access: abstract only (PubMed)

<a id="ex-ajnr2026recanalization-1"></a>
> The multimodal 3D nnU-Net model demonstrated superior performance, achieving mean Dice scores of 35.36% in patients with CR and 50.22% in those with NR. This model substantially outperformed the current clinically used method, providing more accurate outcome estimates than the conventional single-technique threshold-based measures, which yielded Dice scores of 15.73% and 39.71% for CR and NR groups, respectively.
>
> — `ajnr2026recanalization#1` · Abstract (Results) · verbatim, exact match

<a id="ex-ajnr2026recanalization-2"></a>
> evaluated them separately for patients with complete recanalization (successful reperfusion [CR], n = 350) and no recanalization (unsuccessful reperfusion [NR], n = 138) after treatment.
>
> — `ajnr2026recanalization#2` · Abstract (Methods) · verbatim, exact match

### [15] nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation

Fabian Isensee, Tassilo Wald, Constantin Ulrich, et al. · *MICCAI 2024 (LNCS)* · 2024 · <https://arxiv.org/abs/2404.09556>  
Access: full text (arXiv PDF)

<a id="ex-isensee2024revisited-1"></a>
> In contrast to current beliefs, we find that the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models, including ResNet and ConvNeXt variants, 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources.
>
> — `isensee2024revisited#1` · p1 · verbatim, exact match

<a id="ex-isensee2024revisited-2"></a>
> In this study, we demonstrate that many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources.
>
> — `isensee2024revisited#2` · p1 · verbatim, exact match

### [16] Prediction of final infarct volume from native CT perfusion and treatment parameters using deep learning

David Robben, Anna M.M. Boers, Henk A. Marquering, et al. · *Medical Image Analysis* · 2020 · <https://doi.org/10.1016/j.media.2019.101589>  
Access: full text (arXiv PDF)

<a id="ex-robben2020nativectp-1"></a>
> We pursue a data-driven and deconvolution-free approach, where a deep neural network learns to predict the final infarct volume directly from the native CTP images and metadata such as the time parameters and treatment.
>
> — `robben2020nativectp#1` · p1 · verbatim, exact match

<a id="ex-robben2020nativectp-2"></a>
> An ablation study shows that using the native CTP measurements instead of the deconvolved measurements improves the prediction.
>
> — `robben2020nativectp#2` · p1 · verbatim, exact match

<a id="ex-robben2020nativectp-3"></a>
> Additionally, deconvolution is a mathematically ill-posed problem and that is problematic given the low signal to noise ratio of perfusion images (Fieselmann et al., 2011).
>
> — `robben2020nativectp#3` · p2 · verbatim, exact match

<a id="ex-robben2020nativectp-4"></a>
> This results in 188 included subjects.
>
> — `robben2020nativectp#4` · p6 · verbatim, exact match

### [17] Predicting treatment-specific lesion outcomes in acute ischemic stroke from 4D CT perfusion imaging using spatio-temporal convolutional neural networks

Kimberly Amador, Matthias Wilms, Anthony Winder, et al. · *Medical Image Analysis* · 2022 · <https://doi.org/10.1016/j.media.2022.102610>  
Access: abstract only (PubMed)

<a id="ex-amador2022spatiotemporal-1"></a>
> The proposed method was evaluated on 147 patients using a 10-fold cross validation, which demonstrated that the proposed 3D+time model (mean Dice=0.45) significantly outperforms both a 2D+time variant of our approach (mean Dice=0.43) and a state-of-the-art method that uses perfusion maps (mean Dice=0.38).
>
> — `amador2022spatiotemporal#1` · Abstract · verbatim, exact match

<a id="ex-amador2022spatiotemporal-2"></a>
> By merging a U-Net-like architecture with temporal convolutional networks, we efficiently process the spatio-temporal information available in CTP datasets to make a tissue outcome prediction.
>
> — `amador2022spatiotemporal#2` · Abstract · verbatim, exact match

### [18] PerfU-Net: Baseline infarct estimation from CT perfusion source data for acute ischemic stroke

Lucas de Vries, Bart J. Emmer, Charles B.L.M. Majoie, et al. · *Medical Image Analysis* · 2023 · <https://doi.org/10.1016/j.media.2023.102749>  
Access: abstract only (PubMed)

<a id="ex-devries2023perfunet-1"></a>
> The outcomes and perfusion maps of CT perfusion software, however, show many discrepancies between vendors.
>
> — `devries2023perfunet#1` · Abstract · verbatim, exact match

<a id="ex-devries2023perfunet-2"></a>
> To this end, we present a symmetry-aware spatio-temporal segmentation model that encodes the micro-perfusion dynamics in the brain, while decoding a static segmentation map for infarct core assessment.
>
> — `devries2023perfunet#2` · Abstract · verbatim, exact match

<a id="ex-devries2023perfunet-3"></a>
> We achieve state-of-the-art results compared to methods that only use CT perfusion source imaging with a Dice score of 0.46. We are almost on par with methods that use perfusion maps from third party software
>
> — `devries2023perfunet#3` · Abstract · verbatim, exact match

### [19] nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation

Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, et al. · *Nature Methods* · 2021 · <https://doi.org/10.1038/s41592-020-01008-z>  
Access: abstract only (PubMed)

<a id="ex-isensee2021nnunet-1"></a>
> We developed nnU-Net, a deep learning-based segmentation method that automatically configures itself, including preprocessing, network architecture, training and post-processing for any new task.
>
> — `isensee2021nnunet#1` · Abstract · verbatim, exact match

### [20] Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images

Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, et al. · *BrainLes 2021 (MICCAI workshop), Lecture Notes in Computer Science* · 2022 · <https://doi.org/10.1007/978-3-031-08999-2_22>  
Access: full text (arXiv PDF)

<a id="ex-hatamizadeh2022swinunetr-1"></a>
> The swin transformer encoder extracts features at five different resolutions by utilizing shifted windows for computing self-attention and is connected to an FCNN-based decoder at each resolution via skip connections.
>
> — `hatamizadeh2022swinunetr#1` · p1 · verbatim, exact match

### [21] SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation

Shehan Perera, Pouyan Navard, Alper Yilmaz · *CVPR Workshops (CVPRW) 2024* · 2024 · <https://doi.org/10.1109/CVPRW63382.2024.00503>  
Access: full text (arXiv PDF)

<a id="ex-perera2024segformer3d-1"></a>
> SegFormer3D democratizes deep learning for 3D medical image segmentation by offering a model with 33× less parameters and a 13× reduction in GFLOPS compared to the current state-of-theart (SOTA).
>
> — `perera2024segformer3d#1` · p1 · verbatim, exact match

### [22] VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

Zhan Tong, Yi-Bing Song, Jue Wang, et al. · *NeurIPS 2022* · 2022 · <https://arxiv.org/abs/2203.12602>  
Access: full text (arXiv PDF)

<a id="ex-tong2022videomae-1"></a>
> (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data.
>
> — `tong2022videomae#1` · p1 · verbatim, exact match

<a id="ex-tong2022videomae-2"></a>
> The temporally redundant video content enables higher masking ratio than that of images.
>
> — `tong2022videomae#2` · p1 · verbatim, exact match

<a id="ex-tong2022videomae-3"></a>
> We adopt the joint space-time cube embedding [3, 22, 39] in our VideoMAE, where we treat each cube of size 2× 16× 16 as one token embedding.
>
> — `tong2022videomae#3` · p4 · verbatim, PDF spacing normalised

<a id="ex-tong2022videomae-4"></a>
> we perform in-depth ablation studies on VideoMAE design with the default backbone of 16-frame ViT-B on Something-Something V2 (SSV2) and Kinetics-400 (K400).
>
> — `tong2022videomae#4` · p5 · verbatim, exact match

### [23] Video Swin Transformer

Ze Liu, Jia Ning, Yue Cao, et al. · *CVPR 2022* · 2022 · <https://doi.org/10.1109/CVPR52688.2022.00320>  
Access: full text (arXiv PDF)

<a id="ex-liu2022videoswin-1"></a>
> In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization.
>
> — `liu2022videoswin#1` · p1 · verbatim, exact match

<a id="ex-liu2022videoswin-2"></a>
> spatiotemporal locality of videos, in which pixels that are closer to each other in spatiotemporal distance are more likely to be correlated.
>
> — `liu2022videoswin#2` · p2 · verbatim, exact match

### [24] Is Space-Time Attention All You Need for Video Understanding?

Gedas Bertasius, Heng Wang, L. Torresani · *ICML 2021* · 2021 · <https://arxiv.org/abs/2102.05095>  
Access: full text (arXiv PDF)

<a id="ex-bertasius2021timesformer-1"></a>
> Our experimental study compares different self-attention schemes and suggests that “divided attention,” where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered.
>
> — `bertasius2021timesformer#1` · p1 · verbatim, exact match

### [25] Revisiting Feature Prediction for Learning Visual Representations from Video

Adrien Bardes, Q. Garrido, Jean Ponce, et al. · *Transactions on Machine Learning Research (TMLR) 2024* · 2024 · <https://arxiv.org/abs/2404.08471>  
Access: full text (arXiv PDF)

<a id="ex-bardes2024vjepa-1"></a>
> Our results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters; e.g., using a frozen backbone.
>
> — `bardes2024vjepa#1` · p1 · verbatim, exact match

### [26] SAM 2: Segment Anything in Images and Videos

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. · *ICLR 2025* · 2025 · <https://arxiv.org/abs/2408.00714>  
Access: full text (arXiv PDF)

<a id="ex-ravi2024sam2-1"></a>
> Our model is a simple transformer architecture with streaming memory for real-time video processing.
>
> — `ravi2024sam2#1` · p1 · verbatim, exact match

<a id="ex-ravi2024sam2-2"></a>
> For images, the SA task can be considered a subset of PVS with the video reduced to a single frame.
>
> — `ravi2024sam2#2` · p15 · verbatim, exact match

<a id="ex-ravi2024sam2-3"></a>
> SAM 2 is capable of interactively segmenting regions through prompts (clicks, boxes, or masks) on one or multiple video frames by utilizing a streaming memory that stores previous prompts and predictions.
>
> — `ravi2024sam2#3` · p2 · verbatim, exact match

### [27] Medical SAM 2: Segment medical images as video via Segment Anything Model 2

Jia-Yuan Zhu, Yunli Qi, Jun-De Wu · *arXiv preprint arXiv:2408.00874* · 2024 · <https://arxiv.org/abs/2408.00874>  
Access: full text (arXiv PDF)

<a id="ex-zhu2024medicalsam2-1"></a>
> The core concept is to leverage the Segment Anything Model 2 (SAM2) pipeline to treat all 2D and 3D medical segmentation tasks as a video object tracking problem.
>
> — `zhu2024medicalsam2#1` · p1 · verbatim, exact match

<a id="ex-zhu2024medicalsam2-2"></a>
> To put it into practice, we propose a novel self-sorting memory bank mechanism that dynamically selects informative embeddings based on confidence and dissimilarity, regardless of temporal order.
>
> — `zhu2024medicalsam2#2` · p1 · verbatim, exact match

### [28] MedSAM2: Segment Anything in 3D Medical Images and Videos

Jun Ma, Zongxin Yang, Sumin Kim, et al. · *arXiv preprint arXiv:2504.03600* · 2025 · <https://arxiv.org/abs/2504.03600>  
Access: full text (arXiv PDF)

<a id="ex-ma2025medsam2-1"></a>
> The model is developed by fine-tuning the Segment Anything Model 2 on a large medical dataset with over 455,000 3D image-mask pairs and 76,000 frames, outperforming previous models across a wide range of organs, lesions, and imaging modalities.
>
> — `ma2025medsam2#1` · p1 · verbatim, exact match

<a id="ex-ma2025medsam2-2"></a>
> To reuse the pre-trained model weights and avoid prohibitive computing costs, MedSAM2 adopted the SAM2 network architecture, including an image encoder, a memory attention module, a prompt encoder, and a mask decoder (Fig 1b).
>
> — `ma2025medsam2#2` · p2 · verbatim, exact match

<a id="ex-ma2025medsam2-3"></a>
> involving the annotation of 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames, demonstrating that MedSAM2 can reduce manual costs by more than 85%.
>
> — `ma2025medsam2#3` · p1 · verbatim, exact match

### [29] Video Pretraining Advances 3D Deep Learning on Chest CT Tasks

Alexander Ke, Shih-Cheng Huang, Chloe O'Connell, et al. · *MIDL 2023 (PMLR)* · 2023 · <https://arxiv.org/abs/2304.00546>  
Access: full text (arXiv PDF)

<a id="ex-ke2023videopretraining-1"></a>
> We demonstrate video pretraining improves the average performance of seven 3D models on two chest CT datasets, regardless of finetuning dataset size, and that video pretraining allows 3D models to outperform 2D baselines.
>
> — `ke2023videopretraining#1` · p1 · verbatim, exact match

<a id="ex-ke2023videopretraining-2"></a>
> Lastly, we observe that pretraining on the large-scale out-of-domain Kinetics dataset improves performance more than pretraining on a typically-sized in-domain CT dataset.
>
> — `ke2023videopretraining#2` · p1 · verbatim, exact match

<a id="ex-ke2023videopretraining-3"></a>
> We then transferred these weights and finetuned them on RSNA for PE detection and LIDC for lung nodule detection.
>
> — `ke2023videopretraining#3` · p2 · verbatim, exact match

### [30] Segment anything in medical images

Jun Ma, Yuting He, Feifei Li, et al. · *Nature Communications* · 2024 · <https://doi.org/10.1038/s41467-024-44824-z>  
Access: full text (arXiv PDF)

<a id="ex-ma2024medsam-1"></a>
> However, the model exhibited substantial limitations in segmenting typical medical targets with weak boundaries or low contrast.
>
> — `ma2024medsam#1` · p2 · verbatim, exact match

<a id="ex-ma2024medsam-2"></a>
> Essentially, SAM is a promptable segmentation method that requires points or bounding boxes to specify the segmentation targets.
>
> — `ma2024medsam#2` · p2 · verbatim, exact match

### [31] SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images

Haoyu Wang, Sizheng Guo, Jin Ye, et al. · *ECCV Workshops, Lecture Notes in Computer Science* · 2025 · <https://doi.org/10.1007/978-3-031-91721-9_4>  
Access: full text (arXiv PDF)

<a id="ex-wang2023sammed3d-1"></a>
> Given only a few 3D prompt points, SAM-Med3D can accurately segment diverse anatomical structures and lesions across various modalities.
>
> — `wang2023sammed3d#1` · p1 · verbatim, exact match

### [32] Large-vocabulary segmentation for medical images with text prompts

Ziheng Zhao, Yao Zhang, Chaoyi Wu, et al. · *npj Digital Medicine* · 2025 · <https://doi.org/10.1038/s41746-025-01964-w>  
Access: full text (arXiv PDF)

<a id="ex-zhao2023sat-1"></a>
> However, while transferring to medical scenarios, these models trained on natural images suffer from fundamental limitations: (i) models typically perform 2D slice segmentation, which is later fused into 3D volumes through post-processing.
>
> — `zhao2023sat#1` · p1 · verbatim, exact match

<a id="ex-zhao2023sat-2"></a>
> SAT-Pro achieves comparable performance to 72 nnU-Nets—the strongest specialist models trained on each dataset (over 2.2B parameters combined)—over 497 categories.
>
> — `zhao2023sat#2` · p1 · verbatim, PDF spacing normalised

### [33] Generalist foundation models from a multimodal dataset for 3D computed tomography

Ibrahim Ethem Hamamci, Sezgin Er, Chenyu Wang, et al. · *Nature Biomedical Engineering* · 2026 · <https://doi.org/10.1038/s41551-025-01599-y>  
Access: full text (arXiv PDF)

<a id="ex-hamamci2024ctclip-1"></a>
> CT-RATE comprises 25,692 non-contrast 3D chest CT scans from 21,304 unique patients.
>
> — `hamamci2024ctclip#1` · p1 · verbatim, exact match

### [34] Merlin: a computed tomography vision–language foundation model and dataset

Louis Blankemeier, Ashwin Kumar, Joseph Paul Cohen, et al. · *Nature* · 2026 · <https://doi.org/10.1038/s41586-026-10181-8>  
Access: full text (arXiv PDF)

<a id="ex-blankemeier2024merlin-1"></a>
> To overcome these shortcomings for abdominal CT interpretation, we introduce Merlin - a 3D VLM that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining without requiring additional manual annotations.
>
> — `blankemeier2024merlin#1` · p1 · verbatim, PDF spacing normalised

<a id="ex-blankemeier2024merlin-2"></a>
> Recent concurrent research has started to investigate 3D vision language models for radiology, particularly focusing on chest and head CT scans.79, 67 However, the extent of experiments and 3D evaluations are limited.
>
> — `blankemeier2024merlin#2` · p9 · verbatim, exact match

### [35] M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models

Fan Bai, Yu-Xin Du, Tiejun Huang, et al. · *arXiv preprint arXiv:2404.00578* · 2024 · <https://arxiv.org/abs/2404.00578>  
Access: full text (arXiv PDF)

<a id="ex-bai2024m3d-1"></a>
> However, previous research has primarily focused on 2D medical images, leaving 3D images under-explored, despite their richer spatial information.
>
> — `bai2024m3d#1` · p1 · verbatim, exact match

### [36] Towards Generalist Foundation Model for Radiology

Chaoyi Wu, Xiaoman Zhang, Ya Zhang, et al. · *arXiv preprint arXiv:2308.02463* · 2023 · <https://arxiv.org/abs/2308.02463>  
Access: full text (arXiv PDF)

<a id="ex-wu2023radfm-1"></a>
> However, the limitation can also be observed that the reasoning results are still general and more like background medical knowledge, yet not specific to the input case.
>
> — `wu2023radfm#1` · p25 · verbatim, exact match

### [37] Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation

Cheng-Yi Li, Kao-Jung Chang, Cheng-Fu Yang, et al. · *Nature Communications* · 2025 · <https://doi.org/10.1038/s41467-025-57426-0>  
Access: full text (arXiv PDF)

<a id="ex-li2025braingpt-1"></a>
> Toward deploying MLLM for more applicable diagnostic context, we noticed that the (1) scarcity of 3D image training dataset, (2) the direct use of undifferentiated foundation MLLMs, and (3) the lack of pertinent caption evaluation metrics were independent domain-specific constraints that integratively hobbles the iteration of next-generation medical MLLM research.
>
> — `li2025braingpt#1` · p1 · verbatim, exact match

<a id="ex-li2025braingpt-2"></a>
> In this regard, this study collected a 3D-BrainCT dataset (18,885 text-scan pairs) and applied clinical visual instruction tuning (CVIT) to train volumetric anatomy-sensible BrainGPT models to generate radiology-adherent 3D brain CT reports.
>
> — `li2025braingpt#2` · p1 · verbatim, exact match

### [38] Automatic prediction of stroke treatment outcomes: latest advances and perspectives

Zeynel A. Samak, Philip Clatworthy, Majid Mirmehdi · *Biomedical Engineering Letters* · 2025 · <https://doi.org/10.1007/s13534-025-00462-y>  
Access: full text (arXiv PDF)

<a id="ex-samak2024review-1"></a>
> This article provides a broad review of recent advances and applications of deep learning in the prediction of stroke outcomes, including (i) the data and models used, (ii) the prediction tasks and measures of success, (iii) the current challenges and limitations, and (iv) future directions and potential benefits.
>
> — `samak2024review#1` · p1 · verbatim, exact match

<a id="ex-samak2024review-2"></a>
> Hence, there is also a lack of such data for researchers to apply today’s enhanced language models [11, 12] for outcome predictions.
>
> — `samak2024review#2` · p3 · verbatim, exact match

### [39] CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection

Jie Liu, Yixiao Zhang, Jie-Neng Chen, et al. · *ICCV 2023* · 2023 · <https://doi.org/10.1109/ICCV51070.2023.01934>  
Access: full text (arXiv PDF)

<a id="ex-liu2023clipdriven-1"></a>
> To address these issues, we propose the CLIP-Driven Universal Model, which incorporates text embedding learned from Contrastive Language-Image Pre-training (CLIP) to segmentation models.
>
> — `liu2023clipdriven#1` · p1 · verbatim, exact match

<a id="ex-liu2023clipdriven-2"></a>
> We first concatenate the CLIP embedding (wk) and the global image feature (f) and then input it to a multi-layer perceptron (MLP), namely text-based controller [71], to generate parameters (θk), i.e., θk = MLP(wk⊕f ), where ⊕ is the concatenation.
>
> — `liu2023clipdriven#2` · p4 · verbatim, exact match

### [40] Final Stroke Infarct Segmentation Using Deep Neural Networks

Luan Matheus Trindade Dalmazo, Kumaradevan Punithakumar · *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science* · 2025 · <https://doi.org/10.1007/978-3-031-81101-2_11>  
Access: not accessible (paywalled; no abstract in Semantic Scholar/OpenAlex; publisher blocks automated access) — listed only, not cited for claims

*No accessible text. Listed for completeness and not cited for any claim.*

### [41] A Multi-modal Deep Learning Framework for Final Infarct Prediction in Acute Ischemic Stroke: Combining CTA, NCCT, and Clinical Data

Eneko Uruñuela, Annabella Bregazzi, Matthias Wilms · *Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), Lecture Notes in Computer Science* · 2025 · <https://doi.org/10.1007/978-3-031-81101-2_12>  
Access: not accessible (paywalled; no abstract in Semantic Scholar/OpenAlex; publisher blocks automated access) — listed only, not cited for claims

*No accessible text. Listed for completeness and not cited for any claim.*

<a id="screening"></a>

## C. Screening log: excluded candidates

Inclusion rules: R1, reports on ISLES'24 or describes it; R2, predicts final infarct or core from acute CT/CTP; R3, original source of a method we rely on; R4, evidence for a specific claim.

| Candidate | Found via | Reason for exclusion |
|---|---|---|
| Isensee et al. 2018, nnU-Net: Self-adapting Framework (arXiv:1809.10486) | keyword search | superseded by Isensee et al. 2021 (Nature Methods) |
| Gava et al. 2021, Neural Network-derived perfusion maps (arXiv:2101.05992) | arXiv search | off-task: synthesises perfusion maps, does not predict final infarct |
| Bertels et al. 2022, Final infarct prediction in acute ischemic stroke (arXiv:2211.04850) | arXiv search | tutorial overview; primary sources (Robben 2020) cited instead |
| Deep Learning-Based Automatic Segmentation of Ischemic Stroke Lesions in CT Perfusion Imaging (Biomimetics 2026, PMC13204576) | web search | off-task: ISLES 2018 core segmentation only |
| Early Prediction of Ischaemic Stroke Outcomes: A Novel Computational Approach (bioRxiv 2024) | forward citation of ISLES'24 | off-task: biomechanical brain-swelling model, not infarct prediction |
| Infarct-volume prognostic value depends on outcome ascertainment... (Eur J Radiol 2026) | forward citation of ISLES'24 | off-task: functional-outcome prognosis from MRI datasets |
| TopCoW challenge (NEJM AI 2023, arXiv:2312.17670) | forward citation of ISLES'24 | off-task: vessel segmentation |
| Albers et al. 2018 (DEFUSE 3), Saver 2006, Feigin 2025 | problem statement | clinical background; cited in problem_statement.md, not methods literature |

<a id="verification"></a>

## D. Verification

Each source was downloaded and its text extracted locally from the arXiv or publisher PDF, PubMed Central XML, or the PubMed abstract. Excerpts were copied from that text, not paraphrased by a model. A script then checked each excerpt against the source after Unicode normalisation, ligature expansion, re-joining of words hyphenated across line breaks, and whitespace collapsing.

**119** excerpts matched exactly. **9** matched once all whitespace was ignored: the PDF extractor had split or merged words, for example "lesion o utcome". In those cases the characters are identical. **0** failed.

What verification does not cover: it proves that each quote exists in its source. It does not prove that the surrounding sentence in this review characterises the quote fairly, so judge that by reading the quote. Two sources are abstracts taken from non-publisher pages: Mojtahedi et al. from the Amsterdam UMC repository, and Amador et al. 2024 from the authors' GitHub README. The ISLES'24 dataset paper is quoted from its arXiv preprint (2408.11142), because the Radiology: AI version is paywalled.

