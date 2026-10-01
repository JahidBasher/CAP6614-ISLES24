"""Builds excerpts.json: paper metadata + verbatim excerpts + screening log.
Excerpts are copied from text/<source>.txt (extracted from the downloaded PDFs / PMC XML / abstracts)
and checked by verify.py."""
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import json

META = json.load(open('data/cache/meta_raw.json'))

# key: (section, rule, access, text_file, url, venue_override, year_override)
P = {}
def paper(key, section, rule, access, url, venue=None, year=None, text=None, title=None):
    P[key] = dict(section=section, rule=rule, access=access, url=url, venue_override=venue,
                  year_override=year, text_file=text or key, title_override=title, excerpts=[])

def ex(key, loc, quote, topic):
    p = P[key]
    p['excerpts'].append(dict(id=f"{key}#{len(p['excerpts'])+1}", loc=loc, quote=quote, topic=topic))

# ---------------- A. Benchmarks ----------------
paper('winzeck2018isles', 'A', 'R1/R2: ISLES 2016-17 outcome-prediction benchmark', 'full text (publisher PDF)', 'https://doi.org/10.3389/fneur.2018.00679')
ex('winzeck2018isles', 'p1', 'ISLES 2016 and 2017 focused on lesion outcome prediction after ischemic stroke', 'benchmark history')
ex('winzeck2018isles', 'p1', 'Top ranked teams almost always employed deep learning tools, which were predominately convolutional neural networks (CNNs). Despite the great efforts, lesion outcome prediction persists challenging.', 'benchmark history')

paper('hakim2021isles18', 'A', 'R2: ISLES 2018 CTP core benchmark', 'full text (PMC XML)', 'https://doi.org/10.1161/STROKEAHA.120.030696')
ex('hakim2021isles18', 'Abstract', 'the aim of the ISLES 2018 was to predict the infarction core based on CTP imaging from acute ischemic stroke patients, using DWI-based manual segmentation from magnetic resonance imaging (MRI) acquired shortly after.', 'benchmark history')
ex('hakim2021isles18', 'Abstract', "Based on the ranking criteria, the top team’s algorithm demonstrated for average Dice score coefficient and average absolute volume difference 0.51 and 10.2 mL, respectively, outperforming the conventional threshold-based method (dice score coefficient, 0.3; volume difference, 15.3).", 'DL vs thresholds')
ex('hakim2021isles18', 'Abstract', 'Diverse algorithms were used, almost all based on deep learning, with top-ranked approaches making use of the raw perfusion data as well as methods to synthetically generate complementary information to boost prediction performance.', 'native CTP')

paper('hernandez2022isles22', 'A', 'R1 (context): ISLES 2022 MRI segmentation dataset', 'full text (arXiv PDF)', 'https://doi.org/10.1038/s41597-022-01875-5')
ex('hernandez2022isles22', 'p1', 'This dataset comprises 400 multi-vendor MRI cases with high variability in stroke lesion size, quantity and location.', 'benchmark history')

paper('delarosa2025deepisles', 'A', 'R1: DeepISLES generated the ISLES\'24 label masks', 'full text (publisher PDF)', 'https://doi.org/10.1038/s41467-025-62373-x')
ex('delarosa2025deepisles', 'p1', 'We present DeepISLES, a robust ensemble algorithm developed from top submissions to the 2022 Ischemic Stroke Lesion Segmentation challenge we organized.', 'benchmark history')
ex('delarosa2025deepisles', 'p1', "Neuroradiologists prefer DeepISLES’ segmentations over manual annotations in a Turing-like test.", 'labels')

paper('delarosa2024isles24', 'A', 'R1: official ISLES\'24 challenge report', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2408.10966', venue='arXiv preprint arXiv:2408.10966', year=2024)
ex('delarosa2024isles24', 'p1', 'This work introduces the Ischemic Stroke Lesion Segmentation (ISLES) 2024 challenge, which focuses on the prediction of final infarct volumes from pre-interventional acute stroke imaging and clinical data.', 'task')
ex('delarosa2024isles24', 'p2', 'On the hidden test set (N = 98), the top-performing model, a multimodal nnU-Net-based architecture, achieved a Dice score of 0.285 ± 0.213 and an absolute volume difference of 21.2 ± 37.2 mL, underlining the significant challenges posed by this task and the need for further advances in multimodal learning.', 'SOTA')
ex('delarosa2024isles24', 'p6', 'The four following segmentation and task-specific, clinically relevant metrics are used: i) Dice Similarity Coefficient', 'metrics')
ex('delarosa2024isles24', 'p7', 'All three top-performing methods are based on nnU-Net [75].', 'SOTA')
ex('delarosa2024isles24', 'p7', 'Unlike the winning team Kurtlab, the second-ranked team AMC-Axolots employed a cascaded nnU-Net approach rather than a single-model strategy.', 'SOTA')
ex('delarosa2024isles24', 'p7', 'Notably, the third-ranked team, Ninjas, relied exclusively on the 4D CTP perfusion modality and achieved strong individual-metric performance for Dice and absolute lesion count difference, ranking second and first, respectively.', 'native CTP')
ex('delarosa2024isles24', 'p17', 'Notably, the first- and second-ranked teams integrated all available imaging modalities—excluding 4D CTP—and outperformed nnU-Net-based models relying solely on single modalities, such as teams utilizing only CTP or CTA data.', 'modalities')
ex('delarosa2024isles24', 'p17', 'Notably, the perfusion-based baseline models using rCBF achieved infarct volume estimates comparable to those of the best-performing deep learning methods. However, deep learning approaches outperformed in voxel-wise metrics, demonstrating superior spatial localization and identification of infarction.', 'DL vs thresholds')
ex('delarosa2024isles24', 'p17', 'Although one team incorporated clinically relevant variables -such as stroke severity scores at admission, patient demographics, clinical history, and critical time metrics- there was no demonstrated tangible improvement in segmentation performance through the integration of these features.', 'clinical data')
ex('delarosa2024isles24', 'p17', 'A standard concatenation-based fusion strategy was used, which proved insufficient to capture the complex relationships between imaging and clinical variables. Whether more sophisticated feature fusion strategies, such as attention-based mechanisms, cross- and multi-modal techniques like CLIP [93], or advanced tabular data architectures such as TabFPN [94], can enhance segmentation performance, remains an open question.', 'clinical data / VLM')
ex('delarosa2024isles24', 'p17', 'However, our challenge benchmark reveals two key findings: ( i) models utilizing standard deconvolution-derived perfusion maps consistently outperform those trained directly on raw 4D CTP series, and ( ii) models that extract spatial features from', 'native CTP')
ex('delarosa2024isles24', 'p18', 'the 4D CTP data achieve superior performance compared to those attempting to leverage the full spatiotemporal nature of the modality.', 'native CTP')
ex('delarosa2024isles24', 'p18', 'Specifically, the top-performing model utilizing raw 4D CTP data (team Ninjas) selected a subset of meaningful CTP timepoints and applied a spatial convolutional nnU-Net architecture, whereas models designed to explicitly capture temporally-encoded contrast propagation dynamics consistently underperformed.', 'native CTP')
ex('delarosa2024isles24', 'p18', 'We thus advocate for the development of advanced spatiotemporal models', 'native CTP')
ex('delarosa2024isles24', 'p17', 'Distinct infarct growth phenotypes—ranging from slow to fast progressors—have been identified, influenced by patient-specific pathophysiological variables, particularly the status of leptomeningeal collateral circulation, which sustains residual perfusion and limits infarct expansion [15, 16, 91].', 'difficulty')
ex('delarosa2024isles24', 'p17', 'Moreover, technical challenges related to data preprocessing—such as image registration and interpolation across differing image spaces, resolutions, and modalities (e.g., acute CT and follow-up MRI)—further hinder accurate lesion segmentation.', 'difficulty')
ex('delarosa2024isles24', 'p13', 'When focusing on cases with small infarcts (infarct< 5 mL,N = 25), all methods show limited volumetric agreement (Pearsonr< 0.4).', 'difficulty')
ex('delarosa2024isles24', 'p13', 'Methods based solely on CTP may be more sensitive to differences in acquisition protocols, resulting in decreased generalizability.', 'generalization')
ex('delarosa2024isles24', 'p18', 'As a more standardized and higher-resolution modality, NCCT offers a stable input source that may reduce model sensitivity to domain shifts in CTP data.', 'generalization')
ex('delarosa2024isles24', 'p18', 'Furthermore, the inclusion criteria of the ISLES’24 dataset were restricted to patients with favorable recanalization outcomes (i.e., thrombolysis in cerebral infarction -TICI- scores 2B and 3).', 'dataset')
ex('delarosa2024isles24', 'p9', 'The solution aimed to predict the final infarct by leveraging baseline infarct and penumbra masks derived from NCCT and CTA scans, as well as CTP perfusion maps (CBF, CBV, MTT, and TMAX).', 'SOTA')
ex('delarosa2024isles24', 'p9', 'A sequence of 20 time points centered around the peak contrast concentration was extracted to form a 20-channel input.', 'native CTP')
ex('delarosa2024isles24', 'p5', 'Lesion masks are derived from the follow-up MRI using DeepISLES [14].', 'labels')
ex('delarosa2024isles24', 'p13', 'While Center #1 provides higher resolution NCCT images, its CTP acquisitions exhibit consistently a lower resolution compared to Center #2.', 'generalization')

paper('riedel2026isles24dataset', 'A', 'R1: official ISLES\'24 dataset descriptor', 'full text of arXiv preprint (2408.11142); published version paywalled', 'https://doi.org/10.1148/ryai.250603', text='riedel2024isles24arxiv')
ex('riedel2026isles24dataset', 'p4 (arXiv v2)', 'Data preparation, which is briefly illustrated in Figure 1 and described in detail in the methods section, led to a dataset of 245 cases divided into a publicly available train set (n = 149) and - to ensure fair and continuous evaluation of machine learning algorithms - a hidden test set (n = 96).', 'dataset')
ex('riedel2026isles24dataset', 'p4 (arXiv v2)', 'by providing longitudinal imaging at two time points: (sub-)acute CT imaging before therapy and follow-up MRI 2-9 days after intracranial reperfusion therapy.', 'dataset')
ex('riedel2026isles24dataset', 'p3 (arXiv v2)', 'The current clinical standard for estimating perfusion volumes involves deconvolution analysis, which entails generating perfusion maps through perfusion CT and MR deconvolution and applying thresholds to these maps 10.', 'clinical standard')

# ---------------- B. ISLES'24 methods & recent work ----------------
paper('ren2025isles24', 'B', 'R1: winning ISLES\'24 method (Kurtlab)', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2505.18424', venue='arXiv preprint arXiv:2505.18424 (states PMLR 2025)', year=2025)
ex('ren2025isles24', 'p1', 'Our winning solution shows that a carefully designed preprocessing pipeline including deep-learning-based skull stripping and custom intensity windowing is beneficial for accurate segmentation.', 'preprocessing')
ex('ren2025isles24', 'p2', 'For our submission, we used a 3D large residual encoder nnUNet, referred to as “nnU-Net ResEnc L” in the nnU-Net documentation.', 'SOTA')
ex('ren2025isles24', 'p3', 'On this fold, the standard pipeline with only Z-score normalization and [1,99] percentile windowing achieved a Dice score of 21.8%, custom windowing alone improved performance to 31.0%, and our submission combining custom windowing with Z-score normalization further increased the validation Dice score to 31.8%.', 'preprocessing')
ex('ren2025isles24', 'p3', 'Results 10-fold cross validation was performed, and the fold with the highest validation Dice score was selected for submission.', 'protocol')
ex('ren2025isles24', 'p4', 'Kurtlab (Ours) 28.50 (21.27) 21.23 (37.22) 14.39 (21.19) 7.18 (7.67) AMC-Axolotls 26.27 (24.73) 21.31 (35.23) 14.94 (25.12) 7.66 (7.94) Ninjas 25.46 (19.08) 26.29 (39.73) 9.92 (13.46) 5.98 (6.46)', 'SOTA')
ex('ren2025isles24', 'p3', 'Table 1: Windowing ranges for each CT modality used as input to the nnU-Net segmentation model for the final submission to the ISLES’24 challenge.', 'SOTA')
ex('ren2025isles24', 'p3', 'Modality CTA (HU) CBF (mL/100g/min) CBV (mL/100g) MTT (s) Tmax (s) Window Range (0, 90) (0, 35) (0, 10) (0, 20) (0, 7)', 'SOTA')

paper('mojtahedi2025switch', 'B', 'R1: AMC-Axolotls (2nd place) method paper', 'abstract only (Amsterdam UMC repository page; PDF blocked)', 'https://doi.org/10.1007/978-3-031-81101-2_10')
ex('mojtahedi2025switch', 'Abstract', 'Using the nnU-Net framework, we observe a 16% improvement in Dice when baseline lesion masks are added in the absence of CTP. However, when CTP maps are included, the addition of baseline lesion masks is no longer beneficial.', 'modalities')

paper('amador2025switch', 'B', 'R1: ISLES\'24 spatio-temporal 4D CTP submission', 'abstract only (authors\' GitHub README; chapter paywalled)', 'https://doi.org/10.1007/978-3-031-81101-2_9')
ex('amador2025switch', 'Abstract', 'The model integrates a shared-weight convolutional neural network (CNN) encoder, a Transformer encoder, and a CNN decoder to capture both spatial and temporal dependencies within the data. We evaluated this approach on a multicenter dataset of 143 patients from the ISLES 2024 challenge.', 'native CTP')
ex('amador2025switch', 'Abstract', 'The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.', 'native CTP')

paper('cegarra2026isles', 'B', 'R1: controlled CNN vs Transformer comparison on ISLES\'24', 'full text (PMC XML)', 'https://doi.org/10.3390/s26165021', venue='Sensors')
ex('cegarra2026isles', 'Introduction', 'The main limitation of drawing conclusions from comparisons derived from the challenge leaderboard is that the results of the competing strategies are subject to confounding variables such as differences in preprocessing protocols, training data volume, and cross-validation strategy, making it difficult to attribute performance gaps to architectural choice alone.', 'protocol')
ex('cegarra2026isles', 'Abstract', 'The cascade model achieved the highest Dice Similarity Coefficient (0.224).', 'architectures')
ex('cegarra2026isles', 'Abstract', 'On the other hand, SegFormer3D returned the lowest overall metrics (DSC 0.163, AVD 27.28 mL, F1 2.30%).', 'architectures')
ex('cegarra2026isles', 'Abstract', 'Within these limits, the local inductive bias of the convolutional models retained an empirical advantage over the single Transformer evaluated when processing this moderately sized neuroimaging dataset, and complex cascade topologies offered only marginal gains compared with a well-calibrated single-stage network.', 'architectures')
ex('cegarra2026isles', 'Results', 'This section presents the performance results of each of the trained models when evaluated on the held-out test set of 15 patients.', 'protocol')
ex('cegarra2026isles', 'Methods', 'It must be stated explicitly that the two frameworks used (nnU-Net for M1 and M2, MONAI/SegFormer3D for M3) do not share every training-time setting', 'protocol')
ex('cegarra2026isles', 'Related work', 'the only pure Transformer entry (QuiiL, using the MoReT architecture) obtained 0.068 [7].', 'architectures')
ex('cegarra2026isles', 'Abstract', 'The first of these is a single-stage 5-channel nnU-Net, followed by a two-stage cascaded nnU-Net (2 and 7 channels) and a lightweight Transformer (SegFormer3D).', 'architectures')
ex('cegarra2026isles', 'Introduction', '(i) a single-stage nnU-Net approach replicating the preprocessing pipeline of the challenge-winning team under controlled experimental conditions, using 5 input channels (CTA, CBF, CBV, MTT, Tmax)', 'architectures')

paper('ajnr2026recanalization', 'B', 'R2: recent CTP-map nnU-Net outcome model, reperfusion-specific', 'abstract only (PubMed)', 'https://pubmed.ncbi.nlm.nih.gov/40973461/')
ex('ajnr2026recanalization', 'Abstract (Results)', 'The multimodal 3D nnU-Net model demonstrated superior performance, achieving mean Dice scores of 35.36% in patients with CR and 50.22% in those with NR. This model substantially outperformed the current clinically used method, providing more accurate outcome estimates than the conventional single-technique threshold-based measures, which yielded Dice scores of 15.73% and 39.71% for CR and NR groups, respectively.', 'DL vs thresholds')
ex('ajnr2026recanalization', 'Abstract (Methods)', 'evaluated them separately for patients with complete recanalization (successful reperfusion [CR], n = 350) and no recanalization (unsuccessful reperfusion [NR], n = 138) after treatment.', 'dataset')

paper('zhang2025dinov3stroke', 'B', 'R1/R3: frozen foundation model (DINOv3) evaluated on ISLES\'24 NCCT', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2509.23132', venue='arXiv preprint arXiv:2509.23132', year=2025)
ex('zhang2025dinov3stroke', 'p6', 'We also extracted the labeled slices from this dataset to form our dataset. Finally, we compiled 3982 slices, randomly dividing them into 3185 for training and 797 for testing.', 'protocol')
ex('zhang2025dinov3stroke', 'p10', 'On ISLES24, FPN+Refine with layers [2,5,8,11] yields the highest Dice of 0.5678', 'foundation models')
ex('zhang2025dinov3stroke', 'p2', 'Among them, DINOv3 [17] represents a state-of-the-art vision transformer, pretrained on more than 1.6 billion images, and capable of extracting high-fidelity, fine-grained dense features.', 'foundation models')
ex('zhang2025dinov3stroke', 'p11', 'This work presents a unified, reproducible evaluation of a frozen self-supervised transformer backbone (DINOv3) on NCCT-based stroke analysis', 'foundation models')

paper('shah2026asymfex', 'B', 'R1: uses ISLES\'24 with re-annotated pre-treatment labels', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2608.19769', venue='arXiv preprint arXiv:2608.19769', year=2026)
ex('shah2026asymfex', 'p8', 'ISLES’24 targets prediction of postinterventional infarct from pre-interventional CT, but since infarcts evolve after the initial scan, the released masks do not reflect the pre-interventional state. An expert neuroradiologist therefore manually re-delineated pre-interventional ischemic core and penumbra on the CT perfusion scans using ITK-SNAP [31].', 'protocol')
ex('shah2026asymfex', 'p14', 'NCCT alone collapses to near-failing performance at this timepoint regardless of architecture, underscoring that imaging modality, not model design, is the binding constraint at early stroke stages.', 'modalities')
ex('shah2026asymfex', 'p2', 'A 2-stage 3D segmentation pipeline that integrates with nnUNet and incorporates stroke-specific prior knowledge via an explicit geometric tiltcorrection step', 'architectures')
ex('shah2026asymfex', 'p8', 'we incorporate CT perfusion parameter maps, including Cerebral Blood Volume (CBV), Cerebral Blood Flow (CBF)', 'modalities')

paper('britt2026ncct', 'B', 'R1: NCCT model fine-tuned on ISLES\'24', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2609.20151', venue='arXiv preprint arXiv:2609.20151', year=2026)
ex('britt2026ncct', 'p2', 'Including smaller lesions, which predominated in ISLES, median Dice was 0.54 (interquartile range [IQR] 0.30 – 0.70) for acute lesion segmentation (Boston dataset) and 0.20 (IQR 0.03–0.41) for NCCT lesion segmentations when compared to post-treatment infarct (primary target of the ISLES challenge).', 'modalities')
ex('britt2026ncct', 'p2', 'To adapt to new domains, the model was fine-tuned on target-domain subsets from Boston (n=11) and ISLES (n=75), with evaluation on held-out cases not used for fine-tuning.', 'protocol')

paper('samak2024review', 'B', 'R2: survey of DL stroke outcome prediction', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2412.04812')
ex('samak2024review', 'p1', 'This article provides a broad review of recent advances and applications of deep learning in the prediction of stroke outcomes, including (i) the data and models used, (ii) the prediction tasks and measures of success, (iii) the current challenges and limitations, and (iv) future directions and potential benefits.', 'review')
ex('samak2024review', 'p3', 'Hence, there is also a lack of such data for researchers to apply today’s enhanced language models [11, 12] for outcome predictions.', 'VLM limits')

# ---------------- C. CTP outcome prediction ----------------
paper('robben2020nativectp', 'C', 'R2: native-CTP final infarct prediction', 'full text (arXiv PDF)', 'https://doi.org/10.1016/j.media.2019.101589')
ex('robben2020nativectp', 'p1', 'We pursue a data-driven and deconvolution-free approach, where a deep neural network learns to predict the final infarct volume directly from the native CTP images and metadata such as the time parameters and treatment.', 'native CTP')
ex('robben2020nativectp', 'p1', 'An ablation study shows that using the native CTP measurements instead of the deconvolved measurements improves the prediction.', 'native CTP')
ex('robben2020nativectp', 'p2', 'Additionally, deconvolution is a mathematically ill-posed problem and that is problematic given the low signal to noise ratio of perfusion images (Fieselmann et al., 2011).', 'native CTP')
ex('robben2020nativectp', 'p6', 'This results in 188 included subjects.', 'dataset')

paper('amador2022spatiotemporal', 'C', 'R2: 4D CTP spatio-temporal outcome prediction', 'abstract only (PubMed)', 'https://doi.org/10.1016/j.media.2022.102610')
ex('amador2022spatiotemporal', 'Abstract', 'The proposed method was evaluated on 147 patients using a 10-fold cross validation, which demonstrated that the proposed 3D+time model (mean Dice=0.45) significantly outperforms both a 2D+time variant of our approach (mean Dice=0.43) and a state-of-the-art method that uses perfusion maps (mean Dice=0.38).', 'native CTP')
ex('amador2022spatiotemporal', 'Abstract', 'By merging a U-Net-like architecture with temporal convolutional networks, we efficiently process the spatio-temporal information available in CTP datasets to make a tissue outcome prediction.', 'native CTP')

paper('devries2023perfunet', 'C', 'R2: spatio-temporal core segmentation from CTP source data', 'abstract only (PubMed)', 'https://doi.org/10.1016/j.media.2023.102749')
ex('devries2023perfunet', 'Abstract', 'The outcomes and perfusion maps of CT perfusion software, however, show many discrepancies between vendors.', 'native CTP')
ex('devries2023perfunet', 'Abstract', 'To this end, we present a symmetry-aware spatio-temporal segmentation model that encodes the micro-perfusion dynamics in the brain, while decoding a static segmentation map for infarct core assessment.', 'native CTP')
ex('devries2023perfunet', 'Abstract', 'We achieve state-of-the-art results compared to methods that only use CT perfusion source imaging with a Dice score of 0.46. We are almost on par with methods that use perfusion maps from third party software', 'native CTP')

# ---------------- D. Backbones ----------------
paper('isensee2021nnunet', 'D', 'R3: baseline framework', 'abstract only (PubMed)', 'https://doi.org/10.1038/s41592-020-01008-z')
ex('isensee2021nnunet', 'Abstract', 'We developed nnU-Net, a deep learning-based segmentation method that automatically configures itself, including preprocessing, network architecture, training and post-processing for any new task.', 'baseline')

paper('isensee2024revisited', 'D', 'R3: ResEnc presets used by the winner; validation pitfalls', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2404.09556', venue='MICCAI 2024 (LNCS)', year=2024)
ex('isensee2024revisited', 'p1', 'In contrast to current beliefs, we find that the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models, including ResNet and ConvNeXt variants, 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources.', 'baseline')
ex('isensee2024revisited', 'p1', 'In this study, we demonstrate that many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources.', 'protocol')

paper('hatamizadeh2022swinunetr', 'D', 'R3: transformer 3D segmentation backbone', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2201.01266')
ex('hatamizadeh2022swinunetr', 'p1', 'The swin transformer encoder extracts features at five different resolutions by utilizing shifted windows for computing self-attention and is connected to an FCNN-based decoder at each resolution via skip connections.', 'architectures')

paper('perera2024segformer3d', 'D', 'R3: transformer evaluated on ISLES\'24 by Cegarra et al.', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2404.10156')
ex('perera2024segformer3d', 'p1', 'SegFormer3D democratizes deep learning for 3D medical image segmentation by offering a model with 33× less parameters and a 13× reduction in GFLOPS compared to the current state-of-theart (SOTA).', 'architectures')

# ---------------- E. Video models ----------------
paper('tong2022videomae', 'E', 'R3: candidate pretrained video encoder', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2203.12602', venue='NeurIPS 2022', year=2022)
ex('tong2022videomae', 'p1', '(2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data.', 'video')
ex('tong2022videomae', 'p1', 'The temporally redundant video content enables higher masking ratio than that of images.', 'video')

paper('liu2022videoswin', 'E', 'R3: candidate pretrained video encoder', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2106.13230')
ex('liu2022videoswin', 'p1', 'In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization.', 'video')
ex('liu2022videoswin', 'p2', 'spatiotemporal locality of videos, in which pixels that are closer to each other in spatiotemporal distance are more likely to be correlated.', 'video')

paper('bertasius2021timesformer', 'E', 'R3: space-time attention design', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2102.05095', venue='ICML 2021', year=2021)
ex('bertasius2021timesformer', 'p1', 'Our experimental study compares different self-attention schemes and suggests that “divided attention,” where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered.', 'video')

paper('bardes2024vjepa', 'E', 'R3: candidate pretrained video encoder (frozen use)', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2404.08471', venue='Transactions on Machine Learning Research (TMLR) 2024', year=2024)
ex('bardes2024vjepa', 'p1', 'Our results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters; e.g., using a frozen backbone.', 'video')

paper('ravi2024sam2', 'E', 'R3/R4: video segmentation foundation model', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2408.00714', venue='ICLR 2025', year=2025)
ex('ravi2024sam2', 'p1', 'Our model is a simple transformer architecture with streaming memory for real-time video processing.', 'video')
ex('ravi2024sam2', 'p15', 'For images, the SA task can be considered a subset of PVS with the video reduced to a single frame.', 'video')

paper('zhu2024medicalsam2', 'E', 'R4: 3D medical volumes treated as video', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2408.00874', venue='arXiv preprint arXiv:2408.00874', year=2024)
ex('zhu2024medicalsam2', 'p1', 'The core concept is to leverage the Segment Anything Model 2 (SAM2) pipeline to treat all 2D and 3D medical segmentation tasks as a video object tracking problem.', 'video on volumes')
ex('zhu2024medicalsam2', 'p1', 'To put it into practice, we propose a novel self-sorting memory bank mechanism that dynamically selects informative embeddings based on confidence and dissimilarity, regardless of temporal order.', 'video on volumes')

paper('ma2025medsam2', 'E', 'R4: video foundation model fine-tuned for 3D medical images', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2504.03600', venue='arXiv preprint arXiv:2504.03600', year=2025)
ex('ma2025medsam2', 'p1', 'The model is developed by fine-tuning the Segment Anything Model 2 on a large medical dataset with over 455,000 3D image-mask pairs and 76,000 frames, outperforming previous models across a wide range of organs, lesions, and imaging modalities.', 'video on volumes')

paper('ke2023videopretraining', 'E', 'R4: natural-video pretraining transferred to 3D CT', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2304.00546', venue='MIDL 2023 (PMLR)', year=2023)
ex('ke2023videopretraining', 'p1', 'We demonstrate video pretraining improves the average performance of seven 3D models on two chest CT datasets, regardless of finetuning dataset size, and that video pretraining allows 3D models to outperform 2D baselines.', 'video on volumes')
ex('ke2023videopretraining', 'p1', 'Lastly, we observe that pretraining on the large-scale out-of-domain Kinetics dataset improves performance more than pretraining on a typically-sized in-domain CT dataset.', 'video on volumes')

# ---------------- F. Foundation models & VLMs ----------------
paper('ma2024medsam', 'F', 'R3: medical segmentation foundation model', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2304.12306')
ex('ma2024medsam', 'p2', 'However, the model exhibited substantial limitations in segmenting typical medical targets with weak boundaries or low contrast.', 'foundation models')
ex('ma2024medsam', 'p2', 'Essentially, SAM is a promptable segmentation method that requires points or bounding boxes to specify the segmentation targets.', 'foundation models')

paper('wang2023sammed3d', 'F', 'R3: volumetric promptable foundation model', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2310.15161')
ex('wang2023sammed3d', 'p1', 'Given only a few 3D prompt points, SAM-Med3D can accurately segment diverse anatomical structures and lesions across various modalities.', 'foundation models')

paper('liu2023clipdriven', 'F', 'R3/R4: text embeddings as segmentation conditioning', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2301.00785')
ex('liu2023clipdriven', 'p1', 'To address these issues, we propose the CLIP-Driven Universal Model, which incorporates text embedding learned from Contrastive Language-Image Pre-training (CLIP) to segmentation models.', 'VLM option')

paper('zhao2023sat', 'F', 'R3/R4: text-prompted 3D segmentation', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2312.17183')
ex('zhao2023sat', 'p1', 'However, while transferring to medical scenarios, these models trained on natural images suffer from fundamental limitations: (i) models typically perform 2D slice segmentation, which is later fused into 3D volumes through post-processing.', 'VLM limits')
ex('zhao2023sat', 'p1', 'SAT-Pro achieves comparable performance to 72 nnU-Nets—the strongest specialist models trained on each dataset (over 2.2B parameters combined)—over 497 categories.', 'foundation models')

paper('hamamci2024ctclip', 'F', 'R3: 3D CT vision-language model', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2403.17834')
ex('hamamci2024ctclip', 'p1', 'CT-RATE comprises 25,692 non-contrast 3D chest CT scans from 21,304 unique patients.', 'VLM limits')

paper('blankemeier2024merlin', 'F', 'R3: 3D CT VLM using EHR + reports', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2406.06512')
ex('blankemeier2024merlin', 'p1', 'To overcome these shortcomings for abdominal CT interpretation, we introduce Merlin - a 3D VLM that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining without requiring additional manual annotations.', 'VLM option')
ex('blankemeier2024merlin', 'p9', 'Recent concurrent research has started to investigate 3D vision language models for radiology, particularly focusing on chest and head CT scans.79, 67 However, the extent of experiments and 3D evaluations are limited.', 'VLM limits')

paper('wu2023radfm', 'F', 'R3: generalist radiology foundation model', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2308.02463', venue='arXiv preprint arXiv:2308.02463', year=2023)
ex('wu2023radfm', 'p25', 'However, the limitation can also be observed that the reasoning results are still general and more like background medical knowledge, yet not specific to the input case.', 'VLM limits')

paper('bai2024m3d', 'F', 'R3: 3D multimodal LLM', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2404.00578', venue='arXiv preprint arXiv:2404.00578', year=2024)
ex('bai2024m3d', 'p1', 'However, previous research has primarily focused on 2D medical images, leaving 3D images under-explored, despite their richer spatial information.', 'VLM limits')

paper('li2025braingpt', 'F', 'R3/R4: 3D brain CT multimodal LLM', 'full text (arXiv PDF)', 'https://arxiv.org/abs/2407.02235')
ex('li2025braingpt', 'p1', 'Toward deploying MLLM for more applicable diagnostic context, we noticed that the (1) scarcity of 3D image training dataset, (2) the direct use of undifferentiated foundation MLLMs, and (3) the lack of pertinent caption evaluation metrics were independent domain-specific constraints that integratively hobbles the iteration of next-generation medical MLLM research.', 'VLM limits')
ex('li2025braingpt', 'p1', 'In this regard, this study collected a 3D-BrainCT dataset (18,885 text-scan pairs) and applied clinical visual instruction tuning (CVIT) to train volumetric anatomy-sensible BrainGPT models to generate radiology-adherent 3D brain CT reports.', 'VLM option')

# ---------------- Added 2026-09-30 for methodology_proposal/methodology_proposals ----------------
ex('tong2022videomae', 'p4', 'We adopt the joint space-time cube embedding [3, 22, 39] in our VideoMAE, where we treat each cube of size 2× 16× 16 as one token embedding.', 'video')
ex('tong2022videomae', 'p5', 'we perform in-depth ablation studies on VideoMAE design with the default backbone of 16-frame ViT-B on Something-Something V2 (SSV2) and Kinetics-400 (K400).', 'video')
ex('ravi2024sam2', 'p2', 'SAM 2 is capable of interactively segmenting regions through prompts (clicks, boxes, or masks) on one or multiple video frames by utilizing a streaming memory that stores previous prompts and predictions.', 'video')
ex('ke2023videopretraining', 'p2', 'We then transferred these weights and finetuned them on RSNA for PE detection and LIDC for lung nodule detection.', 'video on volumes')
ex('delarosa2024isles24', 'p4', 'Moreover, the clinical data contains demographics (age, sex, etc.), patient history, admission scores (e.g., admission NIHSS), and clinical outcomes (e.g., 3month mRS).', 'clinical data')
ex('delarosa2024isles24', 'p5', 'The 4D CTP series are motion-corrected through image co-registration and temporally resampled at 1 frame/second.', 'dataset')
ex('delarosa2024isles24', 'p9', 'Training was performed with a batch size of 2 and a patch size of 28 × 256 × 192 on two NVIDIA A6000 GPUs.', 'compute')
ex('delarosa2024isles24', 'p9', 'All scans underwent preprocessing, including voxel value clipping to specific ranges (NCCT [0, 100] Hounsfield units, CTA [0, 200] Hounsfield units, CBF [0, 400] mL/100mL/min, CBV [0, 400] mL/100mL, MTT [0, 20] s, TMAX [0, 20] s), background value setting to zero, and skull stripping.', 'preprocessing')
ex('riedel2026isles24dataset', 'p4 (arXiv v2)', 'Complemented is longitudinal clinical data up to a three-month outcome, which enables taking into account patient-specific characteristics like demographics, pre-existing risk factors, medication, and the type of referral or intervention times.', 'clinical data')
ex('ren2025isles24', 'p2', 'First, we applied SynthStrip on the non-contrast CT (NCCT) scans to obtain a brain masks.', 'preprocessing')
ex('ren2025isles24', 'p2', 'Then, we applied this brain mask to the other co-registered scans (CTP, CTA, etc.) to obtain skull-stripped versions of the data.', 'preprocessing')
ex('ren2025isles24', 'p3', 'When clinical windowing guidelines were available in the literature, they informed the initial settings, with empirical adjustments made to optimize model performance.', 'preprocessing')
ex('liu2023clipdriven', 'p4', 'We first concatenate the CLIP embedding (wk) and the global image feature (f) and then input it to a multi-layer perceptron (MLP), namely text-based controller [71], to generate parameters (θk), i.e., θk = MLP(wk⊕f ), where ⊕ is the concatenation.', 'VLM option')
ex('ma2025medsam2', 'p2', 'To reuse the pre-trained model weights and avoid prohibitive computing costs, MedSAM2 adopted the SAM2 network architecture, including an image encoder, a memory attention module, a prompt encoder, and a mask decoder (Fig 1b).', 'video on volumes')
ex('ma2025medsam2', 'p1', 'involving the annotation of 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames, demonstrating that MedSAM2 can reduce manual costs by more than 85%.', 'video on volumes')

ex('delarosa2024isles24', 'p5', 'For anonymization purposes, all scans are defaced based on brain and face masks obtained with TotalSegmentator [56], and the clinical tabular data is de-identified by adding ± 5% random noise to the laboratory variables.', 'clinical data')

ex('delarosa2024isles24', 'p17', 'While acute CT imaging is widely accessible and routinely used in clinical practice, it is inherently less sensitive than the gold-standard DWI and is often affected by artifacts (e.g., streak artifacts) and a low signal-to-noise ratio.', 'difficulty')
ex('cegarra2026isles', 'Methods', 'First, morphological skull-stripping is applied to the NCCT of each patient by thresholding to (0,120) HU, binary erosion (3 iterations), selection of the largest 3D connected component with the cc3d library [42], dilation (5 iterations) and filling of internal holes.', 'preprocessing')

ex('delarosa2024isles24', 'p13', '(i) core tissue, defined as regions with rCBF< 30%; (ii) hypoperfused tissue, defined as regions with Tmax> 6 s', 'clinical standard')

ex('delarosa2024isles24', 'p5', 'The train (test) set contains N = 100 (N = 50) scans from the University Hospital of Munich and N = 50 (N = 48) scans from the University Hospital of Zurich.', 'dataset')

# ---------------- Listed but not quotable ----------------
for k, u in [('dalmazo2025switch', 'https://doi.org/10.1007/978-3-031-81101-2_11'), ('urunuela2025switch', 'https://doi.org/10.1007/978-3-031-81101-2_12')]:
    paper(k, 'B', 'R1: ISLES\'24 submission chapter', 'not accessible (paywalled; no abstract in Semantic Scholar/OpenAlex; publisher blocks automated access) — listed only, not cited for claims', u)

SCREENING_EXCLUDED = [
    dict(candidate='Isensee et al. 2018, nnU-Net: Self-adapting Framework (arXiv:1809.10486)', found_via='keyword search', decision='excluded', reason='superseded by Isensee et al. 2021 (Nature Methods)'),
    dict(candidate='Gava et al. 2021, Neural Network-derived perfusion maps (arXiv:2101.05992)', found_via='arXiv search', decision='excluded', reason='off-task: synthesises perfusion maps, does not predict final infarct'),
    dict(candidate='Bertels et al. 2022, Final infarct prediction in acute ischemic stroke (arXiv:2211.04850)', found_via='arXiv search', decision='excluded', reason='tutorial overview; primary sources (Robben 2020) cited instead'),
    dict(candidate='Deep Learning-Based Automatic Segmentation of Ischemic Stroke Lesions in CT Perfusion Imaging (Biomimetics 2026, PMC13204576)', found_via='web search', decision='excluded', reason='off-task: ISLES 2018 core segmentation only'),
    dict(candidate='Early Prediction of Ischaemic Stroke Outcomes: A Novel Computational Approach (bioRxiv 2024)', found_via='forward citation of ISLES\'24', decision='excluded', reason='off-task: biomechanical brain-swelling model, not infarct prediction'),
    dict(candidate='Infarct-volume prognostic value depends on outcome ascertainment... (Eur J Radiol 2026)', found_via='forward citation of ISLES\'24', decision='excluded', reason='off-task: functional-outcome prognosis from MRI datasets'),
    dict(candidate='TopCoW challenge (NEJM AI 2023, arXiv:2312.17670)', found_via='forward citation of ISLES\'24', decision='excluded', reason='off-task: vessel segmentation'),
    dict(candidate='Albers et al. 2018 (DEFUSE 3), Saver 2006, Feigin 2025', found_via='problem statement', decision='excluded from review', reason='clinical background; cited in problem_statement.md, not methods literature'),
]

out = {'generated': '2026-09-30', 'citation_counts_source': 'Semantic Scholar API, retrieved 2026-09-30',
       'verification': None, 'papers': [], 'screening_excluded': SCREENING_EXCLUDED}
for k, p in P.items():
    m = META[k]; cr = m.get('cr') or {}
    out['papers'].append(dict(
        key=k, section=p['section'], inclusion_rule=p['rule'], access=p['access'],
        url=('https://doi.org/'+m['doi']) if m.get('doi') and not m['doi'].lower().startswith('10.48550') else p['url'],
        title=' '.join((p['title_override'] or cr.get('title') or m['s2_title']).split()),
        authors=cr.get('authors') or m['authors'],
        year=p['year_override'] or cr.get('year') or m['s2_year'],
        venue=p['venue_override'] or cr.get('container') or m['s2_venue'],
        volume=cr.get('volume'), issue=cr.get('issue'), pages=cr.get('page'),
        doi=m.get('doi'), arxiv=m.get('arxiv'), citations_s2=m['citations_s2'],
        text_file=f"data/text/{p['text_file']}.txt", excerpts=p['excerpts']))
json.dump(out, open('data/excerpts.json', 'w'), indent=1, ensure_ascii=False)
print(len(out['papers']), 'papers,', sum(len(p['excerpts']) for p in out['papers']), 'excerpts')
