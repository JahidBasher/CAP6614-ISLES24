# Problem Statement: Predicting the Final Infarct from Pre-Treatment CT in Acute Ischemic Stroke (ISLES'24)

## 1. The task

Acute ischemic stroke happens when an artery supplying the brain is blocked. The tissue it feeds starts to die within minutes. When a patient arrives at the hospital, physicians have to decide quickly whether to reopen the vessel, usually by mechanical thrombectomy. That decision depends on how much brain is already lost and how much can still be saved.

This project addresses the **final infarct prediction** task of the ISLES'24 challenge (de la Rosa et al., 2024; Riedel et al., 2026). The model sees only the CT images and clinical information collected at admission, before treatment. From these, it has to predict which brain tissue will end up permanently infarcted. The reference answer is drawn on follow-up MRI acquired days after treatment.

Unlike conventional lesion segmentation, where the target is visible in the input image, the ISLES'24 target is a *future* state of the tissue. The task is therefore voxel-wise prediction of a patient-specific outcome rather than delineation of a visible structure.

## 2. Inputs and expected outputs

**Inputs.** Each case provides the acute "CT trilogy" acquired at admission, with all images co-registered to the non-contrast CT space:

| Input | What it shows |
|---|---|
| Non-contrast CT (NCCT) | Brain anatomy and early ischemic changes. Excludes hemorrhage. |
| CT angiography (CTA) | The vessels and the site of the occlusion |
| Native 4D CT perfusion (CTP) | A time series of 3D volumes that tracks a contrast bolus passing through the brain |
| Perfusion maps derived from CTP | Cerebral blood flow (CBF), cerebral blood volume (CBV), mean transit time (MTT) and time-to-maximum (Tmax) |
| Clinical data | Demographics, medical history, laboratory values and the admission NIHSS stroke-severity score |

**Output.** A binary 3D mask in the acute CT space that marks every voxel predicted to be infarcted at follow-up. The predicted infarct volume follows directly from the mask.

**Ground truth.** The final infarct is outlined on diffusion-weighted MRI (DWI and ADC) acquired 2 to 9 days after treatment. Masks were generated semi-automatically and then corrected under the supervision of neuroradiologists (Riedel et al., 2026).

**Evaluation.** The challenge uses four metrics:
- Dice similarity coefficient, for voxel overlap
- Absolute volume difference (AVD), in mL
- Lesion-wise F1 score, for detecting individual lesions
- Absolute lesion count difference

The public training set contains 149 patients from two university hospitals, Munich and Zurich. The 96-case test set remains hidden (Riedel et al., 2026). Every patient in the cohort was treated with mechanical thrombectomy and achieved successful reperfusion (TICI 2b or 3).

## 3. Why the problem is important

**Stroke is common and devastating.** More than 7.8 million new ischemic strokes and about 3.6 million ischemic stroke deaths occur worldwide each year. Ischemic strokes make up roughly 65% of all strokes (Feigin et al., 2025). Untreated, a typical large-vessel stroke destroys an estimated 1.9 million neurons per minute (Saver, 2006). Treatment decisions therefore have to be fast, and they have to be right.

**Imaging already drives treatment decisions.** Perfusion imaging is central to selecting patients for thrombectomy. For example, the DEFUSE 3 trial showed that patients treated 6 to 16 hours after onset still benefit if perfusion imaging shows salvageable tissue (Albers et al., 2018).

**Current tools are coarse.** Clinical software estimates the irreversibly damaged "core" and the at-risk "penumbra" with fixed, population-level thresholds, such as relative CBF below 30% for core and Tmax above 6 s for hypoperfused tissue. These rules ignore patient-specific factors such as collateral circulation, lesion location and the dynamics of the CTP signal.

A model that predicts the final infarct voxel by voxel could give a more individualized estimate of how much tissue reperfusion would save. It could also support prognosis and serve as an imaging endpoint in clinical trials. ISLES'24 is among the first public benchmarks to pair the full acute CT trilogy, including native 4D CTP, with follow-up MRI labels, so learning methods for the task can now be compared in a controlled way.

## 4. Why the problem is challenging

**The target is not in the input.** The model has to infer how tissue will evolve, not find a visible lesion. Infarcts can keep growing after successful reperfusion. Patients also range from slow to fast "progressors," depending largely on their collateral blood supply (de la Rosa et al., 2024). Two patients with similar admission scans can end up with different final infarcts.

**Inputs and labels come from different modalities.** Predictions are made from CT, while the labels come from MRI acquired days later at a different resolution. Registration and interpolation errors between these spaces add label noise that no model can fully remove.

**CT is a noisy, low-contrast signal.** Compared with DWI, CT is less sensitive to early ischemia. CTP in particular suffers from a low signal-to-noise ratio, streak and motion artifacts, and acquisition protocols that differ between centers.

**Lesions are small, scattered and imbalanced.** Infarcted voxels are a small fraction of the brain, and many patients have several small or scattered lesions. Missing small lesions is penalized heavily by lesion-wise F1 and lesion count difference, and over-segmentation is penalized by AVD.

**The data are scarce and heterogeneous.** With 149 public cases from two centers, high-capacity models overfit easily and center differences can dominate what is learned. Preprocessing matters as much as architecture: the winning team credited much of its gain to skull stripping and modality-specific intensity windowing (Ren et al., 2025).

**It is unclear how best to use the inputs.** The inputs mix 3D anatomical scans, a 4D perfusion time series, four derived maps and tabular variables. Standard pipelines stack the perfusion maps as independent channels, even though all four are derived from the same contrast bolus. The top-ranked teams left out native 4D CTP entirely, and clinical variables showed no clear benefit in the challenge (de la Rosa et al., 2024).

**State-of-the-art performance remains low.** The winning ISLES'24 method, a residual-encoder 3D nnU-Net, reached a mean test Dice of only 0.285 ± 0.213, an AVD of 21.2 mL and a lesion-wise F1 of about 0.14 (de la Rosa et al., 2024). The challenge organizers conclude that substantial progress in multimodal learning is still needed.

## 5. Research question

Given a strong, consistently preprocessed nnU-Net baseline, we ask:

> **Can an architecture that models relationships across ordered inputs improve final infarct prediction over treating those inputs as independent channels?**

Here, "ordered inputs" means either the sequence of perfusion maps or the native 4D CTP time series, and the candidate architectures are pretrained video encoders. We will measure any improvement with Dice, AVD and lesion-wise F1 under identical data splits, preprocessing and evaluation.

## References

- Albers, G. W., et al. (2018). Thrombectomy for stroke at 6 to 16 hours with selection by perfusion imaging. *New England Journal of Medicine*, 378(8), 708–718.
- de la Rosa, E., et al. (2024). ISLES'24: Final infarct prediction with multimodal imaging and clinical data. Where do we stand? *arXiv:2408.10966*. `[delarosa2024isles24]`
- Feigin, V. L., et al. (2025). World Stroke Organization: Global stroke fact sheet 2025. *International Journal of Stroke*, 20(2), 132–144.
- Ren, T., et al. (2025). How we won the ISLES'24 challenge by preprocessing. *arXiv:2505.18424*. `[ren2025isles24]`
- Riedel, E. O., et al. (2026). The ISLES'24 dataset: A multimodal stroke imaging dataset with hyperacute CT, acute postinterventional MRI, and 3-month clinical outcomes. *Radiology: Artificial Intelligence*, 8(3), e250603. `[riedel2026isles24dataset]`
- Saver, J. L. (2006). Time is brain—quantified. *Stroke*, 37(1), 263–266.
