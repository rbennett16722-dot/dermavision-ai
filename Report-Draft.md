# DermaVision AI: A Transformer-Enhanced Skin Lesion Classifier

**Ellie Lansdown** · Ellie.Lansdown@colorado.edu  
**Kishore Ram Sriramulu Krishnamurthy** · KishoreRam.SriramuluKrishnamurthy@colorado.edu  
**Ryan Bennett** · Ryan.Bennett-3@Colorado.edu  
**Grace Callahan** · Grace.Callahan@colorado.edu  
**Stephanie Furst** · Stephanie.Furst@colorado.edu  
Leeds School of Business, University of Colorado Boulder  
MSBC 5190 Modern AI for Business | Group 7 | Spring 2026

---

## Abstract

Skin cancer is the most diagnosed cancer globally. AI triage tools trained almost exclusively on lighter skin tones (Fitzpatrick I–III) exhibit misdiagnosis rates 30–40% higher for darker-skinned patients. DermaVision AI addresses both the access gap and the equity gap by training a multi-architecture classifier on approximately 38,100 images from ISIC 2019, PAD-UFES-20, and MILK10k across nine diagnostic classes. We compare EfficientNetB0, SwinV2-Tiny, and BiomedCLIP, combining them in a Nelder-Mead optimized weighted ensemble (Silver 2.0). The weighted ensemble achieves balanced accuracy 72.1% and macro AUROC 0.957, with melanoma recall 78.8%, BCC recall 73.3%, and SCC recall 66.8% — substantial gains over the majority-class baseline (11.1%) and the single-dataset bronze model. A held-out fairness audit on Fitzpatrick17k (16,574 images) reveals a maximum accuracy gap of 35.0% across skin tone groups, and the DDI Stanford audit confirms malignant recall of only 22.9% for the darkest skin tone group (Fitzpatrick V–VI). These results demonstrate that architectural improvements and post-training optimization alone cannot close the equity gap without substantially more diverse training data.

---

## 1. Introduction

Melanoma carries a five-year survival rate of 99% when caught early and 20% when caught late. Despite this, a shortage of roughly 75% of needed dermatologists in low- and middle-income countries leaves millions without access to timely screening. In developed nations, specialist wait times of 2–6 months and consultation costs of $150–$300 create their own barriers.

AI-based triage systems offer a partial solution, yet existing published models — almost universally trained on HAM10000, a dataset drawn predominantly from Australian and Austrian clinics — achieve 30–40% lower accuracy on darker skin tones. This is not a minor performance shortfall; it means the populations with the least access to dermatological care are also the most poorly served by AI tools designed to address that gap.

DermaVision AI targets both problems simultaneously: (1) classification accuracy sufficient for credible first-pass triage, and (2) explicit fairness auditing across Fitzpatrick skin types I–VI using held-out datasets never seen during training. Our Bronze model established a controlled baseline on HAM10000 with EfficientNetB0. Our Silver 2.0 model expands to three datasets (~38,100 images), three architectures, and three post-training improvements. We report a weighted ensemble balanced accuracy of 72.1% and macro AUROC of 0.957, with persistent and significant fairness gaps that motivate our planned Gold model multimodal architecture.

---

## 2. Related Work

HAM10000 [1] established the benchmark for dermoscopic classification with 10,015 images across 7 lesion classes. Esteva et al. [2] demonstrated CNN performance matching board-certified dermatologists for binary melanoma classification, generating early optimism about AI-assisted dermatology. Adamson and Smith [3] were among the first to formally document the training data equity problem: models trained on predominantly light-skinned datasets perform substantially worse on patients with darker skin tones, which has direct implications for equitable deployment. Groh et al. [4] introduced Fitzpatrick17k — a dataset of 16,577 images with researcher-assigned Fitzpatrick I–VI labels — specifically to enable skin-type-stratified fairness auditing, finding that without diverse training representation, models invert the equity objective.

On the data side, ISIC 2019 [5] extended HAM10000 to approximately 25,000 images across multiple countries and added squamous cell carcinoma (SCC) as a distinct class. PAD-UFES-20 [6] provided the first large-scale clinical smartphone dataset with explicit Fitzpatrick I–VI labels, drawn from a Brazilian public hospital system serving a diverse lower-income population. On the architecture side, SwinV2 [8] introduced shifted-window self-attention, allowing global spatial reasoning that CNN convolutions cannot replicate. BiomedCLIP [10] pre-trained a Vision Transformer on 15 million biomedical image–text pairs from PubMed Central, providing domain-specific representations before task-specific fine-tuning. Lin et al. [9] introduced focal loss to address class imbalance by down-weighting well-classified examples and focusing learning on hard cases.

Our work is the first to combine ISIC 2019, PAD-UFES-20, and MILK10k and compare EfficientNetB0, SwinV2, and BiomedCLIP in a single controlled framework. It is also the first in this domain to apply post-training temperature calibration and Nelder-Mead ensemble weight optimization alongside a two-dataset held-out fairness audit.

---

## 3. Data

### 3.1 Training Pool

| Dataset | Images | Device | Origin | Fitzpatrick Coverage |
|---------|-------:|--------|--------|---------------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I–III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I–VI |
| MILK10k | 10,480 | Dermoscope | ISIC 2025 Challenge | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | Partial I–VI |

**Table 1: Training data sources.** Labels were standardized to 9 classes: mel, nv, bcc, scc, akiec, bkl, df, vasc, unk. Zero image ID overlap was confirmed across all three sources via SHA-256 hash deduplication. Data were split 70/15/15 (train: 26,676 / val: 5,716 / test: 5,717) with stratification by class.

### 3.2 Descriptive Analysis and Class Distribution

The combined pool is substantially imbalanced. Melanocytic nevi (nv) comprise 38.3% of all images; in the bronze model's HAM10000-only training set this reached 67%. A naive classifier predicting "nv" for every image would achieve 38.3% overall accuracy while catching zero melanoma cases — which is why overall accuracy is a misleading primary metric for this task. The clinically significant classes have the following frequencies: mel=14.4%, bcc=24.2%, scc=4.6%, akiec=4.2%, bkl=10.4%, df=0.9%, vasc=0.9%, unk=2.1%.

EDA across device type revealed meaningful domain shift: ISIC 2019 dermoscopic images average higher contrast and more uniform lighting than PAD-UFES-20 smartphone images, which exhibit variable focal distances, skin oil reflections, and lower resolution. This cross-device variation motivated stronger augmentation in Silver compared to Bronze and the inclusion of source as a tracked metadata field.

Skin tone coverage analysis showed that PAD-UFES-20's 2,298 images with explicit Fitzpatrick I–VI labels provide the only ground-truth skin tone diversity signal in the training set. ISIC 2019 and MILK10k have implicit coverage that is predominantly I–III. This imbalance — roughly 94% of training images from lighter-skin or unverified sources — is the root cause of the fairness gaps reported in Section 5.

### 3.3 Preprocessing and Augmentation

All images were resized to 224×224 RGB. Training augmentation applied horizontal and vertical flips, rotation (±30°), zoom (±20%), brightness jitter (0.7–1.3×), and channel shift (20 units). These settings are deliberately stronger than the Bronze model to handle cross-device variation from PAD-UFES-20 smartphone images. No augmentation was applied to validation or test sets.

---

## 4. Methods

### 4.1 Two-Phase Fine-Tuning Protocol

All three models follow a consistent two-phase fine-tuning strategy. **Phase 1** (8–10 epochs, LR=1×10⁻³) freezes the backbone and trains only the classification head — this initializes the head weights without destabilizing the pre-trained backbone. **Phase 2** (15–20 epochs, LR=1×10⁻⁵) unfreezes the full backbone for end-to-end fine-tuning. Early stopping (patience=5) and ReduceLROnPlateau prevent overfitting. This approach draws on the transfer learning techniques developed in MSBC 5190 coursework, adapting the two-phase protocol from standard ImageNet fine-tuning to a multi-source medical imaging context.

### 4.2 Focal Loss with Risk-Boosted Class Weights

Standard cross-entropy on an imbalanced dataset learns to predict the majority class. We replaced it with focal loss [9] (γ=2.0), which down-weights easy examples so the model concentrates on hard minority cases. Per-class alpha weights were computed from inverse class frequencies, then risk-boosted to reflect clinical cost asymmetry: mel ×1.5, bcc ×1.3, scc ×1.3. Final weights: mel=1.16, nv=0.29, bcc=0.60, scc=3.12, akiec=2.65, bkl=1.07, df=12.35, vasc=12.20, unk=5.22.

### 4.3 EfficientNetB0

EfficientNetB0 [7] compound-scales network depth, width, and input resolution via a fixed coefficient. We used the TensorFlow/Keras pre-trained ImageNet weights as backbone, adding a classification head: GlobalAveragePooling2D → Dense(512, ReLU) → BatchNormalization → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(9, softmax). Phase 2 trainable parameters: 4,798,085. This model serves as the controlled reference point — it is the same architecture as the Bronze model, retrained on the expanded Silver dataset, isolating the contribution of data diversity from architecture changes.

### 4.4 SwinV2-Tiny

SwinV2 [8] uses hierarchical shifted-window self-attention, giving every image patch the ability to attend to every other patch across windows. This captures global lesion structure — border irregularity, color asymmetry — that CNN convolutions, which operate on local receptive fields, cannot access. We used `swinv2_tiny_window8_256` (~28M parameters) from the `timm` library with ImageNet pre-trained weights, adapting the classification head: Linear(768, 512) → ReLU → Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 9). The `timm` library provides modular access to pre-trained transformer checkpoints; our modifications added the custom multi-layer head, two-phase training callbacks, and integrated the SwinV2 forward pass into our unified training pipeline.

### 4.5 BiomedCLIP

BiomedCLIP [10] pre-trains ViT-Base/16 (86.6M parameters) on 15 million biomedical image–text pairs from PubMed Central via contrastive learning. Unlike EfficientNetB0 and SwinV2, which start from ImageNet photographs, BiomedCLIP begins with visual representations already attuned to dermoscopic and clinical medical imagery — textures, lesion morphologies, and color patterns common in biomedical literature. We used the `open_clip` library with the `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` checkpoint, using only the visual encoder (the text encoder is discarded). Our adaptation replaced the default projection head with the same two-layer classification head used in SwinV2, and integrated it into the unified training pipeline with the same focal loss objective.

### 4.6 Equal-Weight Ensemble (Silver Baseline)

The Silver baseline ensemble averages softmax probability vectors equally from all three models: P\_ensemble = (P\_eff + P\_swin + P\_clip) / 3. The rationale is that each model has complementary error modes: EfficientNetB0 captures fine-grained texture features, SwinV2 captures global spatial structure, and BiomedCLIP brings medical domain priors. Ensemble diversity reduces variance without requiring additional training.

### 4.7 Silver 2.0 Post-Training Improvements

Silver 2.0 applies three engineering improvements without retraining the underlying models:

**Weighted Ensemble (Fix 1).** Analysis of the Silver equal-weight ensemble revealed that EfficientNetB0 had near-zero BCC recall (4.3%), routing nearly all BCC predictions to SCC. Including it at one-third weight introduced systematic BCC errors. We used `scipy.optimize.minimize` with the Nelder-Mead method to search for scalar weights (W\_eff, W\_swin, W\_clip) that maximize balanced accuracy on the validation set, starting from (0.2, 0.5, 0.3). The optimizer converged to W\_eff=0.2112, W\_swin=0.5100, W\_clip=0.2788, substantially reducing EfficientNetB0's influence.

**Temperature Scaling (Fix 2).** Raw model logits were overconfident — a model predicting "95% melanoma" may have a true calibrated confidence closer to 60%. This matters clinically: confidence scores are used in the referral threshold logic. We calibrated a scalar temperature parameter T for SwinV2 and BiomedCLIP by minimizing negative log-likelihood on validation logits using L-BFGS. T\_swin=1.3133, T\_clip=1.2545. Predicted class labels are unchanged; only confidence scores are affected.

**Test-Time Augmentation (Fix 3).** Single-pass inference introduces prediction variance on edge cases near class decision boundaries. For each test image, we generated 5 augmented views (same augmentation policy as training) and averaged the predicted probability vectors before taking the argmax. This reduces per-image prediction noise and is especially valuable for ambiguous boundary cases (e.g., mel vs. bkl, bcc vs. scc).

**Additional fixes.** (a) Fitzpatrick17k label mapping: the original mapping covered only uppercase class codes (e.g., `MEL`, `BCC`), causing ~90% of Fitzpatrick17k rows to fall through to `unk` due to its lowercase and long-form disease naming conventions. We extended the mapping with 40+ aliases (e.g., "lentigo maligna melanoma" → mel, "seborrheic keratosis" → bkl, "nevus" → nv). (b) PAD-UFES-20 Fitzpatrick column: the Fitzpatrick labels were silently dropped after `pd.concat` due to column name inconsistency; we standardized to `fitzpatrick` across all source DataFrames before concatenation.

---

## 5. Results

### 5.1 Model Performance on Held-Out Test Set (n = 5,717)

Clinical deployment thresholds established a priori: Mel recall > 0.80 | BCC recall > 0.75 | SCC recall > 0.75 | Balanced Accuracy > 0.75 | Macro AUROC > 0.90 | Fitzpatrick gap < 0.05.

| Model | Bal. Acc | Macro AUROC | Mel Recall | BCC Recall | SCC Recall | Fitz. Gap |
|-------|:--------:|:-----------:|:----------:|:----------:|:----------:|:---------:|
| Majority class baseline | 0.111 | — | 0.000 | 0.000 | 0.000 | — |
| Silver — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Silver — SwinV2-Tiny | 0.713 | 0.957 | 0.761 | 0.748 | 0.600 | 0.358 |
| Silver — BiomedCLIP | 0.602 | 0.930 | 0.797 | 0.653 | 0.566 | 0.193 |
| Sv2.0 — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Sv2.0 — SwinV2 + TTA + Cal | 0.716 | **0.959** | 0.765 | 0.760 | 0.604 | 0.358 |
| Sv2.0 — BiomedCLIP + TTA + Cal | 0.599 | 0.932 | **0.791** | 0.677 | 0.562 | 0.193 |
| **Sv2.0 — Weighted Ensemble** | **0.721** | 0.957 | 0.788 | **0.733** | **0.668** | **0.350** |

**Table 2: Model performance on held-out test set.** Bold values indicate best in class. No model meets all clinical deployment thresholds.

No model meets all six clinical thresholds. The weighted ensemble is the strongest overall performer. BiomedCLIP achieves the highest individual melanoma recall (79.1%), consistent with its pre-training on medical imagery. SwinV2 achieves the best AUROC (0.959 with TTA and calibration).

**On EfficientNetB0 BCC recall (0.043).** The model routes nearly all BCC predictions to SCC. The SCC focal weight (3.12×) is 5× the BCC weight (0.60×), making every SCC miss 5× more costly than a BCC miss during training — so the model learned to predict SCC aggressively. MILK10k's concentration of BCC examples (5,044 of 9,212 total BCC training images) from a single imaging environment may have also contributed. The Nelder-Mead weighted ensemble corrects this by reducing EfficientNetB0's contribution to approximately 0.21, allowing SwinV2's 76.0% BCC recall to dominate the ensemble's BCC predictions.

**Silver 2.0 vs. Silver gains.** The weighted ensemble improves balanced accuracy from 0.699 (Silver equal-weight) to 0.721, BCC recall from 0.043 (if relying on EfficientNetB0 alone) to 0.733 in the ensemble, and reduces the Fitzpatrick gap from 0.358 to 0.350.

### 5.2 Fairness Audit — Individual Model Performance on Fitzpatrick17k (held-out, n = 16,574)

| Fitzpatrick | N | EfficientNetB0 Acc | SwinV2 Acc | BiomedCLIP Acc |
|:-----------:|--:|:-----------------:|:----------:|:--------------:|
| I | 2,947 | 0.250 | 0.390 | 0.230 |
| II | 4,808 | 0.200 | 0.501 | 0.274 |
| III | 3,308 | 0.180 | 0.568 | 0.285 |
| IV | 2,781 | 0.178 | 0.674 | 0.324 |
| V | 1,533 | 0.232 | 0.712 | 0.360 |
| VI | 635 | 0.293 | 0.616 | 0.277 |
| **Max gap** | — | **0.164** | **0.358** | **0.193** |

**Table 3: Per-Fitzpatrick-group accuracy by individual model.** Note the opposing pattern: EfficientNetB0 performs slightly *better* on lighter skin tones while SwinV2 performs better on darker tones — reflecting the different image features each architecture attends to.

| Model | Max Accuracy Gap | Max Mel-Recall Gap | Verdict |
|-------|:----------------:|:------------------:|:-------:|
| EfficientNetB0 | 0.164 | 0.238 | **FAIL** |
| SwinV2 | 0.358 | 0.474 | **FAIL** |
| BiomedCLIP | 0.193 | 0.282 | **FAIL** |
| Weighted Ensemble | 0.350 | — | **FAIL** |

**Table 4: Fairness summary by model.** All models fail both the accuracy equity target (gap < 0.05) and the melanoma-recall equity target (gap < 0.10) by a wide margin.

The Fitzpatrick17k label mapping issue (noted in Section 4.7) means the weighted ensemble's per-group accuracy table is only fully populated for Fitzpatrick I–III; rows IV–VI remain partially impacted by residual `unk` mapping. This is noted as a limitation of the current fairness audit infrastructure.

### 5.3 Fairness Audit — DDI Stanford (held-out, n = 656)

| Skin Tone Group | N Malignant | Malignant Recall (Weighted Ensemble) |
|:----------------|:-----------:|:------------------------------------:|
| Fitzpatrick I–II | 49 | 0.367 |
| Fitzpatrick III–IV | 74 | 0.392 |
| Fitzpatrick V–VI | 48 | **0.229** |

**Table 5: DDI Stanford malignant recall by skin tone group.** Malignant recall is lowest for the darkest skin tone group (22.9%), with a 16.3-point gap between the III–IV and V–VI groups.

The DDI results directly contradict the equity objective: the model is worst at catching malignant lesions in the patients with the darkest skin tones, who are also the patients with the least access to dermatological care. This is the most clinically concerning finding in this report and is the primary motivation for the Gold model's expanded training data strategy.

---

## 6. Conclusion

### 6.1 Summary

DermaVision AI demonstrates that multi-source, multi-architecture training substantially outperforms single-dataset baselines. The Silver 2.0 weighted ensemble achieves balanced accuracy 72.1%, macro AUROC 0.957, melanoma recall 78.8%, and BCC recall 73.3% — all improvements over the equal-weight Silver ensemble and dramatically above the bronze model's performance on the same test set. However, no model meets the pre-specified clinical deployment thresholds. The fairness audit reveals a 35.0% maximum accuracy gap across Fitzpatrick types and a 16.3-point malignant recall gap between lighter and darker skin tone groups. Closing these gaps requires more diverse training data, not just different architectures or post-training techniques.

### 6.2 Future Work (Gold Model)

1. MILK10k paired clinical + dermoscopic two-stream late fusion — multimodal architecture combining complementary image modalities
2. MRA-MIDAS 2025 as a real-world prospective test set
3. Expanded Fitzpatrick IV–VI training data with explicit ground-truth labels
4. Skin-tone-aware augmentation (RandAugment conditioned on estimated Fitzpatrick type) and targeted oversampling
5. Streamlit deployment app wrapping the Gold ensemble
6. Clinical validation with fairness thresholds as mandatory release criteria

### 6.3 Social and Managerial Implications

**Social implications.** The fairness gaps documented in this project are not abstract statistics. A model with 22.9% malignant recall for Fitzpatrick V–VI patients misses more than three out of four malignant lesions in the population most likely to lack access to specialist follow-up care. Deploying a system with known performance disparities of this magnitude could cause direct patient harm — delayed diagnoses, false reassurance, and widened health equity gaps. The ethical obligation is clear: fairness auditing using purpose-built held-out datasets with explicit skin tone labels must be a mandatory release criterion, not an afterthought. Fitzpatrick17k and DDI demonstrated precisely what in-distribution audits would have missed.

There is also a privacy dimension. Skin images are sensitive biometric data. Any deployment system must implement appropriate data minimization (images processed locally where possible), informed consent protocols, and role-based access controls on stored diagnostic outputs.

**Managerial implications.** From a business standpoint, the market opportunity is substantial: the $150–$300 dermatology consultation cost represents a real barrier, and AI triage could reduce unnecessary specialist visits by 40–60% while escalating high-risk cases. However, deploying a model that performs significantly worse on darker-skinned patients creates both legal liability (potential discrimination claims under healthcare equity regulations) and reputational risk that would undermine adoption in exactly the communities the product is intended to serve. The managerial recommendation is staged deployment: pilot in settings with supervised clinical review, collect ground-truth feedback to continuously improve fairness performance, and treat equity metrics as hard release gates rather than aspirational targets. The cost of collecting more diverse labeled data is far smaller than the cost of a discrimination lawsuit or a recall after a preventable missed diagnosis.

### 6.4 Lessons Learned

**Data diversity matters more than architecture.** The move from HAM10000 (~10k images, 1 source) to the three-source Silver pool (~38k images) produced larger gains in balanced accuracy and melanoma recall than switching from EfficientNetB0 to SwinV2 on the same dataset. This was somewhat surprising — we expected the transformer architectures to be the dominant driver of improvement, but the data expansion was at least as impactful.

**Class weight calibration is fragile.** EfficientNetB0's BCC recall collapsing to 4.3% was the most unexpected result of the project. The combination of aggressive SCC focal weights and MILK10k's concentration of BCC examples created a systematic routing error that was invisible from overall balanced accuracy alone. This reinforced the importance of per-class recall monitoring and the danger of optimizing a single aggregate metric.

**Fairness auditing requires purpose-built data.** Fitzpatrick17k and DDI revealed equity gaps that any in-distribution validation set would have missed entirely. If we had only evaluated on the held-out test split (drawn from the same three source datasets), we would have reported misleadingly positive results on skin tone equity.

**What was especially hard.** Aligning label vocabularies across three datasets with different disease naming conventions required extensive manual mapping work. The Fitzpatrick17k label mismatch that caused 90% of rows to be classified as `unk` was not caught until the fairness audit returned near-identical results for every skin tone group — a red flag that prompted investigation. Engineering the ensemble weight optimizer and temperature calibration without access to the test set (to avoid leakage) also required careful validation set management.

**If we had more time and resources.** We would have rebalanced the SCC/BCC focal weight ratio (current 3.12:0.60 is too aggressive; a ratio closer to 1.5:0.8 better reflects relative clinical importance), retrained EfficientNetB0 Phase 2 with a lower learning rate (the backbone was effectively never fine-tuned due to val loss degradation in Phase 2), and sourced substantially more Fitzpatrick IV–VI labeled training data. We would also have built the full multi-model Streamlit app earlier to enable live testing during development rather than post-hoc.

---

## Acknowledgments

The authors thank the organizers of ISIC 2019, PAD-UFES-20, and MILK10k for making their datasets publicly available, and the University of Colorado Boulder Leeds School of Business for support throughout the MSBC 5190 course.

---

## References

[1] Tschandl, P., et al. (2018). The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

[2] Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.

[3] Adamson, A. S., and Smith, A. (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*, 154(11), 1247–1248.

[4] Groh, M., et al. (2021). Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. *CVPR Workshop on Fair, Data-Efficient, and Trusted Computer Vision*.

[5] Combalia, M., et al. (2019). BCN20000: Dermoscopic lesions in the wild. *arXiv preprint arXiv:1908.02288*.

[6] Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221.

[7] Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

[8] Liu, Z., et al. (2022). Swin Transformer V2: Scaling up capacity and resolution. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[9] Lin, T. Y., et al. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

[10] Zhang, S., et al. (2023). BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. *arXiv preprint arXiv:2303.00915*.
