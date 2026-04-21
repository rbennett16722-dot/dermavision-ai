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

Skin cancer is the most diagnosed cancer globally. AI triage tools trained almost exclusively on lighter skin tones (Fitzpatrick I–III) exhibit misdiagnosis rates 30–40% higher for darker-skinned patients. DermaVision AI addresses both the access gap and the equity gap by training a multi-architecture classifier on approximately 38,100 images from ISIC 2019, PAD-UFES-20, and MILK10k across nine diagnostic classes. We compare EfficientNetB0, SwinV2-Tiny, and BiomedCLIP, combining them in a Nelder-Mead optimized weighted ensemble (Silver 2.0). The weighted ensemble achieves balanced accuracy 72.1% and macro AUROC 0.957, with melanoma recall 78.8%, BCC recall 73.3%, and SCC recall 66.8%. A held-out fairness audit on Fitzpatrick17k (16,574 images) reveals a maximum accuracy gap of 35.0% across skin tone groups, and the DDI Stanford audit confirms malignant recall of only 22.9% for the darkest skin tone group (Fitzpatrick V–VI). These results demonstrate that architectural improvements and post-training optimization alone cannot close the equity gap without substantially more diverse training data.

---

## 1. Introduction

Melanoma carries a five-year survival rate of 99% when caught early and 20% when caught late. A shortage of roughly 75% of needed dermatologists in low- and middle-income countries leaves millions without access to timely screening. In developed nations, wait times of 2–6 months and consultation costs of $150–$300 create further barriers.

AI-based triage systems offer a partial solution, yet published models trained on HAM10000 — drawn predominantly from Australian and Austrian clinics — achieve 30–40% lower accuracy on darker skin tones. The populations with the least access to dermatological care are also the most poorly served by existing AI tools. DermaVision AI targets both problems: (1) accuracy sufficient for credible first-pass triage, and (2) explicit fairness auditing across Fitzpatrick skin types I–VI using held-out datasets never seen during training. Our Bronze model established a controlled baseline on HAM10000 with EfficientNetB0. Silver 2.0 expands to three datasets (~38,100 images), three architectures, and three post-training improvements, achieving a weighted ensemble balanced accuracy of 72.1% and macro AUROC of 0.957.

---

## 2. Related Work

HAM10000 [1] established the dermoscopic classification benchmark with 10,015 images across 7 classes. Esteva et al. [2] demonstrated CNN performance matching board-certified dermatologists for melanoma classification. Adamson and Smith [3] documented the training data equity problem: models trained on predominantly light-skinned datasets perform substantially worse on darker-skinned patients. Groh et al. [4] introduced Fitzpatrick17k (16,577 images with Fitzpatrick I–VI labels) for skin-type-stratified fairness auditing, finding that without diverse training data, models invert the equity objective.

ISIC 2019 [5] extended HAM10000 to ~25,000 multi-country images and added SCC. PAD-UFES-20 [6] provided the first large-scale clinical smartphone dataset with explicit Fitzpatrick I–VI labels from a Brazilian public hospital. SwinV2 [8] introduced shifted-window self-attention for global spatial reasoning. BiomedCLIP [10] pre-trained a Vision Transformer on 15 million biomedical image–text pairs from PubMed Central. Lin et al. [9] introduced focal loss to address class imbalance. Our work is the first to combine ISIC 2019, PAD-UFES-20, and MILK10k and compare all three architectures in a single controlled framework with post-training calibration and held-out fairness auditing.

---

## 3. Data

### 3.1 Training Pool

| Dataset | Images | Device | Origin | Fitzpatrick Coverage |
|---------|-------:|--------|--------|---------------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I–III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I–VI |
| MILK10k | 10,480 | Dermoscope | ISIC 2025 Challenge | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | Partial I–VI |

**Table 1: Training data sources.** Labels standardized to 9 classes: mel, nv, bcc, scc, akiec, bkl, df, vasc, unk. Zero image ID overlap confirmed via hash deduplication. Split: 70/15/15 stratified (train: 26,676 / val: 5,716 / test: 5,717).

### 3.2 Descriptive Analysis and Class Distribution

The combined pool is substantially imbalanced: nv=38.3%, mel=14.4%, bcc=24.2%, scc=4.6%, akiec=4.2%, bkl=10.4%, df=0.9%, vasc=0.9%, unk=2.1%. A naive classifier always predicting "nv" achieves 38.3% overall accuracy while catching zero melanoma cases — which is why overall accuracy is a misleading primary metric, and why we use balanced accuracy (mean recall across all 9 classes) as our primary measure.

EDA across device type revealed meaningful domain shift: ISIC 2019 dermoscopic images exhibit higher contrast and more uniform lighting than PAD-UFES-20 smartphone images, which show variable focal distances and lower resolution. This motivated stronger augmentation and source tracking as a metadata field. Skin tone analysis confirmed that PAD-UFES-20's 2,298 explicitly labeled images are the only ground-truth diversity signal; the remaining ~94% of training images are lighter-skin or unverified. This imbalance is the root cause of the fairness gaps in Section 5.

### 3.3 Preprocessing and Augmentation

All images resized to 224×224 RGB. Training augmentation: horizontal/vertical flips, rotation (±30°), zoom (±20%), brightness jitter (0.7–1.3×), channel shift (20 units) — stronger than the Bronze model to handle cross-device variation. No augmentation was applied at validation or test time.

---

## 4. Methods

### 4.1 Two-Phase Fine-Tuning

All three models follow the same protocol. **Phase 1** (8–10 epochs, LR=1e-3): backbone frozen, head trained. **Phase 2** (15–20 epochs, LR=1e-5): full backbone unfrozen, end-to-end fine-tuning. Early stopping (patience=5) and ReduceLROnPlateau prevent overfitting.

### 4.2 Focal Loss with Risk-Boosted Class Weights

We replaced cross-entropy with focal loss [9] (γ=2.0), which down-weights easy examples so the model focuses on hard minority cases. Per-class alpha weights were computed from inverse class frequencies, then risk-boosted: mel ×1.5, bcc ×1.3, scc ×1.3. Final weights: mel=1.16, nv=0.29, bcc=0.60, scc=3.12, akiec=2.65, bkl=1.07, df=12.35, vasc=12.20, unk=5.22.

### 4.3 EfficientNetB0

EfficientNetB0 [7] compound-scales network depth, width, and resolution. We used TensorFlow/Keras with ImageNet pre-trained weights and head: GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(9, softmax). Phase 2 trainable parameters: 4,798,085. This model is the controlled reference point — same architecture as the Bronze model, retrained on the Silver dataset, isolating the contribution of data diversity from architecture changes.

### 4.4 SwinV2-Tiny

SwinV2 [8] uses hierarchical shifted-window self-attention, giving every image patch the ability to attend globally — capturing border irregularity and color asymmetry that CNN convolutions cannot. We used `swinv2_tiny_window8_256` (~28M params) from the `timm` library with a custom multi-layer head: Linear(768,512) → ReLU → Dropout(0.4) → Linear(512,256) → ReLU → Dropout(0.3) → Linear(256,9), and integrated it into our unified two-phase training pipeline.

### 4.5 BiomedCLIP

BiomedCLIP [10] pre-trains ViT-Base/16 (86.6M params) on 15M biomedical image–text pairs from PubMed Central via contrastive learning. Unlike EfficientNetB0 and SwinV2, which start from ImageNet photographs, BiomedCLIP begins with visual representations already attuned to medical imagery. We used the `open_clip` library (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`), discarding the text encoder and replacing the projection head with the same two-layer head as SwinV2. We also implemented Gradient-weighted Class Activation Mapping (Grad-CAM) for all three models to produce visual heatmaps highlighting which image regions drove each prediction — essential for clinical trustworthiness.

### 4.6 Equal-Weight Ensemble (Silver Baseline)

The Silver baseline averages softmax probability vectors equally: P\_ensemble = (P\_eff + P\_swin + P\_clip) / 3. Each model has complementary error modes: EfficientNetB0 captures fine texture, SwinV2 captures global structure, BiomedCLIP brings medical domain priors.

### 4.7 Silver 2.0 Post-Training Improvements

Silver 2.0 applies three engineering improvements without retraining:

**Weighted Ensemble.** EfficientNetB0 had near-zero BCC recall (4.3%), routing nearly all BCC predictions to SCC due to SCC's 5× higher focal weight. We used `scipy` Nelder-Mead optimization on the validation set to find optimal weights maximizing balanced accuracy. Converged to W\_eff=0.2112, W\_swin=0.5100, W\_clip=0.2788.

**Temperature Scaling.** Raw logits were overconfident. We calibrated a scalar temperature T for SwinV2 and BiomedCLIP via L-BFGS on validation logits (T\_swin=1.3133, T\_clip=1.2545). Predicted labels are unchanged; confidence scores become reliable for clinical referral thresholds.

**Test-Time Augmentation (TTA).** For each test image, we generated 5 augmented views and averaged predicted probabilities before taking the argmax, reducing prediction variance on edge cases.

**Additional fixes.** Extended Fitzpatrick17k label mapping (previously ~90% mapped to `unk` due to naming mismatches; fixed with 40+ disease name aliases) and normalized PAD-UFES-20 Fitzpatrick column handling after `pd.concat`.

---

## 5. Results

We evaluate all models on the held-out test set (n=5,717) using **balanced accuracy** (mean recall across 9 classes), **macro AUROC**, per-class recall for mel/bcc/scc, and **Fitzpatrick accuracy gap** on two held-out fairness audit datasets. Overall accuracy is de-emphasized because a majority-class classifier (always predicting "nv") achieves 38.3% while catching zero melanomas. Clinical deployment thresholds established a priori: Mel recall > 0.80 | BCC recall > 0.75 | SCC recall > 0.75 | Balanced Accuracy > 0.75 | Macro AUROC > 0.90 | Fitzpatrick gap < 0.05.

### 5.1 Model Performance on Held-Out Test Set (n = 5,717)

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

**Table 2: Model performance on held-out test set.** Bold = best in class. No model meets all clinical thresholds.

No model meets all six clinical thresholds. The weighted ensemble is the strongest overall performer. BiomedCLIP achieves the highest individual melanoma recall (79.1%), consistent with its medical pre-training. SwinV2 achieves the best AUROC (0.959 with TTA + calibration). **On EfficientNetB0 BCC recall (0.043):** the model routes nearly all BCC predictions to SCC because the SCC focal weight (3.12×) is 5× the BCC weight (0.60×). The Nelder-Mead ensemble corrects this by down-weighting EfficientNetB0 (≈0.21), allowing SwinV2's 76.0% BCC recall to dominate. Silver 2.0 improves balanced accuracy from 0.699 (Silver equal-weight) to 0.721 and reduces the Fitzpatrick gap from 0.358 to 0.350.

### 5.2 Fairness Audit — Fitzpatrick17k (held-out, n = 16,574)

| Model | Max Accuracy Gap | Max Mel-Recall Gap | Verdict |
|-------|:----------------:|:------------------:|:-------:|
| EfficientNetB0 | 0.164 | 0.238 | **FAIL** |
| SwinV2 | 0.358 | 0.474 | **FAIL** |
| BiomedCLIP | 0.193 | 0.282 | **FAIL** |
| Weighted Ensemble | 0.350 | — | **FAIL** |

**Table 3: Fairness summary on Fitzpatrick17k.** All models fail both the accuracy equity target (< 0.05 gap) and the melanoma-recall equity target (< 0.10 gap) by a wide margin.

The weighted ensemble's accuracy on Fitzpatrick I–III ranges from 0.375 to 0.570 (max gap 0.350). Rows IV–VI remain partially impacted by residual label-mapping issues. All models fail equity targets — a key finding.

### 5.3 Fairness Audit — DDI Stanford (held-out, n = 656)

| Skin Tone Group | N Malignant | Malignant Recall (Weighted Ensemble) |
|:----------------|:-----------:|:------------------------------------:|
| Fitzpatrick I–II | 49 | 0.367 |
| Fitzpatrick III–IV | 74 | 0.392 |
| Fitzpatrick V–VI | 48 | **0.229** |

**Table 4: DDI Stanford malignant recall by skin tone group.** Malignant recall is lowest for the darkest group (22.9%), a 16.3-point gap vs. III–IV. The model is worst at catching malignant lesions in exactly the patients with the least access to specialist care.

---

## 6. Conclusion

DermaVision AI demonstrates that multi-source, multi-architecture training substantially outperforms single-dataset baselines. The Silver 2.0 weighted ensemble achieves balanced accuracy 72.1%, macro AUROC 0.957, melanoma recall 78.8%, and BCC recall 73.3% — clear improvements over both the majority-class baseline and the equal-weight Silver ensemble. However, no model meets the pre-specified clinical deployment thresholds. The fairness audits reveal a 35.0% accuracy gap on Fitzpatrick17k and a 22.9% malignant recall for the darkest DDI skin tone group. The central lesson: **data diversity matters more than architecture**. Moving from HAM10000 (~10k images) to the three-source Silver pool (~38k) produced larger gains than switching from EfficientNetB0 to SwinV2. The EfficientNetB0 BCC recall collapse (4.3%) was the most surprising finding — a systematic error invisible from balanced accuracy alone, caused by the interaction of aggressive SCC focal weights and MILK10k's BCC concentration. It reinforced the danger of optimizing a single aggregate metric. If we were to do the project again differently, we would rebalance the SCC/BCC focal weight ratio earlier (3.12:0.60 is too aggressive), retrain EfficientNetB0 Phase 2 at a lower learning rate, and prioritize acquiring Fitzpatrick IV–VI labeled data before training rather than as a post-hoc correction.

**Future work (Gold model):** MILK10k paired clinical + dermoscopic two-stream late fusion; expanded Fitzpatrick IV–VI training data; skin-tone-aware augmentation; Streamlit deployment app; clinical validation with fairness thresholds as mandatory release criteria.

**Social and managerial implications.** A model with 22.9% malignant recall for Fitzpatrick V–VI patients misses more than three in four malignant lesions in the population least likely to receive specialist follow-up — causing direct harm if deployed. Fairness auditing on purpose-built held-out data (Fitzpatrick17k, DDI) is a non-negotiable release criterion; in-distribution audits alone would have missed these gaps entirely. There is also a privacy obligation: skin images are sensitive biometric data requiring data minimization, informed consent, and access controls. From a business standpoint, the $150–$300 consultation cost barrier represents a real market opportunity, and AI triage could reduce unnecessary specialist visits by 40–60% while escalating high-risk cases. However, the managerial recommendation is staged deployment with supervised clinical review and equity metrics treated as hard release gates — not aspirational targets. The cost of diverse training data is far smaller than the legal and reputational risk of a system that discriminates against underserved patients.

---

## Acknowledgments

The authors thank the organizers of ISIC 2019, PAD-UFES-20, and MILK10k for making their datasets publicly available, and the University of Colorado Boulder Leeds School of Business for support throughout the MSBC 5190 course.

---

## References

[1] Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.

[2] Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.

[3] Adamson, A. S., and Smith, A. (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*, 154(11), 1247–1248.

[4] Groh, M., et al. (2021). Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. *CVPR Workshop on Fair, Data-Efficient, and Trusted Computer Vision*.

[5] Combalia, M., et al. (2019). BCN20000: Dermoscopic lesions in the wild. *arXiv:1908.02288*.

[6] Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221.

[7] Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.

[8] Liu, Z., et al. (2022). Swin Transformer V2: Scaling up capacity and resolution. *CVPR*.

[9] Lin, T. Y., et al. (2017). Focal loss for dense object detection. *ICCV*.

[10] Zhang, S., et al. (2023). BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. *arXiv:2303.00915*.
