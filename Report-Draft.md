# DermaVision AI: A Transformer-Enhanced Skin Lesion Classifier

**Authors:** Ellie Lansdown, Kishore, Ryan Bennett, Grace Callahan, Stephanie Furst  
**Course:** MSBC 5190 Modern AI for Business | Group 7 | Spring 2026

---

## Abstract

Skin cancer is the most diagnosed cancer globally. AI triage tools trained almost exclusively on lighter skin tones (Fitzpatrick I–III) exhibit misdiagnosis rates 30–40% higher for darker-skinned patients. DermaVision AI addresses both the access gap and the equity gap by training a multi-architecture classifier on ~38,100 images from ISIC 2019, PAD-UFES-20, and MILK10k across nine diagnostic classes. We compare EfficientNetB0, SwinV2-Tiny, and BiomedCLIP, combining them in a weighted ensemble (Silver 2.0). The weighted ensemble achieves balanced accuracy 72.1% and macro AUROC 0.957, with melanoma recall 78.8%, BCC recall 73.3%, and SCC recall 66.8%. A held-out fairness audit on Fitzpatrick17k (16,574 images) reveals a maximum accuracy gap of 35.0% across skin tone groups, confirming that architectural improvements alone cannot close the equity gap without more diverse training data.

---

## 1. Introduction

Melanoma has a five-year survival rate of 99% when caught early and 20% when caught late. There is a shortage of roughly 75% of needed dermatologists in low- and middle-income countries, and in developed nations wait times of 2–6 months and costs of $150–$300 create barriers for millions.

AI-based triage systems offer a partial solution. Published models trained on HAM10000 achieve 30–40% lower accuracy on darker skin tones because HAM10000 images come almost entirely from Australian and Austrian clinics serving predominantly lighter-skinned populations.

DermaVision AI targets both problems: (1) accuracy sufficient for credible first-pass triage, and (2) explicit fairness auditing across Fitzpatrick skin types I–VI using held-out datasets never seen during training. Our Silver 2.0 results show a weighted ensemble balanced accuracy of 72.1% and macro AUROC 0.957, with persistent and significant fairness gaps that must be addressed before any clinical deployment.

---

## 2. Related Work

HAM10000 (Tschandl et al., 2018) established the benchmark for dermoscopic classification with 10,015 images across 7 classes. Esteva et al. (2017) demonstrated CNN performance matching dermatologists for melanoma classification. Adamson and Smith (2018) documented the training data equity problem, and Groh et al. (2021) introduced Fitzpatrick17k for skin-type-stratified fairness auditing, showing models without diverse representation perform substantially worse on darker skin tones.

ISIC 2019 (Combalia et al., 2019) extended HAM10000 to ~25,000 images and added SCC. PAD-UFES-20 (Pacheco et al., 2020) provided the first large-scale clinical dataset with explicit Fitzpatrick labels from Brazil. BiomedCLIP (Zhang et al., 2023) pre-trained a ViT on 15 million biomedical image-text pairs from PubMed Central, learning medical visual representations without task-specific supervision.

Our work is the first to combine all three datasets and compare EfficientNetB0, SwinV2, and BiomedCLIP in a single controlled framework, and to apply post-training calibration and ensemble optimization in a healthcare skin lesion context.

---

## 3. Data

### 3.1 Training Pool

| Dataset | Images | Device | Origin | Fitzpatrick |
|---------|--------|--------|--------|-------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I–III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I–VI |
| MILK10k | 10,480 | Dermoscope | ISIC 2025 | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | Partial I–VI |

Labels standardized to 9 classes: mel, nv, bcc, scc, akiec, bkl, df, vasc, unk. Zero image ID overlap confirmed across all three sources. Split: 26,676 train / 5,716 val / 5,717 test (70/15/15, stratified by class).

### 3.2 Class Distribution

The combined pool is substantially imbalanced: nv=38.3%, mel=14.4%, bcc=24.2%, scc=4.6%, akiec=4.2%, bkl=10.4%, df=0.9%, vasc=0.9%, unk=2.1%.

### 3.3 Preprocessing and Augmentation

All images resized to 224×224. Training augmentation: horizontal/vertical flips, rotation (30°), zoom (20%), brightness jitter (0.7–1.3×), channel shift (20 units) — stronger than the bronze model to handle cross-device variation from PAD-UFES-20 smartphone images.

---

## 4. Methods

### 4.1 Two-Phase Fine-Tuning

All three models follow the same protocol. Phase 1 (8–10 epochs, LR=1e-3): backbone frozen, head trained. Phase 2 (15–20 epochs, LR=1e-5): full backbone unfrozen, end-to-end fine-tuning. Early stopping (patience=5) and ReduceLROnPlateau prevent overfitting.

### 4.2 Focal Loss with Risk-Boosted Class Weights

We replaced cross-entropy with focal loss (Lin et al., 2017) with γ=2.0. Per-class alpha weights computed from inverse class frequencies, then risk-boosted: mel ×1.5, bcc ×1.3, scc ×1.3. Final weights: mel=1.16, nv=0.29, bcc=0.60, scc=3.12, akiec=2.65, bkl=1.07, df=12.35, vasc=12.20, unk=5.22.

### 4.3 EfficientNetB0

EfficientNetB0 (Tan and Le, 2019) compound-scales depth, width, and resolution. TensorFlow/Keras with head: GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(9, softmax). Phase 2 trainable params: 4,798,085.

### 4.4 SwinV2-Tiny

SwinV2 (Liu et al., 2022) uses shifted window self-attention, allowing every image patch to attend to every other patch. This captures global lesion structure — border irregularity, color asymmetry — that CNN convolutions cannot access. We use `swinv2_tiny_window8_256` (~28M params) via timm with head: Linear(768, 512) → ReLU → Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 9).

### 4.5 BiomedCLIP

BiomedCLIP (Zhang et al., 2023) pre-trains ViT-Base/16 (86.6M params) on 15M biomedical image-text pairs from PubMed Central. Unlike EfficientNetB0 and SwinV2 which start from ImageNet photographs, BiomedCLIP begins with visual representations already tuned to dermoscopic and clinical imagery. Only the visual encoder is used.

### 4.6 Equal-Weight Ensemble (Silver Baseline)

The Silver baseline ensemble averages softmax probability vectors equally from all three models: P_ensemble = (P_eff + P_swin + P_clip) / 3. Each model has complementary error modes — EfficientNetB0 captures fine texture, SwinV2 captures global spatial structure, BiomedCLIP brings medical domain priors.

### 4.7 Silver 2.0 Post-Training Improvements

Silver 2.0 applies three engineering fixes on top of the Silver architecture:

**Fix 1 — Weighted Ensemble:** The equal-weight ensemble over-relies on EfficientNetB0, which has near-zero BCC recall (0.043). We use scipy Nelder-Mead optimization on the validation set to find optimal weights (starting point: Swin=0.5, CLIP=0.3, Eff=0.2) that maximize balanced accuracy. This substantially down-weights EfficientNetB0 and corrects the BCC recall collapse.

**Fix 2 — Temperature Scaling:** Raw model logits are overconfident — a model predicting "95% melanoma" may have true calibrated confidence closer to 60%. We calibrate temperature T via LBFGS on validation logits for SwinV2 and BiomedCLIP. Accuracy is unchanged; confidence scores become reliable, which matters for the clinical referral use case.

**Fix 3 — Test-Time Augmentation (TTA):** Single-pass inference introduces variance on edge cases. For each test image, we generate 5 augmented views and average the predicted probabilities before taking the argmax. This reduces prediction noise, particularly for ambiguous boundary cases between adjacent classes.

**Additional fixes:** Extended Fitzpatrick17k label mapping (previously ~90% of rows mapped to 'unk' due to disease name mismatches) and normalized PAD-UFES-20 Fitzpatrick column handling after pd.concat.

---

## 5. Results

### 5.1 Model Performance on Held-Out Test Set (n = 5,717)

Clinical thresholds: Mel recall > 0.80 | BCC recall > 0.75 | SCC recall > 0.75 | Bal. Acc > 0.75 | AUROC > 0.90 | Fitz. Gap < 0.05

| Model | Bal. Acc | Macro AUROC | Mel Recall | BCC Recall | SCC Recall | Fitz. Gap |
|-------|:--------:|:-----------:|:----------:|:----------:|:----------:|:---------:|
| Majority class baseline | 0.111 | — | 0.000 | 0.000 | 0.000 | — |
| Silver — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Silver — SwinV2-Tiny | 0.713 | 0.957 | 0.761 | 0.748 | 0.600 | 0.358 |
| Silver — BiomedCLIP | 0.602 | 0.930 | 0.797 | 0.653 | 0.566 | 0.193 |
| Silver 2.0 — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Silver 2.0 — SwinV2 + TTA + Cal | 0.716 | **0.959** | 0.765 | 0.760 | 0.604 | 0.358 |
| Silver 2.0 — BiomedCLIP + TTA + Cal | 0.599 | 0.932 | **0.791** | 0.677 | 0.562 | 0.193 |
| **Silver 2.0 — Weighted Ensemble** | **0.721** | 0.957 | 0.788 | **0.733** | **0.668** | **0.350** |

No model meets all clinical thresholds. The weighted ensemble is the strongest overall performer. BiomedCLIP achieves the highest individual melanoma recall (79.1%), reflecting its biomedical pretraining. SwinV2 has the best AUROC (0.959 with TTA + calibration).

**On EfficientNetB0 BCC recall (0.043):** The model routes nearly all BCC predictions to SCC due to the SCC focal weight (3.12×) being 5× the BCC weight (0.60×). Silver 2.0's weighted ensemble corrects this by reducing EfficientNetB0's contribution (≈0.2 weight), allowing SwinV2's 76.0% BCC recall to dominate.

**Silver 2.0 vs Silver gains (weighted ensemble):**
- BCC recall: 0.043 (Eff-only issue corrected) → 0.733 in ensemble
- Balanced accuracy: 0.699 (Silver equal-weight) → 0.721 (Silver 2.0 weighted)
- Fitzpatrick gap: 0.358 (Silver) → 0.350 (Silver 2.0)

### 5.2 Fairness Audit — Fitzpatrick17k (held-out, n = 16,574)

#### Silver 2.0 Weighted Ensemble

| Fitzpatrick | N | Accuracy | Mel Recall | BCC Recall |
|:-----------:|--:|:--------:|:----------:|:----------:|
| I | 2,947 | 0.375 | 0.537 | 0.094 |
| II | 4,808 | 0.498 | 0.346 | 0.135 |
| III | 3,308 | 0.570 | 0.460 | — |
| **Max gap** | — | **0.350** | — | — |

All models **FAIL** both equity targets (accuracy gap target < 0.05; mel-recall gap target < 0.10). The Fitzpatrick17k label mapping remains imperfect — disease nomenclature in Fitzpatrick17k does not fully align with our 9-class system, causing residual mapping to 'unk'. An improved extended label map in Silver 2.0 reduces (but does not eliminate) this issue.

Individual model fairness on Fitzpatrick17k:

| Model | Max Accuracy Gap | Max Mel-Recall Gap | Verdict |
|-------|:----------------:|:------------------:|:-------:|
| EfficientNetB0 | 0.164 | 0.238 | FAIL |
| SwinV2 | 0.358 | 0.474 | FAIL |
| BiomedCLIP | 0.193 | 0.282 | FAIL |
| Weighted Ensemble | 0.350 | — | FAIL |

### 5.3 Fairness Audit — DDI Stanford (held-out, n = 656)

| Skin Tone | N malignant | Malignant Recall (Weighted Ensemble) |
|:----------|:-----------:|:------------------------------------:|
| Fitzpatrick I–II | 49 | 0.367 |
| Fitzpatrick III–IV | 74 | 0.392 |
| Fitzpatrick V–VI | 48 | **0.229** |

Malignant recall is lowest for the darkest skin tone group (22.9%), with a 16.3-point gap between III–IV and V–VI groups. PAD-UFES-20's 2,298 darker-skin images are insufficient to offset ISIC 2019's 25,331 predominantly lighter-skin images.

---

## 6. Conclusion

DermaVision AI demonstrates that multi-source, multi-architecture training substantially outperforms single-dataset baselines. The Silver 2.0 weighted ensemble achieves balanced accuracy 72.1%, macro AUROC 0.957, melanoma recall 78.8%, and BCC recall 73.3% — all improvements over the Silver equal-weight ensemble.

However, no model meets the clinical thresholds for deployment. The fairness audit reveals a 35.0% maximum accuracy gap and 23.8–47.4% mel-recall gaps across Fitzpatrick types. **Closing these gaps requires more diverse training data — not just different architectures or post-training techniques.**

**Next steps (Gold model):**
1. MILK10k paired clinical + dermoscopic two-stream late fusion — multimodal architecture
2. MRA-MIDAS 2025 as real-world prospective test set
3. Expanded Fitzpatrick IV–VI training data with explicit labels
4. Skin-tone-aware augmentation and oversampling
5. Streamlit deployment app wrapping the Gold ensemble
6. Clinical validation with fairness thresholds as explicit release criteria

**Business and social implications:** An accurate, equitable classifier could democratize dermatological triage in low-resource settings. The $150–$300 consultation cost barrier represents a real market opportunity — AI triage could reduce unnecessary specialist visits by 40–60% while escalating high-risk cases. However, deployment without addressing the identified fairness gaps risks causing harm to the patients most in need. We recommend staged deployment with fairness audits as mandatory release criteria.

**Lessons learned:**
- **Data diversity matters more than architecture:** Moving from HAM10000 to three datasets provided larger accuracy gains than switching from EfficientNetB0 to SwinV2
- **Class weight calibration is fragile:** Aggressive focal weights caused unexpected BCC/SCC confusion in EfficientNetB0; the weighted ensemble partially corrects this but does not eliminate it
- **Fairness auditing requires purpose-built held-out data:** Fitzpatrick17k and DDI revealed gaps that in-distribution audits would have missed
- **Post-training improvements have limits:** TTA, temperature scaling, and ensemble weight optimization improve balanced accuracy but cannot compensate for training data bias

---

## References

- Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*.
- Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer. *Nature*.
- Adamson, A. S., and Smith, A. (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*.
- Groh, M., et al. (2021). Evaluating deep neural networks with Fitzpatrick 17k. *CVPR Workshop*.
- Combalia, M., et al. (2019). BCN20000: Dermoscopic lesions in the wild. *arXiv:1908.02288*.
- Pacheco, A. G. C., et al. (2020). PAD-UFES-20. *Data in Brief*.
- Tan, M., and Le, Q. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML*.
- Liu, Z., et al. (2022). Swin Transformer V2. *CVPR*.
- Lin, T., et al. (2017). Focal loss for dense object detection. *ICCV*.
- Zhang, S., et al. (2023). BiomedCLIP. *arXiv:2303.00915*.
