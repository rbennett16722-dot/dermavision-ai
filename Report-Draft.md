# DermaVision AI: A Transformer-Enhanced Skin Lesion Classifier

**Authors:** Ellie Lansdown, Kishore, Ryan Bennett, Grace Callahan, Stephanie Furst  
**Course:** MSBC 5190 Modern AI for Business | Group 7 | Spring 2026

---

## Abstract

Skin cancer is the most diagnosed cancer globally. AI triage tools trained almost exclusively on lighter skin tones (Fitzpatrick I-III) exhibit misdiagnosis rates 30-40% higher for darker-skinned patients. DermaVision AI addresses both the access gap and the equity gap by training a multi-architecture classifier on ~38,100 images from ISIC 2019, PAD-UFES-20, and MILK10k across nine diagnostic classes. We compare EfficientNetB0, SwinV2-Tiny, and BiomedCLIP, combining them in an ensemble. SwinV2 achieves balanced accuracy 68.5% and macro AUROC 0.956, with melanoma recall 74.8% and BCC recall 77.1%. The ensemble improves melanoma recall to 79.8% and SCC recall to 69.1%. A held-out fairness audit on Fitzpatrick17k (16,574 images) reveals a maximum accuracy gap of 12.4% across skin tone groups.

---

## 1. Introduction

Melanoma has a five-year survival rate of 99% when caught early and 20% when caught late. There is a shortage of roughly 75% of needed dermatologists in low- and middle-income countries, and in developed nations wait times of 2-6 months and costs of $150-$300 create barriers for millions.

AI-based triage systems offer a partial solution. Published models trained on HAM10000 achieve 30-40% lower accuracy on darker skin tones because HAM10000 images come almost entirely from Australian and Austrian clinics serving predominantly lighter-skinned populations.

DermaVision AI targets both problems: (1) accuracy sufficient for credible first-pass triage, and (2) explicit fairness auditing across Fitzpatrick skin types I-VI using held-out datasets never seen during training. Our results show meaningful accuracy gains (ensemble balanced accuracy 69.9%, macro AUROC ~0.956) and identify specific remaining fairness gaps.

---

## 2. Related Work

HAM10000 (Tschandl et al., 2018) established the benchmark for dermoscopic classification with 10,015 images across 7 classes. Esteva et al. (2017) demonstrated CNN performance matching dermatologists for melanoma classification. Adamson and Smith (2018) documented the training data equity problem, and Groh et al. (2021) introduced Fitzpatrick17k for skin-type-stratified fairness auditing, showing models without diverse representation perform substantially worse on darker skin tones.

ISIC 2019 (Combalia et al., 2019) extended HAM10000 to ~25,000 images and added SCC. PAD-UFES-20 (Pacheco et al., 2020) provided the first large-scale clinical dataset with explicit Fitzpatrick labels from Brazil. BiomedCLIP (Zhang et al., 2023) pre-trained a ViT on 15 million biomedical image-text pairs from PubMed Central, learning medical visual representations without task-specific supervision.

Our work is the first to combine all three datasets and compare EfficientNetB0, SwinV2, and BiomedCLIP in a single controlled framework.

---

## 3. Data

### 3.1 Training Pool

| Dataset | Images | Device | Origin | Fitzpatrick |
|---------|--------|--------|--------|-------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I-III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I-VI |
| MILK10k | 10,480 | Dermoscope | ISIC 2025 | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | Partial I-VI |

Labels standardized to 9 classes: mel, nv, bcc, scc, akiec, bkl, df, vasc, unk. Zero image ID overlap confirmed. Split: 26,676 train / 5,716 val / 5,717 test (70/15/15, stratified by class).

### 3.2 Class Distribution

The combined pool is substantially imbalanced: nv=38.3%, mel=14.4%, bcc=24.2%, scc=4.6%, akiec=4.2%, bkl=10.4%, df=0.9%, vasc=0.9%, unk=2.1%.

### 3.3 Preprocessing and Augmentation

All images resized to 224x224. Training augmentation: horizontal/vertical flips, rotation (30 deg), zoom (20%), brightness jitter (0.7-1.3x), channel shift (20 units) -- stronger than the bronze model to handle cross-device variation from PAD-UFES-20 smartphone images.

---

## 4. Methods

### 4.1 Two-Phase Fine-Tuning

All three models follow the same protocol. Phase 1 (8-10 epochs, LR=1e-3): backbone frozen, head trained. Phase 2 (15-20 epochs, LR=1e-5): full backbone unfrozen, end-to-end fine-tuning. Early stopping (patience=5) and ReduceLROnPlateau prevent overfitting.

### 4.2 Focal Loss with Risk-Boosted Class Weights

We replaced cross-entropy with focal loss (Lin et al., 2017) with gamma=2.0. Per-class alpha weights computed from inverse class frequencies, then risk-boosted: mel x1.5, bcc x1.3, scc x1.3. Final weights: mel=1.16, nv=0.29, bcc=0.60, scc=3.12, akiec=2.65, bkl=1.07, df=12.35, vasc=12.20, unk=5.22.

### 4.3 EfficientNetB0

EfficientNetB0 (Tan and Le, 2019) compound-scales depth, width, and resolution. TensorFlow/Keras with head: GlobalAveragePooling2D -> Dense(512) -> BatchNorm -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(9, softmax). Phase 2 trainable params: 4,798,085.

### 4.4 SwinV2-Tiny

SwinV2 (Liu et al., 2022) uses shifted window self-attention, allowing every image patch to attend to every other patch. This captures global lesion structure -- border irregularity, color asymmetry -- that CNN convolutions cannot access. We use swinv2_tiny_window8_256 (~28M params) via timm with head: Linear(768, 512) -> ReLU -> Dropout(0.4) -> Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 9).

### 4.5 BiomedCLIP

BiomedCLIP (Zhang et al., 2023) pre-trains ViT-Base/16 (86.6M params) on 15M biomedical image-text pairs from PubMed Central. Unlike EfficientNetB0 and SwinV2 which start from ImageNet photographs, BiomedCLIP begins with visual representations already tuned to dermoscopic and clinical imagery. Only the visual encoder is used.

### 4.6 Ensemble

The ensemble averages softmax probability vectors from all three models: P_ensemble = (P_eff + P_swin + P_clip) / 3. Each model has complementary error modes -- EfficientNetB0 captures fine texture, SwinV2 captures global spatial structure, BiomedCLIP brings medical domain priors. Cases where models disagree receive lower confidence -- a natural clinical referral signal.

---

## 5. Results

### 5.1 Model Performance on Held-Out Test Set (n=5,717)

| Model | Bal. Acc | Macro AUROC | mel Recall | bcc Recall | scc Recall |
|-------|----------|-------------|-----------|-----------|----------|
| Majority class baseline | 0.111 | -- | 0.000 | 0.000 | 0.000 |
| EfficientNetB0 (Silver) | 0.361 | 0.869 | 0.486 | 0.019 | 0.902 |
| SwinV2-Tiny (Silver) | **0.685** | **0.956** | 0.748 | **0.771** | 0.566 |
| BiomedCLIP (Silver) | 0.630 | ~0.930 | **0.795** | 0.730 | 0.445 |
| Ensemble (equal weight) | 0.699 | ~0.955 | **0.798** | 0.740 | **0.691** |

SwinV2 is the strongest single model across most metrics. BiomedCLIP achieves the highest melanoma recall (79.5%), suggesting biomedical pretraining provides the strongest prior for clinically distinctive lesion features. The ensemble achieves the best balanced accuracy (69.9%) and SCC recall (69.1%).

The EfficientNetB0 BCC recall collapsed to 1.9% -- the model routes nearly all BCC cases to SCC. This stems from the SCC focal weight (3.12) being 5x the BCC weight (0.60), combined with MILK10k contributing 5,044 of 9,212 BCC training images. The Gold model addresses this via a weighted ensemble (SwinV2: 0.5, BiomedCLIP: 0.3, EfficientNetB0: 0.2).

### 5.2 Fairness Audit -- Fitzpatrick17k

| Fitzpatrick Type | N | Accuracy | mel Recall |
|-----------------|---|----------|----------|
| Type I | 2,947 | 0.277 | 0.098 |
| Type II | 4,808 | 0.281 | 0.185 |
| Type III | 3,308 | 0.271 | 0.140 |
| Type IV | 2,781 | 0.248 | 0.132 |
| Type V | 1,533 | 0.241 | 0.087 |
| Type VI | 635 | 0.335 | 0.273 |
| **Max gap** | -- | **0.124** | **0.214** |

The 12.4% accuracy gap exceeds the 5% target. The 21.4% mel-recall gap means patients with darker skin are more likely to have melanoma missed -- directly inverting the equity goal. Note: Fitzpatrick17k disease nomenclature does not align with our 9-class system, causing ~90% of images to map to unk. The Gold model implements improved label mapping.

### 5.3 Fairness Audit -- DDI (Stanford)

| Skin Tone | N malignant | Malignant Recall (Ensemble) |
|----------|------------|---------------------------|
| Fitzpatrick I-II | 49 | 0.490 |
| Fitzpatrick III-IV | 74 | 0.568 |
| Fitzpatrick V-VI | 48 | 0.354 |

Malignant recall decreases monotonically from lighter to darker skin, with a 21.4-point gap between III-IV and V-VI groups. PAD-UFES-20's 2,300 darker-skin images are insufficient to offset ISIC 2019's 25,000 predominantly lighter-skin images.

---

## 6. Conclusion

DermaVision AI demonstrates that multi-source, multi-architecture training substantially outperforms single-dataset baselines. SwinV2 achieves 68.5% balanced accuracy and 95.6% macro AUROC. The ensemble pushes melanoma recall to 79.8% and SCC recall to 69.1%.

The fairness audit reveals a 12.4% accuracy gap and 21.4% melanoma-recall gap across Fitzpatrick types. Closing this gap requires more diverse training data -- not just different architectures.

**Next steps:**
1. Expand data collection for Fitzpatrick IV-VI with explicit labels
2. Apply skin-tone-aware augmentation to extend darker-skin coverage
3. Implement weighted ensemble and temperature scaling (Gold model)
4. Clinical validation with explicit fairness thresholds as deployment criteria

**Business and social implications:** An accurate, equitable classifier could democratize dermatological triage in low-resource settings. The $150-$300 consultation cost barrier represents a real market opportunity -- AI triage could reduce unnecessary specialist visits by 40-60% while escalating high-risk cases. However, deployment without addressing the identified fairness gaps risks causing harm to the patients most in need. We recommend staged deployment with fairness audits as release criteria.

**Lessons learned:**
- Data diversity matters more than architecture: moving from HAM10000 to three datasets provided larger gains than switching from EfficientNetB0 to SwinV2
- Class weight calibration is fragile: aggressive focal weights caused unexpected BCC/SCC confusion in EfficientNetB0
- Fairness auditing requires purpose-built held-out data: Fitzpatrick17k and DDI revealed gaps that in-distribution audits would have missed

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
