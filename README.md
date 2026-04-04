# DermaVision AI — Silver Model

A transformer-enhanced skin lesion classifier trained on a multi-source, multi-device dataset, with a held-out fairness audit across Fitzpatrick skin types I–VI.

**Team:** Group 7 | Kishore, Ellie Lansdown, Ryan Bennett, Grace Callahan & Stephanie Furst  
**Course:** MSBC 5190 Modern AI for Business | Spring 2026

---

## Overview

DermaVision AI classifies dermoscopic and clinical skin lesion images into 9 diagnostic categories. The **Silver Model** builds on an EfficientNetB0 bronze baseline by expanding the training data to ~38,100 images across three datasets, introducing two additional architectures (SwinV2 and BiomedCLIP), and adding a held-out fairness evaluation on independently sourced datasets (Fitzpatrick17k, DDI).

---

## Lesion Classes (9 harmonised)

| Code | Diagnosis | Risk |
|------|-----------|------|
| mel | Melanoma | HIGH |
| nv | Melanocytic nevi | Low |
| bcc | Basal cell carcinoma | HIGH |
| scc | Squamous cell carcinoma | HIGH |
| akiec | Actinic keratosis / Bowen's disease | Moderate |
| bkl | Benign keratosis-like lesions | Low |
| df | Dermatofibroma | Low |
| vasc | Vascular lesions | Low |
| unk | Other / Unknown | — |

---

## Training Data

| Dataset | Images | Device | Origin | Fitzpatrick |
|---------|--------|--------|--------|-------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I–III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I–VI |
| MILK10k | 10,480 | Dermoscope | ISIC 2025 challenge | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | I–VI |

The bronze model (HAM10000 only, ~10,000 images, Fitzpatrick I–III dominant) served as the controlled baseline. The silver pool nearly quadruples the training size, adds smartphone images, and meaningfully extends darker skin tone coverage via PAD-UFES-20.

---

## Model Architecture

Three models are trained and compared. All use a **two-phase fine-tuning** strategy: Phase 1 freezes the backbone and trains the classification head; Phase 2 unfreezes the backbone for end-to-end fine-tuning at a low learning rate.

### Model 1 — EfficientNetB0 (Silver Baseline)
- **Framework:** TensorFlow / Keras
- **Backbone:** EfficientNetB0 (ImageNet pre-trained)
- **Head:** GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(9, softmax)
- **Purpose:** Controlled reference — isolates the contribution of data diversity from architecture changes.

### Model 2 — SwinV2-Tiny (Vision Transformer)
- **Framework:** PyTorch + timm
- **Backbone:** `swinv2_tiny_window8_256` (~28M params, ImageNet pre-trained)
- **Head:** Linear(768, 512) → ReLU → Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 9)
- **Why:** Self-attention captures long-range spatial dependencies (global lesion asymmetry, irregular borders) that CNN convolutions cannot.

### Model 3 — BiomedCLIP
- **Framework:** PyTorch + open_clip
- **Backbone:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` — pre-trained on 15M biomedical image–text pairs from PubMed Central
- **Why:** Unlike ImageNet-pretrained models, BiomedCLIP already understands the visual language of medicine (dermoscopy, histology, clinical photography) before fine-tuning.

### Ensemble
Averages softmax probability vectors from all three models. Each model has complementary failure modes: EfficientNetB0 excels at texture, SwinV2 at global structure, BiomedCLIP at domain-specific features. The ensemble is more confident where all three agree and more uncertain where they disagree — a natural referral signal for ambiguous cases.

---

## Training Details

| Setting | Value |
|---------|-------|
| Loss | Focal loss (γ=2.0) + risk-boosted class weights |
| Risk boost | mel ×1.5, bcc ×1.3, scc ×1.3 |
| Split | 70 / 15 / 15 stratified |
| Input size | 224 × 224 RGB |
| Batch size | 32 |
| Phase 1 LR | 1e-3 (head only) |
| Phase 2 LR | 1e-5 (full fine-tune) |
| Augmentation | HFlip, VFlip, Rotation(30°), Zoom, ColorJitter, ChannelShift |
| Mixed precision | float16 compute, float32 weights |

Focal loss replaces standard cross-entropy to down-weight easy majority-class examples and concentrate gradient signal on hard, clinically important minority classes (mel, bcc, scc).

---

## Explainability (Grad-CAM)

Grad-CAM heatmaps are generated for all three models to show which image regions drove each classification decision. This is essential for clinical trustworthiness — the model should attend to the lesion, not to dermoscope artefacts or image borders.

- EfficientNetB0: TensorFlow GradientTape approach targeting the last convolutional block
- SwinV2 / BiomedCLIP: `pytorch-grad-cam` library targeting the final transformer block

---

## Fairness Audit

Two held-out datasets — never seen during training — stress-test performance across skin tones:

| Dataset | Size | Skin Tone Labels |
|---------|------|------------------|
| Fitzpatrick17k | 16,577 images | Fitzpatrick I–VI (explicit) |
| DDI (Stanford) | ~656 images | Skin tone groups 1–2, 3–4, 5–6 |

Per-Fitzpatrick-type accuracy, melanoma recall, and BCC recall are reported for each model and the ensemble. A performance gap between lighter (I–II) and darker (V–VI) skin tones flags potential deployment bias.

---

## Results

Test set: 5,717 images (held-out, never seen during training).

| Model | Bal. Accuracy | Macro AUROC | Mel Recall | BCC Recall | SCC Recall |
|-------|:------------:|:-----------:|:----------:|:----------:|:----------:|
| Bronze — EfficientNetB0 (HAM10000) | — | — | — | — | — |
| EfficientNetB0 (Silver) | 0.361 | 0.869 | 0.486 | 0.019 | 0.902 |
| SwinV2-Tiny (Silver) | **0.685** | **0.956** | 0.748 | **0.771** | 0.566 |
| BiomedCLIP (Silver) | 0.630 | 0.930 | **0.795** | 0.730 | 0.445 |
| Ensemble (equal weight) | 0.699 | 0.955 | **0.798** | 0.740 | **0.691** |

SwinV2 is the strongest single model. BiomedCLIP achieves the highest melanoma recall (79.5%), reflecting its biomedical pretraining. The ensemble is the best overall performer.

> **Note on EfficientNetB0 BCC recall (0.019):** The model routes nearly all BCC predictions to SCC due to a class weight imbalance in the focal loss (SCC weight 5× higher than BCC). This is corrected in the Gold model via a weighted ensemble. See `Silver-Model-Review.md` for full analysis.

### Fairness Audit — Fitzpatrick17k (held-out, n=16,574)

| Fitzpatrick Type | N | Accuracy | Mel Recall |
|:----------------|--:|:--------:|:----------:|
| Type I | 2,947 | 0.277 | 0.098 |
| Type II | 4,808 | 0.281 | 0.185 |
| Type III | 3,308 | 0.271 | 0.140 |
| Type IV | 2,781 | 0.248 | 0.132 |
| Type V | 1,533 | 0.241 | 0.087 |
| Type VI | 635 | 0.335 | 0.273 |
| **Max gap** | — | **0.124** | **0.214** |

### Fairness Audit — DDI Stanford (held-out, n=656)

| Skin Tone | Malignant Recall (Ensemble) |
|:----------|:---------------------------:|
| Fitzpatrick I–II | 0.490 |
| Fitzpatrick III–IV | 0.568 |
| Fitzpatrick V–VI | **0.354** |

Malignant recall is lowest for the darkest skin tone group, confirming that the training data imbalance toward lighter skin tones (ISIC 2019) persists despite PAD-UFES-20's contribution.

---

## Project Structure

```
dermavision-ai/
├── Silver-model.ipynb      # Main notebook: data loading, training, eval, fairness audit
├── Gold-model.ipynb        # Gold improvements: weighted ensemble, TTA, temperature scaling
├── Report-Draft.md         # Full paper draft with real Silver results
├── Silver-Model-Review.md  # Code review: issues found and recommended fixes
├── Gold-Model-Guide.md     # Plain-language guide to Gold improvements and how to run
├── Project-Proposal.md     # Original project proposal
├── data/                   # Raw datasets (git-ignored; download via kagglehub)
│   └── .gitkeep
├── .gitignore
└── README.md
```

All notebooks run in **Google Colab** (GPU recommended: T4 or A100). Datasets are downloaded automatically via `kagglehub`.

> **Note:** `Silver-model.ipynb` is 1.2MB and exceeds GitHub's 1MB notebook preview limit — you will see "Sorry, this file is too large to display." The file downloads and runs correctly. To open it: click the file on GitHub → click the download button, then upload to Colab via **File → Upload notebook**.

---

## Setup & Running

```bash
git clone https://github.com/rbennett16722-dot/dermavision-ai.git
cd dermavision-ai
```

Open `Silver-model.ipynb` in Google Colab. Cell 3 installs all dependencies and subsequent cells download all datasets automatically via `kagglehub` (Kaggle API credentials required).

**Datasets downloaded automatically:**
- `andrewmvd/isic-2019`
- `mahdavi1202/skin-cancer` (PAD-UFES-20)
- `nguyenphucduyloc/milk10k-isic-challenge-2025`
- `nazmusresan/fitzpatrick17k` (fairness audit)
- `souvikda/ddidiversedermatology-multimodal-dataset` (fairness audit)

---

## Silver vs. Bronze: Key Improvements

| Dimension | Bronze | Silver |
|-----------|--------|--------|
| Training data | HAM10000 (~10k, 1 dataset) | ISIC 2019 + PAD-UFES-20 + MILK10k (~32.5k, 3 datasets) |
| Classes | 7 | 9 (adds SCC) |
| Skin tone coverage | Fitzpatrick I–III dominant | I–VI (PAD-UFES-20 adds darker tones) |
| Imaging device | Dermoscope only | Dermoscope + smartphone |
| Architecture | EfficientNetB0 only | EfficientNetB0 + SwinV2 + BiomedCLIP + Ensemble |
| Loss | Cross-entropy | Focal loss + risk-boosted class weights |
| Explainability | Not implemented | Grad-CAM for all 3 models |
| Fairness audit | Within HAM10000 only | Held-out Fitzpatrick17k + DDI |

---

## Limitations & Ethics

- HAM10000 and ISIC 2019 are skewed toward lighter skin tones (Fitzpatrick I–III). PAD-UFES-20 partially addresses this but darker-skin representation remains limited overall.
- This model is **not** a clinical diagnostic tool. Do not use for medical decision-making.
- Fitzpatrick labels in Fitzpatrick17k are researcher-assigned, not self-reported.
- Only the BiomedCLIP visual encoder is used; the text encoder is discarded.

---

## License

MIT
