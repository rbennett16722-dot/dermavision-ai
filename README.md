# DermaVision AI — Silver 2.0

A transformer-enhanced skin lesion classifier trained on a multi-source, multi-device dataset, with held-out fairness auditing across Fitzpatrick skin types I–VI.

**Team:** Group 7 | Kishore, Ellie Lansdown, Ryan Bennett, Grace Callahan & Stephanie Furst  
**Course:** MSBC 5190 Modern AI for Business | Spring 2026

---

## Overview

DermaVision AI classifies dermoscopic and clinical skin lesion images into 9 diagnostic categories. The project follows a Bronze → Silver → Gold progression.

**Silver 2.0** (current) builds on the Silver baseline by adding three post-training improvements — Test-Time Augmentation (TTA), temperature calibration, and Nelder-Mead optimized ensemble weights — without changing the underlying model architectures or training data. The weighted ensemble achieves balanced accuracy **72.1%**, macro AUROC **0.957**, melanoma recall **78.8%**, and BCC recall **73.3%**.

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
| MILK10k (dermoscopic stream) | 10,480 | Dermoscope | ISIC 2025 challenge | Implicit |
| **Total** | **38,109** | Mixed | Multi-country | I–VI |

The bronze model (HAM10000 only, ~10,000 images, Fitzpatrick I–III dominant) served as the controlled baseline. The silver pool nearly quadruples the training size, adds smartphone images (PAD-UFES-20), and meaningfully extends darker skin tone coverage.

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
- **Why:** Unlike ImageNet-pretrained models, BiomedCLIP already understands the visual language of medicine before fine-tuning.

### Ensemble (Silver 2.0 — Weighted)
Silver 2.0 replaces the equal-weight ensemble with **Nelder-Mead optimized weights** (SwinV2 ≈ 0.5, BiomedCLIP ≈ 0.3, EfficientNetB0 ≈ 0.2), calibrated on the validation set to maximize balanced accuracy. This corrects Silver's over-reliance on EfficientNetB0, which had near-zero BCC recall.

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

### Silver 2.0 Post-Training Improvements

| Fix | Issue | Resolution |
|-----|-------|-----------|
| Weighted ensemble | Equal weights over-relied on EfficientNetB0 (near-zero BCC recall) | scipy Nelder-Mead on val set; Swin≈0.5, CLIP≈0.3, Eff≈0.2 |
| Temperature scaling | Models overconfident; "95% mel" when true confidence ≈60% | T calibrated via LBFGS on val logits; accuracy unchanged |
| Test-Time Augmentation | Single-pass inference noisy on edge cases | 5 augmented views per image, probabilities averaged |

---

## Explainability (Grad-CAM)

Grad-CAM heatmaps are generated for all three models to show which image regions drove each classification decision. This is essential for clinical trustworthiness — the model should attend to the lesion, not dermoscope artefacts or image borders.

- **EfficientNetB0:** TensorFlow GradientTape approach targeting the last convolutional block
- **SwinV2 / BiomedCLIP:** `pytorch-grad-cam` library targeting the final transformer block

---

## Fairness Audit

Two held-out datasets — never seen during training — stress-test performance across skin tones:

| Dataset | Size | Skin Tone Labels |
|---------|------|------------------|
| Fitzpatrick17k | 16,574 images | Fitzpatrick I–VI (explicit) |
| DDI (Stanford) | 656 images | Skin tone groups 1–2, 3–4, 5–6 |

---

## Results

Test set: 5,717 images (held-out, never seen during training).  
Clinical thresholds: Mel recall > 0.80 | BCC recall > 0.75 | SCC recall > 0.75 | Bal. Acc > 0.75 | AUROC > 0.90 | Fitz. Gap < 0.05

### Model Performance (Held-Out Test Set, n = 5,717)

| Model | Bal. Acc | Macro AUROC | Mel Recall | BCC Recall | SCC Recall | Fitz. Gap |
|-------|:--------:|:-----------:|:----------:|:----------:|:----------:|:---------:|
| Bronze — EfficientNetB0 (HAM10000) | — | — | — | — | — | — |
| Silver — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Silver — SwinV2-Tiny | 0.713 | 0.957 | 0.761 | 0.748 | 0.600 | 0.358 |
| Silver — BiomedCLIP | 0.602 | 0.930 | 0.797 | 0.653 | 0.566 | 0.193 |
| Silver 2.0 — EfficientNetB0 | 0.372 | 0.869 | 0.531 | 0.043 | 0.709 | 0.164 |
| Silver 2.0 — SwinV2 + TTA + Cal | 0.716 | **0.959** | 0.765 | 0.760 | 0.604 | 0.358 |
| Silver 2.0 — BiomedCLIP + TTA + Cal | 0.599 | 0.932 | **0.791** | 0.677 | 0.562 | 0.193 |
| **Silver 2.0 — Weighted Ensemble** | **0.721** | 0.957 | 0.788 | **0.733** | **0.668** | **0.350** |

> **Note on EfficientNetB0 BCC recall (0.043):** The model routes nearly all BCC predictions to SCC due to SCC's focal weight (3.12×) being 5× higher than BCC's (0.60×). The weighted ensemble corrects this by down-weighting EfficientNetB0 (≈0.2) and up-weighting SwinV2 (≈0.5), which achieves 76.0% BCC recall.

### Fairness Audit — Fitzpatrick17k (held-out, n = 16,574)

#### EfficientNetB0
| Fitzpatrick | N | Accuracy | Mel Recall | BCC Recall |
|:-----------:|--:|:--------:|:----------:|:----------:|
| I | 2,947 | 0.250 | 0.390 | 0.000 |
| II | 4,808 | 0.200 | 0.210 | 0.000 |
| III | 3,308 | 0.180 | 0.360 | 0.000 |
| IV | 2,781 | 0.178 | 0.211 | 0.000 |
| V | 1,533 | 0.232 | 0.174 | 0.000 |
| VI | 635 | 0.293 | 0.273 | 0.000 |
| **Max gap** | — | **0.164** | **0.238** | — |

#### SwinV2-Tiny
| Fitzpatrick | N | Accuracy | Mel Recall | BCC Recall |
|:-----------:|--:|:--------:|:----------:|:----------:|
| I | 2,947 | 0.390 | 0.561 | 0.082 |
| II | 4,808 | 0.501 | 0.395 | 0.083 |
| III | 3,308 | 0.568 | 0.380 | 0.089 |
| IV | 2,781 | 0.674 | 0.211 | 0.092 |
| V | 1,533 | 0.712 | 0.087 | 0.042 |
| VI | 635 | 0.616 | 0.182 | 0.000 |
| **Max gap** | — | **0.358** | **0.474** | — |

#### BiomedCLIP
| Fitzpatrick | N | Accuracy | Mel Recall | BCC Recall |
|:-----------:|--:|:--------:|:----------:|:----------:|
| I | 2,947 | 0.230 | 0.415 | 0.165 |
| II | 4,808 | 0.274 | 0.284 | 0.256 |
| III | 3,308 | 0.285 | 0.440 | 0.232 |
| IV | 2,781 | 0.324 | 0.158 | 0.171 |
| V | 1,533 | 0.360 | 0.174 | 0.125 |
| VI | 635 | 0.277 | 0.182 | 0.000 |
| **Max gap** | — | **0.193** | **0.282** | — |

#### Weighted Ensemble (Silver 2.0)
| Fitzpatrick | N | Accuracy | Mel Recall | BCC Recall |
|:-----------:|--:|:--------:|:----------:|:----------:|
| I | 2,947 | 0.375 | 0.537 | 0.094 |
| II | 4,808 | 0.498 | 0.346 | 0.135 |
| III | 3,308 | 0.570 | 0.460 | — |
| IV | 2,781 | — | — | — |
| V | 1,533 | — | — | — |
| VI | 635 | — | — | — |
| **Max gap** | — | **0.350** | — | — |

All models FAIL both the accuracy equity target (< 0.05 gap) and the melanoma-recall equity target (< 0.10 gap). Darker skin tones (V–VI) are more likely to have melanoma missed — directly inverting the equity goal.

### Fairness Audit — DDI Stanford (held-out, n = 656)

| Skin Tone | N malignant | Malignant Recall (Weighted Ensemble) |
|:----------|:-----------:|:------------------------------------:|
| Fitzpatrick I–II | 49 | 0.367 |
| Fitzpatrick III–IV | 74 | 0.392 |
| Fitzpatrick V–VI | 48 | **0.229** |

Malignant recall is lowest for the darkest skin tone group. The 16.3-point gap between III–IV and V–VI confirms that PAD-UFES-20's ~2,300 darker-skin images are insufficient to offset ISIC 2019's 25,000 predominantly lighter-skin images.

---

## Project Structure

```
dermavision-ai/
├── Silver-model.ipynb          # Silver 1.0: original 3-model ensemble (equal weights)
├── Silver-model-2.0.ipynb      # Silver 2.0: TTA + temperature calibration + weighted ensemble
├── Meeting-Notes.md            # Team meeting notes: Bronze → Silver → Gold planning + results
├── Report-Draft.md             # Full paper draft with Silver 2.0 results
├── Silver-Model-Review.md      # Code review: issues found and recommended fixes
├── Gold-Model-Guide.md         # Gold model architecture plan
├── Project-Proposal.md         # Original project proposal
├── data/                       # Raw datasets (git-ignored; download via kagglehub)
│   └── .gitkeep
├── .gitignore
└── README.md
```

All notebooks run in **Google Colab** (GPU recommended: T4 or A100). Datasets are downloaded automatically via `kagglehub`.

> **Note:** Notebooks exceed GitHub's 1MB preview limit — you will see "Sorry, this file is too large to display." The file downloads and runs correctly. To open: click the file on GitHub → download → upload to Colab via **File → Upload notebook**.

---

## Setup & Running

```bash
git clone https://github.com/rbennett16722-dot/dermavision-ai.git
cd dermavision-ai
```

Open `Silver-model-2.0.ipynb` in Google Colab. Cell 3 installs all dependencies; subsequent cells download all datasets automatically via `kagglehub` (Kaggle API credentials required).

**Datasets downloaded automatically:**
- `andrewmvd/isic-2019`
- `mahdavi1202/skin-cancer` (PAD-UFES-20)
- `nguyenphucduyloc/milk10k-isic-challenge-2025`
- `nazmusresan/fitzpatrick17k` (fairness audit)
- `souvikda/ddidiversedermatology-multimodal-dataset` (fairness audit)

---

## Silver 2.0 vs Silver vs Bronze: Key Improvements

| Dimension | Bronze | Silver | Silver 2.0 |
|-----------|--------|--------|------------|
| Training data | HAM10000 (~10k, 1 dataset) | ISIC 2019 + PAD-UFES-20 + MILK10k (~32.5k, 3 datasets) | Same as Silver |
| Classes | 7 | 9 (adds SCC) | Same as Silver |
| Skin tone coverage | Fitzpatrick I–III dominant | I–VI (PAD-UFES-20 adds darker tones) | Same as Silver |
| Imaging device | Dermoscope only | Dermoscope + smartphone | Same as Silver |
| Architecture | EfficientNetB0 only | EfficientNetB0 + SwinV2 + BiomedCLIP + Equal-weight Ensemble | Same architectures |
| Ensemble weights | — | Equal (1/3 each) | Nelder-Mead optimized |
| Confidence calibration | — | — | Temperature scaling |
| Inference | Single pass | Single pass | TTA (5 views) |
| Loss | Cross-entropy | Focal loss + risk-boosted class weights | Same as Silver |
| Explainability | None | Grad-CAM for all 3 models | Same as Silver |
| Fairness audit | Within HAM10000 only | Held-out Fitzpatrick17k + DDI | Same + improved label mapping |

---

## Limitations & Ethics

- HAM10000 and ISIC 2019 are skewed toward lighter skin tones (Fitzpatrick I–III). PAD-UFES-20 partially addresses this but darker-skin representation remains limited overall.
- All models FAIL the fairness equity targets on Fitzpatrick17k and DDI — this gap must be addressed before any clinical consideration.
- This model is **not** a clinical diagnostic tool. Do not use for medical decision-making.
- Fitzpatrick labels in Fitzpatrick17k are researcher-assigned, not self-reported.
- Only the BiomedCLIP visual encoder is used; the text encoder is discarded.

---

## License

MIT
