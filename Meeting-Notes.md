# DermaVision AI — Team Meeting Notes

**Course:** MSBC 5190 Modern AI for Business | Group 7 | Spring 2026  
**Team:** Kishore, Ellie Lansdown, Ryan Bennett, Grace Callahan, Stephanie Furst

---

## Project Roadmap Overview

| Tier | Focus | Status |
|------|-------|--------|
| Bronze | Single dataset (HAM10000), EfficientNetB0 baseline | Complete |
| Silver | Multi-source datasets, 3 architectures + ensemble, fairness audit | Complete |
| Silver 2.0 | TTA + temperature calibration + optimized ensemble weights | Complete |
| Gold | Multimodal architecture (image + metadata/NLP) | Planned |

---

## Bronze Model

### Guiding Principles
Three things established before any code was written:
1. **Data matters more than the architecture**
2. **Evaluation metrics must have clinical context**
3. **Healthcare AI demands a higher standard of caution than marketing, finance, or recommendation models**

### Dataset: HAM10000
- 10,015 images total (folder 1 = 5,000, folder 2 = 5,015)
- 7 lesion classes with risk tiers:

| Code | Diagnosis | Risk |
|------|-----------|------|
| mel | Melanoma | High (cancerous) |
| bcc | Basal cell carcinoma | High (cancerous) |
| akiec | Actinic keratoses | Moderate (pre-cancerous) |
| nv | Melanocytic nevi | Low (benign) |
| bkl | Benign keratosis | Low (benign) |
| vasc | Vascular lesions | Low (benign) |
| df | Dermatofibroma | Low (benign) |

### Class Imbalance Problem
Melanocytic nevi (nv) accounts for **67% of all images**. Without correction, a model can achieve 67% accuracy by simply predicting nv for every image — and completely miss melanoma, the clinically highest-priority class.

Standard 70/15/15 train/val/test splits preserve this imbalance, giving a false sense of accuracy.

**Resolution:** Focus on recall (sensitivity) for minority high-risk classes as the primary metric. Relying on overall accuracy alone is dangerously misleading.

Train: 7,010 images (70%) | Val: 1,502 (15%) | Test: 1,503 (15%)

### Diagnosis Confirmation Types (dx_type)
Labels in HAM10000 are clinically validated — not guesses:
- **histo**: Histopathology (biopsy — gold standard, 100% accurate)
- **follow_up**: Clinical follow-up confirming benign status
- **consensus**: Multi-expert dermatologist agreement
- **confocal**: Confocal microscopy (advanced in-vivo imaging)

`dx_type` is NOT used in training or testing — it only validates dataset credibility.

### Distribution Insights
After visualizing age, sex, and localization distributions, HAM10000 is effectively a dataset of images mostly of **"middle-aged white men in Europe with lesions formed on their backs."** Training a model on this data alone produces peak poor generalization — it would be unreliable for a 25-year-old Black woman with a lesion on the sole of her foot.

### Bronze Architectures Tried
1. Baseline CNN from scratch
2. EfficientNetB0 (ImageNet pre-trained) as backbone
3. Fine-tuned EfficientNetB0 (data augmentation, learning rate scheduling, early stopping)

### Bronze Evaluation Metric
- Confusion matrix focused on recall for high-risk classes (mel, bcc)
- F1-score (added after Ellie's insight)
- Fitzpatrick skin tone fairness audit

### Assessment After Bronze
HAM10000's 10,015 images are relatively small for deep CNN training. EfficientNetB0 pre-trained on ImageNet already knows edges, textures, gradients, and shapes — directly relevant to dermoscopy. The rational is the same as using VGG19/ResNet50 in Homework 1.

---

## Silver Model

### Why We Expanded Datasets
Bronze exposed that HAM10000 skews toward lighter skin tones (Fitzpatrick I–III), dermoscopic images only, and middle-aged European male patients. This is not what the world of skin lesions looks like.

### Dataset Research and Selection

#### Datasets considered:
1. **HAM10000** — Already in bronze. 10,015 images, 7 classes, Australia + Austria, Fitzpatrick I–III dominant. Known gaps: skin tone bias, nv imbalance at 67%, no smartphone images.

2. **BCN20000** — 18,946 images, 8 classes, Barcelona Hospital Clínic, dermoscopic, includes nails + mucosa lesions. Would nearly double dataset quality. Gap it fills: image volume + class diversity. *Excluded* because ISIC 2019 already contains most of these images — combining all three would cause data leakage.

3. **ISIC 2019 + ISIC 2020** — ~58,000 combined images, multi-country, dermoscopic, with age/sex/anatomy site metadata. ISIC 2019 supersedes HAM10000 and BCN20000 (contains most of their images). *Selected: ISIC 2019.*

4. **PAD-UFES-20** — 2,298 images, 6 classes, Brazil, smartphone clinical photos from 1,373 patients with 26 metadata features including Fitzpatrick skin type. The **only public dataset with smartphone images + full Fitzpatrick I–VI labels**. Gap it fills: darker skin tones, low-income population, real-world conditions. *Selected.*

5. **MILK10k** — 10,480 total images (5,240 clinical + 5,240 dermoscopic paired). 97% biopsied (histopathology-backed labels). Diverse skin tones. Dermoscopic stream (5,240 images) used for silver. Paired clinical+dermoscopic architecture reserved for Gold (multimodal late fusion). *Dermoscopic stream selected.*

#### Final Silver Dataset:
| Dataset | Images | Device | Origin | Fitzpatrick |
|---------|--------|--------|--------|-------------|
| ISIC 2019 | 25,331 | Dermoscope | Multi-country | Implicit (I–III dominant) |
| PAD-UFES-20 | 2,298 | Smartphone | Brazil | Explicit I–VI |
| MILK10k (dermoscopic stream) | ~5,240 | Dermoscope | ISIC 2025 | Implicit |
| **Total** | **~32,500** | Mixed | Multi-country | Partial I–VI |

Using Kaggle notebooks (free 30GB disk + 30hr GPU/week + P100) was considered to handle the larger dataset — an alternative to Colab.

#### Why 9 Classes (up from 7)
Adding ISIC 2019 and PAD-UFES-20 introduced two new clinically significant classes:
- **SCC** (Squamous cell carcinoma) — from BCN20000/PAD-UFES-20, a malignant class
- **Seborrheic keratosis** — merged into existing categories

Final 9 classes: mel, nv, bcc, akiec, bkl, df, vasc, scc, unk

### Silver Model Architecture Plan

**1. Baseline CNN** — Trained on combined dataset with focal loss + class weights. Evaluated on test data and Fitzpatrick17k. Focus: melanoma recall, overall accuracy, Fitzpatrick gap, AUROC.

**2. Fine-Tuned Models:**
- **Model 1 — EfficientNetB0**: Bronze baseline retrained on silver data (reference point)
- **Model 2 — SwinV2 / ViT-B16**: Vision transformer capturing global attention on lesion features
- **Model 3 — BiomedCLIP**: Pre-trained on biomedical domain imagery; best for transfer learning

**3. Ensemble** — Average softmax probabilities from all three models. Pick class with highest confidence. Ensemble technique should exceed any individual model (analogous to random forest bagging).

### Silver Evaluation Framework

| Metric | Definition | Clinical Meaning |
|--------|-----------|-----------------|
| Melanoma Recall | TP / (TP + FN) | Of all actual melanoma cases, what % did we catch? Missing one = potential death |
| Balanced Accuracy | Average recall across all 9 classes equally weighted | Overall fairness across disease types — not dominated by majority nv class |
| AUROC (per class, one-vs-rest) | Area under ROC curve | At any confidence threshold, how well does model separate that class from all others |
| Fitzpatrick Skin Tone Gap | max recall(Fitz I–III) − min recall(Fitz IV–VI) | Does the model perform equally regardless of patient skin tone |
| False Negative Rate (melanoma) | FN / (TP + FN) = 1 − Recall | How often does the model miss a real melanoma — must be driven as low as possible |
| Overall Accuracy | Reported last, contextualized | A model with 88% accuracy but 60% melanoma recall is worse than one with 82% accuracy and 91% melanoma recall |

### Clinical Thresholds (Silver 2.0 targets)
- Melanoma recall > 0.80
- BCC recall > 0.75
- SCC recall > 0.75
- Balanced accuracy > 0.75
- Macro AUROC > 0.90
- Fitzpatrick gap < 0.05

### Explainability: Grad-CAM
AI models in healthcare must be **explainable**. Neural networks don't tell you how they arrived at a classification. Grad-CAM generates a visual heatmap overlaid on the original image, highlighting the exact pixels the AI paid most attention to — essential for clinical trustworthiness.

### Known Cross-Dataset Risks
Images from a dermoscope in Vienna look different from smartphone photos taken in Brazil, which look different from clinical images from Barcelona. The model might learn to recognize the device or the hospital rather than the lesion.

**Mitigations implemented:**
1. **Track data source as metadata** — every image carries a source label (isic_2019, pad_ufes_20, milk10k) to audit whether errors cluster by source
2. **Test set contains images from all sources** — prevents measuring generalization on only training-source images
3. **Normalize preprocessing per source** — PAD-UFES-20 smartphone images need different preprocessing (resolution, lighting, color profile)

### Fairness Audit Datasets (held-out, never seen during training)
- **Fitzpatrick17k** — 16,577 images with explicit Fitzpatrick I–VI labels
- **DDI (Stanford Diverse Dermatology Images)** — ~656 images, skin tone groups 1–2, 3–4, 5–6
- **MRA-MIDAS 2025** — Considered for Gold model evaluation

### Notebook Structure (Silver)
```
Section 1  - Combined data pipeline (shared by all models)
Section 2  - Baseline CNN: train → evaluate → record
Section 3  - EfficientNetB0 silver: train → evaluate → record
Section 4  - SwinV2-Tiny: train → evaluate → record
Section 5  - BiomedCLIP: train → evaluate → record
Section 6  - Ensemble: combine best models → evaluate → record
Section 7  - Final comparison table across all models
Section 8  - Fitzpatrick fairness audit across all models
```

---

## Silver 2.0

Silver 2.0 applies three engineering fixes on top of the Silver architecture without changing the core models or data:

| Fix | Issue Addressed | Resolution |
|-----|----------------|-----------|
| Weighted Ensemble | Equal weights over-relied on EfficientNetB0 (near-zero BCC recall) | scipy Nelder-Mead optimization on val set; optimal weights: Swin≈0.5, CLIP≈0.3, Eff≈0.2 |
| Temperature Scaling | Models overconfident; "95% mel" when true confidence ≈60% | Temperature T calibrated via LBFGS on val logits; accuracy unchanged, confidence trustworthy |
| Test-Time Augmentation (TTA) | Single-pass inference noisy on edge cases | 5 augmented views per image, probabilities averaged before prediction |
| Fitzpatrick17k label fix | ~90% of Fitz17k rows mapped to 'unk' | Extended label map covering all Fitz17k disease name variants |
| PAD-UFES-20 column fix | Fitzpatrick column lost after pd.concat | Normalized to 'fitzpatrick' before concat; explicitly preserved |

---

## Gold Model — Planning

### Vision
The Gold model introduces a **multimodal architecture**: combining visual features from skin images with structured patient metadata (age, sex, lesion location, Fitzpatrick type, lesion diameter) and optionally natural language symptom descriptions.

### Planned Tech Stack

**Deep Learning Frameworks:**
- PyTorch (primary)
- TensorFlow / TF Lite (mobile deployment)

**Computer Vision:**
- Vision Transformer (`google/vit-base-patch16-224`)
- EfficientNet-V2 (ensemble backup)
- Grad-CAM for explainability

**NLP / Text Processing:**
- BioBERT or PubMedBERT (medical text understanding)
- Hugging Face Transformers library

**Text/Symptom Datasets (planned):**
- PubMed medical literature abstracts (dermatology)
- Reddit/StackExchange medical Q&A (patient symptom descriptions)
- Clinical notes datasets (if available)
- Synthetic symptom descriptions generated via GPT-4

**Multimodal Fusion:**
- Custom cross-attention mechanism
- Late fusion as baseline

**Conversational AI:**
- LangChain for orchestration
- OpenAI/Anthropic API for chatbot interface

**Medical Imaging:**
- MONAI (Medical Open Network for AI) framework

**Deployment:**
- Streamlit (web demo)
- AWS SageMaker (model hosting)
- AWS Lambda (serverless)
- Docker (containerization)

**MLOps:**
- Weights & Biases (experiment tracking)
- DVC (data version control)
- GitHub Actions (CI/CD)

**Mobile:**
- TensorFlow Lite for on-device inference
- React Native or Flutter (if time permits)

### Gold Model Goals
- Add NLP component for symptom text analysis
- Implement cross-attention fusion layer
- Expand disease coverage to 15 conditions
- Integrate Fitzpatrick17k dataset for skin tone diversity
- Target accuracy: 90%+ overall
- Add Grad-CAM explainability visualization
- Deliverable: Full multimodal system with explanations

### Data Reserved for Gold
- **MILK10k paired architecture**: Clinical photo + dermoscopic image → two-stream late fusion. Using it in Silver would collapse the Bronze→Silver→Gold progression and remove the ability to measure what multimodal fusion independently contributes.
- **Streamlit / mobile deployment**: Production app wraps the Gold model ensemble.
- **MRA-MIDAS 2025**: Real-world prospective test set for Gold evaluation.

### Bronze → Silver → Gold Progression

| Tier | Architecture | Data | Key Addition |
|------|-------------|------|-------------|
| Bronze | EfficientNetB0 | HAM10000 (~10k) | Baseline |
| Silver 2.0 | Eff + SwinV2 + BiomedCLIP + Weighted Ensemble | ~32.5k multi-source | Transformer diversity + calibration + fairness audit |
| Gold | Two-stream fusion (CNN + metadata/NLP) | Silver + MILK10k paired + MRA-MIDAS 2025 | Multimodal (clinical image + patient metadata) |
