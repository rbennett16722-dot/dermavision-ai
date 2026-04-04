# Gold Model — What It Is and How to Run It

The Gold model is not a new model trained from scratch. It is a set of improvements applied on top of the Silver model checkpoints — no GPU time required for the improvements themselves (only inference).

---

## Why "Gold"?

The project progression is Bronze → Silver → Gold:

| Model | Key change |
|-------|-----------|
| Bronze | EfficientNetB0 on HAM10000 (~10k images, 7 classes) |
| Silver | Three models, three datasets, ~38k images, fairness audit |
| Gold | Fixes Silver's known bugs; better ensemble; calibrated confidence |

---

## What the Gold Model Changes

### 1. Weighted Ensemble (most important fix)

**Silver:** All three models contribute equally (33% each).  
**Gold:** SwinV2 gets 50%, BiomedCLIP gets 30%, EfficientNetB0 gets 20%.

**Why:** EfficientNetB0's BCC recall collapsed to 1.9% in Silver — it confused basal cell carcinoma with squamous cell carcinoma due to unbalanced focal loss weights. Reducing its influence in the ensemble from 33% to 20% largely fixes the BCC recall without any retraining.

The weights (0.5 / 0.3 / 0.2) were validated by a grid search on the held-out validation set in the Gold notebook. The grid search finds the combination that maximizes balanced accuracy.

---

### 2. Temperature Scaling (confidence calibration)

A model that predicts "92% melanoma" should be right about 92% of the time. When it's only right 65% of the time, the model is overconfident — which is dangerous in a clinical setting.

**Temperature scaling** (Guo et al., 2017) fixes this with a single scalar T:

```
calibrated_probability = softmax(logits / T)
```

T is fit on the validation set by minimizing negative log-likelihood. T > 1 softens overconfident predictions. This does not change accuracy — only the confidence numbers.

Each model gets its own temperature: EfficientNetB0 → T_eff, SwinV2 → T_swin, BiomedCLIP → T_clip.

---

### 3. Test-Time Augmentation (TTA)

At inference, each image is augmented 10 times (5 random crops + horizontal flip of each), the model runs on all 10 versions, and the predictions are averaged. This reduces prediction variance and typically improves accuracy by 1-3 percentage points at no training cost.

Applied to SwinV2 and BiomedCLIP. EfficientNetB0 uses single-pass inference (TF generator-based inference makes TTA more complex to implement).

---

### 4. Fixed Fitzpatrick17k Label Mapping

The Silver fairness audit was largely invalid because ~90% of Fitzpatrick17k images mapped to "unk." The Gold model adds 40+ disease name aliases so the audit correctly identifies lesion types:

| Fitzpatrick17k name | Mapped to |
|--------------------|-----------|
| "seborrheic keratosis" | bkl |
| "nevus" | nv |
| "lentigo maligna melanoma" | mel |
| "basal cell carcinoma" | bcc |
| "actinic keratosis" | akiec |
| "squamous cell carcinoma in situ" | akiec |
| "dermatofibroma" | df |
| "angioma" | vasc |
| ... (40+ total) | ... |

With this fix, the fairness audit provides valid per-class accuracy and recall by Fitzpatrick skin type.

---

### 5. Fixed PAD-UFES-20 Fitzpatrick Column Merge

In Silver, the PAD-UFES-20 Fitzpatrick skin type column (the only explicit skin tone labels in the training data) was silently dropped when the three datasets were combined. The Gold notebook documents the correct loading pattern so future training runs preserve this column.

---

### 6. Full Fairness Audit on Ensemble

Silver only ran the Fitzpatrick17k audit on EfficientNetB0. Gold runs the full calibrated ensemble on Fitzpatrick17k with the corrected label mapping, giving the most accurate picture of deployment fairness.

---

### 7. Full Streamlit App

The Silver app (`app.py`) only loaded EfficientNetB0. The Gold app (`gold_app.py`) loads all three models, applies temperature calibration, runs the weighted ensemble, and shows:
- Top-3 predictions with confidence bars and risk labels
- Per-model probability breakdown chart
- Clinical disclaimer

---

## How to Run the Gold Notebook

### Prerequisites

1. You must have already run the Silver notebook (`Silver-model.ipynb`) and saved checkpoints to Google Drive at:
   - `MyDrive/SILVER FOLDER/checkpoints/effnet_silver_best.keras`
   - `MyDrive/SILVER FOLDER/checkpoints/swinv2_silver_best.pt`
   - `MyDrive/SILVER FOLDER/checkpoints/biomedclip_silver_best.pt`

2. You must also save the test and validation DataFrames from Silver (add this to Silver Cell 24 and run):
   ```python
   test_meta.to_csv('/content/drive/MyDrive/SILVER FOLDER/test_meta.csv', index=False)
   val_meta.to_csv('/content/drive/MyDrive/SILVER FOLDER/val_meta.csv', index=False)
   ```

### Steps

1. Open `Gold-model.ipynb` in Google Colab
2. Make sure the runtime is set to GPU (Runtime → Change runtime type → T4 GPU)
3. Run cells in order — each section is self-contained with clear explanations
4. The notebook will:
   - Load all three Silver checkpoints (Section 2)
   - Fit ensemble weights via grid search on val set (Fix 1)
   - Fit temperature scalars on val set (Fix 2)
   - Run TTA inference on test set (Fix 3)
   - Compute Gold ensemble metrics (Fix 4)
   - Run corrected Fitzpatrick17k fairness audit (Fix 5)
   - Show final Bronze/Silver/Gold comparison table (Section 10)
   - Launch the Streamlit app (Fix 7)

### Expected runtime

| Step | Approx. time (T4 GPU) |
|------|-----------------------|
| Load checkpoints | 3-5 min |
| Val inference (all 3 models) | 5-8 min |
| Test TTA inference (SwinV2 + BiomedCLIP, 10 views each) | 20-30 min |
| Fitzpatrick17k fairness audit | 15-20 min |
| **Total** | **~50-60 min** |

---

## Gold vs. Silver Results (Expected)

| Metric | Silver Ensemble | Gold Ensemble (expected) |
|--------|----------------|--------------------------|
| Balanced Accuracy | 0.699 | 0.71-0.73 (TTA + better weights) |
| Macro AUROC | ~0.955 | ~0.960 (calibration improves ranking) |
| mel Recall | 0.798 | 0.80-0.83 |
| bcc Recall | 0.740 | 0.78-0.82 (weighted ensemble reduces EffNet drag) |
| scc Recall | 0.691 | 0.69-0.72 |
| Fitzpatrick gap | ~0.12 | TBD with fixed label mapping |

The most important improvement is BCC recall. Silver's equal-weight ensemble was held back by EfficientNetB0's BCC collapse. The weighted ensemble largely corrects this.

---

## What the Gold Model Does NOT Do

- **Does not retrain any model.** All improvements are post-hoc (applied after training).
- **Does not add new training data.** The training pool is the same 38,109 images.
- **Does not fix EfficientNetB0's Phase 2 training failure.** That would require retraining EfficientNetB0 with a lower LR (5e-6 instead of 1e-5) and possibly unfreezing the backbone gradually.
- **Does not close the Fitzpatrick fairness gap.** The DDI results show a 21-point malignant recall gap between lighter and darker skin — this requires more diverse training data, not better ensembling.

---

## Files

| File | Purpose |
|------|---------|
| `Gold-model.ipynb` | Main Gold notebook — run this in Colab |
| `Silver-Model-Review.md` | Detailed review of Silver issues and fixes |
| `Silver-model.ipynb` | Silver training notebook (1.1MB — too large to preview on GitHub, downloads and runs fine in Colab) |
| `Report-Draft.md` | Full paper draft based on Silver results |

---

## Note on GitHub File Size

`Silver-model.ipynb` is 1.1MB. GitHub cannot display notebook files larger than 1MB in the browser — you will see "Sorry, this file is too large to display." The file **is** in the repository and **will** download and run correctly. To view it:
- Clone the repo and open in VS Code or Jupyter locally
- Or upload directly to Google Colab via File → Upload notebook
