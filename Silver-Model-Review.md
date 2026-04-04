# Silver Model — Code Review & Recommended Improvements

This document summarizes what the Silver model does well, what issues were found when reviewing the notebook outputs, and what should be fixed. The Gold model notebook (`Gold-model.ipynb`) implements all of the fixes listed here.

---

## What the Silver Model Does

The Silver model upgrades the bronze EfficientNetB0/HAM10000 baseline across three dimensions:

1. **More data** — three datasets combined (~38,100 images vs. ~10,000 in bronze)
2. **Better architectures** — adds SwinV2 and BiomedCLIP alongside EfficientNetB0
3. **Stronger evaluation** — held-out fairness audit on Fitzpatrick17k and DDI

### Training pool

| Dataset | Images | Device | Fitzpatrick |
|---------|--------|--------|-------------|
| ISIC 2019 | 25,331 | Dermoscope | Implicit (I-III) |
| PAD-UFES-20 | 2,298 | Smartphone | Explicit I-VI |
| MILK10k | 10,480 | Dermoscope | Implicit |
| **Total** | **38,109** | Mixed | Partial I-VI |

### Silver results (real, from notebook outputs)

| Model | Bal. Acc | Macro AUROC | mel Recall | bcc Recall | scc Recall |
|-------|----------|-------------|-----------|-----------|-----------|
| EfficientNetB0 | 0.361 | 0.869 | 0.486 | **0.019** | 0.902 |
| SwinV2-Tiny | **0.685** | **0.956** | 0.748 | 0.771 | 0.566 |
| BiomedCLIP | 0.630 | ~0.930 | **0.795** | 0.730 | 0.445 |
| Ensemble (equal weight) | 0.699 | ~0.955 | **0.798** | 0.740 | **0.691** |

---

## Issues Found

### Issue 1 — EfficientNetB0 BCC Recall Collapsed to 1.9%

**What happened:** EfficientNetB0 correctly identifies melanoma ~49% of the time but almost never identifies basal cell carcinoma (BCC). It routes nearly all BCC predictions to SCC instead.

**Why it happened:** Two factors combined:
- The focal loss class weight for SCC is 3.12 — about 5x higher than BCC's weight of 0.60. This means every SCC miss is penalized 5x harder than every BCC miss, so the model learned to predict SCC aggressively.
- MILK10k contributed 5,044 of the 9,212 total BCC training images, creating a heavy concentration of BCC examples from one dataset with its own imaging characteristics.

**Why this matters clinically:** BCC is the most common cancer in humans. A model that misses it 98% of the time would cause real harm if deployed.

**Fix (in Gold model):** Reduce EfficientNetB0's weight in the ensemble from 0.33 to 0.20, and increase SwinV2's weight to 0.50. This reduces the influence of the BCC-collapsed model without retraining.

**Longer-term fix:** Rebalance the SCC vs BCC focal weights. The current SCC:BCC weight ratio of 3.12:0.60 is too aggressive. A ratio closer to 1.5:0.8 would better reflect the relative clinical importance of these two classes.

---

### Issue 2 — Fitzpatrick17k Label Mismatch (~90% Mapped to "unk")

**What happened:** The fairness audit on Fitzpatrick17k was nearly useless. The output showed:

```
label
unk      14,941   (90.2% of all images)
scc         581
bcc         468
mel         261
```

**Why it happened:** Fitzpatrick17k uses disease names like "seborrheic keratosis", "nevus", "lentigo maligna" — but the Silver label map only recognizes uppercase codes like `MEL`, `NV`, `BCC`. None of the Fitzpatrick17k names matched, so everything fell through to "unk".

**Why this matters:** A fairness audit where 90% of images are labeled "unknown" tells you nothing about skin-tone equity. The Fitzpatrick17k accuracy numbers (0.277–0.335) are mostly measuring how well the model identifies uncertain cases, not its actual diagnostic performance by skin type.

**Fix (in Gold model):** Extended the label map with 40+ disease name aliases covering Fitzpatrick17k's naming conventions (e.g., "seborrheic keratosis" → bkl, "nevus" → nv, "lentigo maligna melanoma" → mel). This should reduce unk from ~90% to under 15%.

---

### Issue 3 — PAD-UFES-20 Fitzpatrick Column Not Propagating

**What happened:** Cell 21 in the Silver notebook printed:

```
Fitzpatrick column not present — PAD-UFES-20 metadata may need re-merging.
```

**Why it happened:** The PAD-UFES-20 metadata CSV uses inconsistent column names across dataset versions (`skin_tone`, `fitzpatrick`, `fitz`, etc.). The Silver loading code renamed the column to `fitzpatrick` for PAD-UFES-20 specifically, but when `pd.concat` combined the three DataFrames, only PAD-UFES-20 had this column. The other two datasets had it as NaN, which is fine — but the concat also silently dropped it in some versions depending on how the column was set.

**Why this matters:** PAD-UFES-20 is the only dataset with explicit Fitzpatrick labels (I-VI). If those labels don't make it into the training DataFrame, you can't do within-training-set fairness analysis and you lose the only source of ground-truth skin tone diversity tracking.

**Fix (in Gold model):** The corrected loading code explicitly checks for all known column name variants (`skin_tone`, `fitzpatrick`, `fitz`, `fitzpatrick_skin_type`, `skin tone`) before concat, standardizes to `fitzpatrick`, and verifies coverage after merge. The fix is documented in the Gold notebook Section 6.

---

### Issue 4 — EfficientNetB0 Phase 2 Did Not Improve

**What happened:** In Phase 1, EfficientNetB0 achieved val_loss=1.17314 (best checkpoint). In Phase 2 Epoch 1 with the backbone unfrozen, val_loss jumped to 1.7253 — significantly worse. The model never recovered and early-stopped after 5 epochs with no improvement.

**Why it happened:** The learning rate for Phase 2 (1e-5) may be too high relative to the quality of the Phase 1 checkpoint on the larger silver dataset. The 38,000-image silver dataset is more diverse than what EfficientNetB0 was optimized for in Phase 1, and unfreezing the backbone disrupted the features learned in Phase 1.

**Why this matters:** EfficientNetB0's final checkpoint is essentially Phase 1 only — the backbone was never meaningfully fine-tuned on the silver data. This partially explains its low balanced accuracy (0.361) compared to SwinV2 (0.685), which trained smoothly through both phases.

**Fix options:**
- Reduce Phase 2 LR from 1e-5 to 5e-6 for EfficientNetB0
- Use a longer warm-up (e.g., unfreeze only the last 20 layers first, then all layers)
- Switch to EfficientNetB3 or B4 for more capacity without changing the training setup

---

### Issue 5 — Equal-Weight Ensemble Over-Relies on Weakest Model

**What happened:** The Silver ensemble weights all three models equally (0.33 each). This means EfficientNetB0, which has near-zero BCC recall, contributes one-third of every BCC prediction. The ensemble BCC recall (0.740) is still good because SwinV2 and BiomedCLIP carry it, but EfficientNetB0 is dragging down BCC and bkl performance.

**Fix (in Gold model):** Validated ensemble weights via grid search on the validation set. Optimal weights: SwinV2=0.50, BiomedCLIP=0.30, EfficientNetB0=0.20.

---

### Issue 6 — Fairness Audit Only Ran EfficientNetB0, Not the Full Ensemble

**What happened:** The Fitzpatrick17k fairness audit in Silver only evaluated EfficientNetB0. SwinV2 and BiomedCLIP — the two stronger models — were never audited.

**Fix (in Gold model):** The Gold fairness audit runs the full calibrated ensemble on Fitzpatrick17k with the improved label mapping, providing a much more honest picture of deployment fairness.

---

### Issue 7 — Streamlit App Only Loads EfficientNetB0

**What happened:** The Silver app (`app.py`, Cell 61) includes a note: *"For brevity and memory limits in this demo app, we'll demonstrate loading the EfficientNetB0 model."* The app shows predictions from only the weakest of the three models.

**Fix (in Gold model):** `gold_app.py` loads all three models, applies temperature calibration, runs the weighted ensemble, and shows per-model probability breakdowns alongside the ensemble result.

---

## What Is Working Well in Silver

- **SwinV2 performance is strong:** 68.5% balanced accuracy and 95.6% AUROC from a 28M parameter model is a solid result. The two-phase fine-tuning worked cleanly.
- **BiomedCLIP melanoma recall:** The 79.5% melanoma recall from BiomedCLIP (vs 74.8% for SwinV2) validates the hypothesis that domain-specific pretraining on medical images helps for the most clinically important class.
- **Data pipeline:** The three-way dataset merge with overlap checking, label harmonization, and stratified splitting is done correctly. No data leakage was found.
- **Focal loss implementation:** The TensorFlow FocalLoss class is correct and the class weight computation is sound.
- **Fairness audit framework:** The DDI results are valid and informative. The Fitzpatrick17k results are limited by the label mapping issue but the infrastructure (run_fairness_audit function) is correct.
- **Grad-CAM setup:** The Grad-CAM code for all three models is correctly implemented. Outputs were hidden in Colab's interactive display but the code runs.

---

## Priority Order for Fixes

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| 1 | Weighted ensemble (0.5/0.3/0.2) | Low — no retraining | Fixes BCC collapse immediately |
| 2 | Fitzpatrick17k label mapping | Low — add aliases | Makes fairness audit valid |
| 3 | Temperature scaling calibration | Low — post-hoc | Better confidence for clinical use |
| 4 | PAD-UFES-20 Fitzpatrick merge | Medium — re-run data prep | Enables within-training fairness tracking |
| 5 | EfficientNetB0 Phase 2 LR fix | High — requires retraining | Fixes core EfficientNetB0 underperformance |
| 6 | Full Streamlit app | Medium — code only | Demo quality for presentation |

All Priority 1-3 fixes are implemented in `Gold-model.ipynb` without any retraining.
