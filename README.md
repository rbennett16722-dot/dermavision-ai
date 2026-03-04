# DermaVision AI

A CNN-based skin lesion classifier trained on the HAM10000 dataset, with a built-in fairness audit across Fitzpatrick skin types.

## Overview

DermaVision AI fine-tunes **EfficientNet-B0** (ImageNet pre-trained) to classify dermoscopic images into seven diagnostic categories from the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). A fairness evaluation module then measures per-class performance stratified by Fitzpatrick skin phototype (Types I–VI) to surface potential model bias.

## Lesion Classes (HAM10000)

| Code | Diagnosis |
|------|-----------|
| mel  | Melanoma |
| nv   | Melanocytic nevi |
| bcc  | Basal cell carcinoma |
| akiec| Actinic keratosis / Bowen's disease |
| bkl  | Benign keratosis-like lesions |
| df   | Dermatofibroma |
| vasc | Vascular lesions |

## Model Architecture

- **Backbone:** EfficientNet-B0 (torchvision, ImageNet weights)
- **Classifier head:** Dropout(0.3) → Linear(1280, 7)
- **Loss:** Cross-entropy with class-frequency inverse weighting (addresses HAM10000 class imbalance)
- **Optimizer:** AdamW, lr=1e-4, weight decay=1e-2
- **Scheduler:** CosineAnnealingLR
- **Input:** 224×224 RGB, normalized to ImageNet mean/std
- **Augmentation:** RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation(20°)

## Fairness Audit

After training, `src/fairness_audit.py` computes the following metrics stratified by Fitzpatrick skin type:

- Accuracy, sensitivity (recall), specificity per class
- Equalized odds gap across skin type groups
- Demographic parity difference
- Per-group confusion matrices

Results are written to `results/fairness_report.csv` and visualized in `notebooks/03_fairness_audit.ipynb`.

## Project Structure

```
dermavision-ai/
├── data/                   # Raw and processed datasets (git-ignored, see .gitignore)
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis & class distribution
│   ├── 02_training.ipynb   # Training loop, loss curves, validation metrics
│   └── 03_fairness_audit.ipynb  # Fairness analysis by Fitzpatrick type
├── src/
│   ├── dataset.py          # HAM10000Dataset, transforms, train/val split
│   ├── model.py            # EfficientNetB0 wrapper and classifier head
│   ├── train.py            # Training script (CLI entry point)
│   ├── evaluate.py         # Evaluation metrics and confusion matrix
│   └── fairness_audit.py   # Stratified fairness metrics by skin type
├── results/                # Saved checkpoints, metrics, plots
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/rbennett16722-dot/dermavision-ai.git
cd dermavision-ai

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn matplotlib seaborn tqdm jupyter
```

## Data

Download HAM10000 from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and place the images and metadata CSV in `data/ham10000/`.

Expected structure:
```
data/
└── ham10000/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part1/
    └── HAM10000_images_part2/
```

## Training

```bash
python src/train.py \
  --data_dir data/ham10000 \
  --epochs 30 \
  --batch_size 32 \
  --output_dir results/
```

## Fairness Audit

```bash
python src/fairness_audit.py \
  --checkpoint results/best_model.pth \
  --data_dir data/ham10000 \
  --output results/fairness_report.csv
```

## Results

| Metric | Value |
|--------|-------|
| Val Accuracy | TBD |
| Macro F1 | TBD |
| AUC-ROC (avg) | TBD |

Fairness results by Fitzpatrick type are in `results/fairness_report.csv` after running the audit.

## Limitations & Ethics

- HAM10000 is heavily skewed toward lighter skin tones; fairness metrics should be interpreted with this in mind.
- The model is **not** a clinical diagnostic tool. Do not use for medical decision-making.
- Fitzpatrick type labels in HAM10000 are researcher-assigned, not self-reported.

## License

MIT
