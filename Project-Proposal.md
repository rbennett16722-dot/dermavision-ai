# DermaVision AI — Project Proposal

**Course:** Modern AI for Business (MSBC 5190) | Track A — Develop an AI Application  
**Team:** Ellie Lansdown, Kishore, Ryan Bennett, Grace Callahan & Stephanie Furst

---

## 1. Problem

Skin cancer is the world's most diagnosed cancer, yet access to dermatological care is critically unequal. A shortage of roughly 75% of needed dermatologists in low- and middle-income countries leaves hundreds of millions without specialist access. Even in developed nations, wait times of 2–6 months and consultation costs of $150–$300 create prohibitive barriers.

The stakes are high: early detection of melanoma raises the five-year survival rate from 20% to 99%. Every month of delayed triage can mean the difference between a curable and a terminal outcome.

A second, less-discussed failure compounds the access problem: existing AI triage tools are biased. Nearly all published dermatology AI models train almost exclusively on light skin tones (Fitzpatrick I–III), yielding misdiagnosis rates 30–40% higher for patients with darker skin. The communities most underserved by healthcare are also most underserved by the AI tools meant to help them.

**DermaVision AI addresses both failures directly: lack of access and lack of equity.**

---

## 2. Data Sources

Three complementary datasets are used to build a robust, equitable classifier:

| Dataset | Images | Device | Key Contribution |
|---------|--------|--------|-----------------|
| HAM10000 | ~10,000 | Dermoscope | Bronze model baseline; 7 diagnostic classes |
| ISIC 2019 | ~25,000 | Dermoscope | Scale; adds SCC as 8th class |
| PAD-UFES-20 | ~2,300 | Smartphone | Explicit Fitzpatrick I–VI labels; Brazil origin |
| MILK10k | ~5,200 | Dermoscope | ISIC 2025 challenge data; additional diversity |

---

## 3. Methods

The core method is **transfer learning on a convolutional neural network**. The bronze model uses EfficientNetB0 pretrained on ImageNet (14M images, 1,000 classes), replacing the final classification head with a custom softmax layer fine-tuned on the target datasets. Transfer learning is appropriate given the relatively small training set; borrowing general visual features substantially improves generalization.

**Key implementation components:**

- **Class imbalance correction:** Weighted cross-entropy (bronze) / focal loss (silver) penalizes errors on minority classes (e.g., melanoma at ~11%) more heavily than the dominant majority class (melanocytic nevi at ~67%).
- **Data augmentation:** Random flips, rotations, and color jitter expand the training distribution and improve robustness to photographic variation across imaging devices.
- **Explainability (Grad-CAM):** Backward hooks on the final convolutional/transformer layer generate heatmap overlays showing which image regions drove each prediction — replacing black-box outputs with clinically legible evidence.
- **Fairness evaluation:** A custom evaluation loop segments test-set performance by Fitzpatrick skin type (I–VI), producing a quantitative audit of accuracy disparity across groups. The silver model extends this with held-out Fitzpatrick17k and DDI datasets never seen during training.

**Baselines:**
1. Naive majority-class predictor (~67% accuracy)
2. Unweighted ResNet50 without class balancing (78–82%)

**Target:** >85% balanced accuracy with a maximum 5% accuracy gap across any two Fitzpatrick groups.

**Tech stack:** TensorFlow/Keras, PyTorch, timm, open_clip, scikit-learn, Pandas, Matplotlib/Seaborn

---

## 4. Evolution from Proposal to Silver Model

The proposal outlined a single EfficientNetB0 model on HAM10000. The silver model expands this substantially:

| Dimension | Proposal (Bronze) | Silver Model |
|-----------|------------------|--------------|
| Data | HAM10000 (~10k) | 3 datasets (~32.5k) |
| Classes | 7 | 9 (adds SCC) |
| Architecture | EfficientNetB0 only | EfficientNetB0 + SwinV2 + BiomedCLIP + Ensemble |
| Loss | Weighted cross-entropy | Focal loss + risk-boosted class weights |
| Fairness audit | Within test set only | Held-out Fitzpatrick17k + DDI |
| Explainability | Planned | Fully implemented for all 3 models |
