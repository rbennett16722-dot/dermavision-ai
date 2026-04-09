"""
DermaVision AI — Silver 2.1 FastAPI Backend
============================================
Loads the three Silver 2.0 checkpoints and runs the weighted ensemble
(EfficientNetB0 + SwinV2-tiny + BiomedCLIP) on an uploaded image.

Environment variables (all optional, fall back to defaults):
  CKPT_DIR   — directory containing the three .keras / .pt files
                default: ./checkpoints
  EFF_CKPT   — full path to EfficientNetB0 .keras file
  SWIN_CKPT  — full path to SwinV2 .pt file
  CLIP_CKPT  — full path to BiomedCLIP .pt file
  W_EFF      — ensemble weight for EfficientNetB0  (default 0.20)
  W_SWIN     — ensemble weight for SwinV2          (default 0.50)
  W_CLIP     — ensemble weight for BiomedCLIP      (default 0.30)
  T_SWIN     — temperature scalar for SwinV2       (default 1.50)
  T_CLIP     — temperature scalar for BiomedCLIP   (default 1.50)

Run locally:
  uvicorn main:app --reload --port 8000

Run in Colab with ngrok:
  See Silver-model-2.1.ipynb — Section 5.
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_DIR  = os.getenv("CKPT_DIR",  "./checkpoints")
EFF_CKPT  = os.getenv("EFF_CKPT",  f"{CKPT_DIR}/effnet_silver2_best.keras")
SWIN_CKPT = os.getenv("SWIN_CKPT", f"{CKPT_DIR}/swinv2_silver2_best.pt")
CLIP_CKPT = os.getenv("CLIP_CKPT", f"{CKPT_DIR}/biomedclip_silver2_best.pt")

W_EFF  = float(os.getenv("W_EFF",  "0.20"))
W_SWIN = float(os.getenv("W_SWIN", "0.50"))
W_CLIP = float(os.getenv("W_CLIP", "0.30"))
T_SWIN = float(os.getenv("T_SWIN", "1.50"))
T_CLIP = float(os.getenv("T_CLIP", "1.50"))

# ── Class metadata ─────────────────────────────────────────────────────────────
LABEL_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc", "scc", "unk"]
LABEL_FULL  = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / IEC",
    "bkl":   "Benign Keratosis-Like",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular Lesion",
    "scc":   "Squamous Cell Carcinoma",
    "unk":   "Unknown / Other",
}
RISK_TIER = {
    "mel":   "high",
    "bcc":   "high",
    "scc":   "high",
    "akiec": "moderate",
    "nv":    "low",
    "bkl":   "low",
    "df":    "low",
    "vasc":  "low",
    "unk":   "low",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Populated during startup
_models: dict = {}
_ready: bool = False


# ── Model architecture definitions (must mirror Silver-model-2.0.ipynb exactly) ─
def _build_swin_classifier(n_classes: int, device):
    import timm
    import torch.nn as nn

    backbone  = timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                   num_classes=0, global_pool="avg")
    embed_dim = backbone.num_features

    head = nn.Sequential(
        nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    )

    class SwinClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head     = head

        def forward(self, x):
            return self.head(self.backbone(x))

    return SwinClassifier(backbone, head).to(device)


def _build_biomedclip_classifier(n_classes: int, device):
    import torch
    import torch.nn as nn
    import open_clip

    clip_model, _, val_transform = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    vision_encoder = clip_model.visual.to(device)

    with torch.no_grad():
        dummy     = torch.randn(1, 3, 224, 224).to(device)
        embed_dim = vision_encoder(dummy).shape[1]

    class BiomedCLIPClassifier(nn.Module):
        def __init__(self, encoder, embed_dim, n_classes):
            super().__init__()
            self.encoder = encoder
            self.head    = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 256),       nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.head(self.encoder(x))

    return BiomedCLIPClassifier(vision_encoder, embed_dim, n_classes).to(device), val_transform


# ── Preprocessing helpers ──────────────────────────────────────────────────────
def _eff_preprocess(img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = np.array(img.resize((224, 224)), dtype=np.float32)
    return preprocess_input(np.expand_dims(arr, 0))  # [1, 224, 224, 3]


def _swin_preprocess(img: Image.Image):
    import torch
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0)  # [1, 3, 256, 256]


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_single(img: Image.Image) -> dict:
    import torch
    import torch.nn.functional as F

    device = _models["device"]
    img    = img.convert("RGB")

    # EfficientNetB0 (TF/Keras)
    arr      = _eff_preprocess(img)
    prob_eff = _models["eff"].predict(arr, verbose=0)[0]           # [9]

    # SwinV2 (PyTorch + temperature scaling)
    x_swin     = _swin_preprocess(img).to(device)
    with torch.no_grad():
        logits_swin = _models["swin"](x_swin)
        prob_swin   = F.softmax(logits_swin / T_SWIN, dim=1).cpu().numpy()[0]  # [9]

    # BiomedCLIP (PyTorch + temperature scaling)
    x_clip = _models["clip_transform"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_clip = _models["clip"](x_clip)
        prob_clip   = F.softmax(logits_clip / T_CLIP, dim=1).cpu().numpy()[0]  # [9]

    # Weighted ensemble (weights normalised to sum to 1)
    total    = W_EFF + W_SWIN + W_CLIP
    w_e, w_s, w_c = W_EFF / total, W_SWIN / total, W_CLIP / total
    ensemble = w_e * prob_eff + w_s * prob_swin + w_c * prob_clip  # [9]

    top_idx   = int(ensemble.argmax())
    top_label = LABEL_NAMES[top_idx]

    def _to_dict(probs):
        return {lbl: round(float(p), 4) for lbl, p in zip(LABEL_NAMES, probs)}

    return {
        "top_class":    top_label,
        "top_full":     LABEL_FULL[top_label],
        "risk_tier":    RISK_TIER[top_label],
        "ensemble":     _to_dict(ensemble),
        "efficientnet": _to_dict(prob_eff),
        "swinv2":       _to_dict(prob_swin),
        "biomedclip":   _to_dict(prob_clip),
        "meta": {
            "weights": {"eff": round(w_e, 3), "swin": round(w_s, 3), "clip": round(w_c, 3)},
            "temperatures": {"swin": T_SWIN, "clip": T_CLIP},
            "label_full":  LABEL_FULL,
            "risk_tier":   RISK_TIER,
        },
    }


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    import tensorflow as tf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _models["device"] = device
    print(f"[DermaVision 2.1] Device: {device}")

    # ── EfficientNetB0 ──────────────────────────────────────
    print("[DermaVision 2.1] Loading EfficientNetB0 …")

    class FocalLoss(tf.keras.losses.Loss):
        """Must match Silver-model-2.0 definition exactly for .keras deserialization."""
        def __init__(self, gamma=2.0, alpha=None, name="focal_loss", **kwargs):
            super().__init__(name=name, **kwargs)
            self.gamma = gamma
            self.alpha = alpha

        def call(self, y_true, y_pred):
            y_pred   = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
            ce       = -y_true * tf.math.log(y_pred)
            focal_wt = tf.pow(1.0 - y_pred, self.gamma)
            fl       = focal_wt * ce
            if self.alpha is not None:
                fl = self.alpha * fl
            return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"gamma": self.gamma})
            return cfg

    _models["eff"] = tf.keras.models.load_model(
        EFF_CKPT, custom_objects={"FocalLoss": FocalLoss}
    )
    print("[DermaVision 2.1]  ✓ EfficientNetB0")

    # ── SwinV2 ─────────────────────────────────────────────
    print("[DermaVision 2.1] Loading SwinV2 …")
    swin = _build_swin_classifier(len(LABEL_NAMES), device)
    swin.load_state_dict(torch.load(SWIN_CKPT, map_location=device))
    swin.eval()
    _models["swin"] = swin
    print("[DermaVision 2.1]  ✓ SwinV2")

    # ── BiomedCLIP ──────────────────────────────────────────
    print("[DermaVision 2.1] Loading BiomedCLIP …")
    clip_model, clip_transform = _build_biomedclip_classifier(len(LABEL_NAMES), device)
    clip_model.load_state_dict(torch.load(CLIP_CKPT, map_location=device))
    clip_model.eval()
    _models["clip"]           = clip_model
    _models["clip_transform"] = clip_transform
    print("[DermaVision 2.1]  ✓ BiomedCLIP")

    global _ready
    print("[DermaVision 2.1] All models ready — serving on http://localhost:8000")
    _ready = True
    yield
    _ready = False
    _models.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app       = FastAPI(title="DermaVision AI — Silver 2.1", lifespan=lifespan)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _ready:
        raise HTTPException(status_code=503, detail="Models are still loading, please wait a moment and try again.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return predict_single(img)


@app.get("/health")
async def health():
    loaded = [k for k in _models if k != "device"]
    return {"status": "ready" if _ready else "loading", "models_loaded": loaded}
