from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
import numpy as np
from fastapi import UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import io

from transformers import AutoTokenizer

# -----------------------------
# IMPORTANT: allow numpy scalar
# -----------------------------
# This is REQUIRED because your checkpoint contains numpy scalars
# and you trust this checkpoint (self-trained model)

import torch.serialization
torch.serialization.add_safe_globals({
    np.core.multiarray.scalar: np.core.multiarray.scalar
})

# -----------------------------
# IMPORT MODEL ARCHITECTURE
# -----------------------------
from model_def import TransformerCNNBiLSTM
from image_model_def import build_backbone, WeightedSoftVotingEnsemble

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="AI Content Verification Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Chrome extension access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD CHECKPOINT (TRUSTED)
# -----------------------------
CHECKPOINT_PATH = "models/cnn_bert_bilstm_hatespeech.pt"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False   # REQUIRED for this checkpoint
)

# -----------------------------
# METADATA
# -----------------------------
MODEL_NAME = checkpoint.get(
    "model_config", {}
).get("model_name", "bert-base-uncased")

MAX_LENGTH = checkpoint.get(
    "model_config", {}
).get("max_length", 128)

CLASS_NAMES = ["Neutral", "Offensive Language", "Hate Speech"]


# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------
# MODEL
# -----------------------------
model = TransformerCNNBiLSTM(
    transformer_model=MODEL_NAME,
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")
print("Classes:", CLASS_NAMES)

# -----------------------------
# PREPROCESSING (MATCH TRAINING)
# -----------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s!?.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class TextRequest(BaseModel):
    text: str

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/analyze/text")
async def analyze_text(req: TextRequest):
    text = preprocess_text(req.text)

    # Safety checks
    if not text or len(text.split()) > 200:
        return {
            "prediction": "Neutral",
            "confidence": 0.0
        }

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, class_idx = torch.max(probabilities, dim=0)

    return {
        "prediction": CLASS_NAMES[class_idx.item()],
        "confidence": round(confidence.item(), 4)
    }

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "Backend running"}

# -----------------------------
# IMAGE MODEL (ENSEMBLE)
# -----------------------------
IMAGE_MODEL_PATH = "models/image_fusion_model.pth"

image_ckpt = torch.load(
    IMAGE_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

ARCHS = image_ckpt["architectures"]
STATE_LIST = image_ckpt["model_state"]
IMAGE_CLASSES = image_ckpt["labels"]

RATIO_DICT = image_ckpt["ensemble_ratios"]
WEIGHTS = [RATIO_DICT[a] for a in ARCHS]

IMAGE_CLASSES = image_ckpt["labels"]
models_list = []

for arch in ARCHS:
    m = build_backbone(
        name=arch,
        num_classes=len(IMAGE_CLASSES)
    )
    m.to(DEVICE)
    m.eval()
    models_list.append(m)

image_model = WeightedSoftVotingEnsemble(
    models=models_list,
    weights=WEIGHTS,
    num_classes=len(IMAGE_CLASSES)
).to(DEVICE)


print("✅ Image ensemble loaded:", ARCHS)
print("Image classes:", IMAGE_CLASSES)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}

    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = image_model(img_tensor)[0]

    conf, idx = torch.max(probs, dim=0)

    return {
        "prediction": IMAGE_CLASSES[idx.item()],
        "confidence": round(conf.item(), 4)
    }
