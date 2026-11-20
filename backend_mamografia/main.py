import os
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.exposure import equalize_adapthist
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "swin_ssf_final.pt"))
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# SSF: deve ser idêntico ao módulo utilizado no notebook de treinamento
# ---------------------------------------------------------------------
class SSF(nn.Module):
    def __init__(self, channels, channel_dim=-1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channel_dim = channel_dim

    def forward(self, x):
        shape = [1] * x.dim()
        shape[self.channel_dim] = -1
        return (
            x * self.gamma.view(shape).to(x.dtype)
            + self.beta.view(shape).to(x.dtype)
        )


def inject_ssf(model):
    """Insere SSF nas mesmas camadas usadas no treinamento."""
    head = model.get_classifier()

    def _inject(module):
        for name, child in list(module.named_children()):
            if child is head:
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, nn.Sequential(child, SSF(child.out_features)))
            elif isinstance(child, nn.LayerNorm):
                setattr(
                    module,
                    name,
                    nn.Sequential(child, SSF(child.normalized_shape[-1])),
                )
            elif isinstance(child, nn.Conv2d):
                setattr(
                    module,
                    name,
                    nn.Sequential(child, SSF(child.out_channels, channel_dim=1)),
                )
            else:
                _inject(child)

    _inject(model)


def load_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Checkpoint não encontrado em {MODEL_PATH}. "
            "Coloque swin_ssf_final.pt no backend ou configure MODEL_PATH."
        )

    # weights_only=False é necessário para ler o checkpoint completo criado
    # pelo próprio projeto (dicionário com metadados e state_dict).
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=int(checkpoint["num_classes"]),
        img_size=int(checkpoint["img_size"]),
    )

    if checkpoint.get("adapt_mode") == "ssf":
        inject_ssf(model)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------
# Pré-processamento para ROI já recortada pelo usuário
# ---------------------------------------------------------------------
def preprocess_roi(image_bytes: bytes, checkpoint: dict) -> torch.Tensor:
    """
    Recebe uma ROI mamográfica já recortada.
    Reproduz a etapa de CLAHE, redimensionamento e normalização
    usada no notebook para a entrada do Swin.

    Retorna tensor (1, 3, 224, 224).
    """
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im)
            # Preserva a faixa dinâmica de imagens em 16 bits. A conversão direta
            # para "L" poderia descartar informação de intensidade.
            if im.mode in ("P", "PA", "1"):
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            else:
                arr = np.asarray(im)
            if arr.ndim == 3:
                gray = arr[..., :3].astype(np.float32).mean(axis=-1)
            elif arr.ndim == 2:
                gray = arr.astype(np.float32)
            else:
                raise ValueError("Formato de imagem não suportado.")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler a imagem. Envie uma imagem PNG, JPEG ou TIFF válida.",
        )

    if gray.size == 0 or min(gray.shape) < 16:
        raise HTTPException(
            status_code=400,
            detail="A ROI está vazia ou é muito pequena. Envie um recorte com pelo menos 16 × 16 pixels.",
        )

    # Igual ao pipeline do treinamento: min-max antes do CLAHE.
    gray01 = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    roi_clahe = equalize_adapthist(gray01, clip_limit=0.02)

    # O pipeline de treinamento completa com zeros para ficar quadrada.
    h, w = roi_clahe.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top = (side - h) // 2
    left = (side - w) // 2
    square[top:top + h, left:left + w] = roi_clahe

    # Conversão uint8 e redimensionamento como no script de geração.
    img_uint8 = np.asarray(
        Image.fromarray((square * 255).astype(np.uint8)).resize(
            (int(checkpoint["img_size"]), int(checkpoint["img_size"])),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )

    # O notebook replica o canal cinza em RGB e normaliza com ImageNet.
    img = np.repeat(img_uint8[..., None], 3, axis=-1).astype(np.float32) / 255.0
    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, checkpoint = load_model()
    app.state.model = model
    app.state.checkpoint = checkpoint
    yield


app = FastAPI(
    title="API de Classificação de Mamografias",
    description=(
        "Demonstração experimental: classificação de uma ROI mamográfica "
        "já recortada pelo usuário."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restringir ao domínio do Space antes de uso público amplo.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API de mamografia funcionando!"}


@app.get("/health")
def health():
    model_loaded = hasattr(app.state, "model")
    return {
        "status": "online" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado precisa ser uma imagem.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="A imagem enviada está vazia.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="A imagem excede o limite de 25 MB.",
        )

    try:
        x = preprocess_roi(image_bytes, app.state.checkpoint).to(DEVICE)
        with torch.inference_mode():
            logit = app.state.model(x).reshape(-1)
            probability_abnormal = torch.sigmoid(logit).item()
    except HTTPException:
        raise
    except Exception as exc:
        # Detalhes técnicos ficam no log do servidor, não na resposta pública.
        print(f"Erro durante a inferência: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Não foi possível processar a imagem.",
        )

    threshold = float(app.state.checkpoint["threshold"])
    predicted_label = int(probability_abnormal >= threshold)
    probability_normal = 1.0 - probability_abnormal

    return {
        "status": "success",
        "filename": file.filename,
        "message": "Classificação experimental concluída.",
        "prediction": {
            "class_id": predicted_label,
            "class_name": "Anormal" if predicted_label == 1 else "Normal",
            "probability_abnormal": round(probability_abnormal, 6),
            "probability_normal": round(probability_normal, 6),
            "threshold": round(threshold, 6),
        },
        "disclaimer": (
            "Resultado de pesquisa experimental. Não constitui diagnóstico médico "
            "e não deve orientar decisões clínicas."
        ),
    }
