import os
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.exposure import equalize_adapthist
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "beatriz-paz/swin-mamografia")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "swin_ssf_int8.onnx")
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Metadados do pré-processamento/modelo.
# IMG_SIZE, MODEL_THRESHOLD, MODEL_MEAN e MODEL_STD devem corresponder
# exatamente aos valores usados no treinamento/validação.
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.6843695044517517"))
MODEL_MEAN = [float(v) for v in os.getenv(
    "MODEL_MEAN", "0.485,0.456,0.406"
).split(",")]
MODEL_STD = [float(v) for v in os.getenv(
    "MODEL_STD", "0.229,0.224,0.225"
).split(",")]

# Limita as threads para reduzir o pico de RAM em instâncias pequenas.
ORT_INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "1"))
ORT_INTER_OP_THREADS = int(os.getenv("ORT_INTER_OP_THREADS", "1"))


def resolve_model_path() -> Path:
    """Localiza o ONNX localmente ou baixa do Hugging Face Hub."""
    if MODEL_PATH_ENV:
        local_path = Path(MODEL_PATH_ENV)
        if not local_path.is_file():
            raise RuntimeError(f"MODEL_PATH não existe: {local_path}")
        return local_path

    local_model = BASE_DIR / MODEL_FILENAME
    if local_model.is_file():
        return local_model

    try:
        return Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                repo_type="model",
                token=os.getenv("HF_TOKEN") or None,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível baixar o ONNX do Hugging Face. "
            "Confira HF_MODEL_REPO_ID, HF_MODEL_FILENAME e HF_TOKEN."
        ) from exc


def load_model():
    if len(MODEL_MEAN) != 3 or len(MODEL_STD) != 3:
        raise RuntimeError("MODEL_MEAN e MODEL_STD precisam ter três valores.")
    if any(value == 0 for value in MODEL_STD):
        raise RuntimeError("Nenhum valor de MODEL_STD pode ser zero.")

    model_path = resolve_model_path()
    print(f"Carregando modelo ONNX: {model_path}")
    print(f"Tamanho do arquivo: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    options.inter_op_num_threads = ORT_INTER_OP_THREADS
    options.enable_mem_pattern = False

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Entrada ONNX: {input_info.name} | shape={input_info.shape} | type={input_info.type}")
    print(f"Saída ONNX: {output_info.name} | shape={output_info.shape} | type={output_info.type}")

    return session, {
        "input_name": input_info.name,
        "output_name": output_info.name,
        "img_size": IMG_SIZE,
        "mean": MODEL_MEAN,
        "std": MODEL_STD,
        "threshold": MODEL_THRESHOLD,
    }


# ---------------------------------------------------------------------
# Pré-processamento: preservado do main.py anterior
# ---------------------------------------------------------------------
def preprocess_roi(image_bytes: bytes, config: dict) -> np.ndarray:
    """
    Recebe uma ROI mamográfica já recortada.
    Mantém min-max, CLAHE, preenchimento quadrado, redimensionamento,
    replicação do canal cinza e normalização do código anterior.

    Retorna array float32 no formato NCHW: (1, 3, IMG_SIZE, IMG_SIZE).
    """
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im)
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

    # Min-max antes do CLAHE, como no código anterior.
    gray01 = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    roi_clahe = equalize_adapthist(gray01, clip_limit=0.02)

    # Preenchimento central com zeros para formar uma imagem quadrada.
    h, w = roi_clahe.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top = (side - h) // 2
    left = (side - w) // 2
    square[top:top + h, left:left + w] = roi_clahe

    img_uint8 = np.asarray(
        Image.fromarray((square * 255).astype(np.uint8)).resize(
            (config["img_size"], config["img_size"]),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )

    # Escala [0,1], replica cinza em RGB e normaliza.
    img = np.repeat(img_uint8[..., None], 3, axis=-1).astype(np.float32) / 255.0
    mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config["std"], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    # ONNX geralmente recebe NCHW para modelos exportados do PyTorch.
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def predict_probability(session, config, input_array: np.ndarray) -> float:
    """Executa o grafo ONNX e converte o logit binário em probabilidade."""
    outputs = session.run(
        [config["output_name"]],
        {config["input_name"]: input_array},
    )
    output = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    if output.size != 1:
        raise RuntimeError(
            f"Esperava uma saída binária única (1 logit), mas o ONNX retornou shape {np.asarray(outputs[0]).shape}."
        )

    logit = float(output[0])
    # O main.py original aplicava sigmoid ao logit do modelo.
    # Portanto, esta implementação mantém a mesma interpretação.
    logit = float(np.clip(logit, -80.0, 80.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    session, config = load_model()
    app.state.session = session
    app.state.config = config
    yield
    app.state.session = None


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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API de mamografia funcionando com ONNX Runtime!"}


@app.get("/health")
def health():
    model_loaded = getattr(app.state, "session", None) is not None
    return {
        "status": "online" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "runtime": "ONNX Runtime CPU",
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
        config = app.state.config
        x = preprocess_roi(image_bytes, config)
        probability_abnormal = predict_probability(app.state.session, config, x)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Erro durante a inferência: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Não foi possível processar a imagem.",
        )

    threshold = float(config["threshold"])
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
