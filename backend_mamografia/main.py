
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API de Classificação de Mamografias",
    description="Backend experimental para receber imagens de mamografia."
)

# Permite que a interface hospedada no Hugging Face acesse a API.
# Depois podemos restringir ao domínio específico do seu Space.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API de mamografia funcionando!"}


@app.get("/health")
def health():
    return {"status": "online"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado precisa ser uma imagem."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="A imagem enviada está vazia."
        )

    return {
        "status": "success",
        "filename": file.filename,
        "message": "Imagem recebida pela API com sucesso!",
        "image_size_bytes": len(image_bytes),
        "prediction": None
    }
