import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "/app/checkpoints/classifier_v1.pt",
    )
)

ARCHITECTURE = os.getenv(
    "MODEL_ARCHITECTURE",
    "resnet18",
)

NUM_CLASSES = int(
    os.getenv(
        "NUM_CLASSES",
        "10",
    )
)


app = FastAPI(
    title="PyTorch CIFAR-10 Classifier",
    version="1.0.0",
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None


preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)


def load_model() -> None:

    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = get_model(
        architecture=ARCHITECTURE,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()


try:
    load_model()
except Exception as exc:
    print(
        f"Model loading failed: {exc}",
        flush=True,
    )


@app.get("/health")
def health() -> dict:

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ok",
        "model": MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
) -> dict:

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        image_bytes = await image.read()

        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    tensor = preprocess(img)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predictions = [
        {
            "class": CLASSES[index],
            "probability": round(
                float(probabilities[index]),
                6,
            ),
        }
        for index in range(NUM_CLASSES)
    ]

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "predicted_class": predictions[0]["class"],
        "probabilities": predictions,
    }
