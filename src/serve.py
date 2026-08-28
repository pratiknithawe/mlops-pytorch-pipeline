import io,os
from pathlib import Path
import torch
from fastapi import FastAPI,File,HTTPException,UploadFile
from PIL import Image
from torchvision import transforms
from model import get_model

CLASSES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
MODEL_PATH=Path(os.getenv("MODEL_PATH","/app/checkpoints/classifier_v1.pt"))
ARCH=os.getenv("MODEL_ARCHITECTURE","resnet18"); N=int(os.getenv("NUM_CLASSES","10"))
app=FastAPI(title="PyTorch CIFAR-10 Classifier",version="1.0.0")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=None
preprocess=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616])])

def load_model():
    global model
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    ckpt=torch.load(MODEL_PATH,map_location=device,weights_only=False)
    model=get_model(ARCH,N); model.load_state_dict(ckpt["model_state_dict"]); model.to(device); model.eval()
try: load_model()
except Exception as exc: print(f"Model loading failed: {exc}",flush=True)

@app.get("/health")
def health():
    if model is None: raise HTTPException(503,"Model is not loaded")
    return {"status":"ok","model":MODEL_PATH.name}

@app.post("/predict")
async def predict(image: UploadFile=File(...)):
    if model is None: raise HTTPException(503,"Model is not loaded")
    try: img=Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as exc: raise HTTPException(400,f"Invalid image: {exc}") from exc
    x=preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode(): probs=torch.softmax(model(x),dim=1)[0]
    preds=[{"class":CLASSES[i],"probability":round(float(probs[i]),6)} for i in range(N)]
    preds.sort(key=lambda z:z["probability"],reverse=True)
    return {"predicted_class":preds[0]["class"],"probabilities":preds}
