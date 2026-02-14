import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import io

try:
    from .model import PlantCNN
except ImportError:
    from model import PlantCNN


app = FastAPI(title="Plant Disease Classifier")

# -------- Load artifacts --------
DEVICE = "cpu"   # CPU for Docker & <300ms
MODEL_PATH = "artifacts/plant_model.pt"
CLASSES_PATH = "artifacts/class_names.json"

with open(CLASSES_PATH) as f:
    class_names = json.load(f)

model = PlantCNN(num_classes=len(class_names))
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Routes --------
@app.get("/")
def health_check():
    return {"status": "Model is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "prediction": class_names[int(pred.item())],
        "confidence": round(conf.item(), 4)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0', port=8001, log_level="info")
