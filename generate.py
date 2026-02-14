import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
from app.model import PlantCNN

# ---- CONFIG ----
NUM_CLASSES = 5
MODEL_PATH = "artifacts/best_model.pth"
OUTPUT_MODEL = "artifacts/plant_model.pt"

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato__Target_Spot",
    "Tomato_healthy"
]

# ----------------

device = "cpu"

model = PlantCNN(num_classes=NUM_CLASSES)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.eval()

# TorchScript (FAST inference)
example_input = torch.randn(1, 3, 128, 128)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(OUTPUT_MODEL)

# Save class names
with open("artifacts/class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f)

print("✅ Model exported successfully")
