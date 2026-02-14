# Plant Disease Classifier

Simple FastAPI + PyTorch project to classify plant diseases from images.

- API: FastAPI
- Model: PyTorch (scripted `plant_model.pt`)

To run locally:

```bash
python -m app.main
```

To build Docker image:

```bash
docker build -t plant-api .
```