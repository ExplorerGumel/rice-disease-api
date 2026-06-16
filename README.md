# Rice Leaf Disease Detector

A production-ready computer vision system that detects rice leaf diseases from images using transfer learning. Built end-to-end — from model training and experiment tracking to a live, publicly deployed inference API.

**Live demo:** [huggingface.co/spaces/Munzali/RICE_DISEASES_CLASSIFICATION](https://huggingface.co/spaces/Munzali/RICE_DISEASES_CLASSIFICATION)

![Status](https://img.shields.io/badge/status-deployed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Rice is a staple crop across much of the world, and early disease detection directly affects crop yield and food security. This project trains and serves a deep learning model that classifies rice leaf images into five categories, then returns a diagnosis, confidence score, and treatment recommendation through a REST API and a public web demo.

## Problem and approach

Manual disease diagnosis in the field is slow and requires expert knowledge that isn't always available to smallholder farmers. This project automates that diagnosis from a single photograph, using transfer learning on a small labeled dataset rather than training from scratch — a practical choice given the size of agricultural image datasets typically available.

## Classes detected

| Class | Severity | Notes |
|---|---|---|
| Bacterial Leaf Blight | High | Treated with copper-based bactericides |
| Rice Blast | Very High | Most destructive rice disease globally |
| Brown Spot | Medium | Often linked to soil nutrient deficiency |
| Rice Tungro Disease | Very High | Spread by leafhopper vectors |
| Healthy | — | No treatment needed |

## Model development

Three pretrained architectures were fine-tuned and compared under identical conditions using the same data splits and augmentation pipeline — to make an evidence-based choice for the production model rather than picking one architecture by default.

| Model | Best test accuracy | Parameters | Decision |
|---|---|---|---|
| EfficientNet-B0 | 99.86% | ~5.3M | **Selected for deployment** |
| ResNeXt-101-32x8d | 100.00% | ~89M | Rejected — 17x larger for no accuracy gain |
| RegNet-Y-800MF | 100.00% | ~6.4M | Close second |

EfficientNet-B0 was selected for production despite not having the single highest accuracy, because the difference (0.14%) is statistically negligible on this dataset size, while the model is roughly 17 times smaller than ResNeXt-101. Smaller models mean faster inference, lower memory footprint, and cheaper hosting — a real engineering tradeoff, not just a modeling exercise.

### Training details

- **Transfer learning** with selective unfreezing — only the final 3 layers of the backbone were fine-tuned, keeping early layers frozen to retain general visual features learned from ImageNet
- **Augmentation** — random horizontal/vertical flips, rotation, color jitter, and affine translation to improve generalization on a modest dataset size
- **Optimizer** — AdamW with separate learning rates for the classifier head (1e-3) and unfrozen backbone layers (1e-5)
- **Scheduler** — cosine annealing learning rate decay over 15 epochs
- **Loss** — cross-entropy with label smoothing (0.1) to reduce overconfidence
- **Mixed precision training** on GPU to roughly halve training time
- **Compute** — trained on Kaggle's free T4 GPU tier; inference runs on CPU

## Architecture

```
                    ┌─────────────────────┐
                    │   Kaggle GPU         │
                    │   (training only)    │
                    │  3 models compared    │
                    └──────────┬────────────┘
                               │ best model exported
                               ▼
                    ┌─────────────────────┐
                    │  efficientnet_b0     │
                    │  _best.pth (15.6MB)  │
                    └──────────┬────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                                      ▼
  ┌───────────────────┐                ┌──────────────────────┐
  │   FastAPI service   │                │  Gradio app on HF      │
  │  Docker + CI/CD      │                │  Spaces (live demo)    │
  └───────────────────┘                └──────────────────────┘
```

## Tech stack

| Layer | Tools |
|---|---|
| Model training | PyTorch, torchvision, Kaggle GPU |
| Experiment tracking | MLflow |
| API serving | FastAPI, Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions (automated test + build on every push) |
| Testing | pytest, httpx |
| Live deployment | Hugging Face Spaces (Gradio) |

## API usage

The FastAPI service exposes three endpoints:

```bash
GET  /          # health check + welcome message
GET  /health    # service and model status
POST /predict   # upload an image, get a diagnosis
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@rice_leaf.jpg"
```

Example response:

```json
{
  "predicted_class": "blast",
  "confidence": 0.9678,
  "all_probabilities": {
    "bacterial": 0.012,
    "blast": 0.9678,
    "brownspot": 0.015,
    "healthy": 0.003,
    "tungro": 0.002
  },
  "disease_info": {
    "full_name": "Rice Blast",
    "severity": "Very High",
    "treatment": "Apply tricyclazole fungicide. Avoid excess nitrogen."
  }
}
```

## Running locally

```bash
git clone https://github.com/ExplorerGumel/rice-disease-api.git
cd rice-disease-api
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Running with Docker

```bash
docker build -t rice-disease-api .
docker run -p 8000:80 rice-disease-api
```

## CI/CD pipeline

Every push to `main` automatically:
1. Installs dependencies in a clean Ubuntu environment
2. Runs the full pytest suite, including a real inference test against the model
3. Builds the Docker image to confirm it's deployable

This catches environment-specific bugs — like a model that fails to load under a different working directory, or a dependency resolution conflict — before they reach a live deployment.

## Project structure

```
rice-disease-api/
├── app.py                      # FastAPI application
├── inference.py                # Model loading and prediction logic
├── model_training.py           # Training script (for reproducibility)
├── log_model.py                # MLflow experiment logging
├── models/
│   └── efficientnet_b0_best.pth
├── tests/
│   └── test_app.py
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile
└── requirements.txt
```

## Dataset

[Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) — Kaggle, 5 classes, cleaned and balanced.
