# log_model.py
import torch
import mlflow
import mlflow.pytorch
import json
import os

MODEL_PATH = "models/efficientnet_b0_best.pth"
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

mlflow.set_experiment("rice-disease-classification")

with mlflow.start_run(run_name="efficientnet_b0_kaggle"):

    # Log parameters
    mlflow.log_param("model",        "efficientnet_b0")
    mlflow.log_param("num_classes",  5)
    mlflow.log_param("epochs",       15)
    mlflow.log_param("batch_size",   32)
    mlflow.log_param("optimizer",    "AdamW")
    mlflow.log_param("scheduler",    "CosineAnnealingLR")
    mlflow.log_param("augmentation", "flip+rotation+colorjitter")
    mlflow.log_param("loss",         "CrossEntropy+LabelSmoothing")

    # Log metrics
    mlflow.log_metric("best_accuracy", checkpoint["best_accuracy"])

    # Log class names as artifact
    class_info = {"class_names": checkpoint["class_names"]}
    with open("class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)
    mlflow.log_artifact("class_info.json")
    mlflow.log_artifact(MODEL_PATH)

    print(f" Model logged to MLflow")
    print(f" Best accuracy: {checkpoint['best_accuracy']:.4f}")
    print(f" Classes: {checkpoint['class_names']}")