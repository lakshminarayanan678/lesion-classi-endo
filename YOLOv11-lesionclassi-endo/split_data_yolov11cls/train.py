import wandb
from ultralytics import YOLO
import os

# Initialize W&B run
run = wandb.init(project="capsule_vision_challenge_2024", name="ue_internal_yolo11L_cls_run", job_type="train")

# # Log dataset as artifact
# artifact = wandb.Artifact(name="anat_1_yolo_split", type="dataset")
# artifact.add_dir("path/to/mnist160")
# run.log_artifact(artifact)
# artifact.wait()

# Use existing dataset artifact
artifact = run.use_artifact("ue_internal_split:latest", type="dataset")
# dataset_dir = artifact.download()

# Train the model
model = YOLO("yolo11l-cls.pt")
results = model.train(data="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11ncls/data1", epochs=75, imgsz=224)

# Log the trained model as an artifact
model_artifact = wandb.Artifact(name="ue_internal_yolo11L_cls_model", type="model")
model_artifact.add_file(model.ckpt_path if hasattr(model, "ckpt_path") else "runs/classify/train/weights/best.pt")
run.log_artifact(model_artifact)

run.finish()