# import wandb
from ultralytics import YOLO
# import os

# # Initialize W&B run
# run = wandb.init(project="capsule_vision_challenge_2024", name="ue_yolo11n_cls", job_type="train")

# # # Log dataset as artifact
# # artifact = wandb.Artifact(name="ue_yolo_split", type="dataset")
# # artifact.add_dir("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/YOLO_v11n-cls")
# # run.log_artifact(artifact)
# # artifact.wait()

# # Use existing dataset artifact
# artifact = run.use_artifact("ue_yolo_split:latest", type="dataset")

# Train the model
model = YOLO("yolo11n-cls.pt")
results = model.train(data="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11cls", epochs=75, imgsz=224)
test_results = model.val(data="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11cls", save_json=True, plots=True, split="test")

# # Log the trained model as an artifact
# model_artifact = wandb.Artifact(name="yolo11n_cls_model", type="model")
# model_artifact.add_file(model.ckpt_path if hasattr(model, "ckpt_path") else "runs/classify/train/weights/best.pt")
# run.log_artifact(model_artifact)

# run.finish()