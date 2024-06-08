from ultralytics import YOLO
from torchreid import models, utils

# Load a model
model = YOLO("yolov8x.yaml")  # build a new model from scratch
model = YOLO("/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/model_yolo8/weights/best.pt")  # load a pretrained model (recommended for training)

# Evaluate the model with different conf threshold
metrics = model.val()
metrics = model.val(imgsz=640, batch=16, conf=0.25, device="0")
metrics = model.val(imgsz=640, batch=16, conf=0.5, device="0")
metrics = model.val(imgsz=640, batch=16, conf=0.75, device="0")
metrics = model.val(imgsz=640, batch=16, conf=0.9, device="0")
