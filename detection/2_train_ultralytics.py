from ultralytics import YOLO


# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/detection/ee443.yaml', epochs=100, imgsz=640)