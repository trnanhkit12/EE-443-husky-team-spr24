from ultralytics import YOLO


# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='C:/Users/antho/EE-443-husky-team-spr24/data', epochs=100, imgsz=640)
