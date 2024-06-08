from ultralytics import YOLO


# Load a model
model = YOLO('yolov8x.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='/media/cycyang/sda1/EE443_final/detection/ee443.yaml', epochs=100, imgsz=640)