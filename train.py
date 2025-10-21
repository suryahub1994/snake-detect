from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="dataset/Snake_Detect/data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    name="brain_yolo"
)

model.val()
