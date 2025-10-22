from ultralytics import YOLO
import cv2
import os 

model = YOLO('runs/detect/brain_yolo5/weights/best.pt')
input_folder = 'dataset/Snake_Detect/valid/images/'

for file in os.listdir(input_folder):
    if file.lower().endswith('jpg'):
        img_path = os.path.join(input_folder, file)
        results = model.predict(img_path)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        for x1, x2, y1, y2 in xyxy:
            print(f"{x1, x2, y1, y2}")
            x1_int = int(x1)
            x2_int = int(x2)
            y1_int = int(y1)
            y2_int = int(y2)
            annotated = results[0].plot()  # ← draws boxes, labels, confidence
            cv2.imshow("Result", annotated)
            cv2.waitKey(0)
            
