import cv2
import os

train_image = 'dataset/Snake_Detect/train/images/f623c50c9a486a200d5bb171e41733a0_jpeg.rf.82cbf0b15e33750a52cecfa24ae5eddf.jpg'
label_image = 'dataset/Snake_Detect/train/labels/f623c50c9a486a200d5bb171e41733a0_jpeg.rf.82cbf0b15e33750a52cecfa24ae5eddf.txt'

img = cv2.imread(train_image)

if img is None:
    print("❌ Image not found or failed to load:", train_image)
    exit()
else:
    print("✅ Image loaded successfully:", img.shape)

h, w, _ = img.shape

with open(label_image, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"Class {class_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Image with bounding boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
