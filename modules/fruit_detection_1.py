from ultralytics import YOLO
import cv2

model = YOLO("../models/yolo_fruits_and_vegetables_v3.pt")
cap = cv2.VideoCapture("../media/Apples sorting 2.mp4")
codec = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("FruitDetection1.mp4",codec,25,(640,360))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    out.write(annotated)

    cv2.imshow("YOLOv11 Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
