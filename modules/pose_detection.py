from ultralytics import YOLO
import cv2

model = YOLO("../models/yolo11n.pt")
cap = cv2.VideoCapture("../La central TVN.mp4")
codec = cv2.VideoWriter_fourcc(*"X264")
out = cv2.VideoWriter("LaCentral_Analized.mp4",codec,30,(1920,1080))

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
