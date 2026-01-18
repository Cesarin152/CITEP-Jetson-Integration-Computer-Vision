from ultralytics import YOLO
import cv2

model = YOLO("../models/yolo11n.pt")
cap = cv2.VideoCapture("../Cropped_walking.mp4")
codec = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("AvenueDetection_Analized.mp4",codec,60,(1280,720))

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
