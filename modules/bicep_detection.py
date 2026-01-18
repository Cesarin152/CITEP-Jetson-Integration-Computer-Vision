from ultralytics import YOLO
import cv2
import numpy as np



model = YOLO("../models/yolo11n-pose.pt")
cap = cv2.VideoCapture("../media/Curl de bíceps con mancuernas.mp4")
codec = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("BicepDetection_Analized.mp4",codec,60,(1280,720))\

P1_INDEX = 6
P2_INDEX = 8
P3_INDEX = 10

import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    cosine_angle = dot_product / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle = np.degrees(angle)

    if angle > 180.0:
        angle = 360 - angle

    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    if len(results[0].keypoints) > 0 and len(results[0].keypoints.xy) > 0:
        keypoints_person1 = results[0].keypoints.xy[0].cpu().numpy()

        try:
            shoulder = keypoints_person1[P1_INDEX]
            elbow = keypoints_person1[P2_INDEX]
            wrist = keypoints_person1[P3_INDEX]

            angle = calculate_angle(shoulder, elbow, wrist)
            angle_text = f"Codo D: {angle:.1f} grados"

            cv2.putText(annotated, angle_text, 
                        (int(elbow[0]) - 50, int(elbow[1]) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if angle < 30:
                feedback = "Flexion Completa!"
                cv2.putText(annotated, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            elif angle > 160:
                feedback = "Extension Completa!"
                cv2.putText(annotated, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        except IndexError:
            cv2.putText(annotated, "Esperando deteccion de Keypoints...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv11 Pose y Analisis de Angulo", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()