import cv2
import numpy as np
from modules.detector_base import Detector

class BicepDetector(Detector):
    def __init__(self, model_path, video_path):
        super().__init__(model_path, video_path)
        self.p1_index = 6
        self.p2_index = 8
        self.p3_index = 10

    def calculate_angle(self, a, b, c):
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

    def process_frame(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        annotated = results[0].plot()

        if len(results[0].keypoints) > 0 and len(results[0].keypoints.xy) > 0:
            keypoints_person1 = results[0].keypoints.xy[0].cpu().numpy()

            try:
                shoulder = keypoints_person1[self.p1_index]
                elbow = keypoints_person1[self.p2_index]
                wrist = keypoints_person1[self.p3_index]

                angle = self.calculate_angle(shoulder, elbow, wrist)
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
        
        return annotated