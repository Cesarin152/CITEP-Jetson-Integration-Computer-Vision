import numpy as np
import cv2
from modules.detector_base import Detector
# Reuse the user's utils if possible, or inline them if simple. 
# Let's import them to respect the user's code structure.
from modules.utils import draw_text, plot_pose_skeleton

class RobberyDetector(Detector):
    def __init__(self, model_path, video_path):
        super().__init__(model_path, video_path)
        # Dictionary to track suspicion per Track ID: {track_id: frames}
        self.suspicion_meter = {}
        self.alert_threshold = 10 # frames

    def is_hands_up(self, keypoints):
        """
        Check if wrists are above the eyes/ears.
        """
        if keypoints is None: return False
        if keypoints.shape[0] < 11: return False
        
        nose_y = keypoints[0][1]
        l_wrist_y = keypoints[9][1]
        r_wrist_y = keypoints[10][1]
        l_conf = keypoints[9][2]
        r_conf = keypoints[10][2]
        
        if l_conf < 0.5 or r_conf < 0.5: return False
        # In image coordinates, smaller Y is higher
        if l_wrist_y < nose_y and r_wrist_y < nose_y: return True
        return False

    def is_bagging(self, keypoints):
        """
        Check if hands are in a 'bagging' or 'concealing' position.
        """
        if keypoints is None: return False
        if keypoints.shape[0] < 13: return False
        
        l_wrist = keypoints[9]
        r_wrist = keypoints[10]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        
        if l_wrist[2] < 0.5 or r_wrist[2] < 0.5 or l_hip[2] < 0.5 or r_hip[2] < 0.5: return False
        
        # 1. Height Check (wrisst below hips? or similar height?)
        # The original code: l_wrist[1] < l_hip[1] - 20 (wrist ABOVE hip by 20px)
        if l_wrist[1] < l_hip[1] - 20 and r_wrist[1] < r_hip[1] - 20: return False
            
        # 2. Proximity Check
        wrist_dist = np.linalg.norm(l_wrist[:2] - r_wrist[:2])
        l_dist_hip = np.linalg.norm(l_wrist[:2] - l_hip[:2])
        r_dist_hip = np.linalg.norm(r_wrist[:2] - r_hip[:2])
        
        if wrist_dist < 120: return True
        if l_dist_hip < 100 or r_dist_hip < 100: return True
            
        return False

    def process_frame(self, frame):
        # Result of inference
        # We use track=True if checking for persistence, but self.model() in base might not support 'track' easily
        # depending on how Detector calls it. 
        # Base class calls self.model(frame). Let's call self.model.track explicitly here.
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_threshold)
        annotated_frame = frame.copy()
        
        if results and results[0].keypoints is not None and results[0].boxes.id is not None:
             all_keypoints = results[0].keypoints.data.cpu().numpy()
             track_ids = results[0].boxes.id.int().cpu().numpy()
             
             for kps, track_id in zip(all_keypoints, track_ids):
                 # Check behavior
                 suspicious_hands_up = self.is_hands_up(kps)
                 suspicious_bagging = self.is_bagging(kps)
                 
                 # Update Meter
                 current_suspicion = self.suspicion_meter.get(track_id, 0)
                 
                 if suspicious_hands_up or suspicious_bagging:
                     current_suspicion += 1
                 else:
                     current_suspicion = max(0, current_suspicion - 1)
                 
                 self.suspicion_meter[track_id] = current_suspicion
                 
                 # Draw
                 head_x, head_y = int(kps[0][0]), int(kps[0][1])
                 if current_suspicion > self.alert_threshold:
                     if suspicious_hands_up:
                         status = "HANDS UP!"
                         color = (0, 0, 255)
                     elif suspicious_bagging:
                         status = "BAGGING!"
                         color = (0, 165, 255) # Orange
                     else:
                         status = "SUSPICIOUS"
                         color = (0, 165, 255)
                     draw_text(annotated_frame, f"ID {track_id}: {status}", (head_x - 50, head_y - 20), color=color, scale=0.6)
                 else:
                     draw_text(annotated_frame, f"ID {track_id}", (head_x, head_y - 20), color=(0, 255, 0), scale=0.5)

             plot_pose_skeleton(annotated_frame, results[0].keypoints.data)

        # Global Alert
        any_suspicious = any(v > self.alert_threshold for v in self.suspicion_meter.values())
        if any_suspicious:
             draw_text(annotated_frame, "WARNING: SUSPICIOUS ACTIVITY DETECTED", (20, 50), color=(0, 0, 255), scale=1)
             cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 5)
        else:
             draw_text(annotated_frame, "Monitoring...", (20, 50), color=(0, 255, 0))

        return annotated_frame
