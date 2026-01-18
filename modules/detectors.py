from ultralytics import YOLO
import cv2
import numpy as np
from utils import draw_text, plot_pose_skeleton

class BaseDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame):
        raise NotImplementedError

class ObjectDetector(BaseDetector):
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__(model_path)
    
    def predict(self, frame):
        # Use track for consistency, though predict() works too. 
        # persist=True is important for tracking.
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame

class FruitClassifier(BaseDetector):
    def __init__(self, model_path='yolov8n.pt'): 
        # Switched to Object Detection model as requested
        super().__init__(model_path)
        # COCO Classes: 46: banana, 47: apple, 49: orange, 50: broccoli, 51: carrot
        self.fruit_classes = [46, 47, 49, 50, 51] 
    
    def predict(self, frame):
        results = self.model(frame, verbose=False)
        
        # Filter boxes
        # results[0].boxes.cls contains class indices
        # results[0].boxes.xyxy contains coordinates
        
        filtered_boxes = []
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.fruit_classes:
                    filtered_boxes.append(box)
        
        # Create a new Plot with only filtered boxes? 
        # Ultralytics plot() draws everything in the results object.
        # We can modify the results object or draw manually.
        # Drawing manually is safer to avoid internal state hacking.
        
        annotated_frame = frame.copy()
        
        for box in filtered_boxes:
            # Draw Box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = f"{results[0].names[cls_id]} {conf:.2f}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return annotated_frame

class TheftDetector(BaseDetector):
    def __init__(self, model_path='yolov8n-pose.pt'):
        super().__init__(model_path)
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
        
        # 1. Height Check
        if l_wrist[1] < l_hip[1] - 20 and r_wrist[1] < r_hip[1] - 20: return False
            
        # 2. Proximity Check
        wrist_dist = np.linalg.norm(l_wrist[:2] - r_wrist[:2])
        l_dist_hip = np.linalg.norm(l_wrist[:2] - l_hip[:2])
        r_dist_hip = np.linalg.norm(r_wrist[:2] - r_hip[:2])
        
        if wrist_dist < 120: return True
        if l_dist_hip < 100 or r_dist_hip < 100: return True
            
        return False

    def predict(self, frame):
        # Enable tracking
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = frame.copy()
        
        if results and results[0].keypoints is not None and results[0].boxes.id is not None:
            # keypoints.data: (N, 17, 3)
            # boxes.id: (N,) - Track IDs
            all_keypoints = results[0].keypoints.data.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Identify active IDs
            active_ids = set()
            
            for kps, track_id in zip(all_keypoints, track_ids):
                active_ids.add(track_id)
                
                # Check suspicious behavior
                suspicious_hands_up = self.is_hands_up(kps)
                suspicious_bagging = self.is_bagging(kps)
                
                # Update Meter
                current_suspicion = self.suspicion_meter.get(track_id, 0)
                
                if suspicious_hands_up or suspicious_bagging:
                    current_suspicion += 1
                else:
                    current_suspicion = max(0, current_suspicion - 1)
                
                self.suspicion_meter[track_id] = current_suspicion
                
                # Visualization specific to this person
                # Get bounding box center or head to draw status
                head_x, head_y = int(kps[0][0]), int(kps[0][1])
                
                if current_suspicion > self.alert_threshold:
                    if suspicious_hands_up:
                        status = "HANDS UP!"
                        color = (0, 0, 255)
                    elif suspicious_bagging:
                        status = "BAGGING!"
                        color = (0, 165, 255)
                    else:
                        status = "SUSPICIOUS" # Lingering suspicion
                        color = (0, 165, 255)
                        
                    draw_text(annotated_frame, f"ID {track_id}: {status}", (head_x - 50, head_y - 20), color=color, scale=0.6)
                    # Draw a box around them if possible, or just the skeleton with alert color
                else:
                    draw_text(annotated_frame, f"ID {track_id}", (head_x, head_y - 20), color=(0, 255, 0), scale=0.5)

            # Cleanup missing IDs
            # (Optional: remove IDs that haven't been seen for X frames, but simple dict cleanup is okay for now)
            # For simplicity, we keep all for this session or could clear ones not in active_ids if we wanted strict frame usage
            
            # Simple cleanup: if ID not in active_ids, maybe decay it?
            # Let's just keep the history for now, assuming not millions of people.
            
            plot_pose_skeleton(annotated_frame, results[0].keypoints.data)

        # Global Alert check (if ANYONE is suspicious)
        any_suspicious = any(v > self.alert_threshold for v in self.suspicion_meter.values())
        if any_suspicious:
             draw_text(annotated_frame, "WARNING: SUSPICIOUS ACTIVITY DETECTED", (20, 50), color=(0, 0, 255), scale=1)
             cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 5)
        else:
             draw_text(annotated_frame, "Monitoring...", (20, 50), color=(0, 255, 0))

        return annotated_frame
