import cv2
import numpy as np

def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.7, thickness=2):
    """Draws text on the frame with a black outline for better visibility."""
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def plot_pose_skeleton(frame, keypoints, steps=2):
    """
    Plots a skeleton on the frame based on YOLOv8 pose keypoints.
    This is a simplified visualizer.
    """
    # Standard COCO keypoint connections
    skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
        (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (5, 7)
    ]
    
    for kps in keypoints:
        # kps is a tensor/array of shape (17, 3) -> x, y, conf
        if kps is None or len(kps) == 0:
            continue
            
        kp_np = kps.cpu().numpy() if hasattr(kps, 'cpu') else kps
        
        # Draw connections
        for p1, p2 in skeleton:
            if p1-1 < len(kp_np) and p2-1 < len(kp_np):
                pt1 = kp_np[p1-1]
                pt2 = kp_np[p2-1]
                
                # Check confidence (index 2)
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 255), 2)
        
        # Draw points
        for i, pt in enumerate(kp_np):
            if pt[2] > 0.5:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
