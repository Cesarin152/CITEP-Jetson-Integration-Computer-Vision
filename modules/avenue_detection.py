import cv2
from modules.detector_base import Detector

class AvenueDetector(Detector):
    def __init__(self, model_path, video_path):
        super().__init__(model_path, video_path)

    def process_frame(self, frame):
        # Run inference
        results = self.model(frame, conf=self.conf_threshold)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()
        return annotated_frame
