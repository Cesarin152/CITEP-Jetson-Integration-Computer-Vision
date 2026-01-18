from abc import ABC, abstractmethod
from ultralytics import YOLO
import cv2

class Detector(ABC):
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.conf_threshold = 0.25 # Default confidence
    
    def set_confidence(self, conf):
        self.conf_threshold = conf

    @abstractmethod
    def process_frame(self, frame):
        """
        Processes a frame and returns the annotated frame.
        Must be implemented by subclasses.
        """
        pass

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
