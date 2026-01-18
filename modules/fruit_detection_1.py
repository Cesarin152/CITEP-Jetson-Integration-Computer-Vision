from modules.detector_base import Detector

class FruitDetector1(Detector):
    def __init__(self, model_path, video_path):
        super().__init__(model_path, video_path)

    def process_frame(self, frame):
        results = self.model(frame, conf=self.conf_threshold)
        annotated_frame = results[0].plot()
        return annotated_frame
