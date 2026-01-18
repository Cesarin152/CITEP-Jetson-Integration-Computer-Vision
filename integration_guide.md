
## Adding New Modules

To integrate a new computer vision project/script into this application, follow these steps:

1.  **Prepare Files**:
    - Place your YOLO model (`.pt`) in the `models/` directory.
    - Place your test video (`.mp4`) in the `media/` directory.

2.  **Create Module**:
    - Create a new Python file in `modules/` (e.g., `my_new_detection.py`).
    - Define a class that inherits from `Detector` (see `modules/detector_base.py`).
    - Implement the `process_frame(self, frame)` method.

    ```python
    from modules.detector_base import Detector

    class MyNewDetector(Detector):
        def __init__(self, model_path, video_path):
            super().__init__(model_path, video_path)

        def process_frame(self, frame):
            results = self.model(frame, conf=self.conf_threshold)
            return results[0].plot()
    ```

3.  **Update UI**:
    - Open `app/mainwindow.ui` in Qt Designer (or edit the XML).
    - Add a new `QPushButton` (e.g., `btn_new_project`).

4.  **Connect Logic**:
    - In `app/main.py`:
        - Import your new class: `from modules.my_new_detection import MyNewDetector`
        - Connect the button signal: `self.btn_new_project.clicked.connect(self.start_new_detection)`
        - Add the starter method:
          ```python
          def start_new_detection(self):
              self.start_detection(MyNewDetector, "my_model.pt", "my_video.mp4")
          ```
