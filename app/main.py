
import sys
import os
# Hack: Importar torch antes que cualquier otra cosa para evitar errores de DLL (WinError 1114)
try:
    import torch
except ImportError:
    pass

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UI_PATH = os.path.join(BASE_DIR, "app", "mainwindow.ui")

# Importar Clases de Detección
sys.path.append(BASE_DIR) # Permitir imports como 'from modules...'
from modules.avenue_detection import AvenueDetector
from modules.bicep_detection import BicepDetector
from modules.pose_detection import PoseDetector
from modules.fruit_detection_1 import FruitDetector1
from modules.fruit_detection_2 import FruitDetector2
from modules.robbery_detection import RobberyDetector

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._run_flag = True

    def run(self):
        while self._run_flag:
            ret, frame = self.detector.cap.read()
            if ret:
                annotated_frame = self.detector.process_frame(frame)
                self.change_pixmap_signal.emit(annotated_frame)
            else:
                # Loop video
                self.detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Al detenerse
        self.detector.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            uic.loadUi(UI_PATH, self)
        except FileNotFoundError:
            QMessageBox.critical(None, "Error Fatal", f"No se encontró el archivo UI: {UI_PATH}")
            sys.exit(1)

        self.thread = None
        self.detector = None

        # Conectar Botones
        self.btn_avenue.clicked.connect(self.start_avenue_detection)
        self.btn_pose.clicked.connect(self.start_pose_detection)
        self.btn_bicep.clicked.connect(self.start_bicep_detection)
        self.btn_fruit1.clicked.connect(self.start_fruit1_detection)
        self.btn_fruit2.clicked.connect(self.start_fruit2_detection)
        self.btn_robbery.clicked.connect(self.start_robbery_detection)
        
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_exit.clicked.connect(self.close_app)

        # Conectar Controles
        self.slider_conf.valueChanged.connect(self.update_confidence)
        
        # Estado inicial
        self.btn_stop.setEnabled(False)

    def close_app(self):
        self.stop_detection()
        self.close()

    def stop_detection(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.label_video.clear()
        self.label_video.setText("Vista de Video")
        self.btn_stop.setEnabled(False)

    def start_detection(self, detector_class, model_name, video_name):
        # Detener anterior si existe
        self.stop_detection()

        # Rutas
        model_path = os.path.join(MODELS_DIR, model_name)
        video_path = os.path.join(MEDIA_DIR, video_name)

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Modelo no encontrado: {model_name}")
            return
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", f"Video no encontrado: {video_name}")
            return

        # Instanciar Detector
        try:
            self.detector = detector_class(model_path, video_path)
            
            # Configurar Thread
            self.thread = VideoThread(self.detector)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

            self.btn_stop.setEnabled(True)
            # Actualizar slider con valor por defecto
            self.slider_conf.setValue(int(self.detector.conf_threshold * 100))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallo al iniciar el detector:\n{str(e)}")

    def update_image(self, cv_img):
        """Actualiza el QLabel con el nuevo frame de opencv"""
        qt_img = self.convert_cv_qt(cv_img)
        self.label_video.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convierte de OpenCV a QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_confidence(self):
        val = self.slider_conf.value()
        self.label_conf.setText(f"Confianza: {val/100.0:.2f}")
        if self.detector:
            self.detector.set_confidence(val/100.0)

    # Wrappers específicos
    def start_avenue_detection(self):
        self.start_detection(AvenueDetector, "yolo11n.pt", "Cropped_walking.mp4")

    def start_pose_detection(self):
        self.start_detection(PoseDetector, "yolo11n.pt", "La central TVN.mp4")

    def start_bicep_detection(self):
        self.start_detection(BicepDetector, "yolo11n-pose.pt", "Curl de bíceps con mancuernas.mp4")

    def start_fruit1_detection(self):
        self.start_detection(FruitDetector1, "yolo_fruits_and_vegetables_v3.pt", "Apples sorting 2.mp4")

    def start_fruit2_detection(self):
        self.start_detection(FruitDetector2, "yolo_fruits_and_vegetables_v3.pt", "Apples sorting 3.mp4")

    def start_robbery_detection(self):
        # Asumo que usamos el modelo de pose (yolo11n-pose.pt) que ya tienes, 
        # o 'yolov8n-pose.pt' si es especifico. Usaremos el de pose existente.
        # Y necesitamos un video. Usaré 'La central TVN.mp4' por defecto.
        self.start_detection(RobberyDetector, "yolo11n-pose.pt", "La central TVN.mp4")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
