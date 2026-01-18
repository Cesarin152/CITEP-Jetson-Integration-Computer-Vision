import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton
from PyQt5 import uic

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UI_PATH = os.path.join(BASE_DIR, "app", "mainwindow.ui")

# Mapeo de Scripts a Modelos y Videos requeridos
# Script Name -> { 'model': model_filename, 'video': video_filename }
SCRIPT_REQUIREMENTS = {
    "avenue_detection.py": {
        "model": "yolo11n.pt",
        "video": "Cropped_walking.mp4"
    },
    "pose_detection.py": {
        "model": "yolo11n.pt",
        "video": "La central TVN.mp4"
    },
    "bicep_detection.py": {
        "model": "yolo11n-pose.pt",
        "video": "Curl de bíceps con mancuernas.mp4"
    },
    "fruit_detection_1.py": {
        "model": "yolo_fruits_and_vegetables_v3.pt",
        "video": "Apples sorting 2.mp4"
    },
    "fruit_detection_2.py": {
        "model": "yolo_fruits_and_vegetables_v3.pt",
        "video": "Apples sorting 3.mp4"
    }
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Cargar archivo .ui
        try:
            uic.loadUi(UI_PATH, self)
        except FileNotFoundError:
            QMessageBox.critical(None, "Error Fatal", f"No se encontró el archivo de interfaz:\n{UI_PATH}")
            sys.exit(1)

        # Conectar botones
        self.btn_avenue.clicked.connect(lambda: self.ejecutar("avenue_detection.py"))
        self.btn_pose.clicked.connect(lambda: self.ejecutar("pose_detection.py"))
        self.btn_bicep.clicked.connect(lambda: self.ejecutar("bicep_detection.py"))
        self.btn_fruit1.clicked.connect(lambda: self.ejecutar("fruit_detection_1.py"))
        self.btn_fruit2.clicked.connect(lambda: self.ejecutar("fruit_detection_2.py"))
        
        # Botón Salir
        self.btn_exit.clicked.connect(self.close)

    def ejecutar(self, script_name):
        """Ejecuta un script tras validar sus requisitos."""
        
        # 1. Validar existencia del script
        script_path = os.path.join(MODULES_DIR, script_name)
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"No se encontró el script:\n{script_name}\n\nRuta: {script_path}")
            return

        # 2. Validar requisitos (Modelo y Video)
        if script_name in SCRIPT_REQUIREMENTS:
            reqs = SCRIPT_REQUIREMENTS[script_name]
            
            # Validar Modelo
            model_path = os.path.join(MODELS_DIR, reqs["model"])
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error de Modelo", 
                                     f"Falta el modelo requerido:\n{reqs['model']}\n\nDebe estar en: {MODELS_DIR}")
                return

            # Validar Video (se asume en la raiz del proyecto, BASE_DIR)
            video_path = os.path.join(BASE_DIR, reqs["video"])
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "Advertencia de Video", 
                                    f"No se encontró el video de prueba:\n{reqs['video']}\n\nDebe estar en: {BASE_DIR}\n\nEl script podría fallar o cerrarse inmediatamente.")
                # No retornamos, solo advertimos, ya que el usuario podría querer usar la cámara o el script podría manejarlo.
        
        # 3. Ejecutar
        try:
            # Ejecutamos con cwd=MODULES_DIR para que los scripts encuentren ../models correctamente
            subprocess.Popen([sys.executable, script_name], cwd=MODULES_DIR)
        except Exception as e:
            QMessageBox.critical(self, "Error de Ejecución", f"Ocurrió un error al lanzar el script:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
