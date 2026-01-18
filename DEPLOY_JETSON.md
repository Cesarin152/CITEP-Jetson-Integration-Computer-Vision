# Despliegue en NVIDIA Jetson

Ejecutar este proyecto en una NVIDIA Jetson (Nano, Orin Nano, AGX) requiere algunos pasos específicos debido a la arquitectura ARM y la aceleración GPU.

## 1. Preparación del Sistema (JetPack)

Asegúrate de tener **JetPack 5.x o 6.x** instalado.
No uses el `requirements.txt` estándar de Windows directamente, ya que PyTorch y OpenCV deben instalarse de fuentes específicas de NVIDIA para usar la GPU.

### A. Instalar Dependencias del Sistema
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-pyqt5 python3-opencv libopenblas-base libopenmpi-dev
```
*Nota: Es mejor instalar PyQt5 desde `apt` en lugar de `pip` en la Jetson.*

### B. Instalar PyTorch y Torchvision
**NO** uses `pip install torch`. Debes usar los wheels pre-compilados de NVIDIA.

1.  Consulta la [guía oficial de NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) para tu versión de JetPack.
2.  Ejemplo común (para JetPack 5):
    ```bash
    # Descargar wheel (ejemplo)
    wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
    pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
    ```
3.  Compilar Torchvision (debe coincidir con la versión de PyTorch):
    ```bash
    git clone --branch v0.15.2 https://github.com/pytorch/vision torchvision
    cd torchvision
    python3 setup.py install --user
    ```

### C. Instalar Ultralytics (YOLO)
```bash
pip3 install ultralytics
```

## 2. Optimización del Modelo (TensorRT)

Las Jetson son poderosas pero tienen recursos limitados. Para obtener FPS reales (30-60 FPS), DEBES convertir tus modelos `.pt` a TensorRT (`.engine`).

1.  **Exportar el modelo** (puedes hacerlo en tu PC o en la Jetson, pero mejor en la Jetson para evitar problemas de compatibilidad):
    ```bash
    yolo export model=models/yolo11n.pt format=engine device=0
    ```
    Esto generará `yolo11n.engine`.

2.  **Usar el modelo .engine**:
    El código de `ultralytics` carga automáticamente el `.engine` si se le pasa como ruta.
    
    En `app/main.py`, cambia la extensión o asegúrate de que el archivo `.engine` exista en `models/`.
    
    *Recomendación*: Modifica `app/main.py` para buscar `.engine` primero y si no, usar `.pt`.

## 3. Ejecución

1.  Clona este proyecto en la Jetson.
2.  Ejecuta:
    ```bash
    python3 app/main.py
    ```

## 4. Solución de Problemas Comunes

*   **Error "Illegal Instruction"**: Generalmente significa que instalaste una versión de numpy u opencv no compilada para ARM64. Reinstala usando `apt` o wheels oficiales.
*   **Lento (1-2 FPS)**:
    *   Verifica que `torch.cuda.is_available()` sea `True`.
    *   Asegúrate de estar usando el archivo `.engine`.
    *   Ejecuta `sudo jetson_clocks` para activar el modo de máximo rendimiento (fan al máximo).

## 5. Uso de Cámaras en Vivo (CSI / USB)

Por defecto el proyecto usa archivos de video. Para usar una cámara en vivo en la Jetson:

1.  **Cámara USB**:
    En `app/main.py`, cambia la ruta del video por el índice de la cámara (generalmente 0):
    ```python
    self.start_detection(RobberyDetector, "yolo11n-pose.engine", 0)
    ```

2.  **Cámara CSI (Raspberry Pi Camera v2, IMX219, etc.)**:
    Las cámaras CSI requieren una cadena GStreamer en OpenCV.
    ```python
    gstreamer_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    self.start_detection(RobberyDetector, "yolo11n-pose.engine", gstreamer_str)
    ```
