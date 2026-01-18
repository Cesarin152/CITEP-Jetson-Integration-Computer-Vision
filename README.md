# Computer Vision CITEP AI Course Project

Este proyecto contiene una colección de scripts de demostración de visión artificial utilizando YOLOv11 y OpenCV.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

*   **app/**: Contiene la aplicación principal (`main.py`) que lanza la interfaz gráfica para seleccionar las demostraciones.
*   **modules/**: Contiene los scripts individuales de detección y análisis (e.g., detección de avenidas, detección de bíceps, frutas, etc.).
*   **models/**: Almacena los modelos pre-entrenados de YOLO (`.pt`).
*   **videos/**: (Opcional) Debe contener los videos de prueba (los videos actuales se esperan en la raíz del proyecto para compatibilidad con los scripts).

## Requisitos

Para ejecutar el proyecto, asegúrese de tener instaladas las siguientes dependencias:

*   Python 3.8+
*   ultralytics
*   opencv-python
*   numpy

Puede instalarlos ejecutando:

```bash
pip install -r requirements.txt
```

## Como Ejecutar

1.  Navegue a la raíz del proyecto.
2.  Ejecute el script principal:

```bash
python app/main.py
```

3.  Se abrirá una ventana con botones para ejecutar las diferentes demostraciones.

## Notas

*   Algunos scripts requieren archivos de video específicos en la raíz del proyecto (`Cropped_walking.mp4`, `Curl de bíceps con mancuernas.mp4`, etc.).
*   El modelo `yolo_fruits_and_vegetables_v3.pt` es requerido para la detección de frutas. Asegúrese de que esté en la carpeta `models/`.
