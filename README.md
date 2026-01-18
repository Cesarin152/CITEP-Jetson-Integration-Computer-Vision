# Computer Vision CITEP AI Course Project

Este proyecto es una aplicación de escritorio completa para demostraciones de visión artificial utilizando **YOLOv11**, **OpenCV** y **PyQt5**.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

*   **app/**:
    *   `main.py`: Punto de entrada de la aplicación. Gestiona la interfaz gráfica y la lógica.
    *   `mainwindow.ui`: Archivo de diseño de la interfaz (XML de Qt Designer).
*   **modules/**: Contiene la lógica de detección. Cada script es una clase modular (e.g., `AvenueDetector`, `RobberyDetector`).
*   **models/**: Almacena los modelos pre-entrenados de YOLO (`.pt`).
*   **media/**: **(Importante)** Aquí debes colocar los videos de entrada (`.mp4`).
*   **outputs/**: Aquí se guardan automáticamente los videos procesados/analizados.

## Módulos Incluidos

1.  **Avenue Detection**: Conteo y detección de personas en una avenida.
2.  **Pose Detection**: Estimación de pose humana.
3.  **Bicep Detection**: Análisis de ejercicios de gimnasio mediante ángulos de articulaciones.
4.  **Fruit Detection**: Clasificación y detección de frutas.
5.  **Robbery Detection**: Detección de comportamientos sospechosos (manos arriba, ocultamiento).

## Requisitos

Para ejecutar el proyecto:

```bash
pip install -r requirements.txt
```

Incluye:
*   `PyQt5` (Interfaz Gráfica)
*   `ultralytics` (YOLO)
*   `opencv-python`
*   `numpy`
*   `torch`

## Como Ejecutar

1.  Asegúrate de que tus videos estén en la carpeta `media/` y tus modelos en `models/`.
2.  Ejecuta el comando:

```bash
python app/main.py
```

3.  Usa los botones del panel izquierdo para seleccionar un modelo.
4.  Ajusta el slider de "Confianza" en tiempo real para filtrar detecciones.

## Integración de Nuevos Proyectos

Para agregar tus propios módulos, consulta el archivo `integration_guide.md`.
