import tkinter as tk
import subprocess
import sys
import os

# Obtener la ruta base del proyecto (un nivel arriba de `app/`)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(BASE_DIR, "modules")

# Función que ejecuta otro script Python
def ejecutar(script):
    subprocess.Popen([sys.executable, script], cwd=MODULES_DIR)

# -------- Ventana principal --------
root = tk.Tk()
root.title("Menú de Visión Artificial")
root.geometry("500x450")        # Más alto para más botones
root.configure(bg="white")

# Título
titulo = tk.Label(
    root,
    text="Seleccione un script para ejecutar",
    bg="white",
    fg="black",
    font=("Arial", 18)
)
titulo.pack(pady=20)

# ---- Estilo de botones ----
btn_style = {
    "width": 35,
    "height": 2,
    "font": ("Arial", 12)
}

# ---- Botones ----
tk.Button(
    root, text="Avenue Detection",
    command=lambda: ejecutar("avenue_detection.py"),
    **btn_style
).pack(pady=5)

tk.Button(
    root, text="Pose Detection (La Central)",
    command=lambda: ejecutar("pose_detection.py"),
    **btn_style
).pack(pady=5)

tk.Button(
    root, text="Bicep Detection",
    command=lambda: ejecutar("bicep_detection.py"),
    **btn_style
).pack(pady=5)

tk.Button(
    root, text="Fruit Detection 1",
    command=lambda: ejecutar("fruit_detection_1.py"),
    **btn_style
).pack(pady=5)

tk.Button(
    root, text="Fruit Detection 2",
    command=lambda: ejecutar("fruit_detection_2.py"),
    **btn_style
).pack(pady=5)

# Botón Salir
tk.Button(
    root, text="Salir",
    command=root.quit,
    bg="#d9534f",
    fg="white",
    font=("Arial", 12),
    width=15
).pack(pady=20)

# ---- Texto inferior derecho ----
creditos = tk.Label(
    root,
    text="Demo elaborada por Victor Lau y Cesar Vigil",
    bg="white",
    fg="gray",
    font=("Arial", 10)
)
creditos.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

root.mainloop()
