# -*- coding: utf-8 -*-
import os
import subprocess

# Ruta del dataset (ajústalo a tu estructura)

# Ruta del dataset (ajústalo a tu estructura)
DATASET_PATH = "/export/fhome/jmtorres/Projecte/YoloDS/hand_face_detection_dataset/"
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

# Número de épocas para entrenar
EPOCHS = 50

# Crear data.yaml con 2 clases: cara y mano
def crear_data_yaml():
    print("Creando data.yaml con clases: cara, manso")
    content = f"""train: {os.path.join(DATASET_PATH, "images/train")}
val: {os.path.join(DATASET_PATH, "images/val")}

nc: 2
names: ['cara', 'manso']
"""
    with open(DATA_YAML, "w") as f:
        f.write(content)

# Función para clonar YOLOv5
def clone_yolov5():
    if not os.path.exists("yolov5"):
        print("Clonando YOLOv5...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
        subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

# Función para clonar YOLOv8 (Ultralytics)
def clone_yolov8():
    print("Instalando YOLOv8 (Ultralytics)...")
    subprocess.run(["pip", "install", "ultralytics"])

# Entrenamiento con YOLOv5
def train_yolov5():
    print("Entrenando con YOLOv5...")
    subprocess.run([
        "python", "yolov5/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", str(EPOCHS),
        "--data", DATA_YAML,
        "--weights", "yolov5s.pt",
        "--name", "exp_yolov5"
    ])

# Entrenamiento con YOLOv8
def train_yolov8():
    print("Entrenando con YOLOv8...")
    subprocess.run([
        "yolo", "task=detect",
        "mode=train",
        f"data={DATA_YAML}",
        "model=yolov8s.pt",
        "epochs=" + str(EPOCHS),
        "name=exp_yolov8"
    ])

# MAIN
if __name__ == "__main__":
    crear_data_yaml()
    clone_yolov5()
    clone_yolov8()
    
    train_yolov5()
    train_yolov8()
