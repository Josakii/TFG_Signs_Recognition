import cv2
import os

# Ruta al directorio con los videos
video_dir = 'WLASL/videos/'

# Lista para guardar [(frames, nombre_video), ...]
dataset = []

# Especifica la resolución a la que quieres redimensionar los videos
target_resolution = (224, 224)  # Cambia esto a la resolución deseada

# Obtener lista de archivos de video en el directorio
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error al abrir: {video_file}")
        continue

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame a la resolución deseada
        resized_frame = cv2.resize(frame, target_resolution)

        frames.append(resized_frame)

    cap.release()

    # Guardar el conjunto de frames junto con el nombre del video
    dataset.append((frames, video_file))
    print(f"Procesado: {video_file}, Frames: {len(frames)}")

print(f"\nTotal videos procesados: {len(dataset)}")
