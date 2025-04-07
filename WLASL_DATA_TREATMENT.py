import cv2 
import os

# Ruta al directorio con los videos
video_dir = 'WLASL/videos/'

# Lista para guardar [(frames, nombre_video), ...]
dataset = []

# Especifica la resolución a la que quieres redimensionar los videos
target_resolution = (224, 224)

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

# 🔍 Buscar el video con más frames
if dataset:
    max_frames_video = max(dataset, key=lambda x: len(x[0]))
    print(f"🎥 Video con más frames: {max_frames_video[1]} ({len(max_frames_video[0])} frames)")

    # 📊 Calcular media de frames
    total_frames = sum(len(frames) for frames, _ in dataset)
    media_frames = total_frames / len(dataset)
    print(f"📈 Media de frames por video: {media_frames:.2f}")
else:
    print("No se procesaron videos.")
