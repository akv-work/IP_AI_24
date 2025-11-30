from ultralytics import YOLO
import os

model_path = "D:/vs code/Учеба/4/oiis/oiis3/runs/detect/train5/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Ошибка: Файл модели не найден по пути {model_path}")
    print("Укажите правильный путь")
    exit()

model = YOLO(model_path)

video_source = "test_video.mp4" 

if not os.path.exists(video_source):
    print(f"Ошибка: Видеофайл {video_source} не найден.")
    exit()

print("=== Запуск трекинга: BoT-SORT ===")

model.track(
    source=video_source,
    save=True,
    tracker="botsort.yaml",
    project="runs/track",
    name="botsort",
    device=0
)

print("=== Запуск трекинга: ByteTrack ===")

model.track(
    source=video_source,
    save=True,
    tracker="bytetrack.yaml",
    project="runs/track",
    name="bytetrack",
    device=0
)

print("Готово!")