!pip install roboflow ultralytics supervision opencv-python pillow

from roboflow import Roboflow
import os

rf = Roboflow(api_key="2OFRWtLUJqdrNuwZhhPk")

project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
dataset = project.version(10).download("yolov8")

dataset_dir = dataset.location
data_yaml_file = os.path.join(dataset_dir, "data.yaml")

# вывод YAML
with open(data_yaml_file, 'r') as f:
    yaml_content = f.read()
    print("Содержимое data.yaml:")
    print(yaml_content)

# Сохранение YAML в файл
with open("dataset_info.yaml", "w") as f:
    f.write(yaml_content)

# размера датасета
train_image_count = len(os.listdir(os.path.join(dataset_dir, "train", "images")))
val_image_count   = len(os.listdir(os.path.join(dataset_dir, "valid", "images")))
test_dir = os.path.join(dataset_dir, "test", "images")
test_image_count  = len(os.listdir(test_dir)) if os.path.exists(test_dir) else 0

with open("dataset_sizes.txt", "w") as f:
    f.write(f"Train: {train_image_count}\nValid: {val_image_count}\nTest: {test_image_count}\n")

print("\nКлассы: person")

# ------ YOLOv10 TRAIN ------
from ultralytics import YOLO
model = YOLO("yolov10n.pt")

results = model.train(
    data=data_yaml_file,
    epochs=10,
    imgsz=640,
    batch=16,
    name="yolov10n_people_detection",
    device=0
)

# Сохранение логов обучения
with open("train_results.txt", "w") as f:
    f.write(str(results))

# ------ PACK RUNS ------
import shutil
runs_path = "runs/detect"
zip_filename = "detect.zip"

if os.path.exists(runs_path):
    shutil.make_archive("detect", 'zip', runs_path)
    print("Архив создан detect.zip")
else:
    print("Папка runs/detect ещё не создана!")

# ------ VALIDATION ------
metrics = model.val(data=data_yaml_file)

with open("val_metrics.txt", "w") as f:
    f.write(f"mAP50: {metrics.box.map50:.4f}\n")
    f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
    f.write("\nПолный вывод:\n")
    f.write(str(metrics))

print("\nМетрики сохранены в val_metrics.txt")

# ------ VISUALIZATION ------
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt

save_dir = "outputs"
os.makedirs(save_dir, exist_ok=True)

# выбор изображения
if os.path.exists(test_dir):
    lst = os.listdir(test_dir)
    sample_img_path = os.path.join(test_dir, lst[0])
else:
    lst = os.listdir(os.path.join(dataset_dir, "valid", "images"))
    sample_img_path = os.path.join(dataset_dir, "valid", "images", lst[0])

results = model(sample_img_path)

# рисуем, но не показываем
plt.figure(figsize=(10,10))
img = Image.open(sample_img_path)
plt.imshow(img)
plt.axis('off')

for r in results:
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            plt.gca().add_patch(
                plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            )
            plt.text(x1, y1 - 10, f"person {conf:.2f}",
                     color='red', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

out_img_path = os.path.join(save_dir, "detection_result.jpg")
plt.savefig(out_img_path, dpi=300)
plt.close()

print(f"Сохранено: {out_img_path}")

# ultralytics save
r = results[0]
r.save(os.path.join(save_dir, "detection_direct.jpg"))

# ----- VIDEO TRACKING -----
import cv2
import numpy as np
from pathlib import Path
from IPython.display import Video, HTML, display

weights_path = "best.pt"
if not os.path.exists(weights_path):
    auto = "/content/runs/detect/yolov10n_people_detection/weights/best.pt"
    if os.path.exists(auto):
        weights_path = auto
    else:
        print("Нет best.pt. Пропускаю трекинг.")

model = YOLO(weights_path)
model.fuse()

input_video_path = "street3.mp4"

# ПРОВЕРКА видео
if not os.path.exists(input_video_path):
    raise FileNotFoundError("street3.mp4 не найден!")

# ByteTrack
os.makedirs("track_results", exist_ok=True)
bytetrack_cfg = "bytetrack.yaml"

if not os.path.exists(bytetrack_cfg):
    import urllib.request
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml"
    urllib.request.urlretrieve(url, bytetrack_cfg)

model.track(
    source=input_video_path,
    conf=0.25,
    iou=0.45,
    persist=True,
    tracker=bytetrack_cfg,
    save=True,
    name="bytetrack",
    exist_ok=True
)

# копируем результат
if os.path.exists("runs/detect/bytetrack/track.mp4"):
    shutil.copy("runs/detect/bytetrack/track.mp4", "track_results/bytetrack.mp4")

# BoTSORT
botsort_cfg = "botsort.yaml"
if not os.path.exists(botsort_cfg):
    import urllib.request
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/botsort.yaml"
    urllib.request.urlretrieve(url, botsort_cfg)

model.track(
    source=input_video_path,
    conf=0.25,
    iou=0.45,
    persist=True,
    tracker=botsort_cfg,
    save=True,
    name="botsort",
    exist_ok=True
)

if os.path.exists("runs/detect/botsort/track.mp4"):
    shutil.copy("runs/detect/botsort/track.mp4", "track_results/botsort.mp4")

print("ВСЕ ФАЙЛЫ СОХРАНЕНЫ.")
