from roboflow import Roboflow
import os

rf = Roboflow(api_key="2OFRWtLUJqdrNuwZhhPk")
#Загрузка проекта и датасета
project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
dataset = project.version(10).download("yolov8")
dataset_path = dataset.location
data_yaml_path = os.path.join(dataset_path, "data.yaml")
with open(data_yaml_path, 'r') as f:
    print("Содержимое data.yaml:")
    print(f.read())
train_images = len(os.listdir(os.path.join(dataset_path, "train", "images")))
val_images = len(os.listdir(os.path.join(dataset_path, "valid", "images")))
test_images = len(os.path.join(dataset_path, "test", "images")) if os.path.exists(os.path.join(dataset_path, "test")) else 0
print(f"\nРазмеры датасета:")
print(f"Train: {train_images} изображений")
print(f"Valid: {val_images} изображений")
print(f"Test: {test_images} изображений")
print("\nКлассы: person")

from ultralytics import YOLO
model = YOLO("yolov10n.pt")
#Обучение
results = model.train(
    data=data_yaml_path,
    epochs=10,
    imgsz=640,
    batch=16,
    name="yolov10n_people_detection",
    device=0
)

import shutil
from google.colab import files
folder_to_download = "/content/runs/detect"
zip_filename = "detect.zip"
print("Упаковка папки в ZIP...")
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', folder_to_download)
print(f"Архив создан: {zip_filename}")
files.download(zip_filename)

#Валидация на тестовой выборке
metrics = model.val(data=data_yaml_path)
#Вывод метрик
print("Метрики валидации:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("\nПолные метрики:")
print(metrics)

import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt

test_image_path = os.path.join(dataset_path, "test", "images")
if os.path.exists(os.path.join(dataset_path, "test", "images")):
    test_images = os.listdir(os.path.join(dataset_path, "test", "images"))
    test_image_path = os.path.join(dataset_path, "test", "images", test_images[0])
else:
    test_images = os.listdir(os.path.join(dataset_path, "valid", "images"))
    test_image_path = os.path.join(dataset_path, "valid", "images", test_images[0])
results = model(test_image_path)

#Визуализация
plt.figure(figsize=(10, 10))
img = Image.open(test_image_path)
plt.imshow(img)
plt.axis('off')

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f"person {conf:.2f}"
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
            plt.text(x1, y1-10, label, color='red', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title("Визуализация детекций на тестовом изображении")
plt.show()
result.save("detection_result.jpg")
print("Результат сохранен как detection_result.jpg")


import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from IPython.display import Video, HTML, display

print("Зависимости установлены.")
#Загрузка модели best.pt
model_path = "best.pt"
if not os.path.exists(model_path):
    possible = "/content/runs/detect/yolov10n_people_detection/weights/best.pt"
    if os.path.exists(possible):
        model_path = possible
    else:
        print("ЗАГРУЗИТЕ МОДЕЛЬ best.pt:")
        from google.colab import files
        uploaded = files.upload()
        model_path = list(uploaded.keys())[0]
model = YOLO(model_path)
model.fuse()
print(f"Модель загружена: {model_path}")

video_path = "street3.mp4"
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f"Видео: {w}x{h}, FPS: {fps:.1f}, кадров: {total_frames}")

box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

def annotate_frame(frame, results):
    detections = sv.Detections.from_ultralytics(results)
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    labels = [f"ID:{int(id)} {conf:.2f}" for id, conf in zip(results.boxes.id, results.boxes.conf)]
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
    return annotated_frame

#ByteTrack
print("\nЗапуск ByteTrack...")
conf_path = "bytetrack.yaml"
if not os.path.exists(conf_path):
    !wget -q https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml -O {conf_path}
model.track(
    source=video_path,
    conf=0.25,
    iou=0.45,
    persist=True,
    tracker=conf_path,
    save=True,
    name="bytetrack",
    exist_ok=True
)

output_byte = "runs/detect/bytetrack/track.mp4"
print("ByteTrack готов!")
display(Video(output_byte, embed=True, width=800))  # embed=True!

#BoTSORT
print("\nЗапуск BoTSORT...")
conf_path = "botsort.yaml"
if not os.path.exists(conf_path):
    !wget -q https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/botsort.yaml -O {conf_path}
model.track(
    source=video_path,
    conf=0.25,
    iou=0.45,
    persist=True,
    tracker=conf_path,
    save=True,
    name="botsort",
    exist_ok=True
)

output_bot = "runs/detect/botsort/track.mp4"
print("BoTSORT готов!")
display(Video(output_bot, embed=True, width=800))

print("\nЭксперименты с параметрами:")
experiments = [
    {"name": "агрессивный", "conf": 0.1, "iou": 0.7, "tracker": "bytetrack.yaml"},
    {"name": "консервативный", "conf": 0.5, "iou": 0.3, "tracker": "botsort.yaml"},
    {"name": "долгая_память", "conf": 0.25, "iou": 0.45, "tracker": "botsort.yaml"},
]
for exp in experiments:
    print(f"\n--- {exp['name']} ---")
    cfg = exp["tracker"]
    if not os.path.exists(cfg):
        !wget -q https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/{cfg} -O {cfg}
    model.track(
        source=video_path,
        conf=exp["conf"],
        iou=exp["iou"],
        persist=True,
        tracker=cfg,
        save=True,
        name=f"exp_{exp['name']}",
        exist_ok=True
    )
    out_path = f"runs/detect/exp_{exp['name']}/track.mp4"
    print(f"Готово: {out_path}")
    display(Video(out_path, embed=True, width=600))
    
#Сравнение
print("\nСРАВНЕНИЕ:")
display(HTML(f"""
<table border="1" style="width:100%; text-align:center;">
  <tr><th>ByteTrack</th><th>BoTSORT</th><th>Агрессивный</th></tr>
  <tr>
    <td><video src="{output_byte}" width=300 controls></video></td>
    <td><video src="{output_bot}" width=300 controls></video></td>
    <td><video src="runs/detect/exp_агрессивный/track.mp4" width=300 controls></video></td>
  </tr>
</table>
"""))

print("""
ВЫВОДЫ:
1. ByteTrack: Быстрый, подходит для реального времени, но теряет ID при окклюзиях.
2. BoTSORT: Точнее благодаря ReID, лучше для сложных сцен.
3. Параметры:
   • conf ↓ → больше детекций/треков (агрессивно)
   • iou ↑ → строже matching, меньше ложных ID
   • persist=True → держит ID между кадрами
Рекомендация: BoTSORT с conf=0.25, iou=0.45 для толпы.
""")