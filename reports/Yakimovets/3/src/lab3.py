!pip install roboflow ultralytics supervision

from roboflow import Roboflow
import os

# --- Roboflow загрузка датасета ---
rf = Roboflow(api_key="2OFRWtLUJqdrNuwZhhPk")
workspace = rf.workspace("leo-ueno")
project = workspace.project("people-detection-o4rdr")

dataset = project.version(10).download("yolov8")
dataset_path = dataset.location
data_yaml_path = os.path.join(dataset_path, "data.yaml")

# Вывод содержимого YAML
with open(data_yaml_path, 'r') as file:
    print("Содержимое data.yaml:")
    print(file.read())

# Информация о датасете
train_images_count = len(os.listdir(os.path.join(dataset_path, "train", "images")))
valid_images_count = len(os.listdir(os.path.join(dataset_path, "valid", "images")))
test_dir = os.path.join(dataset_path, "test", "images")
test_images_count = len(os.listdir(test_dir)) if os.path.exists(test_dir) else 0

print("\nРазмеры датасета:")
print(f"Train: {train_images_count} изображений")
print(f"Valid: {valid_images_count} изображений")
print(f"Test: {test_images_count} изображений")
print("\nКлассы: person")

# --- Обучение YOLOv10 ---
from ultralytics import YOLO

yolo_model = YOLO("yolov10n.pt")

# Колаб может быть без GPU → выбираем устройство автоматически
device = 0 if torch.cuda.is_available() else "cpu"
print("Используем устройство:", device)

train_results = yolo_model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name="yolov10n_people_detection",
    device=device
)

# --- Валидация ---
val_metrics = yolo_model.val(data=data_yaml_path)

print("Метрики валидации:")
print(f"mAP@0.5: {val_metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {val_metrics.box.map:.4f}")
print("\nПолные метрики:")
print(val_metrics)

# --- Визуализация результата ---
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt

# Выбор тестового изображения
if os.path.exists(test_dir):
    test_images_list = os.listdir(test_dir)
    sample_img_path = os.path.join(test_dir, test_images_list[0])
else:
    valid_dir = os.path.join(dataset_path, "valid", "images")
    valid_images_list = os.listdir(valid_dir)
    sample_img_path = os.path.join(valid_dir, valid_images_list[0])

# YOLO предсказание
prediction_results = yolo_model(sample_img_path)
result = prediction_results[0]  # Берём первый результат

# --- Сохранение YOLO-визуализации ---
result_plotted = result.plot()
yolo_output_file = "yolo_prediction.jpg"
Image.fromarray(result_plotted).save(yolo_output_file)
print(f"YOLO-визуализация сохранена: {yolo_output_file}")

# --- Ручная отрисовка рамок ---
plt.figure(figsize=(10, 10))
image = Image.open(sample_img_path)
plt.imshow(image)
plt.axis('off')

for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0])
    label = f"person {conf:.2f}"

    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
    )
    plt.text(
        x1,
        y1 - 10,
        label,
        color='red',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

plt.title("Детекции на тестовом изображении")

manual_output_file = "yolo_boxes_overlay.png"
plt.savefig(manual_output_file, dpi=200)
plt.close()
print(f"Ручная визуализация сохранена: {manual_output_file}")
