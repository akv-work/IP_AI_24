import os
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from roboflow import Roboflow

def safe_list_images(path):
    if not os.path.exists(path):
        print(f"[ОШИБКА] Путь не найден: {path}")
        return []
    return [
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_dataset(dataset_dir):
    print("\n=== АНАЛИЗ ДАТАСЕТА ===")

    dataset_dir = Path(dataset_dir)

    # Пути изображений
    train_p = dataset_dir / "train" / "images"
    val_p = dataset_dir / "valid" / "images"
    test_p = dataset_dir / "test" / "images"

    # Загрузка data.yaml
    yaml_path = dataset_dir / "data.yaml"
    data = load_yaml(yaml_path)

    print(f"Классы: {data['nc']}")
    print(f"Названия классов: {data['names']}")

    # Подсчёты
    for name, p in [("train", train_p), ("val", val_p), ("test", test_p)]:
        imgs = safe_list_images(str(p))
        print(f"{name}: {len(imgs)} изображений")

    print("Анализ датасета завершён.\n")


def show_random_images(dataset_dir, count=6):
    dataset_dir = Path(dataset_dir)
    train_p = dataset_dir / "train" / "images"

    images = safe_list_images(str(train_p))
    if not images:
        print("Нет изображений для визуализации.")
        return

    chosen = random.sample(images, min(count, len(images)))

    plt.figure(figsize=(12, 8))
    for i, img_path in enumerate(chosen):
        img = Image.open(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def train_yolo(data_yaml, epochs=5):
    print("\n=== ОБУЧЕНИЕ YOLO ===")

    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,  
        batch=16,
        device="cpu",
        workers=8,
        project="runs/license_plate_detector",
        name="experiment",
        exist_ok=True
    )

    print("Обучение завершено.\n")
    return model



def validate_model(model, data_yaml):
    print("\n=== ВАЛИДАЦИЯ МОДЕЛИ ===")
    model.val(data=data_yaml)
    print("Валидация завершена.\n")

def run_prediction(model, img_path):
    print("\n=== ПРЕДСКАЗАНИЕ ===")
    results = model(img_path)
    results[0].show()  # Покажет окно с bbox
    print("Предсказание завершено.\n")


def main():
    print("Загрузка Roboflow…")

    rf = Roboflow(api_key="Aw9Jet0111VWlnIJz9cL")
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    dataset = project.version(11).download("yolov11")

    dataset_dir = dataset.location
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    print(f"Датасет загружен в: {dataset_dir}")

    analyze_dataset(dataset_dir)

    show_random_images(dataset_dir)

    model = train_yolo(data_yaml)

    validate_model(model, data_yaml)

    # Тестовое изображение
    test_img = safe_list_images(os.path.join(dataset_dir, "test", "images"))
    if test_img:
        run_prediction(model, test_img[0])


if __name__ == "__main__":
    main()
