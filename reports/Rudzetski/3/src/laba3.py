import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import random

def main():
    model = YOLO("yolov12s.pt")

    DATA_YAML = "datasets/cats_dataset/data.yaml"
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"data.yaml не найден: {DATA_YAML}")

    print(f"data.yaml найден: {DATA_YAML}")

    print("Запуск fine-tuning YOLOv12s...")
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        name="yolov12s_cats_final",
        patience=10,
        save=True,
        plots=True,
        val=True,
        device=0,
        workers=0
    )

    metrics = model.val()
    print(f"mAP@0.50:     {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")

    best_model = YOLO(results.best)

    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/Kitten_stretching.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
    ]

    def show(url, conf=0.25):
        img = Image.open(BytesIO(requests.get(url).content))
        res = best_model(img, conf=conf, verbose=False)[0]
        plt.figure(figsize=(10,7))
        plt.imshow(res.plot())
        plt.title(f"conf > {conf} | найдено: {len(res.boxes)}")
        plt.axis('off')
        plt.show()

    for c in [0.25, 0.5, 0.75]:
        print(f"\n--- conf = {c} ---")
        show(random.choice(urls), c)

    with open("final_results.txt", "w", encoding="utf-8") as f:
        f.write(f"mAP@0.50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.50:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Модель: {results.best}\n")
    print("Результаты сохранены в final_results.txt")

if __name__ == '__main__':
    main()