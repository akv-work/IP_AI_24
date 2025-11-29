import os
import random
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


def safe_list_images(path):
    if not os.path.exists(path):
        print(f"[ОШИБКА] Путь не найден: {path}")
        return []
    return [
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def predict_and_show_many(model, image_paths, cols=5):
    """Показывает N предсказаний в одной фигуре"""

    num_images = len(image_paths)
    rows = (num_images + cols - 1) // cols  

    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    ax = ax.flatten()  

    for i, img_path in enumerate(image_paths):
        print(f"\n=== ПРЕДСКАЗАНИЕ ДЛЯ: {img_path} ===")

        result = model(img_path)[0]
        result.save(filename=f"temp_{i}.jpg")  

        img = Image.open(f"temp_{i}.jpg")
        ax[i].imshow(img)
        ax[i].set_title(os.path.basename(img_path))
        ax[i].axis("off")

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"Объект: class={cls}, conf={conf:.3f}, coords={xyxy}")

    for j in range(i+1, rows * cols):
        ax[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    dataset_dir = r"C:\4curs1sem\ОИИС\labs\lab3\dataset"
    test_images = Path(dataset_dir) / "test" / "images"

    imgs = safe_list_images(str(test_images))
    if not imgs:
        print("Нет изображений в test/images!")
        return

    selected_images = random.sample(imgs, 5)
    print("Выбраны изображения:")
    for img in selected_images:
        print(" •", img)

    model_path = r"C:\4curs1sem\ОИИС\labs\lab3\weights\best.pt"
    model = YOLO(model_path)

    predict_and_show_many(model, selected_images)


if __name__ == "__main__":
    main()
