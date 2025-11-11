import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

DATA_YAML = "./dataset/data.yaml"
OUTPUT_DIR = "./runs/train/license_plate_yolov12n"

WEIGHTS = "runs/train/license_plate_yolov12n/weights/best.pt"
IMG_DIR = "./images_in"
OUT_DIR = "./images_out"
os.makedirs(OUT_DIR, exist_ok=True)


def draw_boxes_and_save(image_path, results, save_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes = results.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2, conf, class]
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        text = f"{results.names[int(cls_id)]} {conf:.2f}"
        draw.text((x1 + 5, y1 + 5), text, fill="red")

    img.save(save_path)


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=704,
        batch=32,
        save=True,
        name="license_plate_yolov12n",
        project="runs/train",
        device=0,
        workers=8,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        cos_lr=True,
    )

    print("Training finished. Results:", results)

    model = YOLO(WEIGHTS)

    images = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        print(f"⚠В папке {IMG_DIR} нет изображений!")
        return

    for i, img_file in enumerate(images):
        img_path = os.path.join(IMG_DIR, img_file)
        out_path = os.path.join(OUT_DIR, f"pred_{img_file}")

        preds = model.predict(img_path, imgsz=1280, conf=0.25, save=False)
        res = preds[0]
        draw_boxes_and_save(img_path, res, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
