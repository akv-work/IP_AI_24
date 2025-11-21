import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


VIDEO_PATH = "data/How_to_Play_Rock_Paper_Scissors.mp4"
OUTPUT_DIR = "tracking_results"
MODEL_WEIGHTS = "D:/University/4kurs/OIIS/lab3/my/runs/detect/yolo12m8/weights/best.pt"

TRACKER_TYPES = ["bytetrack.yaml", "botsort.yaml"]

EXPERIMENT_PARAMS = [
    {"name": "low_conf", "conf": 0.15, "iou": 0.45, "imgsz": 640},
    {"name": "high_conf", "conf": 0.50, "iou": 0.45, "imgsz": 640},
]


def visualize_image(img_path, title="Result"):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_default_tracking():

    for tracker in TRACKER_TYPES:
        print("=" * 60)
        print(f"Standard run: {tracker}")
        print("=" * 60)

        model = YOLO(MODEL_WEIGHTS)

        result_dir = os.path.join(OUTPUT_DIR, f"default_{tracker.replace('.yaml', '')}")

        model.track(
            source=VIDEO_PATH,
            tracker=tracker,
            save=True,
            project=OUTPUT_DIR,
            name=f"default_{tracker.replace('.yaml', '')}",
            exist_ok=True
        )


def run_parameter_experiments():
    model = YOLO(MODEL_WEIGHTS)

    for exp in EXPERIMENT_PARAMS:
        for tracker in TRACKER_TYPES:

            run_name = f"exp_{exp['name']}_{tracker.replace('.yaml', '')}"
            result_dir = os.path.join(OUTPUT_DIR, run_name)

            print("=" * 60)
            print(f"Custom run {exp['name']} | Tracker: {tracker}")
            print(f"conf={exp['conf']}  iou={exp['iou']}  imgsz={exp['imgsz']}")
            print("=" * 60)

            model.track(
                source=VIDEO_PATH,
                tracker=tracker,
                conf=exp["conf"],
                iou=exp["iou"],
                imgsz=exp["imgsz"],
                save=True,
                project=OUTPUT_DIR,
                name=run_name,
                exist_ok=True
            )


def main():
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError("Model file not found (best.pt).")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError("Video not found. Check VIDEO_PATH.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_default_tracking()
    run_parameter_experiments()

if __name__ == "__main__":
    main()
