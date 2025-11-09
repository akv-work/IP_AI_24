import os
from ultralytics import YOLO
import ultralytics
print(ultralytics.__file__)
VIDEO_PATH = "data/How_to_Play_Rock_Paper_Scissors.mp4"
OUTPUT_DIR = "tracking_results"
MODEL_WEIGHTS = "D:/University/4kurs/OIIS/lab3/my/runs/detect/yolo12m8/weights/best.pt"

TRACKER_TYPES = ["bytetrack.yaml", "botsort.yaml"]


def run_object_tracking(video_path, model_path, tracker_type="bytetrack"):
    print("=" * 50)
    print(f"Start tracking {tracker_type.upper()}...")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(model_path)

    results = model.track(
        source=video_path,
        tracker=tracker_type,
        show=False,
        save=True,
        project=OUTPUT_DIR,
        name=f"{tracker_type}_results",
        exist_ok=True
    )

    print(f"Finished {tracker_type} tracking. Results saved to {OUTPUT_DIR}/{tracker_type}_results")
    return results


def main_lab4_tracking():
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError("Model file not found (best.pt).")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError("Video not found. Check VIDEO_PATH.")

    for tracker in TRACKER_TYPES:
        run_object_tracking(VIDEO_PATH, MODEL_WEIGHTS, tracker_type=tracker)

if __name__ == "__main__":
    main_lab4_tracking()
