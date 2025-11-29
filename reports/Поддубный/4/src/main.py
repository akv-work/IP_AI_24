import os
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/train/license_plate_yolov12n12/weights/best.pt")
    p.add_argument("--source", type=str, required=True, help="path to video file or directory of videos")
    p.add_argument("--tracker", type=str, required=True, help="tracker configs YAML (e.g. configs/bytetrack.yaml)")
    p.add_argument("--project", type=str, default="runs/track", help="where to save results")
    p.add_argument("--name", type=str, default="track_run", help="run name")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=1280, help="inference image size")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.project, args.name), exist_ok=True)

    model = YOLO(args.weights)
    print(f"Running tracker {args.tracker} on {args.source} ...")
    model.track(
        source=args.source,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        persist=True,
        save=True,
        save_txt=True,
        project=args.project,
        name=args.name,
    )
    print("Done. Results in:", os.path.join(args.project, args.name))

if __name__ == "__main__":
    main()
