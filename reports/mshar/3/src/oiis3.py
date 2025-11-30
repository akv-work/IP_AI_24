
from roboflow import Roboflow
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    rf = Roboflow(api_key="NyvDyUSaMPXReGLbyjq1")
    project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
    version = project.version(14)
    dataset = version.download("yolov8")

    model = YOLO("yolo11m.pt")

    model.train(data=f"{dataset.location}/data.yaml", epochs=5, imgsz=640, device=0, batch=2, workers=0)

    metrics = model.val()
    
    filename = "my.jpg"

    results = model(filename)
    results[0].save("result_detection.jpg")

    print(metrics.box.map)