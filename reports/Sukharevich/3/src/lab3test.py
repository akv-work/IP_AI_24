import cv2
from ultralytics import YOLO

def realtime_detection(model_path, cam_index=0):
    """Детекция объектов в реальном времени через камеру"""
    print("\nЗапуск детекции в реальном времени... Нажми 'q' для выхода.")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Ошибка: камера не найдена или занята!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр с камеры")
            break
        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv12 Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "D:/University/4kurs/OIIS/lab3/my/runs/detect/yolo12m8/weights/best.pt"
    realtime_detection(model_path)
