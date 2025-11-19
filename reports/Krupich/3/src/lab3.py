import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11: Детектор людей")
        self.root.geometry("1000x600")
        
        self.model = YOLO("best.pt")

        self.btn_load = Button(root, text="Загрузить изображение", command=self.load_img, font=("Arial", 14), bg="#ddd")
        self.btn_load.pack(pady=20)

        self.image_frame = Frame(root)
        self.image_frame.pack(expand=True, fill="both")

        self.lbl_original = Label(self.image_frame, text="Здесь будет оригинал", bg="gray", width=50, height=25)
        self.lbl_original.pack(side="left", padx=20, expand=True)

        self.lbl_result = Label(self.image_frame, text="Здесь будет результат", bg="gray", width=50, height=25)
        self.lbl_result.pack(side="right", padx=20, expand=True)

    def load_img(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return

        original_pil = Image.open(path)
        original_pil.thumbnail((450, 450))
        self.img_orig_tk = ImageTk.PhotoImage(original_pil)
        self.lbl_original.config(image=self.img_orig_tk, text="", width=0, height=0)

        results = self.model.predict(path, conf=0.5)
        
        res_array = results[0].plot()
        res_array = cv2.cvtColor(res_array, cv2.COLOR_BGR2RGB)
        
        result_pil = Image.fromarray(res_array)
        result_pil.thumbnail((450, 450))
        self.img_res_tk = ImageTk.PhotoImage(result_pil)
        self.lbl_result.config(image=self.img_res_tk, text="", width=0, height=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()