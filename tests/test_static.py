# AI Model Test | Static @rasvet
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2, os
from ultralytics import YOLO


OBJECT_NAME = 'proto_enemy'
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = '../best.pt'

model = YOLO(MODEL_PATH)

def get_color_from_confidence(confidence):
    """ Градация цвета рамки в зависимости от уверенности """
    RED_THRESHOLD = 0.2

    if confidence <= RED_THRESHOLD:
        return (0, 0, 255)

    normalized_conf = (confidence - RED_THRESHOLD) / (1.0 - RED_THRESHOLD)

    red = int(255 * (1 - normalized_conf))
    green = int(255 * normalized_conf)
    blue = 0

    return (blue, green, red)

def collect_images_from_folder(folder_path):
    """ Рекурсивная сборка путей изображений в папке """
    supported_formats = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))

    return image_paths

class StaticTest:

    def __init__(self, root):
        self.root = root
        self.current_index = 0
        self.image_paths = []
        self.total_images = 0

        self.root.title("AI Model Test | Static")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

        self.create_widgets()
        self.show_current_image()

    def create_widgets(self):
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(pady=10)

        self.folder_btn = ttk.Button(nav_frame, text="Выбрать папку", command=self.select_folder)
        self.folder_btn.pack(side=tk.TOP, pady=5)

        self.prev_btn = ttk.Button(nav_frame, text="←", command=self.prev_image, width=5)
        self.prev_btn.pack(side=tk.LEFT, padx=10)

        self.image_info = ttk.Label(nav_frame, text="", font=('Arial', 12))
        self.image_info.pack(side=tk.LEFT, padx=20)

        self.next_btn = ttk.Button(nav_frame, text="→", command=self.next_image, width=5)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.focus_set()

    def select_folder(self):
        """ Диалоговое окно выбора папки, сбор изображений и обновление интерфейса """
        folder_path = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder_path:
            self.image_paths = collect_images_from_folder(folder_path)
            self.image_paths.sort()
            self.total_images = len(self.image_paths)
            self.current_index = 0
            self.show_current_image()

    def show_current_image(self):
        """ Отображение изображения с результатом детекции """
        if not self.image_paths:
            self.image_info.config(text="Изображения не найдены")
            return

        self.image_info.config(text=f"{self.current_index + 1}/{self.total_images}")

        self.prev_btn.config(state='normal' if self.current_index > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_index < self.total_images - 1 else 'disabled')

        img_path = self.image_paths[self.current_index]
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Ошибка загрузки изображения: {img_path}")
            self.next_image()
            return

        # Детекция объектов
        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        found_count = 0

        if boxes is not None:
            for box in boxes:
                conf = box.conf.item()

                if conf > CONFIDENCE_THRESHOLD:
                    found_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = get_color_from_confidence(conf)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    percent_confidence = int(conf * 100)
                    label = f'{OBJECT_NAME} {percent_confidence}%'
                    cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f'{os.path.basename(img_path)}: найдено {OBJECT_NAME} = {found_count}')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight() - 150

        pil_image.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(
            screen_width // 2,
            screen_height // 2,
            image=self.photo,
            anchor='center'
        )

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def next_image(self):
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.show_current_image()

def main():
    root = tk.Tk()
    app = StaticTest(root)
    root.mainloop()

if __name__ == "__main__":
    main()