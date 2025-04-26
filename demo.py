import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import re
import os

# Khởi tạo model
model = YOLO("best.pt")
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=r'models\ch_PP-OCRv4_det_infer',
    rec_model_dir=r'models\en_PP-OCRv4_rec_infer',
    cls_model_dir=r'models\ch_ppocr_mobile_v2.0_cls_train'
)

# Ánh xạ số thành chữ
def process_string(text):
    number_to_char = {
        '1': 'L', '2': 'Z', '5': 'S',
        '6': 'G', '8': 'B', '0': 'O'
    }
    if len(text) == 3 and text[2] in number_to_char:
        text = text[:2] + number_to_char[text[2]] + text[3:]
    return text

# Tăng tương phản
def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel, max_pixel = np.min(gray), np.max(gray)
    if max_pixel == min_pixel:
        return gray
    return ((gray - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)

# Xử lý một frame từ video hoặc ảnh
def process_frame(img, texts, filename_base="frame"):
    results = model(img)
    for result in results:
        if result.boxes is None or result.boxes.xyxy is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            pre_img = img[y1:y2, x1:x2]
            if pre_img.size == 0: continue

            enhanced_img = enhance_contrast(pre_img)
            enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            resized_img = cv2.resize(enhanced_img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            try:
                ocr_result = ocr.ocr(resized_img, cls=True)
                text_lines = []
                if ocr_result and isinstance(ocr_result, list):
                    for line in ocr_result:
                        for word in line:
                            if isinstance(word, list) and len(word) > 1:
                                text_raw = word[1][0].strip().replace(" ", "").upper()
                                if text_raw:
                                    if len(text_raw) == 3:
                                        text_raw = process_string(text_raw)
                                    text_lines.append(text_raw)
                                    texts.append(text_raw)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for i, line in enumerate(text_lines[::-1]):
                    y_offset = max(y1 - 10 - (i * 20), 5)
                    cv2.putText(img, line, (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                texts.append(f"OCR error: {e}")

    return img

# Mở ảnh
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path: return

    img = cv2.imread(file_path)
    texts = []
    processed_img = process_frame(img, texts, os.path.basename(file_path))

    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil.resize((500, 400)))

    image_label.config(image=img_tk)
    image_label.image = img_tk

# Mở video và tua nhanh (frame_skip = số frame bỏ qua giữa các lần xử lý)
def open_video():
    file_path = filedialog.askopenfilename()
    if not file_path: return

    cap = cv2.VideoCapture(file_path)
    frame_skip = 5  # tua nhanh: xử lý mỗi 5 frame
    texts = []

    while cap.isOpened():
        for _ in range(frame_skip - 1):
            cap.read()  # Bỏ qua frame

        ret, frame = cap.read()
        if not ret: break

        processed_frame = process_frame(frame, texts)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((500, 400)))

        image_label.config(image=img_tk)
        image_label.image = img_tk
        root.update_idletasks()
        root.update()

    cap.release()

# Giao diện GUI
root = tk.Tk()
root.title("YOLO + PaddleOCR GUI")
root.geometry("600x600")

Button(root, text="Chọn ảnh", command=open_image).pack(pady=10)
Button(root, text="Chọn video (tua nhanh)", command=open_video).pack(pady=5)

image_label = Label(root)
image_label.pack()

text_label = Label(root, text="", wraplength=500, justify="left")
text_label.pack(pady=10)

root.mainloop()
