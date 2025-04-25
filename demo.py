import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import re
import os


# Khởi tạo model YOLO và PaddleOCR
model = YOLO("best.pt")
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=r'models\ch_PP-OCRv4_det_infer',
    rec_model_dir=r'models\en_PP-OCRv4_rec_infer',
    cls_model_dir=r'models\ch_ppocr_mobile_v2.0_cls_train'
)

# Xử lý chuỗi sau khi OCR
def process_string(text):
    while True:
        match = re.search(r'[A-Za-z]{3}|[^\w\s]', text)
        if match:
            end_pos = match.end()
            next_char = text[end_pos:end_pos+1]
            if next_char.isdigit():
                break
            else:
                text = text[:match.start()] + text[end_pos:]
        else:
            break
    return text

# Tăng tương phản ảnh xám
def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel = np.min(gray)
    max_pixel = np.max(gray)
    if max_pixel == min_pixel:
        contrast = gray
    else:
        contrast = ((gray - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
    return contrast

# Xử lý OCR và lưu biển số
def process_ocr(img, texts, filename_base):
    results = model(img)

    output_folder = "plates"
    output_folder1 = "crop"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder1, exist_ok=True)

    for result in results:
        if result.boxes is None or result.boxes.xyxy is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            pre_img = img[y1:y2, x1:x2]
            if pre_img.size == 0:
                continue

            save_path_crop = os.path.join(output_folder1, f"{filename_base}_plate_{idx}.jpg")
            cv2.imwrite(save_path_crop, pre_img)

            enhanced_img = enhance_contrast(pre_img)
            enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

            save_path = os.path.join(output_folder, f"{filename_base}_plate_{idx}.jpg")
            cv2.imwrite(save_path, enhanced_img_bgr)

            try:
                esized_img = cv2.resize(enhanced_img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ocr_result = ocr.ocr(esized_img, cls=True)
                print("OCR result:", ocr_result)

                text_lines = []

                if ocr_result and isinstance(ocr_result, list):
                    for line in ocr_result:
                        if isinstance(line, list):
                            for word in line:
                                if isinstance(word, list) and len(word) > 1:
                                    text_raw = word[1][0].strip().replace(" ", "").upper()
                                    if text_raw:
                                        text_lines.append(text_raw)
                                        texts.append(text_raw)

                # Vẽ khung và viết các dòng text trên box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for i, line in enumerate(text_lines[::-1]):  # Viết từ dưới lên
                    y_offset = y1 - 10 - (i * 20)
                    if y_offset < 0: y_offset = 5  # tránh ra ngoài ảnh
                    cv2.putText(img, line, (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                texts.append(f"OCR error: {e}")

    return " ".join(texts) + "\n"

# Mở ảnh và hiển thị
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    texts = []

    filename_base = os.path.splitext(os.path.basename(file_path))[0]
    full_text = process_ocr(img, texts, filename_base)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil.resize((500, 400)))

    image_label.config(image=img_tk)
    image_label.image = img_tk

    text_label.config(text=f"Nhận dạng: {process_string(full_text)}\nXử lý: {full_text}")
    print(full_text + "\n")
    print(process_string(full_text))

# Giao diện GUI
root = tk.Tk()
root.title("YOLO + PaddleOCR GUI")
root.geometry("600x600")

btn = Button(root, text="Chọn ảnh", command=open_image)
btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

text_label = Label(root, text="", wraplength=500, justify="left")
text_label.pack(pady=10)

root.mainloop()
