from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

# Khởi tạo PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

path2 = r'best (5).pt'
path = r'test/AQUA7_19909_checkin_2020-10-29-0-20XbQxBkO41n_jpg.rf.64799d671d67ec1d963d12779442f8be.jpg'

model = YOLO(path2)

detected_texts = [] 
results3 = model(path)
image = cv2.imread(path)

def straighten_image(gray, output_path):
    # Chuyển đổi ảnh sang màu xám
    
    cv2.imshow('adfasdf', gray)
    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Tìm các đường thẳng trong ảnh
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Tính toán góc trung bình của các đường thẳng
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90  # Chuyển đổi radian sang độ
            angles.append(angle)
    
    if angles:
        # Tính góc trung bình
        mean_angle = np.mean(angles)
        
        # Lấy kích thước của ảnh
        (h, w) = gray.shape[:2]
        
        # Tạo ma trận xoay
        M = cv2.getRotationMatrix2D((w // 2, h // 2), mean_angle, 1.0)
        
        # Xoay ảnh
        rotated = cv2.warpAffine(gray, M, (w, h))
        
        cv2.imshow('phang ',rotated )
        # Lưu ảnh đã làm phẳng
        cv2.imwrite(output_path, rotated)
        
        print(f"Đã lưu ảnh đã làm phẳng tại: {output_path}")
        return rotated  # Trả về ảnh đã làm phẳng
    else:
        print("Không tìm thấy đường thẳng nào trong ảnh.")
        return gray  # Trả về ảnh gốc nếu không tìm thấy đường thẳng

def pre(cropped_image):
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
    min_pixel = np.min(gray_image)
    max_pixel = np.max(gray_image)
    contrast_stretch = ((gray_image - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
    blurred_image2 = cv2.GaussianBlur(contrast_stretch, (5, 5), 0)
    blurred_image = cv2.GaussianBlur(blurred_image2, (5, 5), 0)
    thresholded_image2 = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Làm phẳng ảnh
    straightened_image = straighten_image(thresholded_image2, 'out.jpg')
    
    cv2.imshow('Thresholded Image', thresholded_image2)
    cv2.imwrite('contrast_stretch.jpg', thresholded_image2) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return thresholded_image2

if image is None:
    print("Không thể tải ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    for r in results3:
        boxes = r.boxes.xyxy  
        if len(boxes) == 0:
            continue 

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  

            border_color = (0, 255, 0) 
            border_thickness = 2  
            cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border_thickness)

            cropped_image = image[y1:y2, x1:x2]
            
            thresholded_image = pre(cropped_image)
            result = ocr.ocr(thresholded_image, cls=True)

            # In ra văn bản đã trích xuất
            detected_texts = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    text = line[1][0]
                    print("Detected text: " + text)
                    detected_texts.append(text)


    cv2.imshow('Kết quả nhận diện', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()