from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
path2 = r'best (4).pt'
path = r'test\27-bien-so-xe-may_jpg.rf.eb139d311c6be0584cfac0c45d786920.jpg'

model = YOLO(path2)

detected_texts = [] 
results3 = model(path)
image = cv2.imread(path)
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
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            min_pixel = np.min(gray_image)
            max_pixel = np.max(gray_image)
            contrast_stretch = ((gray_image - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
            blurred_image = cv2.GaussianBlur(contrast_stretch, (5, 5), 0)
            thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            
            cv2.imshow('Thresholded Image', thresholded_image)
            cv2.imwrite('contrast_stretch.jpg', thresholded_image) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # nhan dien bang easy ocr
            reader = easyocr.Reader(['en'], gpu = False)
            result = reader.readtext(thresholded_image)
            for (bbox, text, prob) in result:
                print(text)
                detected_texts.append(text) 

    cv2.imshow('easy ocr   ', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()