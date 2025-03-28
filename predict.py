from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr

# Đường dẫn đến ảnh và mô hình
path = r'test\AQUA7_77379_checkin_2020-10-29-22-4Ftq5x7prc9_jpg.rf.8a793fca6975e6c70cd465d8871840ed.jpg'
path2 = r'best (5).pt'

# Tải mô hình YOLO
model = YOLO(path2)
results3 = model(path)

# Kiểm tra kết quả từ YOLO
for r in results3:
    boxes = r.boxes.xyxy  # Tọa độ (x1, y1, x2, y2) của các hộp bao quanh
    if len(boxes) == 0:
        print("Không phát hiện đối tượng nào.")
        continue  # Nếu không có hộp nào, bỏ qua vòng lặp

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  
        image = cv2.imread(path)
        cropped_image = image[y1:y2, x1:x2]
        
        # Lưu và hiển thị ảnh đã cắt
        cv2.imwrite(f'cropped_plate_{i}.jpg', cropped_image)
        im = Image.fromarray(cropped_image[..., ::-1])  # Chuyển đổi từ BGR sang RGB
        im.show(f'Cropped Plate {i}')

        # Làm to ảnh (tăng kích thước)
        scale_factor = 2  # Thay đổi hệ số tỷ lệ theo nhu cầu
        enlarged_image = cv2.resize(cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Chuyển ảnh đã làm to sang màu xám
        gray_image = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)
        
        # Tách ngưỡng
        _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Hiển thị ảnh xám và ảnh đã tách ngưỡng
        cv2.imshow('Gray Image', gray_image)
        cv2.imshow('Thresholded Image', thresholded_image)

        # Nhận diện chữ trong biển số
        reader = easyocr.Reader(['en'])
        result = reader.readtext(thresholded_image)

        # In kết quả nhận diện
        for (bbox, text, prob) in result:
            print(f'Text: {text}, Probability: {prob}')

# Chờ cho đến khi người dùng nhấn phím
cv2.waitKey(0)
cv2.destroyAllWindows()