from ultralytics import YOLO
import cv2
import os


# Đường dẫn đến thư mục chứa ảnh và mô hình
image_path = r'test\AQUA7_20784_checkin_2020-11-1-15-54V7qhZNYbg5_jpg.rf.aef07f88699022f3e5cb0b58ee017e28.jpg'  # Đường dẫn đến hình ảnh đầu vào
output_folder = r'output'  # Thư mục để lưu ảnh đã xử lý
model_path = r'best (5).pt'  # Đường dẫn đến mô hình YOLO

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Tải mô hình YOLO
model = YOLO(model_path)

# Dự đoán trên hình ảnh
results = model(image_path)

# Lấy hình ảnh đầu vào
image = cv2.imread(image_path)

# Vẽ bounding boxes lên hình ảnh
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Lấy tọa độ của bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ của bounding box
        conf = box.conf[0]  # Độ tin cậy
        label = box.cls[0]  # Nhãn

        # cropped_image = image_np[y1:y2, x1:x2]
        # Vẽ bounding box lên hình ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Vẽ hình chữ nhật
        cv2.putText(image, f'{label:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Lưu hình ảnh đã xử lý
output_image_path = os.path.join(output_folder, 'result_with_text.jpg')
cv2.imwrite(output_image_path, image)

print(f"Kết quả đã được lưu tại: {output_image_path}")