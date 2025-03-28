from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")
results = model.train(data="mydata.yaml", epochs=5)

