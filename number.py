from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best (6).pt')  # Path to your YOLO model

# Load the input image
image = cv2.imread('contrast_stretch.jpg')  # Path to your input image

# Run the model on the image
results = model(image)

# Draw bounding boxes and annotations
for result in results:
    boxes = result.boxes.xyxy  # Get bounding box coordinates
    scores = result.boxes.conf  # Get confidence scores
    classes = result.boxes.cls  # Get class indices
    names = model.names  # Get class names

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
        label = f"{names[int(cls)]}: {score:.2f}"  # Create label with class name and score
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Add text

# Display the annotated image
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the display window