import cv2
from ultralytics import YOLO

# Load model once (global)
model = YOLO("yolov8n.pt")  # auto-downloads first time

def count_people(image_path):
    """
    Runs YOLO locally to detect people.
    Returns: (head_count: int, annotated_bytes: bytes)
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Invalid image file")

    # Run detection
    results = model(image)[0]

    count = 0

    # Loop through detections
    for box in results.boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls == 0:
            count += 1

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put count text
    cv2.putText(
        image,
        f"People Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Convert image → bytes
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise Exception("Failed to encode image")

    return count, buffer.tobytes()