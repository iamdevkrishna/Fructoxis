import sys
import json
import os
from PIL import Image
from ultralytics import YOLO

def run_detection(image_path, output_json_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(BASE_DIR, 'yolov8l.pt')
    yolo = YOLO(yolo_path)
    image = Image.open(image_path).convert('RGB')
    results = yolo(image, device='cpu', verbose=False)
    boxes = results[0].boxes
    yolo_dictionary = results[0].names

    if len(boxes) == 0:
        res = {"status": "no_boxes"}
    else:
        allowed_objects = ['apple', 'banana', 'orange', 'sports ball', 'bowl']
        target_box = None
        yolo_guess = ""

        # PRIORITY 1: Hunt for allowed objects first
        for box in boxes:
            guess = yolo_dictionary[int(box.cls[0])]
            if guess in allowed_objects:
                coords = box.xyxy[0].tolist()
                target_box = tuple(map(int, coords))
                yolo_guess = guess
                break

        # PRIORITY 2: Grab the largest box if no fruit was directly recognized
        if target_box is None:
            max_area = 0
            for box in boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = (x1, y1, x2, y2)
                    yolo_guess = yolo_dictionary[int(box.cls[0])]

        res = {
            "status": "success",
            "target_box": list(target_box) if target_box else None,
            "yolo_guess": yolo_guess
        }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_detection(sys.argv[1], sys.argv[2])
