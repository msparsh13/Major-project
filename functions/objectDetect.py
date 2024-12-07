
import torch
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load YOLO model


path = os.path.abspath(os.getcwd()+"/models/object detection pascal voc/yolov10_pascal_voc.pt")
model = YOLO(model=path)


# COCO dataset class names
"""
COCO_CLASSES = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog",
    18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra",
    24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase",
    30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat",
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle",
    41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana",
    48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog",
    54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant",
    60: "bed", 61: "dining table", 62: "toilet", 63: "TV", 64: "laptop", 65: "mouse",
    66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster",
    72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors",
    78: "teddy bear", 79: "hair drier", 80: "toothbrush"
}
"""

PASCAL_CLASS = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}
def objectdetect(img):
    """
    Detects objects using YOLO and returns directional guidance based on object positions.
    """
    obj = {}

    # Convert image to RGB as required by the model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLO
    results = model.predict(img_rgb, conf=0.9)

    # Get image dimensions
    img_height, img_width = img.shape[:2]
    img_center_x = img_width / 2

    # Iterate through detection results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy.cpu().numpy().tolist()
            x_min, y_min, x_max, y_max = xyxy[0]

            # Calculate the center of the bounding box
            box_center_x = (x_min + x_max) / 2

            # Determine direction based on the center
            if box_center_x < img_center_x - img_width * 0.1:  # Left threshold
                direction = "toward your left"
            elif box_center_x > img_center_x + img_width * 0.1:  # Right threshold
                direction = "toward your right"
            else:
                direction = "toward the front"

            # Extract object class ID and convert to an integer
            cls_id = int(box.cls.cpu().numpy().item())

            # Add object data to the dictionary
            obj[str(cls_id)] = {
                "direction": direction
            }

    return generate_prompt(obj_data=obj)
##

def generate_prompt(obj_data):
    """
    Generates a descriptive prompt based on detected object data.
    """
    prompts = []

    for obj_id, details in obj_data.items():
        # Map object ID to its class name
        obj_name = PASCAL_CLASS[int(obj_id) + 1]

        # Generate a prompt with direction
        prompts.append(
            f"A {obj_name} is detected {details['direction']} from your point of view."
        )

    return prompts
