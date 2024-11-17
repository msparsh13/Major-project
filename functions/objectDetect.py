

import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model

## if not accessing make sure to use an absolute path  
model = YOLO("../models/Object Detection/content/runs/detect/train/weights/best.pt")

# COCO dataset class names
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

def objectdetect(img):
    """
    Detects objects using YOLO and returns detected objects with their data.
    """
    obj = {}
    
    # Convert image to RGB as required by the model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform object detection with YOLO
    results = model.predict(img_rgb, conf=0.6)
    
    # Iterate through detection results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy.cpu().numpy().tolist()
            
            # Extract object class ID and convert to an integer
            cls_id = int(box.cls.cpu().numpy().item())  # Fix: Extract scalar value
            
            # Add object data to the dictionary
            obj[str(cls_id)] = {
                "coordinates": xyxy
            }
    
    return generate_prompt(obj_data=obj)

def generate_prompt(obj_data):
    """
    Generates a descriptive prompt based on detected object data.
    """
    prompts = []
    
    for obj_id, details in obj_data.items():
        # Map object ID to its class name
        obj_name = COCO_CLASSES[int(obj_id)]
        
        # Generate a prompt for the object
        prompts.append(
            f"A {obj_name} is detected at coordinates {details['coordinates']} from your point of view."
        )
    
    return prompts
