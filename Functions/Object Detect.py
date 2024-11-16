import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("Untitled Folder 1/Major Project/models/Object detection/content/runs/detect/train/weights/best.pt")

def objectdetect(img):
    """
    Detects objects using YOLO and estimates their depth and position using MiDaS.
    Returns detected objects with their depth, distance, and position classifications.
    """
    # Initialize an empty dictionary for object data
    obj = {}

    # Convert image to RGB as required by the model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLO
    results = model.predict(img_rgb, conf=0.6)

    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy.cpu().numpy().tolist()
            
            # Placeholder for depth information; call checkdepth() for depth estimation
            depth = checkdepth(img)

            # Determine position classification (e.g., far, near, very near) based on thresholds
            # This can depend on the depth value
            if depth < 1:
                position = "very near"
            elif depth < 3:
                position = "near"
            else:
                position = "far"

            # Add object data to the dictionary
            obj[str(box.cls.cpu().numpy())] = {
                "coordinates": xyxy,
                "depth": depth,
                "position": position,
            }

    return obj

def checkdepth(img):
    """
    Estimate depth from an input image using a MiDaS model.
    Returns the depth estimation as a numpy array.
    """
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    # Prepare input for the MiDaS model
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
    input_batch = transform(img).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert depth prediction to numpy array
    output = prediction.cpu().numpy()
    return np.mean(output)  # Return the average depth value for simplicity

def generateprompt(obj_data):
    """
    Generates a prompt based on the detected object's data.
    """
    prompts = []
    for obj_class, details in obj_data.items():
        prompts.append(
            f"Object {obj_class} is {details['position']} from your point of view, "
            f"located at coordinates {details['coordinates']}, "
            f"and at a depth of approximately {details['depth']} meters."
        )
    return prompts

