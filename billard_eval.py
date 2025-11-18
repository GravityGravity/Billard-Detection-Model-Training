# billard_eval.py — Computer Vision Billiard Ball Detector
#
# Description:
#   Loads a single image, preprocesses it to 512×512, runs the trained
#   billiard detector model, and draws the detected balls (solid/striped)
#   on the image. Prints coordinates + class and displays the result.
#
# Example:
#   python billard_eval.py ...path/to/image.png
#

import torch
import cv2
import sys
import numpy as np
import torchvision
from pathlib import Path
from billard_train import *

# Select GPU if not possible use CPU for tensor computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the input image path from CLI
img_path = Path(sys.argv[1])
img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create scaling factors so predictions can be mapped back correctly
h, w, c = img.shape
scaleH = 512.0 / h
scaleW = 512.0 / w

# Resize to the 512×512 shape used by the training pipeline
img_resized = cv2.resize(img, (512, 512))

# Normalize to [0,1] and convert to PyTorch tensor format (C, H, W)
img_normalize = np.array(img_resized, dtype=np.float32) / 255.0
img_tensor = torch.tensor(np.transpose(img_normalize, (2, 0, 1)))
img_tensor = img_tensor.unsqueeze(0).to(device)

# Load the trained detector model
model = torch.load("billard_detect_model.pth",
                   map_location=device, weights_only=False)
model.to(device)

# Run the model in evaluation mode
model.eval()

with torch.no_grad():
    outputs = model(img_tensor)
    TotalBalls = 0
    ball_coordinates = []
    # Loop through the 16×16 prediction grid
    for g_row in range(16):
        for g_col in range(16):
            out_conf = torch.sigmoid(outputs[0, g_col, g_row, 0])
            out_rel_x = outputs[0, g_col, g_row, 1].item()
            out_rel_y = outputs[0, g_col, g_row, 2].item()
            out_rel_w = outputs[0, g_col, g_row, 3].item()
            out_rel_h = outputs[0, g_col, g_row, 4].item()
            out_solid = outputs[0, g_col, g_row, 5].item()
            out_striped = outputs[0, g_col, g_row, 6].item()
            # Only count strong detections
            if out_conf > 0.8:
                TotalBalls += 1
                rad = int(((out_rel_w * 32) / scaleW) // 2)
                x = int(((out_rel_x * 32) + (g_row * 32)) / scaleW)
                y = int(((out_rel_y * 32) + (g_col * 32)) / scaleH)
                if out_striped > out_solid:
                    ball_cls = 1
                else:
                    ball_cls = 0
                ball_coordinates.append(
                    (x, y, rad, ball_cls))
                continue
    print(TotalBalls)

    for x, y, rad, type in ball_coordinates:
        cv2.circle(img, (x, y), rad, ((255, 0, 0)
                   if type == 1 else (0, 255, 0)), 3)
        print(f'{x} {y} {rad} {type}')

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow(str({img_path}), img)
cv2.waitKey(0)
