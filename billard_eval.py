# Project3.py for CompVision U72902494

import torch
import cv2
import sys
import numpy as np
import torchvision
from pathlib import Path
from billard_train import *

# Select GPU if not possible use CPU for tensor computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# single image path
img_path = Path(sys.argv[1])  # folder with .png/.jpg files (224x224)
img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create scalings
h, w, c = img.shape
scaleH = 512.0 / h
scaleW = 512.0 / w

# Ensure 512x5512 image size (if not already)
img_resized = cv2.resize(img, (512, 512))

# Normalize and turn image into a tensor with dimensiosn (C, W, H)
img_normalize = np.array(img_resized, dtype=np.float32) / 255.0
img_tensor = torch.tensor(np.transpose(img_normalize, (2, 0, 1)))
img_tensor = img_tensor.unsqueeze(0).to(device)


# Load model
model = torch.load("billard_detect_model.pth",
                   map_location=device, weights_only=False)
model.to(device)

# Evaluate Model
model.eval()

with torch.no_grad():
    outputs = model(img_tensor)
    TotalBalls = 0
    ball_coordinates = []
    # print('\n')
    # print(f'EVAL IMAGE: {img_path}')
    for g_row in range(16):
        for g_col in range(16):
            out_conf = torch.sigmoid(outputs[0, g_col, g_row, 0])
            out_rel_x = outputs[0, g_col, g_row, 1].item()
            out_rel_y = outputs[0, g_col, g_row, 2].item()
            out_rel_w = outputs[0, g_col, g_row, 3].item()
            out_rel_h = outputs[0, g_col, g_row, 4].item()
            out_solid = outputs[0, g_col, g_row, 5].item()
            out_striped = outputs[0, g_col, g_row, 6].item()
            # if out_conf > 0:
            #     # print(
            #     #     f'c {out_conf}, rel_x {out_rel_x}, rel_y {out_rel_y} sol {out_solid}, str {out_striped}, g_row{g_row}, g_col {g_col}')
            #     # print(f'{i, g_row, g_col, 0}')
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
        print(f'{x} {y} {rad} {type}')
