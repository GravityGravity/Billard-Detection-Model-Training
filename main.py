# FILE: Main.py
#
# Description:
#
#
# Test file locations:
#   - C:\Users\Work\_repos\CompVision\Billard-Train\test\annotated_images

# 1 for striped balls and 0 for solid balls


import sys
import os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import colorama as cl
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch as tc
import torchvision
import numpy as np  # Mathmatical data structs
import pandas as pd  # Handle large datasets
import PIL as pl  # Pillow basic image manipulation

cl.init(autoreset=True)  # Reset color to default after every print

# Tells Pytroch to use GPU for tensor operations
# device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

# Classes for COCO Style data format standard
class ballDataset(tc.utils.data.Dataset):
    """Gather and read in images for model training
    """

    def __init__(self,
                 exts: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
                 scale_01: bool = True):
        self.exts = exts
        self.scale_01 = scale_01
        self.rows = rows = defaultdict(list)
        self.Categories = Categories = [{"id": 0, "name": 'solid'},
              {"id": 1, "name": 'striped'}]
        self.ann = {}
        print(cl.Fore.CYAN + 'Class ballDataset(tc.Dataset)')

        try:
            print(f'Image path passed: {sys.argv[1]}')
            self.root = Path(sys.argv[1])
        except IndexError:
            sys.exit(cl.Fore.RED + '  ERROR: No Argument for image path')

        # Create Categories
        rows["categories"] = Categories

        img_iter = 1

        # Loop through all files in training image data
        for f in self.root.glob('*'):

            if not f.suffix.lower() in exts:
                continue

            # Read in image as color
            img = cv.imread(str(f), cv.IMREAD_COLOR_RGB)
            if img is None:
                print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
                      {f.name} + ' img did not get read')  # debug

            # Add image to COCO data format standard
            rows['images'].append(
                {"id": img_iter, "file_name": f.name, "height": img.shape[0], "width": img.shape[1]})

            # Find image annotation file path
            label_path = f.with_suffix('.txt')
            if not label_path.exists():
                print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
                      {label_path.name} + ' img annotation did not exist')  # debug
                continue

            # Print file number read
            tqdm.write(f"{f.stem}")

            # Write image ball labels into annotationsz
            with open(label_path, 'r') as label:
                lines = [line.strip() for line in label.readlines()[1:]]
                labels = [line.split() for line in lines]

                for boundary in labels:
                    Radius = int(boundary[2])
                    TL_x = int(boundary[0]) - Radius
                    TL_y = int(boundary[1]) - Radius
                    B_class = int(boundary[3])

                    rows['annotations'].append(
                        {"id": (len(rows['annotations'])+1), "image_id": img_iter, "bbox": [TL_x, TL_y, Radius*2, Radius*2], "category_id": B_class})

            img_iter += 1
            self.images = rows['images']
            self.ann = rows['annotations']

    def __len__(self) -> int:
        return len(self.rows['images'])

    def print_dataset(self):
        for key, value in self.rows.items():
            print(f'{key}')
            for v in value:
                print(f'{v}')
        return None

    def __getitem__(self, idx):
        img_info = self.images[idx]

        # Read in image as color
        img = cv.imread(os.path.join(self.root, img_info['file_name']), cv.IMREAD_COLOR_RGB)
        if img is None:
            print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
                    {img_info['file_name']} + ' img did not get read')  # debug
        
        # Normalize image
        img = img.astype(np.float32) / 255.0

        # OpenCV reads img as numpy array H, W, C.  create tensor with order of img shape as C, H, W.
        img = tc.as_tensor(img).permute(2, 0, 1)

        # Get ball annotations for this img
        img_anns = [a for a in self.ann if a['image_id'] == img_info['id']]
        boxes = []
        labels = []

        # Create bounding boxes for each ball in img and add classification of ball into labels
        for a in img_anns:
            min_x, min_y, w, h = a['bbox']
            max_x, max_y = min_x + w, min_y + h
            boxes.append([min_x, min_y, max_x, max_y])
            labels.append(a['category_id'])

        # Target data standard format that pytorch vision takes in for model training
        target = {"boxes": tc.as_tensor(boxes, dtype=tc.float32),
                  "labels": tc.as_tensor(labels, dtype=tc.int64),
                  "image_id": tc.tensor([img_info['id   ']], dtype=tc.int64)}
        
        return img, target


bDet = ballDataset()

bDet.print_dataset()

class ballDetector():
    def __init__(self):
        backbone = torchvision.models.resnet.resnet18()
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'feat4'})





class ballDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Instructor Code
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(4*4*16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Flatten(x, 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
