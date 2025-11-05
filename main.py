# FILE: Main.py
#
# Description:
#
#
# Test file locations:
#   - C:\Users\Work\_repos\CompVision\Billard-Train\test\annotated_images

import sys
import os
from pathlib import Path
import colorama as cl
import cv2 as cv
import torch as tc
import torchvision as tcv
import numpy as np  # Mathmatical data structs
import pandas as pd  # Handle large datasets
import PIL as pl  # Pillow basic image manipulation

cl.init(autoreset=True)  # Reset color to default after every print

# Tells Pytroch to use GPU for tensor operations
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

rows = []


def image_gather(normalization: bool):
    """Gather and read in images for model training
    """
    print(cl.Fore.CYAN + 'image_gather()')

    try:
        print(f'Image path passed: {sys.argv[1]}')
        folder = Path(sys.argv[1])
    except IndexError:
        sys.exit(cl.Fore.RED + '  ERROR: No Argument for image path')

    for i, f in enumerate(folder.glob('*')):
        if f.suffix != '.png':
            continue

        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
        if img is None:
            print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
                  {f.name} + ' img did not get read')  # debug

        label_path = f.with_suffix('.txt')
        if not label_path.exists():
            print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
                  {label_path.name} + ' img annotation did not exist')  # debug
            continue

        if normalization is True:
            img = img.astype(np.float32) / 255.0

        with open(label_path, 'r') as label:
            label = [line.strip() for line in label]

        rows.append({'file_name': f.name, 'image': img, 'annotation': label})


image_gather(True)

for r in rows:
    print(r['file_name'])
