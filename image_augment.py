# FILE: image_augment
# Description:
#   Takes in test file image numbers and creates an image then outputs augmentations of that image:
#               Augments:
#                       - Rotate 180
#                       - Zoom 2x
#                       - Zoomout 2x
#                            + its txt annotation labels


import cv2 as cv
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import colorama as cl
# Auto reset colored console print
cl.init(autoreset=True)

print('\n   ---image_augment.py---  ')

root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
print(f'    Root: {root}')


def image_gather(normalization: bool):
    """Gather and read in images for model training
    """
    print(cl.Fore.CYAN + 'image_gather()')
    img_180str = ''
    img_90str = ''
    img_zoom2str = ''
    img_zoom4str = ''
    img_graystr = ''
    img_rgbstr = ''

    try:
        print(f'Image path passed: {sys.argv[1]}')
        folder = Path(sys.argv[1])
    except IndexError:
        sys.exit(cl.Fore.RED + '  ERROR: No Argument for image path')

    image_names = [(f.name, f.stem) for f in folder.glob('*.png')]
    for f_name, name in image_names:

        img = cv.imread(os.path.join(root, f_name), cv.IMREAD_COLOR)
        if img is None:
            print(f"Error: {f_name} Image not found or unable to read.")
            sys.exit(cl.Fore.Red + '    Exiting...')
        else:
            print(f"{f_name} Image loaded successfully!")
        print("Image shape:", img.shape)

        h, w, c = img.shape

        img_180str = ''
        img_90str = ''
        img_zoom2str = ''
        img_graystr = ''
        img_rgbstr = ''

        img_180 = cv.rotate(img, cv.ROTATE_180)
        img_90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img_zoom2 = img_zoom(img, h, w, 2)
        img_zoom2_bbox_list = []
        img_zoom4 = img_zoom(img, h, w, 4)
        img_zoom4_bbox_list = []

        # Label Processing
        label_path = os.path.join(root, name + '.txt')
        if not os.path.exists(label_path):
            print(cl.Fore.RED + '    X  ' +
                  (name + '.txt') + ' img annotation did not exist.')
            continue
        else:
            df = pd.read_csv(label_path, sep='\r', header=None)
            for idx, row in df.iterrows():
                img_graystr = img_graystr + row[0] + '\n'
                img_rgbstr = img_rgbstr + row[0] + '\n'

                if idx == 0:
                    img_90str = img_90str + row[0] + '\n'
                    img_180str = img_180str + row[0] + '\n'
                else:
                    x_c, y_c, rad, cls = map(int, (row[0].split()))
                    img_180str = img_180str + \
                        coordinates_180_rotate(x_c, y_c, rad, h, w, cls)
                    img_90str = img_90str + \
                        coordinates_90_rotate(x_c, y_c, rad, h, w, cls, img_90)
                    img_zoom2_bbox = coordinates_Zoom(
                        x_c, y_c, rad, h, w, cls, 2)
                    if img_zoom2_bbox:
                        img_zoom2_bbox_list.append(img_zoom2_bbox)
                    img_zoom4_bbox = coordinates_Zoom(
                        x_c, y_c, rad, h, w, cls, 4)
                    if img_zoom4_bbox:
                        img_zoom4_bbox_list.append(img_zoom4_bbox)

        img_zoom2str = str(len(img_zoom2_bbox_list)) + '\n'
        for bbox in img_zoom2_bbox_list:
            img_zoom2str += bbox

        img_zoom4str = str(len(img_zoom4_bbox_list)) + '\n'
        for bbox in img_zoom4_bbox_list:
            img_zoom4str += bbox

        texts = {name + '_flip.txt': img_180str,
                 name + '_rot90.txt': img_90str,
                 name + '_gray.txt': img_graystr,
                 name + '_CC.txt': img_rgbstr,
                 name + '_zoom2x.txt': img_zoom2str,
                 name + 'zoom4x.txt': img_zoom4str}

        cv.imwrite(os.path.join(output_dir, name + '_flip.png'), img_180)
        cv.imwrite(os.path.join(output_dir, name + '_rot90.png'), img_90)
        cv.imwrite(os.path.join(output_dir, name + '_gray.png'), img_gray)
        cv.imwrite(os.path.join(output_dir, name + '_CC.png'), img_rgb)
        cv.imwrite(os.path.join(output_dir, name + '_zoom2x.png'), img_zoom2)
        cv.imwrite(os.path.join(output_dir, name + '_zoom4x.png'), img_zoom4)

        for filename, bboxs in texts.items():
            with open(os.path.join(output_dir, filename), 'w') as ftxt:
                ftxt.write(bboxs)

    print(f'image_names: {image_names}')


def coordinates_180_rotate(x, y, rad, h, w, cls):
    x_180 = w - x - 1
    y_180 = h - y - 1
    rad_180 = rad
    cls = cls

    return f"{x_180} {y_180} {rad_180} {cls}\n"


def coordinates_90_rotate(x, y, rad, h, w, cls, img):
    x_90 = h - y - 1
    y_90 = x
    rad_90 = rad
    cls = cls

    return f"{x_90} {y_90} {rad_90} {cls}\n"


def img_zoom(img, h, w, zoom_factor):
    # compute crop size (50% zoom = take center 1/zoom_factor)
    new_h = h // zoom_factor
    new_w = w // zoom_factor

    # top-left corner of the crop box
    y1 = (h - new_h) // 2
    y2 = y1 + new_h

    x1 = (w - new_w) // 2
    x2 = x1 + new_w

    # crop the center
    cropped = img[y1:y2, x1:x2]

    # resize back to original size
    zoomed = cv.resize(cropped, (w, h))

    return zoomed


def coordinates_Zoom(x, y, rad, h, w, cls, zoom_factor):

    z = zoom_factor

    # size of the cropped region
    new_w = w // z
    new_h = h // z

    # top-left corner of crop
    crop_x1 = (w - new_w) // 2
    crop_y1 = (h - new_h) // 2

    # BR corner of crop
    crop_x2 = (crop_x1 + new_w)
    crop_y2 = (crop_y1 + new_h)

    # translate into crop coordinates
    x_c = x - crop_x1
    y_c = y - crop_y1

    # scale back up
    x_new = x_c * z
    y_new = y_c * z
    rad_new = rad * z

    if (x - rad < crop_x1 or
        x + rad > crop_x2 or
        y - rad < crop_y1 or
            y + rad > crop_y2):
        return ''   # circle NOT fully inside

    return f"{x_new} {y_new} {rad_new} {cls}\n"


image_gather(True)
