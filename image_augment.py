# FILE: image_augment.py
# Description:
#   Creates augmented versions of PNG images and their annotation TXT files.
#   Augments include:
#       - 180° rotate
#       - 90° rotate
#       - 2× zoom
#       - 4× zoom
#       - RGB color convert
#       - Grayscale convert
#
# Example (terminal):
#   python image_augment.py ...path/input_folder/ ...path/output_folder/


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

# root path -> Images Stored.  output_dir -> output directory to place new augmented images
root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
print(f'    Root: {root}')  # Debug


def image_gather(normalization: bool):
    """Gather and read in images that are augmented for model training
    """
    print(cl.Fore.CYAN + 'image_gather()')
    # Declare each images txt annotations string
    try:
        print(f'Image path passed: {sys.argv[1]}')
        folder = Path(sys.argv[1])
    except IndexError:
        sys.exit(cl.Fore.RED + '  ERROR: No Argument for image path')

    # List of png image file names
    image_names = [(f.name, f.stem) for f in folder.glob('*.png')]
    for f_name, name in image_names:

        # Read image data
        img = cv.imread(os.path.join(root, f_name), cv.IMREAD_COLOR)
        if img is None:
            print(f"Error: {f_name} Image not found or unable to read.")
            sys.exit(cl.Fore.Red + '    Exiting...')
        else:
            print(f"{f_name} Image loaded successfully!")
        print("Image shape:", img.shape)

        # Original image shape
        h, w, c = img.shape

        # Declare each images txt annotations string
        img_180str = ''
        img_90str = ''
        img_zoom2str = ''
        img_graystr = ''
        img_rgbstr = ''

        # CV processing images
        img_180 = cv.rotate(img, cv.ROTATE_180)
        img_90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Zoom image processing
        img_zoom2 = img_zoom(img, h, w, 2)
        img_zoom2_bbox_list = []
        img_zoom4 = img_zoom(img, h, w, 4)
        img_zoom4_bbox_list = []

        # Label Processing + Label check
        label_path = os.path.join(root, name + '.txt')
        if not os.path.exists(label_path):
            print(cl.Fore.RED + '    X  ' +
                  (name + '.txt') + ' img annotation did not exist.')
            continue
        else:
            # Read in Annotation txt of original image
            df = pd.read_csv(label_path, sep='\r', header=None)

            # Translate ball annotations into corresponding augments (zoom2x, zoom4x, flip, 90_rotate)
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

        # Write annotations for 2x image zoom due to balls out of boundary
        img_zoom2str = str(len(img_zoom2_bbox_list)) + '\n'
        for bbox in img_zoom2_bbox_list:
            img_zoom2str += bbox

        # Write annotations for 4x image zoom due to balls out of boundary
        img_zoom4str = str(len(img_zoom4_bbox_list)) + '\n'
        for bbox in img_zoom4_bbox_list:
            img_zoom4str += bbox

        # Dict of new file names and their corresponding annotations
        texts = {name + '_flip.txt': img_180str,
                 name + '_rot90.txt': img_90str,
                 name + '_gray.txt': img_graystr,
                 name + '_CC.txt': img_rgbstr,
                 name + '_zoom2x.txt': img_zoom2str,
                 name + 'zoom4x.txt': img_zoom4str}

        # Output all augmented images
        cv.imwrite(os.path.join(output_dir, name + '_flip.png'), img_180)
        cv.imwrite(os.path.join(output_dir, name + '_rot90.png'), img_90)
        cv.imwrite(os.path.join(output_dir, name + '_gray.png'), img_gray)
        cv.imwrite(os.path.join(output_dir, name + '_CC.png'), img_rgb)
        cv.imwrite(os.path.join(output_dir, name + '_zoom2x.png'), img_zoom2)
        cv.imwrite(os.path.join(output_dir, name + '_zoom4x.png'), img_zoom4)

        # Write all txt annotations for augmented images
        for filename, bboxs in texts.items():
            with open(os.path.join(output_dir, filename), 'w') as ftxt:
                ftxt.write(bboxs)

    # DEBUG: Print all image names
    print(f'image_names: {image_names}')


def coordinates_180_rotate(x, y, rad, h, w, cls):
    """
    Rotates object coordinates 180 degrees around the image center.
    - Flips x and y across both axes based on image width (w) and height (h).
    - Radius (rad) and class (cls) remain unchanged.
    - Returns a formatted string of the transformed circle parameters.
    """
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
    """
    Performs a centered zoom on an image.
    - Crops a centered area of size (h/zoom_factor, w/zoom_factor).
    - Resizes the cropped region back to the original image shape.
    - Returns the zoomed image.
    """
    # Compute crop size (50% zoom = take center 1/zoom_factor)
    new_h = h // zoom_factor
    new_w = w // zoom_factor

    # Top-left corner of the crop box
    y1 = (h - new_h) // 2
    y2 = y1 + new_h

    x1 = (w - new_w) // 2
    x2 = x1 + new_w

    # Crop the visible area
    cropped = img[y1:y2, x1:x2]

    # Resize image
    zoomed = cv.resize(cropped, (w, h))

    return zoomed


def coordinates_Zoom(x, y, rad, h, w, cls, zoom_factor):
    """
    Adjusts circle coordinates for a centered zoom transformation.
    - Computes how (x, y, rad) shift when the image is zoom-cropped.
    - Converts original absolute coordinates into the cropped region,
      then scales them back up by the zoom factor.
    - Ensures the entire circle remains inside the cropped area.
      Returns an empty string if the circle is partially cut off.
    - Returns a formatted string of the new zoom-adjusted parameters.
    """

    z = zoom_factor

    # Size of the cropped region
    new_w = w // z
    new_h = h // z

    # Top-left corner of crop
    crop_x1 = (w - new_w) // 2
    crop_y1 = (h - new_h) // 2

    # BR corner of crop
    crop_x2 = (crop_x1 + new_w)
    crop_y2 = (crop_y1 + new_h)

    # Translate into crop coordinates
    x_c = x - crop_x1
    y_c = y - crop_y1

    # Scale coordinates
    x_new = x_c * z
    y_new = y_c * z
    rad_new = rad * z

    # Filter balls outside of zoomed/croppedi mage
    if (x - rad < crop_x1 or
        x + rad > crop_x2 or
        y - rad < crop_y1 or
            y + rad > crop_y2):
        return ''

    return f"{x_new} {y_new} {rad_new} {cls}\n"


image_gather(True)
