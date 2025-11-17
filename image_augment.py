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
import colorama as cl

print('\n   ---image_augment.py---  ')

root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
print(f'    Root: {root}')

def image_gather(normalization: bool):
    """Gather and read in images for model training
    """
    print(cl.Fore.CYAN + 'image_gather()')

    try:
        print(f'Image path passed: {sys.argv[1]}')
        folder = Path(sys.argv[1])
    except IndexError:
        sys.exit(cl.Fore.RED + '  ERROR: No Argument for image path')

    image_names = [(f.name, f.stem) for f in folder.glob('*.png')]
    for f_name, name in image_names:

        
        img = cv.imread(os.path.join(root, f_name), cv.IMREAD_COLOR_BGR)
        if img is None:
            print(f"Error: {f_name} Image not found or unable to read.")
            sys.exit(cl.Fore.Red + '    Exiting...')
        else:
            print(f"{f_name} Image loaded successfully!")
        print("Image shape:", img.shape)

        label_path = os.path.join(root, name + '.txt')
        if label_path is None:
            print('    X' + {label_path.name} + ' img annotation did not exist')  # debug
            sys.exit(cl.Fore.Red + '    Exiting...')

        img_180 = (img, cv.ROTATE_180)


    print(f'image_names: {image_names}')


    

        # if f.suffix != '.png':
        #     continue

        # img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
        # if img is None:
        #     print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
        #           {f.name} + ' img did not get read')  # debug

        # label_path = f.with_suffix('.txt')
        # if not label_path.exists():
        #     print(cl.Fore.RED + '    X' + cl.Fore.WHITE +
        #           {label_path.name} + ' img annotation did not exist')  # debug
        #     continue

        # print(f.stem)

        # if normalization is True:
        #     img = img.astype(np.float32) / 255.0

        # with open(label_path, 'r') as label:
        #     label = [line.strip() for line in label]
        # rows[int(f.stem)] = {'file_name': f.name, 'image': img, 'annotation': label}


image_gather(True)

# for r in rows:
#     print(r)
#     print(rows[r])
#     print('\n')