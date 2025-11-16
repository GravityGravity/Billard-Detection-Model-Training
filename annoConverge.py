# File: Annotation Converge.py

# Description:
#       Takes in all annotations for all image billard balls txt files and outputs them into a single csv
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

headers = ['filename', 'x_center', 'y_center', 'width', 'height', 'ball_type']
root = Path(sys.argv[1])
ann = []

csv = pd.DataFrame()

for f in root.glob('*.txt'):

    label_path = f.with_suffix('.txt')
    if not label_path.exists():
        print('    X' +
              {label_path.name} + ' img annotation did not exist')  # debug
        continue

        # Write image ball labels into annotations

    with open(label_path, 'r') as label:
        lines = [line.strip() for line in label.readlines()[1:]]
        labels = [line.split() for line in lines]

        for boundary in labels:
            Radius = int(boundary[2])
            TL_x = int(boundary[0]) - Radius
            TL_y = int(boundary[1]) - Radius
            BR_x = TL_x + (Radius*2)
            BR_y = TL_y + (Radius*2)
            B_class = int(boundary[3])

            new_dict = dict.fromkeys(
                headers)

            new_dict = dict(zip(new_dict.keys(), (
                f.stem+'.png', int(boundary[0]), int(boundary[1]), Radius*2, Radius*2, B_class)))

            print(new_dict)

            ann.append(new_dict)

csv = pd.DataFrame(ann)

csv.to_csv("test/annotated_images/annotations.csv", index=False)
