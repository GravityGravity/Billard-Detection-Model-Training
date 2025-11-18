# File: Annotation_Converge.py
#
# Description:
#   Reads every .txt annotation file in a folder, extracts all billiard-ball
#   label entries, converts them into a consistent CSV format, and writes the
#   combined dataset to:
#       test/annotated_images/annotations.csv
#
# Example:
#   python Annotation_Converge.py ...path/to/annotation_folder/
#
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# The final CSV will use these column names
headers = ['filename', 'x_center', 'y_center', 'width', 'height', 'ball_type']

# Folder containing all .txt annotation files
root = Path(sys.argv[1])

# List to store each annotation row before converting to a DataFrame
ann = []

# Loop through every .txt file in the directory
for f in root.glob('*.txt'):

    label_path = f.with_suffix('.txt')
    # Safety check — skip files that do not exist
    if not label_path.exists():
        print('    X' +
              {label_path.name} + ' img annotation did not exist')  # debug
        continue

    # Read annotation lines (skipping the first line which stores count)
    with open(label_path, 'r') as label:
        lines = [line.strip() for line in label.readlines()[1:]]
        labels = [line.split() for line in lines]

        # Process every ball boundary in the annotation file
        for boundary in labels:
            Radius = int(boundary[2])

            # Top-left and bottom-right bounding box coords
            TL_x = int(boundary[0]) - Radius
            TL_y = int(boundary[1]) - Radius
            BR_x = TL_x + (Radius*2)
            BR_y = TL_y + (Radius*2)

            B_class = int(boundary[3])

            # Build annotation dictionary for CSV
            new_dict = dict.fromkeys(
                headers)

            new_dict = dict(zip(new_dict.keys(), (
                f.stem+'.png',
                int(boundary[0]),
                int(boundary[1]),
                Radius*2,
                Radius*2,
                B_class)))

            print(new_dict)  # Debug print
            ann.append(new_dict)

# Convert collected annotations to a DataFrame and export as CSV
csv = pd.DataFrame(ann)
csv.to_csv("test/annotated_images/annotations.csv", index=False)
