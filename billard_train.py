# File: billard_train.py
# Model: ResNet-50 backbone, 512x512 inputs, 16×16 prediction grid.
#
# Description:
#   Trains a YOLO-style billiard ball detector (solid vs striped).
#   Loads images + CSV annotations, builds a 16×16 target grid, runs training,
#   saves the model, then performs a quick confidence-based evaluation.
#
# Example (terminal):
#   python billard_train.py ...path/to/annotated_images/


import os
import sys
import cv2 as cv2
import torch
from pathlib import Path
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
import pandas as pd
from PIL import Image


class BallDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root = Path(sys.argv[1])  # Root of labeled images
        self.img_size = 512  # Model accepted image sizes
        self.gridsize = 16  # Grid size
        self.labels = pd.read_csv(
            'test/annotated_images/annotations.csv')  # Image labels

        # Create dict of image names with list of ball boundaries as labels
        self.image_to_labels = defaultdict(list)
        for _, row in self.labels.iterrows():
            self.image_to_labels[row['filename']].append({
                'x_center': float(row['x_center']),
                'y_center': float(row['y_center']),
                'width': float(row['width']),
                'height': float(row['height']),
                'ball_type': int(row['ball_type'])
            })

        # Create a list of image names with .jpg, .png. jpeg
        valid_exts = {".jpg", ".jpeg", ".png"}
        all_imgs = [p.name for p in self.root.iterdir(
        ) if p.suffix.lower() in valid_exts]

        # Sort list of image names
        self.image_files = sorted(all_imgs)

    def __len__(self):  # Number of images
        return len(self.image_files)

    def _load_and_resize(self, path):  # Resize all images to 512x512 with RGB conversion
        img = Image.open(path).convert('RGB')
        w, h = img.size
        # Return scales for ball center coordinates
        scaleW = self.img_size / w
        scaleH = self.img_size / h

        img = img.resize((self.img_size, self.img_size))
        return img, scaleW, scaleH

    def _img_to_tensor(self, img):
        img_arr = np.array(img, dtype=np.float32) / 255.0  # Normalize Image
        img_arr = np.transpose(img_arr, (2, 0, 1))  # W, H, C -> C, W, H
        # Create tensor of normalized, transposed images
        return torch.from_numpy(img_arr)

    def __getitem__(self, idx):
        filename = self.image_files[idx]  # Grab image name
        img_path = os.path.join(sys.argv[1], filename)  # Grab image path

        img, scaleW, scaleH = self._load_and_resize(img_path)
        img_tensor = self._img_to_tensor(img)

        g = self.gridsize  # gridsize 16

        # A 16 x 16 x 7 image grid with 7 layers of answers / weights
        target = torch.zeros((g, g, 7), dtype=torch.float32)
        cell_size = self.img_size / g

        # Creating bounding boxes of billards scaled to 512x512
        for ann in self.image_to_labels[filename]:
            # Scale ball box cords to 512
            x_c = (ann['x_center'] * scaleW)
            y_c = (ann['y_center'] * scaleH)
            w = (ann['width'] * scaleW)
            h = (ann['height'] * scaleH)
            b_class = (float(ann['ball_type']))

            # Which grid in the 16x16 gride size does the center fall into
            g_x = int(x_c / cell_size)
            g_y = int(y_c / cell_size)

            # Check Out of bounds box
            if not (0 <= g_x < g and 0 <= g_y < g):
                print(
                    f'  X - FILE: ({idx}) bbox not read -> g_x/y: {g_x, g_y} x/y {ann['x_center'], ann['y_center']} scaled x/y_c {x_c, y_c}\n')
                continue

            # cell-relative coordinates
            rel_x = (x_c - (g_x * cell_size)) / cell_size
            rel_y = (y_c - (g_y * cell_size)) / cell_size
            rel_w = (w) / cell_size
            rel_h = (h) / cell_size

            # Target / Ground Truth Tensor
            target[g_y, g_x, 0] = 1.0  # Target confidence
            target[g_y, g_x, 1] = rel_x  # X target
            target[g_y, g_x, 2] = rel_y  # Y target
            target[g_y, g_x, 3] = rel_w  # W target
            target[g_y, g_x, 4] = rel_h  # H target

            if b_class == 0:
                target[g_y, g_x, 5] = 1.0  # Solid balls
                target[g_y, g_x, 6] = 0
            if b_class == 1:
                target[g_y, g_x, 5] = 0  # Striped ball
                target[g_y, g_x, 6] = 1.0

        return img_tensor, target, img_path


class ResNet50Backbone(nn.Module):  # Resnet 50 backbone creation

    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-made ResNet50
        if pretrained:
            backbone = resnet50(weights="IMAGENET1K_V1")
        else:
            backbone = resnet50(weights=None)

        # Drop avgpool + fc, keep everything up to layer4
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.out_channels = 2048  # resnet50's last conv channels

    def forward(self, x):
        # Input: (N, 3, 512, 512)
        # Output: (N, 2048, 16, 16) for 512x512 input
        return self.body(x)


class BillardDetector(nn.Module):
    def __init__(self, pretrained_backbone=True, num_classes=2, grid_size=16):
        super().__init__()
        # Resnet50 as backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.grid_size = grid_size

        # obj (confidence) + bbox (x, y, w, h) + class types
        head_out = 1 + 4 + num_classes
        # Convlution Layer
        self.head = nn.Conv2d(self.backbone.out_channels,
                              head_out, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)          # (N, 2048, 16, 16)
        out = self.head(feats)            # (N, 7, 16, 16)
        out = out.permute(0, 2, 3, 1)     # (N, 16, 16, 7)
        return out


class YOLOLikeLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        """
        pred: (N, G, G, 7) raw logits
        target: (N, G, G, 7) ground truth
            target[...,0] = obj
            target[...,1:5] = bbox
            target[...,5:7] = class one-hot
        """
        obj_target = target[..., 0]
        bbox_target = target[..., 1:5]
        cls_target = target[..., 5:7]

        obj_logit = pred[..., 0]
        bbox_pred = pred[..., 1:5]
        cls_logit = pred[..., 5:7]

        # Objectness loss
        obj_loss = self.bce(obj_logit, obj_target)

        # Coordinates loss only where obj == 1
        obj_mask = obj_target.unsqueeze(-1)  # (N,G,G,1)
        bbox_loss = self.mse(bbox_pred, bbox_target) * obj_mask
        bbox_loss = bbox_loss.sum(-1)  # sum over 4 bbox components

        # Class loss only where obj == 1
        cls_loss = self.bce(cls_logit, cls_target) * obj_mask
        cls_loss = cls_loss.sum(-1)  # sum over 2 class logits

        # Weighted sum
        noobj_mask = 1.0 - obj_target
        obj_loss = obj_loss * obj_target + self.lambda_noobj * obj_loss * noobj_mask

        # Reduce over grid & batch
        obj_loss = obj_loss.mean()
        bbox_loss = bbox_loss.mean()
        cls_loss = cls_loss.mean()

        total = self.lambda_coord * bbox_loss + obj_loss + cls_loss
        return total, {'obj': obj_loss.item(), 'bbox': bbox_loss.item(), 'ball_type': cls_loss.item()}

# Epoch Training


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, targets, filename) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss, components = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)} "
                  f"loss={loss.item():.4f} "
                  f"obj={components['obj']:.4f} "
                  f"bbox={components['bbox']:.4f} "
                  f"ball_type={components['ball_type']:.4f}")
    return running_loss / len(dataloader)


def train(model, dataloader, optimizer, device, criterion):
    for epoch in range(30):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, device, criterion)
        print(f"        Epoch {epoch+1}, batch avg loss={avg_loss:.4f}\n\n")


def main():
    # Set up training model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BallDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True)
    model = BillardDetector(pretrained_backbone=True).to(device)
    criterion = YOLOLikeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Begin training model then save model
    train(model, dataloader, optimizer, device, criterion)
    torch.save(model, "billard_detect_model.pth")

    # Evaluate model quickly after training
    model.eval()
    for i, (img_tensors, target, filename) in enumerate(dataloader):
        with torch.no_grad():
            TotalBalls = 0
            img_tensors = img_tensors.to(device)
            outputs = model(img_tensors)
            for N in range(dataloader.batch_size):  # For loop in range of Batch Size
                print(TotalBalls)  # Print total # of balls
                TotalBalls = 0
                print('\n')
                # Print image name being evaluated
                print(f'EVAL IMAGE: {filename[N]}')
                for g_row in range(16):  # For loop through 16x16 Grid
                    for g_col in range(16):
                        # Output confidence score of detected ball
                        out_conf = torch.sigmoid(outputs[N, g_row, g_col, 0])
                        if out_conf > 0.9:  # SUPER CONFIDENT
                            TotalBalls += 1
                            print('     FOUND BALL WITH 0.9 CONF')
                            continue
                        if out_conf > 0.5:  # DECENTLY CONFIDENT
                            TotalBalls += 1
                            print('     FOUND BALL WITH 0.5 CONF')
                            continue


if __name__ == "__main__":
    main()
