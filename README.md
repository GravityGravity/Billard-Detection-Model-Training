# 🎱 Billiard Ball Detection & Classification  
*A YOLO-style PyTorch model for detecting solids vs. stripes*

![Billiard Banner](banner_image_here.png)  
*Replace this with your banner image (recommended size ~1200×250)*

---

##  Overview

This project implements a YOLO-style billiard ball detector using a **ResNet-50 backbone**, **512×512 images**, and a **16×16 prediction grid**.  
It includes scripts for training the model, generating augmentations, merging annotations, and running single-image inference.

---

##  Features

- YOLO-style detection head (objectness, bbox, class logits)  
- Pretrained **ResNet-50** backbone  
- 512×512 input resolution  
- 16×16 grid-based ball localization  
- Solid vs. striped classification  
- Automatic dataset augmentation  
- Annotation conversion to CSV  
- Single-image inference + result visualization

---
## Reflection

This was my first full object-detection training pipeline, and it was a huge learning experience.
- My first attempt (DIY YOLO) failed — too complex and unstable.
- My early ResNet18 model (224×224, 7×7 grid) lost too much detail for small balls.
- Switching to ResNet50 + 512×512 + 16×16 grid significantly improved results.
- With only 40 training images, I expanded to ~400 using augmentation.
- Current accuracy: ~30% detection, with a high false-positive rate.
- Future improvements:
- Gather more real images (most important).
- Tune YOLO loss weights and learning rate.
- Add NMS to reduce false alarms.
- Try pretrained YOLO models (YOLOv5n, YOLOv8n) for comparison.

TEST OUTPUT
----------------------------------------
Detection rate: 10.87% - 15.5 points
False alarm rate: 71.15% - 41.2 points
Classification rate: 100.00% - 100.0 points

## 🗂 Project Structure

| File | Purpose |
|------|---------|
| `billard_train.py` | Trains the YOLO-style detector |
| `image_augment.py` | Creates augmented images + updated labels |
| `Annotation_Converge.py` | Combines all `.txt` annotations into one CSV |
| `Project3.py` | Runs inference and draws detected balls |
| `annotations.csv` | Final merged annotations |

---

##  Before & After

### **Before**  
![Before](before_image_here.png)

### **After**  
![After](after_image_here.png)


## Summary

### 1. Annotation Merge  
`Annotation_Converge.py` converts all `.txt` label files into a unified CSV.

### 2. Data Augmentation  
`image_augment.py` creates extra data via:
- 180° rotation  
- 90° rotation  
- 2× zoom  
- 4× zoom  
- RGB swap  
- Grayscale  

Each augmentation includes recalculated bounding-circle coordinates.

### 3. Training (`billard_train.py`)
- Backbone: ResNet-50 (pretrained ImageNet)  
- Output: YOLO-style 7 values per grid cell  
- Image Size: 512×512  
- Grid: 16×16  
- Optimizer: Adam  
- Epochs: 100  
- Custom YOLO-like loss function  

### 4. Inference 
Loads the trained model, predicts ball centers/radii/classes, and draws them on the image.

---


## 📦 Requirements

Install all dependencies via:

opencv-python
numpy
torch
torchvision
pandas
Pillow
colorama
