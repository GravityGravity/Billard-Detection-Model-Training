# Billiard Ball Finder (PyTorch • Solids vs Stripes)

This project identifies all billiard balls in a table image and classifies each as **solid** or **striped** using **PyTorch**. The entire process runs in **under 10 seconds per image**, even on a modest GPU or laptop CPU.

## Overview
The system combines fast image processing with deep learning to:
- **Detect** circular ball candidates on the table  
- **Classify** each detected ball as *solid* or *striped*  
- **Output** ball centers, radii, labels, and confidence scores

## How It Works
1. **Preprocessing** – Color normalization, masking the table area, and noise reduction to highlight ball edges.  
2. **Ball Localization** – Circle detection using Hough transforms or contour analysis, optionally combined with a lightweight PyTorch detector (e.g., MobileNet or YOLO-Nano).  
3. **PyTorch Classification** – Cropped ball patches are passed into a trained CNN (MobileNetV3-Small or EfficientNet-Lite) for solid/striped prediction.  
4. **Post-Processing** – Confidence checks and heuristic corrections (stripe area ratio, edge intensity) to refine results.

## Performance Goals
- Full pipeline executes in **<10 seconds per image**  
- Evaluation metrics: precision, recall, classification accuracy, and total runtime  
- Deterministic and reproducible evaluation with fixed seeds

## 📦 Dependencies (requirements.txt)
cv2,
numpy,
torch,
torchvision,
pandas,
Pillow
