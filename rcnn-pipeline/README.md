# RCNN Pipeline (Original Logic Reproduction)

> This project reproduces the core inference logic of RCNN from Girshick et al., 2014 using PyTorch.

---

## ⚠️ Important Notice (Must Read)

**This repository is created solely for educational purposes, specifically for studying the pipeline of the paper:  
"Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation".**

- ❗ This is a **PyTorch-based implementation for pipeline understanding only**
- ❗ **No training has been performed**
- ❗ **The correctness and practical usability are NOT guaranteed**
- ❗ This code should NOT be used for production or benchmarking

---

## Project Structure

rcnn_pipeline/
├── 1_selective_search.py # Stage 1: Region Proposal
├── 2_feature_extraction.py # Stage 2: CNN Feature Extraction + Affine Warp
├── 3_svm_classification.py # Stage 3: SVM Classification (NOT Softmax!)
├── 4_bbox_regression.py # Stage 4: BBox Regression (Ridge Regression)
└── 5_nms_and_inference.py # Stage 5: NMS + Full Inference Pipeline


---

## Installation

```bash
pip install torch torchvision
pip install selectivesearch
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install joblib

```

Input Image
   │
   ▼
┌─────────────────────────────────────────┐
│ Stage 1: Selective Search               │
│  - Generate ~2000 region proposals      │
│  - Category-independent                │
│  - Runs on CPU, independent of CNN      │
└─────────────────────┬───────────────────┘
                      │ ~2000 boxes (x1,y1,x2,y2)
                      ▼
┌─────────────────────────────────────────┐
│ Stage 2: CNN Feature Extraction         │
│  - Affine Warp → 227×227 (with 16px padding) │
│  - Each proposal passes through AlexNet │
│    independently (~2000 forward passes!)│
│  - Extract fc7 features (4096-dim)      │
└─────────────────────┬───────────────────┘
                      │ Feature matrix (2000, 4096)
                      ▼
┌─────────────────────────────────────────┐
│ Stage 3: SVM Classification             │
│  - One LinearSVC per class              │
│  - IoU >= 0.3 → positive samples        │
│  - Output class confidence scores       │
└─────────────────────┬───────────────────┘
                      │ Class predictions + scores
                      ▼
┌─────────────────────────────────────────┐
│ Stage 4: BBox Regression                │
│  - One Ridge regressor per class        │
│  - Learn (dx, dy, dw, dh) transformations│
│  - Trained with IoU >= 0.6 samples      │
└─────────────────────┬───────────────────┘
                      │ Refined bounding boxes
                      ▼
┌─────────────────────────────────────────┐
│ Stage 5: Non-Maximum Suppression (NMS)  │
│  - Per-class NMS                        │
│  - IoU threshold = 0.3 (paper setting)  │
│  - Output final detections              │
└─────────────────────┬───────────────────┘
                      │
                      ▼
            Final Detection Results
        (Bounding boxes + class + score)


```markdown
## Reference

- Ross Girshick et al., 2014
- Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation