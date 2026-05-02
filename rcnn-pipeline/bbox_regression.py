import numpy as np
from sklearn.linear_model import Ridge
from typing import List, Tuple
import joblib
import os


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def compute_regression_targets(
    proposals: np.ndarray,
    gt_boxes: np.ndarray
) -> np.ndarray:
    Px, Py, Pw, Ph = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    Gx, Gy, Gw, Gh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    dx = (Gx - Px) / Pw
    dy = (Gy - Py) / Ph
    dw = np.log(Gw / (Pw + 1e-6))
    dh = np.log(Gh / (Ph + 1e-6))
    return np.stack([dx, dy, dw, dh], axis=1)

def apply_regression(
    proposals: np.ndarray,
    deltas: np.ndarray
) -> np.ndarray:
    Px, Py, Pw, Ph = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    Gx = dx * Pw + Px
    Gy = dy * Ph + Py
    Gw = Pw * np.exp(dw)
    Gh = Ph * np.exp(dh)
    x1 = Gx - Gw / 2.0
    y1 = Gy - Gh / 2.0
    x2 = Gx + Gw / 2.0
    y2 = Gy + Gh / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


class RCNNBBoxRegressor:


    def __init__(self, num_classes: int, alpha: float = 1000.0):
      
        self.num_classes = num_classes
        self.alpha = alpha
        self.regressors = {}

    def train(
        self,
        features: np.ndarray,
        proposals: np.ndarray,
        gt_boxes: np.ndarray,
        labels: np.ndarray,
        iou_threshold: float = 0.6
    ):
        from svm_classification import compute_iou

        proposals_xywh = xyxy_to_xywh(proposals)
        gt_xywh = xyxy_to_xywh(gt_boxes)

        for cls_id in range(self.num_classes):
            cls_mask = labels == cls_id

            valid_ious = np.array([
                compute_iou(tuple(proposals[i]), tuple(gt_boxes[i]))
                for i in range(len(proposals))
                if cls_mask[i]
            ])

            cls_indices = np.where(cls_mask)[0]
            high_iou_mask = valid_ious >= iou_threshold
            train_indices = cls_indices[high_iou_mask]

            if len(train_indices) == 0:
                print(f"  [class_{cls_id}] ")
                continue

            X = features[train_indices]                       
            G = gt_xywh[train_indices]                          
            targets = compute_regression_targets(P, G)           

            regressor = Ridge(alpha=self.alpha)
            regressor.fit(X, targets)
            self.regressors[cls_id] = regressor

            print(f"  [class_{cls_id}] BBox samples={len(train_indices)}")

    def predict(
        self,
        features: np.ndarray,
        proposals: np.ndarray,
        pred_classes: np.ndarray
    ) -> np.ndarray:

        proposals_xywh = xyxy_to_xywh(proposals)
        refined_boxes = proposals.copy().astype(np.float32)

        for cls_id, regressor in self.regressors.items():
            cls_mask = pred_classes == cls_id
            if not np.any(cls_mask):
                continue

            X = features[cls_mask]
            P = proposals_xywh[cls_mask]

            deltas = regressor.predict(X)                
            refined = apply_regression(P, deltas)       
            refined_boxes[cls_mask] = refined

        return refined_boxes

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for cls_id, reg in self.regressors.items():
            joblib.dump(reg, os.path.join(save_dir, f"bbox_reg_class_{cls_id}.pkl"))
        print(f"[Stage 4]  {save_dir}/")



if __name__ == "__main__":
    print("=" * 50)
    print("Stage 4: Bounding Box Regression")
    print("=" * 50)

    N = 200
    NUM_CLASSES = 20

    features = np.random.randn(N, 4096).astype(np.float32)
    proposals = np.random.randint(0, 400, size=(N, 4)).astype(np.float32)
    proposals[:, 2] = proposals[:, 0] + np.random.randint(20, 100, size=N)
    proposals[:, 3] = proposals[:, 1] + np.random.randint(20, 100, size=N)

    gt_boxes = proposals + np.random.randn(N, 4) * 10  
    labels = np.random.randint(0, NUM_CLASSES, size=N)
    pred_classes = np.random.randint(0, NUM_CLASSES, size=N)


    regressor = RCNNBBoxRegressor(num_classes=NUM_CLASSES, alpha=1000.0)
    regressor.train(features, proposals, gt_boxes, labels, iou_threshold=0.6)


    refined_boxes = regressor.predict(features, proposals, pred_classes)
    print(f"\n")
    for i in range(3):
        print(f"  originall: {proposals[i].astype(int)}")
        print(f"  fixed: {refined_boxes[i].astype(int)}")
        print()
