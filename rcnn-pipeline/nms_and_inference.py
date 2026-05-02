import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import os


import sys
sys.path.append(os.path.dirname(__file__))
from selective_search import run_selective_search, filter_proposals
from feature_extraction import RCNNFeatureExtractor, extract_features
from svm_classification import RCNNClassifier, compute_iou
from bbox_regression import RCNNBBoxRegressor


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> List[int]:

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h

        iou = intersection / (areas[i] + areas[rest] - intersection + 1e-6)

        keep_mask = iou < iou_threshold
        order = rest[keep_mask]

    return keep


class RCNNDetector:

    def __init__(
        self,
        feature_extractor: RCNNFeatureExtractor,
        classifier: RCNNClassifier,
        bbox_regressor: RCNNBBoxRegressor,
        class_names: List[str],
        device: torch.device,
        score_threshold: float = 0.0,   
        nms_threshold: float = 0.3,    
        max_proposals: int = 2000       
    ):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bbox_regressor = bbox_regressor
        self.class_names = class_names
        self.device = device
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_proposals = max_proposals

    def detect(self, image_path: str) -> Dict:

        print(f"\n{'='*50}")
        print(f" {image_path}")
        print(f"{'='*50}")

        print("\n[Stage 1]  Selective Search")
        image, regions = run_selective_search(image_path)
        proposals = filter_proposals(regions, max_proposals=self.max_proposals)
        proposals_array = np.array(proposals, dtype=np.float32)
        print(f"  : {len(proposals)}")

        print("\n[Stage 2] CNN")
        features = extract_features(
            image, proposals, self.feature_extractor,
            self.device, batch_size=64
        )
        print(f" : {features.shape}")  


        print("\n[Stage 3] SVM ")
        scores = self.classifier.predict_scores(features)   
        pred_classes = np.argmax(scores, axis=1)           
        pred_scores = np.max(scores, axis=1)               
        print(f": {len(pred_classes)}")


        print("\n[Stage 4] BBox ")
        refined_boxes = self.bbox_regressor.predict(
            features, proposals_array, pred_classes
        )
        print(f"  fix complete")


        print("\n[Stage 5] NMS ")
        final_boxes, final_labels, final_scores = [], [], []

        for cls_id in range(len(self.class_names)):
            cls_mask = pred_classes == cls_id
            if not np.any(cls_mask):
                continue

            cls_boxes = refined_boxes[cls_mask]
            cls_scores = pred_scores[cls_mask]

            score_mask = cls_scores > self.score_threshold
            cls_boxes = cls_boxes[score_mask]
            cls_scores = cls_scores[score_mask]

            if len(cls_boxes) == 0:
                continue

            keep_indices = nms(cls_boxes, cls_scores, self.nms_threshold)

            final_boxes.extend(cls_boxes[keep_indices])
            final_labels.extend([cls_id] * len(keep_indices))
            final_scores.extend(cls_scores[keep_indices])

        if len(final_boxes) == 0:
            print("  no object")
            return {'boxes': [], 'labels': [], 'scores': [], 'class_names': []}

        final_boxes = np.array(final_boxes)
        final_labels = np.array(final_labels)
        final_scores = np.array(final_scores)

        print(f"  NMS: {len(final_boxes)} ")

        return {
            'boxes': final_boxes,
            'labels': final_labels,
            'scores': final_scores,
            'class_names': [self.class_names[l] for l in final_labels]
        }


COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]

def visualize_detections(image_path: str, results: Dict, save_path: str = "output_detection.png"):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image)

    for i, (box, label, score, name) in enumerate(zip(
        results['boxes'], results['labels'],
        results['scores'], results['class_names']
    )):
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[label % len(COLORS)]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"{name}: {score:.2f}"
        ax.text(
            x1, y1 - 5, label_text,
            color='white', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
        )

    ax.set_title(f"RCNN Detection Results ({len(results['boxes'])} objects detected)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n: {save_path}")


if __name__ == "__main__":

    PASCAL_VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" {device}")

    feature_extractor = RCNNFeatureExtractor(pretrained=True).to(device)

    classifier = RCNNClassifier(num_classes=20)

    bbox_regressor = RCNNBBoxRegressor(num_classes=20)


    detector = RCNNDetector(
        feature_extractor=feature_extractor,
        classifier=classifier,
        bbox_regressor=bbox_regressor,
        class_names=PASCAL_VOC_CLASSES,
        device=device,
        score_threshold=0.0,
        nms_threshold=0.3,
        max_proposals=2000
    )

    IMAGE_PATH = "test_image.jpg"  
    results = detector.detect(IMAGE_PATH)

    if len(results['boxes']) > 0:
        visualize_detections(IMAGE_PATH, results)
        for name, score, box in zip(results['class_names'], results['scores'], results['boxes']):
            print(f"  {name:15s} score={score:.3f}  box={box.astype(int)}")
