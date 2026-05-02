import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Tuple, Dict


def compute_iou(box1: Tuple, box2: Tuple) -> float:

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def assign_labels(
    proposals: List[Tuple],
    gt_boxes: List[Tuple],
    gt_labels: List[int],
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:

    labels = np.full(len(proposals), -1, dtype=np.int32)   
    max_ious = np.zeros(len(proposals), dtype=np.float32)

    for i, proposal in enumerate(proposals):
        best_iou = 0.0
        best_label = -1

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            iou = compute_iou(proposal, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_label = gt_label

        max_ious[i] = best_iou

        if best_iou >= iou_threshold:
            labels[i] = best_label


    pos_count = np.sum(labels >= 0)
    neg_count = np.sum(labels == -1)
    print(f"[Stage 3] pos {pos_count} neg {neg_count} 个")

    return labels, max_ious



    def __init__(self, num_classes: int, C: float = 0.001):

        self.num_classes = num_classes
        self.C = C
        self.svms: Dict[int, LinearSVC] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None
    ):

        features_scaled = self.scaler.fit_transform(features)

        for cls_id in range(self.num_classes):
            cls_name = class_names[cls_id] if class_names else f"class_{cls_id}"

            binary_labels = np.where(labels == cls_id, 1, -1)

            pos_count = np.sum(binary_labels == 1)
            neg_count = np.sum(binary_labels == -1)

            if pos_count == 0:
                print(f"  [{cls_name}] ")
                continue

            print(f"  [{cls_name}] SVM: pos={pos_count}, neg={neg_count}")

            svm = LinearSVC(C=self.C, max_iter=10000, dual=True)
            svm.fit(features_scaled, binary_labels)
            self.svms[cls_id] = svm

        self.is_fitted = True
        print(f"[Stage 3] SVM {len(self.svms)} ")

    def predict_scores(self, features: np.ndarray) -> np.ndarray:

        assert self.is_fitted, " train() SVM"
        features_scaled = self.scaler.transform(features)

        scores = np.full((len(features), self.num_classes), -np.inf)

        for cls_id, svm in self.svms.items():
            scores[:, cls_id] = svm.decision_function(features_scaled)

        return scores

    def predict_classes(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        scores = self.predict_scores(features)
        pred_classes = np.argmax(scores, axis=1)
        pred_scores = np.max(scores, axis=1)
        return pred_classes, pred_scores

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for cls_id, svm in self.svms.items():
            joblib.dump(svm, os.path.join(save_dir, f"svm_class_{cls_id}.pkl"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        print(f"[Stage 3]  {save_dir}/")

    def load(self, save_dir: str, class_ids: List[int]):

        self.scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
        for cls_id in class_ids:
            path = os.path.join(save_dir, f"svm_class_{cls_id}.pkl")
            if os.path.exists(path):
                self.svms[cls_id] = joblib.load(path)
        self.is_fitted = True
        print(f"[Stage 3]  {len(self.svms)} ")


if __name__ == "__main__":
    print("=" * 50)
    print("Stage 3: SVM Classification")
    print("=" * 50)

    # PASCAL VOC 20类
    PASCAL_VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    NUM_CLASSES = len(PASCAL_VOC_CLASSES)
    N_PROPOSALS = 500

    features = np.random.randn(N_PROPOSALS, 4096).astype(np.float32)
    labels = np.random.randint(-1, NUM_CLASSES, size=N_PROPOSALS)
    print(f"inp_feat: {features.shape}")
    print(f"pos: {np.sum(labels >= 0)}，neg: {np.sum(labels == -1)}")

    classifier = RCNNClassifier(num_classes=NUM_CLASSES, C=0.001)
    classifier.train(features, labels, class_names=PASCAL_VOC_CLASSES)


    scores = classifier.predict_scores(features)
    print(f"\n: {scores.shape}")  

    pred_classes, pred_scores = classifier.predict_classes(features)

    for i in range(5):
        print(f" {PASCAL_VOC_CLASSES[pred_classes[i]]}, scores:={pred_scores[i]:.3f}")

    classifier.save("svm_models/")
