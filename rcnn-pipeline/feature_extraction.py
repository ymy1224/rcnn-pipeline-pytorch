import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple

def affine_warp(image: np.ndarray, box: Tuple, output_size: int = 227, padding: int = 16) -> Image.Image:

    x1, y1, x2, y2 = box
    h_img, w_img = image.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w_img, x2 + padding)
    y2 = min(h_img, y2 + padding)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        cropped = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    warped = cv2.resize(cropped, (output_size, output_size))
    return Image.fromarray(warped)

class RCNNFeatureExtractor(nn.Module):


    def __init__(self, pretrained: bool = True):
        super(RCNNFeatureExtractor, self).__init__()

        alexnet = models.alexnet(pretrained=pretrained)

        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.fc6 = alexnet.classifier[0:3]  
        self.fc7 = alexnet.classifier[3:6] 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)      
        x = self.avgpool(x)      
        x = torch.flatten(x, 1)  
        x = self.fc6(x)          
        x = self.fc7(x)           
        return x


def get_transform() -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]    
        )
    ])


def extract_features(
    image: np.ndarray,
    proposals: List[Tuple],
    model: RCNNFeatureExtractor,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:

    model.eval()
    transform = get_transform()
    all_features = []

    print(f"[Stage 2]  {len(proposals)} ")

    with torch.no_grad():
        for i in range(0, len(proposals), batch_size):
            batch_boxes = proposals[i:i + batch_size]
            batch_tensors = []

            for box in batch_boxes:
                warped = affine_warp(image, box, output_size=227, padding=16)
                tensor = transform(warped)
                batch_tensors.append(tensor)

            batch_input = torch.stack(batch_tensors).to(device)  
            features = model(batch_input)                         
            all_features.append(features.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"  : {min(i+batch_size, len(proposals))}/{len(proposals)}")

    all_features = np.vstack(all_features)  
    print(f"[Stage 2] shape: {all_features.shape}")
    return all_features



if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from selective_search import run_selective_search, filter_proposals

    IMAGE_PATH = "test_image.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f": {device}")

    print("=" * 50)
    print("Stage 2: CNN Feature Extraction")
    print("=" * 50)


    image, regions = run_selective_search(IMAGE_PATH)
    proposals = filter_proposals(regions, max_proposals=2000)


    model = RCNNFeatureExtractor(pretrained=True).to(device)
    print(f"backbone: AlexNet (fc7)")


    features = extract_features(image, proposals, model, device, batch_size=32)
    print(f": {features.shape}") 

    np.save("region_features.npy", features)

