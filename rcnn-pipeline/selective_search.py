import cv2
import numpy as np
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def run_selective_search(image_path: str, mode: str = "fast") -> tuple:

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, regions = selectivesearch.selective_search(
        image_rgb,
        scale=500,
        sigma=0.9,
        min_size=10
    )

    return image_rgb, regions


def filter_proposals(regions: list, min_size: int = 20, max_proposals: int = 2000) -> list:

    candidates = set()

    for region in regions:
        x, y, w, h = region['rect']
        if w < min_size or h < min_size:
            continue

        box = (x, y, x + w, y + h)
        candidates.add(box)

    filtered = list(candidates)[:max_proposals]
    return filtered


def visualize_proposals(image: np.ndarray, proposals: list, max_show: int = 100):

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for (x1, y1, x2, y2) in proposals[:max_show]:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='red', facecolor='none', alpha=0.5
        )
        ax.add_patch(rect)

    ax.set_title(f"Selective Search Proposals (showing {min(max_show, len(proposals))}/{len(proposals)})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output_proposals.png", dpi=150)
    plt.show()
    print(f"[Stage 1] : {len(proposals)}")

if __name__ == "__main__":
    IMAGE_PATH = "test_image.jpg"  

    print("=" * 50)
    print("Stage 1: Selective Search Region Proposal")
    print("=" * 50)

    image, regions = run_selective_search(IMAGE_PATH, mode="fast")
    print(f"Selective Search : {len(regions)} ")

    proposals = filter_proposals(regions, min_size=20, max_proposals=2000)
    print(f": {len(proposals)} ")
    visualize_proposals(image, proposals, max_show=50)
