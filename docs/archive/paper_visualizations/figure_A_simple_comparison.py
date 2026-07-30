import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

def visualize_comparison(image_path, baseline_results, seahunter_results, output_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('(a) Original Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')

    baseline_img = baseline_results[0].plot()
    baseline_img_rgb = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(baseline_img_rgb)
    axes[0, 1].set_title('(b) YOLOv8 Baseline Detection', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')

    seahunter_img = seahunter_results[0].plot()
    seahunter_img_rgb = cv2.cvtColor(seahunter_img, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(seahunter_img_rgb)
    axes[1, 0].set_title('(c) SeaHunter (Ours) Detection', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')

    baseline_boxes = len(baseline_results[0].boxes)
    seahunter_boxes = len(seahunter_results[0].boxes)

    comparison_text = f"""Detection Comparison:

YOLOv8 Baseline: {baseline_boxes} objects
SeaHunter (Ours): {seahunter_boxes} objects

Key Improvements:
• MultiSEAM suppresses wave interference
• EMA focuses on small swimmers
• SPD-Conv preserves fine details
"""
    axes[1, 1].text(0.1, 0.5, comparison_text, fontsize=14,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison visualization to {output_path}")
    plt.close()

def main():
    base_dir = Path("D:/UAV_Project")

    baseline_model_path = base_dir / "runs/detect_ablation/1_Baseline/weights/best.pt"
    seahunter_model_path = base_dir / "runs/detect_seahunter/8_SeaHunter_Final/weights/best.pt"

    predict_images = list((base_dir / "yolo_source/runs/detect/predict").glob("*.jpg"))

    if not predict_images:
        val_images_dir = base_dir / "datasets/SeaDronesSee/images/val"
        if val_images_dir.exists():
            predict_images = list(val_images_dir.glob("*.jpg"))[:10]

    if not predict_images:
        print("❌ No test images found!")
        return

    selected_images = [
        predict_images[len(predict_images) // 4],
        predict_images[len(predict_images) // 2],
        predict_images[3 * len(predict_images) // 4]
    ]

    print(f"Loading models...")
    baseline_model = YOLO(str(baseline_model_path))
    seahunter_model = YOLO(str(seahunter_model_path))

    output_dir = base_dir / "paper_visualizations"
    output_dir.mkdir(exist_ok=True)

    for idx, image_path in enumerate(selected_images, 1):
        print(f"\n[{idx}/3] Processing {image_path.name}...")

        baseline_results = baseline_model.predict(image_path, conf=0.25, verbose=False)
        seahunter_results = seahunter_model.predict(image_path, conf=0.25, verbose=False)

        output_path = output_dir / f"figure_A_comparison_{idx}_{image_path.stem}.png"
        visualize_comparison(image_path, baseline_results, seahunter_results, output_path)

    print(f"\n✓ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
