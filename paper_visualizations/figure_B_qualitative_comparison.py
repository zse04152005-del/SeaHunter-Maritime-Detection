import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def create_zoom_box(ax, img, bbox, zoom_factor=3, position='upper right'):
    x1, y1, x2, y2 = map(int, bbox)

    margin = 20
    x1_m = max(0, x1 - margin)
    y1_m = max(0, y1 - margin)
    x2_m = min(img.shape[1], x2 + margin)
    y2_m = min(img.shape[0], y2 + margin)

    zoomed_region = img[y1_m:y2_m, x1_m:x2_m]

    if zoomed_region.size == 0:
        return

    h, w = img.shape[:2]
    zh, zw = zoomed_region.shape[:2]

    zoom_w = int(zw * 1.5)
    zoom_h = int(zh * 1.5)

    positions = {
        'upper right': (w - zoom_w - 20, 20),
        'upper left': (20, 20),
        'lower right': (w - zoom_w - 20, h - zoom_h - 20),
        'lower left': (20, h - zoom_h - 20)
    }

    px, py = positions.get(position, positions['upper right'])

    rect_main = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=3, edgecolor='yellow',
                                   facecolor='none', linestyle='--')
    ax.add_patch(rect_main)

    zoomed_resized = cv2.resize(zoomed_region, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    imagebox = OffsetImage(zoomed_resized, zoom=1)
    ab = AnnotationBbox(imagebox, (px + zoom_w//2, py + zoom_h//2),
                        frameon=True, pad=0.5,
                        bboxprops=dict(edgecolor='yellow', linewidth=3, facecolor='white'))
    ax.add_artist(ab)

    connection = patches.ConnectionPatch((x1 + (x2-x1)/2, y1), (px + zoom_w//2, py + zoom_h),
                                          "data", "data", arrowstyle="->",
                                          shrinkA=0, shrinkB=0,
                                          mutation_scale=20, fc="yellow", linewidth=2)
    ax.add_artist(connection)

def visualize_with_zoom(baseline_img, seahunter_img, baseline_boxes, seahunter_boxes,
                        image_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(baseline_img)
    axes[0].set_title('YOLOv8 Baseline', fontsize=18, fontweight='bold', pad=20)
    axes[0].axis('off')

    axes[1].imshow(seahunter_img)
    axes[1].set_title('SeaHunter (Ours)', fontsize=18, fontweight='bold', pad=20)
    axes[1].axis('off')

    if len(seahunter_boxes) > 0:
        smallest_idx = 0
        min_area = float('inf')

        for idx, box in enumerate(seahunter_boxes):
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                min_area = area
                smallest_idx = idx

        zoom_box = seahunter_boxes[smallest_idx][:4]
        create_zoom_box(axes[1], seahunter_img, zoom_box, position='upper right')

        if len(baseline_boxes) > smallest_idx:
            create_zoom_box(axes[0], baseline_img, baseline_boxes[smallest_idx][:4], position='upper right')

    plt.suptitle(f'Qualitative Comparison: {image_name}', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved zoomed comparison to {output_path}")
    plt.close()

def main():
    base_dir = Path("D:/UAV_Project")

    baseline_model_path = base_dir / "runs/detect_ablation/1_Baseline/weights/best.pt"
    seahunter_model_path = base_dir / "runs/detect_seahunter/8_SeaHunter_Final/weights/best.pt"

    predict_images = list((base_dir / "yolo_source/runs/detect/predict").glob("*.jpg"))

    if not predict_images:
        val_dir = base_dir / "datasets/SeaDronesSee/images/val"
        if val_dir.exists():
            predict_images = list(val_dir.glob("*.jpg"))

    if not predict_images:
        print("❌ No test images found!")
        return

    selected_images = [
        predict_images[len(predict_images) // 5],
        predict_images[2 * len(predict_images) // 5],
        predict_images[3 * len(predict_images) // 5]
    ]

    print("Loading models...")
    baseline_model = YOLO(str(baseline_model_path))
    seahunter_model = YOLO(str(seahunter_model_path))

    output_dir = base_dir / "paper_visualizations"

    for idx, image_path in enumerate(selected_images, 1):
        print(f"\n[{idx}/3] Processing {image_path.name}...")

        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        baseline_results = baseline_model.predict(image_path, conf=0.25, verbose=False)
        seahunter_results = seahunter_model.predict(image_path, conf=0.25, verbose=False)

        baseline_plotted = baseline_results[0].plot()
        baseline_rgb = cv2.cvtColor(baseline_plotted, cv2.COLOR_BGR2RGB)

        seahunter_plotted = seahunter_results[0].plot()
        seahunter_rgb = cv2.cvtColor(seahunter_plotted, cv2.COLOR_BGR2RGB)

        baseline_boxes = baseline_results[0].boxes.xyxy.cpu().numpy() if len(baseline_results[0].boxes) > 0 else []
        seahunter_boxes = seahunter_results[0].boxes.xyxy.cpu().numpy() if len(seahunter_results[0].boxes) > 0 else []

        output_path = output_dir / f"figure_B_qualitative_comparison_{idx}.png"

        visualize_with_zoom(baseline_rgb, seahunter_rgb,
                            baseline_boxes, seahunter_boxes,
                            image_path.stem, output_path)

    print(f"\n✓ All qualitative comparisons saved to {output_dir}")

if __name__ == "__main__":
    main()
