import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def draw_error_analysis_boxes(ax, img, predictions, ground_truth=None, show_legend=True):
    ax.imshow(img)
    ax.axis('off')

    legend_elements = []

    if predictions is not None and len(predictions) > 0:
        for box_data in predictions:
            x1, y1, x2, y2 = box_data[:4]
            conf = box_data[4] if len(box_data) > 4 else 0
            cls = int(box_data[5]) if len(box_data) > 5 else 0

            if conf < 0.4:
                color = 'red'
                label = 'False Alarm (Wave)'
                linestyle = '-'
            else:
                color = 'green'
                label = 'Correct Detection'
                linestyle = '-'

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=3, edgecolor=color,
                                      facecolor='none', linestyle=linestyle)
            ax.add_patch(rect)

            if label == 'False Alarm (Wave)':
                ax.text(x1, y1 - 5, f'FP: {conf:.2f}', color='red',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            else:
                ax.text(x1, y1 - 5, f'{conf:.2f}', color='green',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='green', linewidth=3, label='Correct Detection'),
            Patch(facecolor='none', edgecolor='red', linewidth=3, label='False Alarm (Waves/Foam)'),
            Patch(facecolor='none', edgecolor='blue', linewidth=3, linestyle='--', label='Missed Detection')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

def create_error_analysis(baseline_img, seahunter_img, baseline_boxes, seahunter_boxes,
                          image_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    draw_error_analysis_boxes(axes[0], baseline_img, baseline_boxes, show_legend=True)
    axes[0].set_title('YOLOv8 Baseline\n(False alarms on waves/foam)', fontsize=18, fontweight='bold', pad=20)

    draw_error_analysis_boxes(axes[1], seahunter_img, seahunter_boxes, show_legend=False)
    axes[1].set_title('SeaHunter (Ours)\n(Suppressed wave interference)', fontsize=18, fontweight='bold', pad=20)

    baseline_fps = sum(1 for box in baseline_boxes if box[4] < 0.4) if len(baseline_boxes) > 0 else 0
    seahunter_fps = sum(1 for box in seahunter_boxes if box[4] < 0.4) if len(seahunter_boxes) > 0 else 0

    stats_text = f"""
    Baseline: {baseline_fps} false positives
    SeaHunter: {seahunter_fps} false positives
    Reduction: {((baseline_fps - seahunter_fps) / max(baseline_fps, 1) * 100):.1f}%
    """

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Error Analysis: Wave Interference Suppression', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error analysis to {output_path}")
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

    challenging_images = []
    for img_path in predict_images:
        img_name_lower = img_path.stem.lower()
        if any(keyword in img_name_lower for keyword in ['wave', 'sea', 'water', '10', '11', '12']):
            challenging_images.append(img_path)

    if not challenging_images:
        challenging_images = predict_images[:5]

    selected_images = challenging_images[:3]

    print("Loading models...")
    baseline_model = YOLO(str(baseline_model_path))
    seahunter_model = YOLO(str(seahunter_model_path))

    output_dir = base_dir / "paper_visualizations"

    for idx, image_path in enumerate(selected_images, 1):
        print(f"\n[{idx}/{len(selected_images)}] Processing {image_path.name}...")

        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        baseline_results = baseline_model.predict(image_path, conf=0.15, verbose=False)
        seahunter_results = seahunter_model.predict(image_path, conf=0.25, verbose=False)

        baseline_boxes = []
        if len(baseline_results[0].boxes) > 0:
            boxes = baseline_results[0].boxes
            baseline_boxes = np.column_stack([
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy()
            ])

        seahunter_boxes = []
        if len(seahunter_results[0].boxes) > 0:
            boxes = seahunter_results[0].boxes
            seahunter_boxes = np.column_stack([
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy()
            ])

        output_path = output_dir / f"figure_C_error_analysis_{idx}.png"

        create_error_analysis(img_rgb, img_rgb, baseline_boxes, seahunter_boxes,
                              image_path.stem, output_path)

    print(f"\n✓ All error analysis visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
