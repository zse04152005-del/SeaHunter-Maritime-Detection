import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def plot_training_curves_comparison(baseline_csv, seahunter_csv, output_path):
    baseline_df = pd.read_csv(baseline_csv)
    seahunter_df = pd.read_csv(seahunter_csv)

    baseline_df.columns = baseline_df.columns.str.strip()
    seahunter_df.columns = seahunter_df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ('metrics/mAP50(B)', 'mAP@0.5'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall')
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        if metric in baseline_df.columns:
            epochs_baseline = baseline_df['epoch'].values
            values_baseline = baseline_df[metric].values
            ax.plot(epochs_baseline, values_baseline, label='YOLOv8 Baseline',
                    linewidth=2.5, color='#FF6B6B', linestyle='--', marker='o', markersize=3)

        if metric in seahunter_df.columns:
            epochs_seahunter = seahunter_df['epoch'].values
            values_seahunter = seahunter_df[metric].values
            ax.plot(epochs_seahunter, values_seahunter, label='SeaHunter (Ours)',
                    linewidth=2.5, color='#4ECDC4', linestyle='-', marker='s', markersize=3)

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel(title, fontsize=13, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        if metric in seahunter_df.columns:
            best_value = seahunter_df[metric].max()
            ax.axhline(y=best_value, color='green', linestyle=':', linewidth=1.5, alpha=0.6,
                       label=f'Best: {best_value:.3f}')

    plt.suptitle('Training Metrics Comparison: Baseline vs SeaHunter', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves comparison to {output_path}")
    plt.close()

def plot_confusion_matrix_comparison(baseline_cm_path, seahunter_cm_path, output_path):
    baseline_cm = plt.imread(str(baseline_cm_path))
    seahunter_cm = plt.imread(str(seahunter_cm_path))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(baseline_cm)
    axes[0].set_title('YOLOv8 Baseline\nConfusion Matrix', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')

    axes[1].imshow(seahunter_cm)
    axes[1].set_title('SeaHunter (Ours)\nConfusion Matrix', fontsize=16, fontweight='bold', pad=15)
    axes[1].axis('off')

    plt.suptitle('Confusion Matrix Comparison', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix comparison to {output_path}")
    plt.close()

def plot_loss_curves(baseline_csv, seahunter_csv, output_path):
    baseline_df = pd.read_csv(baseline_csv)
    seahunter_df = pd.read_csv(seahunter_csv)

    baseline_df.columns = baseline_df.columns.str.strip()
    seahunter_df.columns = seahunter_df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    loss_types = [
        ('val/box_loss', 'Box Loss'),
        ('val/cls_loss', 'Classification Loss'),
        ('val/dfl_loss', 'DFL Loss')
    ]

    for idx, (loss_col, title) in enumerate(loss_types):
        ax = axes[idx]

        if loss_col in baseline_df.columns:
            epochs_baseline = baseline_df['epoch'].values
            loss_baseline = baseline_df[loss_col].values
            ax.plot(epochs_baseline, loss_baseline, label='Baseline',
                    linewidth=2.5, color='#FF6B6B', linestyle='--', alpha=0.8)

        if loss_col in seahunter_df.columns:
            epochs_seahunter = seahunter_df['epoch'].values
            loss_seahunter = seahunter_df[loss_col].values
            ax.plot(epochs_seahunter, loss_seahunter, label='SeaHunter',
                    linewidth=2.5, color='#4ECDC4', linestyle='-', alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    plt.suptitle('Validation Loss Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss curves to {output_path}")
    plt.close()

def main():
    base_dir = Path("D:/UAV_Project")

    baseline_results_csv = base_dir / "runs/detect_ablation/1_Baseline/results.csv"
    seahunter_results_csv = base_dir / "runs/detect_seahunter/8_SeaHunter_Final/results.csv"

    baseline_cm = base_dir / "runs/detect_ablation/1_Baseline/confusion_matrix_normalized.png"
    seahunter_cm = base_dir / "runs/detect_seahunter/8_SeaHunter_Final/confusion_matrix_normalized.png"

    output_dir = base_dir / "paper_visualizations"
    output_dir.mkdir(exist_ok=True)

    print("Generating training curves comparison...")
    plot_training_curves_comparison(
        baseline_results_csv,
        seahunter_results_csv,
        output_dir / "figure_D1_training_curves.png"
    )

    print("\nGenerating confusion matrix comparison...")
    if baseline_cm.exists() and seahunter_cm.exists():
        plot_confusion_matrix_comparison(
            baseline_cm,
            seahunter_cm,
            output_dir / "figure_D2_confusion_matrix.png"
        )
    else:
        print("⚠ Confusion matrices not found, skipping...")

    print("\nGenerating loss curves comparison...")
    plot_loss_curves(
        baseline_results_csv,
        seahunter_results_csv,
        output_dir / "figure_D3_loss_curves.png"
    )

    print(f"\n✓ All analysis plots saved to {output_dir}")

if __name__ == "__main__":
    main()
