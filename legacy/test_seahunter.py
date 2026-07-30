#!/usr/bin/env python3
"""
SeaHunter Inference Script
Test the trained SeaHunter model on images or video
"""

import argparse
from pathlib import Path
import sys

# Add yolo_source to path
sys.path.insert(0, str(Path(__file__).parent / 'yolo_source'))

from ultralytics import YOLO
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='SeaHunter Inference')
    parser.add_argument('--weights', type=str, default='weights/seahunter_best.pt',
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source (image/video/directory)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Inference image size (default: 1024)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU threshold (default: 0.7)')
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    parser.add_argument('--save-dir', type=str, default='runs/detect/predict',
                        help='Directory to save results')
    parser.add_argument('--show', action='store_true',
                        help='Display results in window')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0/1/2...) or cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if weights exist
    if not Path(args.weights).exists():
        print(f"❌ Error: Model weights not found at {args.weights}")
        print(f"\n📥 Please download SeaHunter weights from:")
        print(f"   Google Drive: [LINK]")
        print(f"   百度网盘: [LINK]")
        print(f"\nThen place the file at: {args.weights}")
        return

    # Check if source exists
    if not Path(args.source).exists():
        print(f"❌ Error: Source not found at {args.source}")
        return

    print("="*60)
    print("🌊 SeaHunter Maritime Object Detection")
    print("="*60)
    print(f"📦 Model: {args.weights}")
    print(f"📂 Source: {args.source}")
    print(f"🖼️  Image Size: {args.imgsz}x{args.imgsz}")
    print(f"🎯 Confidence: {args.conf}")
    print(f"🔗 IoU Threshold: {args.iou}")
    print(f"💾 Save Results: {args.save}")
    print(f"🖥️  Device: {args.device}")
    print("="*60 + "\n")

    # Load model
    print("⏳ Loading SeaHunter model...")
    model = YOLO(args.weights)
    print("✅ Model loaded successfully!\n")

    # Run inference
    print("🚀 Running inference...\n")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
        show=args.show,
        device=args.device
    )

    # Print detection summary
    print("\n" + "="*60)
    print("📊 Detection Summary")
    print("="*60)

    total_detections = 0
    class_counts = {}

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            total_detections += len(boxes)

            # Count detections per class
            for cls_id in boxes.cls:
                cls_name = result.names[int(cls_id)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f"\n✅ Total Detections: {total_detections}")

    if class_counts:
        print("\nDetections by Class:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {cls_name}: {count}")
    else:
        print("\n⚠️  No objects detected")

    if args.save:
        print(f"\n💾 Results saved to: {args.save_dir}")

    print("\n" + "="*60)
    print("✅ Inference complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
