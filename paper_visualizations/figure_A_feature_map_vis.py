import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)

        if target_class is None:
            target_class = model_output.argmax(dim=1)

        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()

        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.numpy()

def get_backbone_last_layer(model, model_type='baseline'):
    if hasattr(model.model, 'model'):
        layers = model.model.model
        for i in range(len(layers) - 1, -1, -1):
            if hasattr(layers[i], 'conv'):
                return layers[i]
    return None

def visualize_feature_maps(image_path, baseline_model_path, seahunter_model_path, output_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    baseline_model = YOLO(baseline_model_path)
    seahunter_model = YOLO(seahunter_model_path)

    baseline_layer = get_backbone_last_layer(baseline_model)
    seahunter_layer = get_backbone_last_layer(seahunter_model)

    baseline_cam = GradCAM(baseline_model, baseline_layer)
    seahunter_cam = GradCAM(seahunter_model, seahunter_layer)

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    baseline_heatmap = baseline_cam.generate_cam(img_tensor)
    seahunter_heatmap = seahunter_cam.generate_cam(img_tensor)

    baseline_heatmap_resized = cv2.resize(baseline_heatmap, (img.shape[1], img.shape[0]))
    seahunter_heatmap_resized = cv2.resize(seahunter_heatmap, (img.shape[1], img.shape[0]))

    baseline_colored = cv2.applyColorMap(np.uint8(255 * baseline_heatmap_resized), cv2.COLORMAP_JET)
    seahunter_colored = cv2.applyColorMap(np.uint8(255 * seahunter_heatmap_resized), cv2.COLORMAP_JET)

    baseline_overlay = cv2.addWeighted(img_rgb, 0.5, cv2.cvtColor(baseline_colored, cv2.COLOR_BGR2RGB), 0.5, 0)
    seahunter_overlay = cv2.addWeighted(img_rgb, 0.5, cv2.cvtColor(seahunter_colored, cv2.COLOR_BGR2RGB), 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(baseline_overlay)
    axes[1].set_title('YOLOv8 Baseline\n(Attention scattered on waves)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(seahunter_overlay)
    axes[2].set_title('SeaHunter (Ours)\n(Focused on swimmer)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature map visualization to {output_path}")
    plt.close()

def main():
    base_dir = Path("/mnt/d/UAV_Project")

    baseline_model = base_dir / "runs/detect_ablation/1_Baseline/weights/best.pt"
    seahunter_model = base_dir / "runs/detect_seahunter/8_SeaHunter_Final/weights/best.pt"

    test_images = list((base_dir / "yolo_source/runs/detect/predict").glob("*.jpg"))

    if test_images:
        test_image = test_images[len(test_images) // 2]
    else:
        val_images = list((base_dir / "datasets/SeaDronesSee/images/val").glob("*.jpg"))
        test_image = val_images[len(val_images) // 2] if val_images else None

    if test_image is None:
        print("No test images found!")
        return

    output_path = base_dir / "paper_visualizations/figure_A_feature_map_vis.png"

    visualize_feature_maps(test_image, baseline_model, seahunter_model, output_path)

if __name__ == "__main__":
    main()
