import os
from ultralytics import YOLO
import torch

# === 基础配置 ===
BASE_DIR = os.getcwd()
DATA_YAML = os.path.join(BASE_DIR, 'seadrones_local.yaml')

# === 实验列表 ===
experiments = [
    {
        'name': '8_SeaHunter_Final',
        # 请确保这个 yaml 文件已经包含了 P2 + SPD + MultiSEAM + EMA
        'model': 'yolo_source/ultralytics/cfg/models/v8/yolov8s_p2_seahunter.yaml',
        'pretrained': True, 
        'info': 'YOLOv8s + P2 + SPD(User) + SEAM + EMA + NWD Loss'
    }
]

def run_experiments():
    # 1. 检查数据配置文件
    if not os.path.exists(DATA_YAML):
        print(f"❌ 错误：找不到配置文件 {DATA_YAML}")
        return

    # 2. 检查硬件
    print(f"🔍 PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🚀 显卡就位: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 未检测到 GPU！")

    # 3. 开始训练循环
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"🚀 启动终极实验: {exp['name']}")
        print(f"ℹ️  配置: {exp['info']}")
        print(f"{'='*60}\n")
        
        try:
            # 加载模型结构
            model = YOLO(exp['model'])
            
            # 加载预训练权重
            try:
                print("   -> 正在加载 yolov8s.pt 预训练权重...")
                model.load('yolov8s.pt')
            except Exception as e:
                print(f"   -> ⚠️ 权重加载提示 (正常): {e}")

            # 开始训练
            model.train(
                data=DATA_YAML,
                epochs=150,       # 训练 150 轮
                imgsz=1024,       # 🔥 核心：大图才能看清小目标
                
                # === 显存控制 (RTX 4060 专用) ===
                batch=4,          # 显存紧张改 2
                workers=2,        # 减少数据加载线程
                # ============================
                
                # === 优化器策略 ===
                optimizer='SGD',  # 小 Batch 下 SGD 更稳
                lr0=0.01,         # 初始学习率
                lrf=0.01,         # 最终学习率
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                
                # === 数据增强 (已修正参数) ===
                mosaic=1.0,       # ✅ 修正：直接设置概率 1.0 (开启) 或 0.5
                copy_paste=0.3,   # 复制粘贴增强
                mixup=0.1,        # 少量 Mixup
                degrees=0.0,      # 不旋转
                
                # === 杂项 ===
                project='runs/detect_seahunter', # 结果保存在这里
                name=exp['name'],
                patience=30,      # 早停轮数
                exist_ok=True,    # 允许覆盖
                amp=True          # 开启混合精度加速
            )
            print(f"✅ 实验 {exp['name']} 完成！")
            
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    run_experiments()