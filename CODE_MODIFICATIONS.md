# 🔧 Code Modifications Guide

This document provides **detailed instructions** on how to modify the Ultralytics YOLOv8 source code to implement SeaHunter's custom modules.

---

## ⚠️ Why Modifications Are Needed

SeaHunter introduces three custom architectural components that are **not available** in the standard Ultralytics YOLOv8 framework:

1. **SPDConv** (Space-to-Depth Convolution) - Lossless downsampling
2. **EMA** (Efficient Multi-Scale Attention) - Channel attention with GroupNorm
3. **MultiSEAM** (Multi-Scale Separation and Enhancement Attention Module) - Context aggregation

These modules must be manually added to the YOLOv8 source code before training or inference.

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Cloned the SeaHunter repository
- ✅ Installed PyTorch 2.0.1+ with CUDA 11.8
- ✅ Verified that `yolo_source/` directory exists in the project root

---

## 🛠️ Modification Steps

### Step 1: Locate the Modified Files

The required modifications have **already been applied** in the included `yolo_source/` directory. You can use this directly **without** installing `ultralytics` from PyPI.

**Modified Files:**
```
yolo_source/ultralytics/nn/modules/block.py         (Lines 1961-2081)
yolo_source/ultralytics/nn/modules/__init__.py      (Import statements)
yolo_source/ultralytics/nn/tasks.py                 (Module parsing logic)
```

---

### Step 2: Verify Custom Modules in `block.py`

Open `yolo_source/ultralytics/nn/modules/block.py` and **scroll to line 1961**. You should see:

```python
# ===================================================================
#      SeaHunter Custom Modules
#      1. SPDConv: 无损下采样 (保留微小特征)
#      2. MultiSEAM: 复杂背景分离 (Context核心)
#      3. EMA: 强力通道注意力 (GroupNorm版)
# ===================================================================

# === 1. SPD-Conv (User's Exact Implementation) ===

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # Space-to-Depth 会将通道数变成 4 倍，所以输入卷积的通道是 c1 * 4
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 核心逻辑：切片拼接 (Space-to-Depth)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))
```

**SPDConv Explanation:**
- **Purpose:** Replace standard strided convolution (2×2 downsampling) with lossless space-to-depth transformation
- **Mechanism:** Splits input into 4 sub-patches (top-left, top-right, bottom-left, bottom-right) and concatenates along channel dimension
- **Benefit:** Preserves fine-grained features critical for detecting tiny objects (swimmers as small as 10×10 pixels)

---

### Step 3: Verify EMA Module (Line 1998)

```python
# === 2. EMA (Attention - GroupNorm Optimized) ===
# 针对小目标和 BatchSize=4 的情况，GroupNorm 优于 BN
class EMA(nn.Module):
    def __init__(self, c1, c2, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert c1 // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 优化点：使用 GroupNorm
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)

        self.conv1x1 = nn.Conv2d(c1 // self.groups, c1 // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c1 // self.groups, c1 // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
```

**EMA Explanation:**
- **Purpose:** Enhance cross-channel feature interactions with spatial attention
- **Key Innovation:** Uses GroupNorm instead of BatchNorm (more stable for small batch sizes like 4)
- **Mechanism:** Combines horizontal and vertical pooling with grouped channel attention

---

### Step 4: Verify MultiSEAM Module (Line 2060)

```python
# === 3. MultiSEAM (Context - 解决海浪干扰) ===
class SEAMResidual(nn.Module):
    def __init__(self, fn):
        super(SEAMResidual, self).__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class SeamDcovN(nn.Module):
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.SiLU(),
            nn.BatchNorm2d(c2),
            *[nn.Sequential(
                SEAMResidual(nn.Sequential(
                    nn.Conv2d(c2, c2, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=c2),
                    nn.SiLU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            ) for i in range(depth)]
        )
    def forward(self, x):
        return self.net(x)

class MultiSEAM(nn.Module):
    def __init__(self, c1, c2, depth=1, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
        super(MultiSEAM, self).__init__()
        if c1 != c2: c2 = c1
        self.DCovN0 = SeamDcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = SeamDcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = SeamDcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x); y1 = self.DCovN1(x); y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c); y1 = self.avg_pool(y1).view(b, c); y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.exp(y).expand_as(x)
```

**MultiSEAM Explanation:**
- **Purpose:** Aggregate multi-scale context to suppress wave-induced false positives
- **Mechanism:** Three parallel branches with different patch sizes (3×3, 5×5, 7×7) capture context at multiple scales
- **Output:** Channel-wise attention weights that emphasize true targets and suppress wave patterns

---

### Step 5: Register Modules in `__init__.py`

Open `yolo_source/ultralytics/nn/modules/__init__.py` and **verify** these lines exist:

```python
from .block import (
    # ... existing imports ...
    SPDConv,
    EMA,
    MultiSEAM,
)

__all__ = (
    # ... existing exports ...
    "SPDConv",
    "EMA",
    "MultiSEAM",
)
```

If missing, add them to the import and export lists.

---

### Step 6: Enable YAML Parsing in `tasks.py`

Open `yolo_source/ultralytics/nn/tasks.py` and find the `parse_model()` function (around line 300-400). Ensure the following modules are recognized:

```python
# Inside parse_model() function
if m in {
    Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
    DWConv, Focus, BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR,
    C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
    SPDConv, EMA, MultiSEAM,  # ⬅️ Add custom modules here
}:
    c1, c2 = ch[f], args[0]
    # ... existing logic ...
```

**Note:** The included `yolo_source/` directory already has this modification. You only need to verify it.

---

## ✅ Verification Checklist

After completing modifications, verify everything works:

### Test 1: Import Custom Modules
```bash
cd yolo_source
python -c "from ultralytics.nn.modules import SPDConv, EMA, MultiSEAM; print('✅ Modules imported successfully')"
```

### Test 2: Load Model Configuration
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo_source/ultralytics/cfg/models/v8/yolov8s_p2_seahunter.yaml'); print('✅ Model loaded successfully')"
```

### Test 3: Check Module Instantiation
```bash
python -c "
import torch
from ultralytics.nn.modules import SPDConv, EMA, MultiSEAM

x = torch.randn(1, 128, 64, 64)
spd = SPDConv(128, 256)
ema = EMA(256, 256)
seam = MultiSEAM(256, 256)

y1 = spd(x)
y2 = ema(y1)
y3 = seam(y2)
print(f'✅ SPDConv output: {y1.shape}')
print(f'✅ EMA output: {y2.shape}')
print(f'✅ MultiSEAM output: {y3.shape}')
"
```

Expected output:
```
✅ SPDConv output: torch.Size([1, 256, 32, 32])
✅ EMA output: torch.Size([1, 256, 32, 32])
✅ MultiSEAM output: torch.Size([1, 256, 32, 32])
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:** Use the local modified version:
```bash
cd yolo_source
pip install -e .
```

Or set Python path manually:
```bash
export PYTHONPATH="/path/to/SeaHunter-Maritime-Detection/yolo_source:$PYTHONPATH"
```

---

### Error: "Unknown module 'SPDConv' in YAML"

**Cause:** `tasks.py` hasn't been updated to recognize the custom module.

**Solution:** Edit `yolo_source/ultralytics/nn/tasks.py` and add `SPDConv, EMA, MultiSEAM` to the module recognition list (see Step 6).

---

### Error: "GroupNorm channels must be divisible by groups"

**Cause:** EMA module's `factor=32` doesn't divide evenly into input channels.

**Solution:** Adjust `factor` in the YAML config or ensure backbone output channels are multiples of 32 (e.g., 256, 512, 1024).

---

## 📚 Additional Resources

- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **Custom Module Tutorial:** https://docs.ultralytics.com/usage/cfg/#custom-modules
- **PyTorch nn.Module Guide:** https://pytorch.org/docs/stable/generated/torch.nn.Module.html

---

## 📝 Summary

To use SeaHunter, you must:

1. ✅ Use the included `yolo_source/` directory (already modified)
2. ✅ **DO NOT** install `ultralytics` from PyPI
3. ✅ Verify custom modules are present in `block.py` (lines 1961-2081)
4. ✅ Ensure modules are registered in `__init__.py` and `tasks.py`
5. ✅ Run verification tests to confirm everything works

**If you need to start from a fresh Ultralytics installation**, follow Steps 2-6 to manually add the three custom modules. Otherwise, the provided `yolo_source/` is ready to use out-of-the-box.
