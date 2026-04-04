# 数据集说明

## 目录

1. [ML数据集](#ml数据集)
2. [公开数据集](#公开数据集)
3. [数据下载](#数据下载)
4. [数据格式](#数据格式)

---

## ML数据集

**Metallographic Low-light Dataset** - 世界首个电镀金相截面真实工业场景配对基准数据集。

### 采集设置

| 参数 | 值 |
|------|-----|
| 显微镜 | Leica DM4000 M LED 金相显微镜 |
| 相机 | Leica DFC450 5MP CCD |
| 分辨率 | 2560×1920 |
| 物镜 | 50× |
| 曝光时间 | 1-5 ms（极低光照） |
| 参考曝光 | 优化参数（正常光照） |

### 数据集规模

| 类别 | 数量 |
|------|------|
| 训练集 | 1000对 |
| 测试集 | 200对 |
| **总计** | **1200对** |

### 缺陷类型

1. **微裂纹 (Micro-cracks)**: 材料表面的微小裂缝
2. **孔隙 (Porosities)**: 材料内部的气孔或空洞
3. **分层 (Delamination)**: 层状结构之间的分离

### 采集协议

为确保训练监督信号的绝对可靠性，制定了严格的配对采集协议：

1. 所有硬件设置（载物台位置、光阑、对焦）在采集期间锁定
2. 首先收集正常曝光地面真值
3. 然后仅通过减少曝光时间至1-5ms来捕获对应的极端低光图像
4. 确保严格的像素级空间对齐和泊松散粒噪声主导的退化

### 数据可用性

ML数据集将在论文录用后在GitHub和Zenodo上公开，以促进该跨学科领域的可重复研究。

---

## 公开数据集

### LOL数据集

用于低光图像增强研究的真实世界配对数据集。

- 来源: https://daooshee.github.io/ROG/
- 包含485对低/正常光照图像
- 训练/测试分割已提供

### SICE数据集

多曝光序列数据集。

- 来源: https://github.com/csjcai/SICE
- 包含589个多曝光序列
- 涵盖不同光照条件

---

## 数据下载

### 自动下载脚本

```bash
# 下载LOL数据集
python scripts/download_lol.py

# 下载SICE数据集
python scripts/download_sice.py
```

### 手动下载

1. 访问数据集官方网站
2. 解压到 `datasets/` 目录
3. 按照以下结构组织：

```
datasets/
├── LOL/
│   └── LOL_v2/
│       ├── Synced_low/
│       ├── Synced_normal/
│       └── Test/
├── SICE/
│   └── SICE_Dataset/
└── ML/          # 待公开
    ├── train/
    ├── test/
    └── README.md
```

---

## 数据格式

### 图像格式

- 格式: PNG或JPEG
- 色彩空间: RGB
- 原始分辨率: 2560×1920（ML）、多种分辨率（LOL/SICE）

### 标注格式 (ML数据集)

标注使用JSON格式存储：

```json
{
  "image_id": "sample_001",
  "file_name": "sample_001.png",
  "defects": [
    {
      "category": "micro_cracks",
      "polygon": [[x1, y1], [x2, y2], ...],
      "area": 1234.5
    }
  ]
}
```

### 数据加载

使用项目提供的DataLoader：

```python
from datasets.lol_dataset import LOLDataset
from datasets.ml_dataset import MLDataset

# 加载LOL数据集
lol_dataset = LOLDataset(
    data_root='./datasets/LOL',
    split='train',
    img_size=256
)

# 加载ML数据集
ml_dataset = MLDataset(
    data_root='./datasets/ML',
    split='train',
    img_size=512,
    augment=True
)
```

---

## 预处理

### 标准化

所有输入图像进行以下预处理：

1. 归一化到 [0, 1]
2. 减去ImageNet均值：[0.485, 0.456, 0.406]
3. 除以ImageNet标准差：[0.229, 0.224, 0.225]

### 数据增强

训练时应用以下增强：

- 随机水平翻转
- 随机旋转 (±15°)
- 随机裁剪
- 颜色抖动（亮度、对比度、饱和度）

---

## 许可证

ML数据集将基于CC BY-NC-SA 4.0许可证发布。

---

## 引用

如果使用ML数据集，请引用我们的论文：

```bibtex
@article{zhang2025ppflie,
  title={PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition},
  author={Zhang, Yi and Song, Jian and Liu, Xiyang and Yang, Miaosen},
  journal={arXiv preprint},
  year={2025}
}
```
