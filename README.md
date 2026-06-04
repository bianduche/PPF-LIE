# PPF-LIE: Physical Photon Field Driven Low-Light Enhancement

<div align="center">

**PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition**

*Yi Zhang, Jian Song, Xiyang Liu, Miaosen Yang*

**School of Materials, Shanghai Dianji University, Shanghai, China**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Main Contributions](#main-contributions)
- [Method Overview](#method-overview)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## 📖 Project Overview

This project proposes **PPF-LIE**, a physical photon field prior-based extreme low-light image enhancement framework, specifically designed for industrial microscopic imaging scenarios.

### Research Background

In extreme low-light environments, traditional machine vision systems face severe challenges:
- Extremely low image visibility
- Poor contrast
- Severe statistical noise

Most existing Low-Light Image Enhancement (LLIE) methods are based on **continuous deterministic mapping** and **Gaussian noise assumptions**, which fundamentally contradict the discrete photon statistical characteristics in extremely dark imaging.

### Core Idea

The core innovation of PPF-LIE lies in:
> Abandoning continuous noise assumptions, starting from the physical nature of discrete photon statistics, and constructing a discrete diffusion process that perfectly aligns with photon imaging physics to recover continuous radiance probability fields from sparse photon observations.

---

## 🎯 Main Contributions

1. **PPF Modeling Framework**
   - First explicit modeling of discrete photon statistical characteristics in extreme low-light imaging
   - Modeling low-light images as Poisson-sampled sparse photon observations
   - Providing a new theoretical perspective for extreme low-light visual modeling

2. **Discrete Photon Space Diffusion Generative Model**
   - Designing a diffusion framework operating in discrete photon space
   - Directly predicting continuous expected photon rates through pure additive discrete Markov chains
   - Introducing time-weighted Poisson NLL as the optimization objective

3. **Systematic Downstream Vision Task Evaluation Framework**
   - Establishing a multi-level evaluation system: "Pixel Quality - Perceptual Quality - Downstream Task Performance"
   - Conducting systematic experiments on multiple public benchmark datasets

4. **Industrial Metallographic Dataset Construction**
   - Constructing the **ML Dataset**, the world's first paired benchmark for real industrial scenarios of electroplated metallographic cross-sections

---

## 🔧 Method Overview

### 3.1 Photon Distribution Prior

In extremely dark environments, the discrete photon count N(x) at pixel x strictly follows a **Poisson distribution** with the continuous radiance field λ(x) as its expectation:

$$N(x) \sim \text{Poisson}(\lambda(x)) = \frac{\lambda(x)^{N(x)} e^{-\lambda(x)}}{N(x)!}$$

### 3.2 Discrete Forward Diffusion Process

Unlike continuous Gaussian scaling, we model the forward diffusion process as a **discrete Markov chain**, gradually destroying photon states by injecting random Poisson scattering noise:

$$q(N_t|N_{t-1}) = \text{Poisson}(N_t - N_{t-1}; \beta_t)$$

### 3.3 Network Parameterization and Training Objective

The network is parameterized to directly predict the underlying continuous expected photon rate $\lambda_0 = f_\theta(N_t, t)$, using a time-weighted Poisson negative log-likelihood loss:

$$\mathcal{L}_{diffusion} = \mathbb{E}_{t,N_0,\epsilon}[\lambda_0(x) - N_0(x)\log\lambda_0(x) + \delta]$$

### 3.4 Radiance Field Reconstruction and Color Consistency

Introducing the radiance reconstruction and color consistency module (CCM), combining the recovered continuous field $\lambda_0$ with conditional features $\Phi_{I_{in}}$ extracted from the original low-light input:

$$I_{out}(x) = \mathcal{M}_{color}(\lambda_0(x), \Phi_{I_{in}}(x))$$

---

## 📊 Datasets

### ML Dataset (Metallographic Low-light)

The **world's first** paired benchmark dataset for real industrial scenarios of electroplated metallographic cross-sections.

| Attribute | Description |
|-----------|-------------|
| Acquisition Device | Leica DM4000 M LED Metallographic Microscope + Leica DFC450 5MP CCD |
| Resolution | 2560×1920 Full Resolution |
| Objective Magnification | 50× |
| Exposure Time | 1-5 ms (Extreme Low-light) |
| Sample Count | 1200 paired images |
| Defect Types | Micro-cracks, Porosities, Delamination |

### Dataset Download

| Dataset | Link | Password |
|---------|------|----------|
| **ML Dataset** | [Baidu Pan](https://pan.baidu.com/s/1cjWTkVeTMEt98erGA1_6ww) | `j9bj` |
| LOL | https://daooshee.github.io/ROG/ | - |
| SICE | https://github.com/csjcai/SICE | - |

### Public Datasets

This project also evaluates on the following public datasets:
- **LOL** - Real-world low/normal light paired dataset
- **SICE** - Multi-exposure sequence dataset

---

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the project
git clone https://github.com/your-repo/PPF-LIE.git
cd PPF-LIE

# Create conda environment
conda create -n ppflie python=3.8
conda activate ppflie

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Train on LOL dataset
python train.py --config configs/train_lol.yaml

# Train on ML dataset
python train.py --config configs/train_ml.yaml
```

### Testing the Model

```bash
# Evaluate model performance
python test.py --config configs/test.yaml --model_path experiments/best_model.pth

# Test on specific dataset
python test.py --dataset lol
python test.py --dataset ml
```

### Inference Example

```python
import torch
from models.ppf_lie import PPFLIE

# Load model
model = PPFLIE()
checkpoint = torch.load('experiments/best_model.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Process low-light image
low_light_image = torch.randn(1, 3, 256, 256)
enhanced_image = model(low_light_image)
```

---

## 📈 Experimental Results

### Quantitative Evaluation

#### LOL Dataset

| Method | PSNR↑ | SSIM↑ | FSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| Retinexformer | 25.32 | 0.91 | 0.97 | 0.06 |
| Restormer | 24.85 | 0.89 | 0.97 | 0.07 |
| DiffLLE | 24.60 | 0.87 | 0.96 | 0.08 |
| **PPF-LIE (Ours)** | **25.71** | **0.92** | **0.98** | **0.04** |

#### ML Dataset (Industrial Metallographic)

| Method | PSNR↑ | SSIM↑ | FSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| LLFormer | 8.06 | 0.46 | 0.86 | 0.42 |
| GSAD | 9.49 | 0.50 | 0.84 | 0.33 |
| D3PM | 18.20 | 0.76 | 0.88 | 0.22 |
| **PPF-LIE (Ours)** | **25.96** | **0.89** | **0.97** | **0.15** |

### Ablation Study

| Variant | LOL PSNR | SICE PSNR | ML PSNR |
|---------|----------|-----------|---------|
| w/o PPF | 23.41 | 22.15 | 21.84 |
| w/o Poisson NLL | 24.82 | 23.84 | 23.51 |
| w/o CCM | 25.10 | 24.22 | 24.68 |
| **Full Model** | **25.71** | **24.66** | **25.96** |

### Downstream Task Performance

Using YOLO26 for instance segmentation, downstream detection mAP@0.5 reaches **85.5%**.

---

## 📁 Project Structure

```
PPF-LIE/
├── configs/                 # Configuration files
│   ├── train_lol.yaml
│   ├── train_ml.yaml
│   └── test.yaml
├── datasets/               # Dataset processing
│   └── README.md
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── ppf_lie.py         # Main model
│   ├── diffusion.py       # Diffusion module
│   ├── ccm.py             # Color consistency module
│   └── networks.py        # Network architecture
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── losses.py          # Loss functions
│   ├── metrics.py         # Evaluation metrics
│   └── data_utils.py      # Data processing
├── experiments/           # Experimental results
│   └── README.md
├── results/              # Output results
├── docs/                 # Documentation
│   ├── paper_summary.md
│   ├── method_details.md
│   └── experiment_details.md
├── train.py              # Training script
├── test.py               # Testing script
├── requirements.txt      # Dependencies
├── README.md             # This file
└── LICENSE              # License
```

---

## 🔬 Method Details Documentation

For detailed method descriptions and experimental analysis, please refer to:

- [Paper Summary and Method Overview](docs/paper_summary.md)
- [Technical Details](docs/method_details.md)
- [Experimental Results and Analysis](docs/experiment_details.md)

---

## 📜 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{zhang2025ppflie,
  title={PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition},
  author={Zhang, Yi and Song, Jian and Liu, Xiyang and Yang, Miaosen},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📧 Contact Information

- **Corresponding Author**: Xiyang Liu
- **Email**: 19526000027@163.com
- **Institution**: School of Materials, Shanghai Dianji University, Shanghai, China

---

## 📄 License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this project helpful, please give us a star!**

</div>
