"""
Evaluation metrics for PPF-LIE.

Includes PSNR, SSIM, FSIM, and LPIPS metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import filters
from typing import Tuple


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 1.0  # Assuming normalized images
    psnr = 20 * torch.log10(torch.tensor(max_pixel) / torch.sqrt(mse))

    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM).

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        window_size: Size of the Gaussian window

    Returns:
        SSIM value between -1 and 1
    """
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # Convert to numpy for computation
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Handle batch dimension
    if img1_np.ndim == 4:
        img1_np = img1_np.mean(axis=1)  # Average across channels
        img2_np = img2_np.mean(axis=1)

    # Create Gaussian window
    sigma = 1.5
    gauss = np.array(
        [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)]
    )
    gauss = gauss / gauss.sum()
    window = np.outer(gauss, gauss)

    # Compute means
    mu1 = filters.convolve(img1_np, window, mode='reflect')
    mu2 = filters.convolve(img2_np, window, mode='reflect')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = filters.convolve(img1_np ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = filters.convolve(img2_np ** 2, window, mode='reflect') - mu2_sq
    sigma12 = filters.convolve(img1_np * img2_np, window, mode='reflect') - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def calculate_fsim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Feature Similarity Index (FSIM).

    Based on phase congruency and gradient magnitude.

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)

    Returns:
        FSIM value
    """
    # Convert to numpy
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Handle batch/channel dimensions
    if img1_np.ndim == 4:
        img1_np = img1_np.mean(axis=1)
    if img2_np.ndim == 4:
        img2_np = img2_np.mean(axis=1)

    # Compute gradient magnitude using Sobel filters
    grad_x1 = filters.sobel(img1_np, axis=1)
    grad_y1 = filters.sobel(img1_np, axis=0)
    grad1 = np.sqrt(grad_x1 ** 2 + grad_y1 ** 2)

    grad_x2 = filters.sobel(img2_np, axis=1)
    grad_y2 = filters.sobel(img2_np, axis=0)
    grad2 = np.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # Compute gradient magnitude similarity
    T = 0.85  # Constant to avoid division by zero
    GMS = (2 * grad1 * grad2 + T) / (grad1 ** 2 + grad2 ** 2 + T)

    # Phase congruency (simplified using edge density)
    PC1 = (grad1 - grad1.min()) / (grad1.max() - grad1.min() + 1e-8)
    PC2 = (grad2 - grad2.min()) / (grad2.max() - grad2.min() + 1e-8)

    # Feature similarity
    alpha = 1.0
    beta = 1.0
    S_L = (2 * PC1 * PC2 + T) / (PC1 ** 2 + PC2 ** 2 + T)
    S_G = GMS ** alpha
    S_PC = S_L ** beta

    FSIM = S_G * S_PC

    return float(np.mean(FSIM))


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, net: str = 'vgg') -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).

    Requires lpips package: pip install lpips

    Args:
        img1: First image tensor (B, C, H, W), should be in [-1, 1]
        img2: Second image tensor (B, C, H, W), should be in [-1, 1]
        net: Feature network to use ('vgg', 'alex', 'squeeze')

    Returns:
        LPIPS distance
    """
    try:
        import lpips
        lpips_model = lpips.LPIPS(net=net)
        device = img1.device

        # Convert from [0, 1] to [-1, 1]
        img1_lpips = img1 * 2 - 1
        img2_lpips = img2 * 2 - 1

        # Move to same device as model
        img1_lpips = img1_lpips.to(device)
        img2_lpips = img2_lpips.to(device)

        # Compute LPIPS distance
        with torch.no_grad():
            distance = lpips_model(img1_lpips, img2_lpips)

        return distance.mean().item()
    except ImportError:
        print("Warning: lpips not installed. Using MSE as fallback.")
        return torch.mean((img1 - img2) ** 2).item()


class MetricCalculator:
    """
    Calculator for multiple evaluation metrics.
    """

    def __init__(self, include_lpips: bool = True):
        self.include_lpips = include_lpips
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.fsim_values = []
        self.lpips_values = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with batch of images."""
        self.psnr_values.append(calculate_psnr(pred, target))
        self.ssim_values.append(calculate_ssim(pred, target))
        self.fsim_values.append(calculate_fsim(pred, target))

        if self.include_lpips:
            self.lpips_values.append(calculate_lpips(pred, target))

    def compute(self) -> dict:
        """Compute average metrics."""
        metrics = {
            'psnr': np.mean(self.psnr_values),
            'ssim': np.mean(self.ssim_values),
            'fsim': np.mean(self.fsim_values),
        }

        if self.include_lpips:
            metrics['lpips'] = np.mean(self.lpips_values)

        return metrics
