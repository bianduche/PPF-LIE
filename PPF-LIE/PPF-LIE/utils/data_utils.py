"""
Data utilities for PPF-LIE.

Includes dataset classes, data augmentation, and dataloader creation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, List
import torchvision.transforms as T


class LOLDataset(Dataset):
    """
    LOL (Low-Light) Dataset for low-light image enhancement.

    Args:
        root_dir: Root directory containing LOL dataset
        split: 'train' or 'test'
        transform: Optional transforms to apply
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Set paths based on split
        if split == 'train':
            self.low_dir = os.path.join(root_dir, 'LOL_v2', 'Synced_low')
            self.high_dir = os.path.join(root_dir, 'LOL_v2', 'Synced_normal')
        else:
            self.low_dir = os.path.join(root_dir, 'LOL_v2', 'Test', 'Synced_low')
            self.high_dir = os.path.join(root_dir, 'LOL_v2', 'Test', 'Synced_normal')

        # Get image files
        self.low_files = sorted(os.listdir(self.low_dir))
        self.high_files = sorted(os.listdir(self.high_dir))

        assert len(self.low_files) == len(self.high_files), \
            "Number of low and high light images must match"

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get low-light and corresponding normal-light image pair."""
        # Load images
        low_path = os.path.join(self.low_dir, self.low_files[idx])
        high_path = os.path.join(self.high_dir, self.high_files[idx])

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # Resize
        low_img = low_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        high_img = high_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensors
        low_tensor = T.ToTensor()(low_img)
        high_tensor = T.ToTensor()(high_img)

        if self.transform:
            low_tensor, high_tensor = self.transform(low_tensor, high_tensor)

        return low_tensor, high_tensor


class MLDataset(Dataset):
    """
    ML (Metallographic Low-light) Dataset for industrial defect detection.

    Args:
        root_dir: Root directory containing ML dataset
        split: 'train' or 'test'
        img_size: Image size for training
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 512,
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # Set paths
        self.low_dir = os.path.join(root_dir, split, 'low')
        self.high_dir = os.path.join(root_dir, split, 'high')

        # Get image files
        if os.path.exists(self.low_dir):
            self.low_files = sorted(os.listdir(self.low_dir))
            self.high_files = sorted(os.listdir(self.high_dir))
        else:
            self.low_files = []
            self.high_files = []

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get low-light and corresponding normal-light image pair."""
        low_path = os.path.join(self.low_dir, self.low_files[idx])
        high_path = os.path.join(self.high_dir, self.high_files[idx])

        # Load images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # Resize
        low_img = low_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        high_img = high_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensors
        low_tensor = T.ToTensor()(low_img)
        high_tensor = T.ToTensor()(high_img)

        # Apply augmentation if enabled
        if self.augment:
            low_tensor, high_tensor = self._augment(low_tensor, high_tensor)

        return low_tensor, high_tensor

    def _augment(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation to image pair."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            low = torch.flip(low, dims=[2])
            high = torch.flip(high, dims=[2])

        # Random rotation (90 degrees)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            low = torch.rot90(low, k, dims=[1, 2])
            high = torch.rot90(high, k, dims=[1, 2])

        return low, high


class RandomCrop(nn.Module):
    """Random crop for paired images."""

    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = img1.shape[1:]
        th, tw = self.crop_size, self.crop_size

        if h == th and w == tw:
            return img1, img2

        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()

        img1_crop = img1[:, i:i+th, j:j+tw]
        img2_crop = img2[:, i:i+th, j:j+tw]

        return img1_crop, img2_crop


class RandomFlip(nn.Module):
    """Random horizontal flip for paired images."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) > self.p:
            img1 = torch.flip(img1, dims=[2])
            img2 = torch.flip(img2, dims=[2])
        return img1, img2


class ToTensor:
    """Convert PIL Image to Tensor."""

    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, Image.Image):
            return T.ToTensor()(img)
        return torch.from_numpy(img).float()


def create_dataloader(
    dataset_name: str,
    data_root: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 256,
    augment: bool = True,
) -> DataLoader:
    """
    Create dataloader for specified dataset.

    Args:
        dataset_name: 'lol' or 'ml'
        data_root: Root directory containing dataset
        split: 'train' or 'test'
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size
        augment: Whether to apply augmentation

    Returns:
        DataLoader instance
    """
    if dataset_name.lower() == 'lol':
        dataset = LOLDataset(
            root_dir=data_root,
            split=split,
            img_size=img_size,
        )
    elif dataset_name.lower() == 'ml':
        dataset = MLDataset(
            root_dir=data_root,
            split=split,
            img_size=img_size,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return dataloader
