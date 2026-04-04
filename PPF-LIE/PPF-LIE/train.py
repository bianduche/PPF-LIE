"""
Training script for PPF-LIE.

Usage:
    python train.py --config configs/train_lol.yaml
    python train.py --config configs/train_ml.yaml --resume path/to/checkpoint.pth
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models import PPFLIE
from utils.losses import CombinedLoss
from utils.metrics import MetricCalculator
from utils.data_utils import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPF-LIE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create PPF-LIE model."""
    model = PPFLIE(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        num_layers=config['model']['num_layers'],
        time_steps=config['model']['time_steps'],
        beta_start=config['model']['beta_start'],
        beta_end=config['model']['beta_end'],
        use_ccm=config['ccm']['enable'],
    )
    return model.to(device)


def create_optimizer(model, config):
    """Create optimizer."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
        betas=config['train']['optimizer']['betas'],
        eps=config['train']['optimizer']['eps'],
    )
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train']['epochs'],
        eta_min=config['train']['learning_rate'] * 0.01,
    )
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metric_calculator = MetricCalculator(include_lpips=False)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (low_images, high_images) in enumerate(pbar):
        low_images = low_images.to(device)
        high_images = high_images.to(device)

        # Forward pass
        enhanced_images = model(low_images)

        # Compute loss
        loss_dict = criterion(enhanced_images, high_images, return_dict=True)
        loss = loss_dict['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        metric_calculator.update(enhanced_images.detach(), high_images)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{metric_calculator.psnr_values[-1]:.2f}',
        })

    avg_loss = total_loss / len(dataloader)
    metrics = metric_calculator.compute()

    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    metric_calculator = MetricCalculator(include_lpips=False)

    with torch.no_grad():
        for low_images, high_images in tqdm(dataloader, desc='Validating'):
            low_images = low_images.to(device)
            high_images = high_images.to(device)

            # Forward pass
            enhanced_images = model(low_images)

            # Compute loss
            loss_dict = criterion(enhanced_images, high_images, return_dict=True)
            loss = loss_dict['total']

            total_loss += loss.item()
            metric_calculator.update(enhanced_images, high_images)

    avg_loss = total_loss / len(dataloader)
    metrics = metric_calculator.compute()

    return avg_loss, metrics


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(config.get('seed', 42))

    # Create directories
    checkpoint_dir = Path(config['train']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(config, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create loss function
    criterion = CombinedLoss(
        gamma_perceptual=config['train']['loss_weights']['perceptual'],
        gamma_color=config['train']['loss_weights']['color'],
        gamma_tv=config['train']['loss_weights']['tv'],
    )

    # Load data
    dataset_name = 'lol' if 'lol' in str(args.config).lower() else 'ml'
    train_loader = create_dataloader(
        dataset_name=dataset_name,
        data_root=config['train']['data_root'],
        split='train',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['train']['img_size'],
        augment=True,
    )

    val_loader = create_dataloader(
        dataset_name=dataset_name,
        data_root=config['train']['data_root'],
        split='test',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['train']['img_size'],
        augment=False,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['train']['epochs']):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train PSNR: {train_metrics['psnr']:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val PSNR: {val_metrics['psnr']:.2f}")
        print(f"Val SSIM: {val_metrics['ssim']:.4f} | Val FSIM: {val_metrics['fsim']:.4f}")

        # Save checkpoint
        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']

        if (epoch + 1) % config['train']['save_interval'] == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_psnr': best_psnr,
            }

            # Save latest
            torch.save(checkpoint, checkpoint_dir / 'latest_model.pth')

            # Save best
            if is_best:
                torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
                print(f"Saved best model with PSNR: {best_psnr:.2f}")

    print("Training completed!")


if __name__ == '__main__':
    main()
