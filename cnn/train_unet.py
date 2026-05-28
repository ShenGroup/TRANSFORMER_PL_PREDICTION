"""
UNet training script for path loss prediction.

This script:
1. Loads preprocessed training and testing data
2. Computes normalization statistics from training set
3. Defines UNet model with 4 encoder/decoder layers
4. Trains the model with L1 loss
5. Evaluates periodically and saves checkpoints
6. Logs metrics to Weights & Biases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import wandb


# ========== Dataset ==========

class PathLossDataset(Dataset):
    """Dataset for path loss prediction."""
    
    def __init__(self, data_list, norm_stats=None):
        """
        Args:
            data_list: List of data dictionaries
            norm_stats: Dictionary with normalization statistics (optional)
        """
        self.data_list = data_list
        self.norm_stats = norm_stats
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Extract data
        height_matrix = item['height_matrix'].copy()  # (240, 240)
        pathloss_matrix = item['pathloss_matrix']  # (240, 240)
        frq = item['frq']
        pol = item['pol']
        tx_height = item['tx_height']
        rx_height = item['rx_height']
        pwr = item['pwr']
        
        # Apply normalization if provided
        if self.norm_stats is not None:
            height_matrix = (height_matrix - self.norm_stats['height_mean']) / self.norm_stats['height_std']
            frq = (frq - self.norm_stats['frq_mean']) / self.norm_stats['frq_std']
            tx_height = (tx_height - self.norm_stats['tx_height_mean']) / self.norm_stats['tx_height_std']
            rx_height = rx_height / 50.0  # Normalize by dividing by 50
            pwr = (pwr - self.norm_stats['pwr_mean']) / self.norm_stats['pwr_std']
            # pol is binary (0 or 1), no normalization needed
        
        # Create constant matrices for parameters (256, 256)
        crop_size = height_matrix.shape[0]  # Should be 256
        frq_matrix = np.full((crop_size, crop_size), frq, dtype=np.float32)
        pol_matrix = np.full((crop_size, crop_size), pol, dtype=np.float32)
        tx_height_matrix = np.full((crop_size, crop_size), tx_height, dtype=np.float32)
        rx_height_matrix = np.full((crop_size, crop_size), rx_height, dtype=np.float32)
        pwr_matrix = np.full((crop_size, crop_size), pwr, dtype=np.float32)
        
        # Stack into 6-channel input (6, crop_size, crop_size)
        input_tensor = np.stack([
            height_matrix,
            frq_matrix,
            pol_matrix,
            tx_height_matrix,
            rx_height_matrix,
            pwr_matrix
        ], axis=0)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_tensor).float()
        target_tensor = torch.from_numpy(pathloss_matrix).float().unsqueeze(0)  # (1, crop_size, crop_size)
        
        return input_tensor, target_tensor


def compute_normalization_stats(data_list):
    """
    Compute normalization statistics from training data.
    
    Args:
        data_list: List of data dictionaries
    
    Returns:
        Dictionary with mean and std for each parameter
    """
    print("Computing normalization statistics...")
    
    # Collect all values (excluding rx_height since it's normalized by /50)
    heights = []
    frqs = []
    tx_heights = []
    pwrs = []
    
    for item in tqdm(data_list):
        heights.append(item['height_matrix'].flatten())
        frqs.append(item['frq'])
        tx_heights.append(item['tx_height'])
        pwrs.append(item['pwr'])
    
    # Concatenate and compute statistics
    heights = np.concatenate(heights)
    frqs = np.array(frqs)
    tx_heights = np.array(tx_heights)
    pwrs = np.array(pwrs)
    
    stats = {
        'height_mean': float(np.mean(heights)),
        'height_std': float(np.std(heights)),
        'frq_mean': float(np.mean(frqs)),
        'frq_std': float(np.std(frqs)),
        'tx_height_mean': float(np.mean(tx_heights)),
        'tx_height_std': float(np.std(tx_heights)),
        'pwr_mean': float(np.mean(pwrs)),
        'pwr_std': float(np.std(pwrs)),
    }
    
    print(f"Height: mean={stats['height_mean']:.4f}, std={stats['height_std']:.4f}")
    print(f"Frequency: mean={stats['frq_mean']:.4f}, std={stats['frq_std']:.4f}")
    print(f"Tx Height: mean={stats['tx_height_mean']:.4f}, std={stats['tx_height_std']:.4f}")
    print(f"Rx Height: normalized by /50 (fixed)")
    print(f"Power: mean={stats['pwr_mean']:.4f}, std={stats['pwr_std']:.4f}")
    
    return stats


# ========== UNet Model ==========

class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> Dropout"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet model for path loss prediction.
    
    Architecture:
    - Input: 6 channels (height + 5 parameters)
    - Encoder: 4 levels (64, 128, 256, 512)
    - Bottleneck: 1024 channels
    - Decoder: 4 levels (512, 256, 128, 64)
    - Output: 1 channel (path loss)
    """
    
    def __init__(self, in_channels=6, out_channels=1, dropout_rate=0.1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout_rate)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512, dropout_rate)  # 1024 = 512 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout_rate)  # 512 = 256 (upconv) + 256 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout_rate)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout_rate)  # 128 = 64 (upconv) + 64 (skip)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # (B, 64, 240, 240)
        x = self.pool1(enc1)  # (B, 64, 120, 120)
        
        enc2 = self.enc2(x)  # (B, 128, 120, 120)
        x = self.pool2(enc2)  # (B, 128, 60, 60)
        
        enc3 = self.enc3(x)  # (B, 256, 60, 60)
        x = self.pool3(enc3)  # (B, 256, 30, 30)
        
        enc4 = self.enc4(x)  # (B, 512, 30, 30)
        x = self.pool4(enc4)  # (B, 512, 15, 15)
        
        # Bottleneck
        x = self.bottleneck(x)  # (B, 1024, 15, 15)
        
        # Decoder with skip connections
        x = self.upconv4(x)  # (B, 512, 30, 30)
        x = torch.cat([x, enc4], dim=1)  # (B, 1024, 30, 30)
        x = self.dec4(x)  # (B, 512, 30, 30)
        
        x = self.upconv3(x)  # (B, 256, 60, 60)
        x = torch.cat([x, enc3], dim=1)  # (B, 512, 60, 60)
        x = self.dec3(x)  # (B, 256, 60, 60)
        
        x = self.upconv2(x)  # (B, 128, 120, 120)
        x = torch.cat([x, enc2], dim=1)  # (B, 256, 120, 120)
        x = self.dec2(x)  # (B, 128, 120, 120)
        
        x = self.upconv1(x)  # (B, 64, 240, 240)
        x = torch.cat([x, enc1], dim=1)  # (B, 128, 240, 240)
        x = self.dec1(x)  # (B, 64, 240, 240)
        
        # Output
        x = self.out_conv(x)  # (B, 1, 240, 240)
        
        return x


# ========== Evaluation Functions ==========

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluation function.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        (loss, mae) tuple
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        
        loss = F.l1_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_count += batch_size
    
    if total_count == 0:
        return float('nan'), float('nan')
    
    return total_loss / total_count, total_mae / total_count


@torch.no_grad()
def evaluate_worstcase(model, loader, device):
    """
    Worst-case evaluation with error distribution statistics.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        (stats_dict, error_tensor) tuple
    """
    model.eval()
    error_list = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        
        # Compute absolute error
        err = (outputs - targets).abs()
        error_list.append(err.flatten().detach().cpu())
    
    if not error_list:
        return {
            'max': float('nan'), 'p95': float('nan'), 'p75': float('nan'),
            'p50': float('nan'), 'p20': float('nan'), 'min': float('nan'),
            'mean': float('nan'), 'std': float('nan'), 'count': 0
        }, torch.empty(0)
    
    v = torch.cat(error_list, dim=0)
    
    # Convert to numpy for quantile computation (handles large arrays better)
    v_np = v.numpy()
    
    stats = {
        'max': float(v_np.max()),
        'p95': float(np.percentile(v_np, 95)),
        'p75': float(np.percentile(v_np, 75)),
        'p50': float(np.percentile(v_np, 50)),
        'p20': float(np.percentile(v_np, 20)),
        'min': float(v_np.min()),
        'mean': float(v_np.mean()),
        'std': float(v_np.std()),
        'count': v_np.size,
    }
    
    return stats, v


# ========== Training ==========

def save_checkpoint(model, optimizer, step, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint: {path}")


def train():
    """Main training function."""
    
    # Configuration
    data_dir = '/scratch/tvs9by/ntia/cnn/unet_data/'
    checkpoint_dir = '/scratch/tvs9by/ntia/cnn/unet_checkpoints/'
    
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 1000
    eval_every = 1000  # Evaluate every N steps
    save_every = 10000  # Save checkpoint every N steps
    dropout_rate = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("\nLoading training data...")
    with open(os.path.join(data_dir, 'training_data.pkl'), 'rb') as f:
        training_data = pickle.load(f)
    print(f"Training samples: {len(training_data)}")
    
    print("\nLoading testing data...")
    with open(os.path.join(data_dir, 'testing_data.pkl'), 'rb') as f:
        testing_data = pickle.load(f)
    print(f"Testing samples: {len(testing_data)}")
    
    # Compute normalization statistics from training data
    norm_stats = compute_normalization_stats(training_data)
    
    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unet_training_{timestamp}"
    
    wandb.login(key='458bc70094e51a25582798845f13b9208bb80cc4')
    wandb.init(
        project="ntia-unet-pathloss",
        name=run_name,
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "dropout_rate": dropout_rate,
            "eval_every": eval_every,
            "save_every": save_every,
            "train_samples": len(training_data),
            "test_samples": len(testing_data),
            "architecture": "UNet",
            "in_channels": 6,
            "out_channels": 1,
        }
    )
    
    # Save normalization statistics
    norm_stats_path = os.path.join(checkpoint_dir, f'norm_stats_{timestamp}.pt')
    torch.save(norm_stats, norm_stats_path)
    print(f"\nSaved normalization statistics to {norm_stats_path}")
    
    # Create datasets
    train_dataset = PathLossDataset(training_data, norm_stats)
    test_dataset = PathLossDataset(testing_data, norm_stats)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Create model
    print("\nInitializing UNet model...")
    model = UNet(in_channels=6, out_channels=1, dropout_rate=dropout_rate).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    step_global = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        epoch_loss = 0.0
        epoch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in pbar:
            step_global += 1
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss (L1 loss)
            loss = F.l1_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            batch_size_actual = inputs.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_count += batch_size_actual
            
            # Log training loss to wandb
            wandb.log({"train/loss": loss.item(), "train/step": step_global})
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Periodic evaluation
            if step_global % eval_every == 0:
                print(f"\n[Evaluating at step {step_global}]")
                
                eval_loss, eval_mae = evaluate(model, test_loader, device)
                print(f"  Test set | loss {eval_loss:.4f} | mae {eval_mae:.4f}")
                
                eval_stats, _ = evaluate_worstcase(model, test_loader, device)
                print(
                    f"  [error distribution] "
                    f"max={eval_stats['max']:.4f} | p95={eval_stats['p95']:.4f} | p75={eval_stats['p75']:.4f} | "
                    f"p50={eval_stats['p50']:.4f} | p20={eval_stats['p20']:.4f} | min={eval_stats['min']:.4f} | "
                    f"mean={eval_stats['mean']:.4f} | std={eval_stats['std']:.4f} | n={eval_stats['count']}"
                )
                
                # Log evaluation metrics to wandb
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/mae": eval_mae,
                    "eval/error_max": eval_stats['max'],
                    "eval/error_p95": eval_stats['p95'],
                    "eval/error_p75": eval_stats['p75'],
                    "eval/error_p50": eval_stats['p50'],
                    "eval/error_p20": eval_stats['p20'],
                    "eval/error_min": eval_stats['min'],
                    "eval/error_mean": eval_stats['mean'],
                    "eval/error_std": eval_stats['std'],
                    "train/step": step_global,
                })
                
                # Save checkpoint (only at save_every intervals)
                if step_global % save_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_step{step_global}.pt")
                    save_checkpoint(model, optimizer, step_global, checkpoint_path)
                    
                    # Log checkpoint to wandb
                    wandb.save(checkpoint_path)
                
                model.train()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_count if epoch_count > 0 else float('nan')
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics to wandb
        wandb.log({"train/epoch_loss": avg_epoch_loss, "train/epoch": epoch+1})
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_final_step{step_global}.pt")
    save_checkpoint(model, optimizer, step_global, final_checkpoint_path)
    
    # Log final checkpoint to wandb
    wandb.save(final_checkpoint_path)
    wandb.save(norm_stats_path)
    
    # Finish wandb run
    wandb.finish()
    
    print("\nTraining complete!")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final evaluation on test set...")
    print("="*80)
    
    eval_loss, eval_mae = evaluate(model, test_loader, device)
    print(f"FINAL - Test set | loss {eval_loss:.4f} | mae {eval_mae:.4f}")
    
    eval_stats, eval_errors = evaluate_worstcase(model, test_loader, device)
    print(
        f"[FINAL error distribution] "
        f"max={eval_stats['max']:.4f} | p95={eval_stats['p95']:.4f} | p75={eval_stats['p75']:.4f} | "
        f"p50={eval_stats['p50']:.4f} | p20={eval_stats['p20']:.4f} | min={eval_stats['min']:.4f} | "
        f"mean={eval_stats['mean']:.4f} | std={eval_stats['std']:.4f} | n={eval_stats['count']}"
    )
    
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
