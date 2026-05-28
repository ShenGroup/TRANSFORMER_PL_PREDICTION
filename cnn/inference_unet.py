
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm
import csv
import rasterio
import argparse
import time
from datetime import datetime

# ========== UNet Model (Copied from train_unet.py) ==========

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
        self.dec4 = DoubleConv(1024, 512, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout_rate)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout_rate)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout_rate)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out_conv(x)
        
        return x

# ========== Helper Functions ==========

def crop_center(img, cropx, cropy):
    """Crop the center of a 2D array."""
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]

def normalize_sample(data_dict, norm_stats, crop_size=256):
    """Normalization logic from PathLossDataset but for a single sample."""
    height_matrix = data_dict['height_matrix'].copy()
    frq = data_dict['frq']
    pol = data_dict['pol']
    tx_height = data_dict['tx_height']
    rx_height = data_dict['rx_height']
    pwr = data_dict['pwr']

    if norm_stats is not None:
        height_matrix = (height_matrix - norm_stats['height_mean']) / norm_stats['height_std']
        frq = (frq - norm_stats['frq_mean']) / norm_stats['frq_std']
        tx_height = (tx_height - norm_stats['tx_height_mean']) / norm_stats['tx_height_std']
        rx_height = rx_height / 50.0
        pwr = (pwr - norm_stats['pwr_mean']) / norm_stats['pwr_std']
    
    # Create constant matrices
    frq_matrix = np.full((crop_size, crop_size), frq, dtype=np.float32)
    pol_matrix = np.full((crop_size, crop_size), pol, dtype=np.float32)
    tx_height_matrix = np.full((crop_size, crop_size), tx_height, dtype=np.float32)
    rx_height_matrix = np.full((crop_size, crop_size), rx_height, dtype=np.float32)
    pwr_matrix = np.full((crop_size, crop_size), pwr, dtype=np.float32)

    input_tensor = np.stack([
        height_matrix,
        frq_matrix,
        pol_matrix,
        tx_height_matrix,
        rx_height_matrix,
        pwr_matrix
    ], axis=0)

    return torch.from_numpy(input_tensor).float().unsqueeze(0) # Add batch dim

# ========== Main Processing ==========

def main():
    parser = argparse.ArgumentParser(description="UNet Inference on raw TIFFs")
    parser.add_argument('--pid_min', type=int, default=2, help='Minimum PID')
    parser.add_argument('--pid_max', type=int, default=2, help='Maximum PID')
    parser.add_argument('--oid_min', type=int, default=181, help='Minimum OID')
    parser.add_argument('--oid_max', type=int, default=193, help='Maximum OID')
    args = parser.parse_args()

    # Paths
    model_path = '/scratch/tvs9by/ntia/cnn/unet_checkpoints/unet_training_20251208_085718_step90000.pt'
    norm_stats_path = '/scratch/tvs9by/ntia/cnn/unet_checkpoints/norm_stats_20251208_085718.pt'
    data_dir = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/'
    params_csv = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/parameters.csv'
    catalog_csv = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/analysis_catalog.csv'
    
    output_dir = '/scratch/tvs9by/ntia/cnn/inference_results'
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, 'inference_results.csv')
    txt_output_path = os.path.join(output_dir, 'quantiles.txt')

    print(f"Starting inference with PID range [{args.pid_min}, {args.pid_max}] and OID range [{args.oid_min}, {args.oid_max}]")
    start_time = time.time()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {model_path}...")
    model = UNet(in_channels=6, out_channels=1).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # Load Norm Stats
    print(f"Loading normalization stats from {norm_stats_path}...")
    try:
        # Note: torch.load might return a dict directly or contained in a 'norm_stats' key
        # verify structure based on train_unet.py: torch.save(norm_stats, ...) where norm_stats is a dict.
        norm_stats = torch.load(norm_stats_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading norm stats: {e}")
        return

    # Load and Filter Metadata
    print("Loading metadata...")
    df_params = pd.read_csv(params_csv)
    df_catalog = pd.read_csv(catalog_csv)
    
    pol_map = {'Horizontal': 0, 'Vertical': 1}
    df_params['pol'] = df_params['pol'].map(pol_map)
    
    merged_data = pd.merge(df_catalog, df_params, on='PID', how='inner')
    
    # Filter
    filtered_data = merged_data[
        (merged_data['PID'] >= args.pid_min) & (merged_data['PID'] <= args.pid_max) &
        (merged_data['OID'] >= args.oid_min) & (merged_data['OID'] <= args.oid_max)
    ]
    
    print(f"Found {len(filtered_data)} files matching criteria.")
    if len(filtered_data) == 0:
        return

    all_errors = []
    processed_count = 0
    crop_size = 256

    # Processing Loop
    for idx, row in tqdm(filtered_data.iterrows(), total=len(filtered_data)):
        filename = row['RID'] + '.tiff'
        tiff_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(tiff_path):
            continue
            
        try:
            t0 = time.time()
            with rasterio.open(tiff_path) as tiff:
                pathloss_full = tiff.read(1)
                height_full = tiff.read(2)
            
            # Preprocessing
            mask = np.abs(pathloss_full) < 1e38
            if not mask.any():
                continue

            pathloss_full = np.where(np.abs(pathloss_full) > 1e38, np.min(pathloss_full[mask]), pathloss_full)
            height_full = np.where(np.abs(height_full) > 1e38, np.min(height_full[np.abs(height_full) < 1e38]), height_full)

            if height_full.shape[0] < crop_size or height_full.shape[1] < crop_size:
                continue
            
            height_matrix = crop_center(height_full, crop_size, crop_size) # (256, 256)
            pathloss_matrix = crop_center(pathloss_full, crop_size, crop_size)
            
            # IMPORTANT: Negate path loss as in prepare_unet_data
            pathloss_matrix = -pathloss_matrix
            
            # Prepare data dict for normalization
            data_dict = {
                'height_matrix': height_matrix.astype(np.float32),
                'frq': float(row['frq']),
                'pol': int(row['pol']),
                'tx_height': float(row['height']),
                'rx_height': 50.0, # Assumed 50.0
                'pwr': float(row['pwr'])
            }
            
            input_tensor = normalize_sample(data_dict, norm_stats, crop_size).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            pred_matrix = output.squeeze().cpu().numpy() # (256, 256)
            
            # Apply center 4x4 substitution (predicted = actual)
            H_mat, W_mat = pred_matrix.shape
            i_center = H_mat // 2
            j_center = W_mat // 2
            
            # Define center 4x4 region (indices) matching range(center-1, center+3)
            i_start = max(0, i_center - 1)
            i_end = min(H_mat, i_center + 3)
            j_start = max(0, j_center - 1)
            j_end = min(W_mat, j_center + 3)
            
            pred_matrix[i_start:i_end, j_start:j_end] = pathloss_matrix[i_start:i_end, j_start:j_end]
            
            # Calculate Error
            abs_error_matrix = np.abs(pathloss_matrix - pred_matrix)
            file_errors = abs_error_matrix.flatten()
            all_errors.append(file_errors)
            
            # Write per-file stats to TXT
            file_stats = {
                'max': float(np.max(file_errors)),
                'p99': float(np.percentile(file_errors, 99)),
                'p95': float(np.percentile(file_errors, 95)),
                'p75': float(np.percentile(file_errors, 75)),
                'p50': float(np.percentile(file_errors, 50)),
                'p20': float(np.percentile(file_errors, 20)),
                'min': float(np.min(file_errors)),
                'mean': float(np.mean(file_errors)),
                'std': float(np.std(file_errors)),
            }
            
            with open(txt_output_path, 'a') as f:
                f.write(f"File: {filename} | PID: {row['PID']} | OID: {row['OID']}\n")
                f.write(f"Prediction Error (Predicted - Actual):\n")
                f.write(f"  Mean: {file_stats['mean']:.4f}\n")
                f.write(f"  Std:  {file_stats['std']:.4f}\n")
                f.write(f"  Min:  {file_stats['min']:.4f}\n")
                f.write(f"  Max:  {file_stats['max']:.4f}\n")
                f.write(f"  20th percentile: {file_stats['p20']:.4f}\n")
                f.write(f"  50th percentile: {file_stats['p50']:.4f}\n")
                f.write(f"  75th percentile: {file_stats['p75']:.4f}\n")
                f.write(f"  95th percentile: {file_stats['p95']:.4f}\n")
                f.write(f"  99th percentile: {file_stats['p99']:.4f}\n")
                f.write("-" * 40 + "\n")
            
            print(f"Processed {filename}: Mean Err={file_stats['mean']:.4f}, Max={file_stats['max']:.4f}")

            # Write to Individual CSV
            pid_oid_csv_path = os.path.join(output_dir, f'inference_pid{row["PID"]}_oid{row["OID"]}.csv')
            
            # Use numpy indices helper
            H, W = pred_matrix.shape
            # Meshgrid
            j_coords, i_coords = np.meshgrid(np.arange(W), np.arange(H))
            
            flat_i = i_coords.flatten()
            flat_j = j_coords.flatten()
            flat_h = height_matrix.flatten()
            flat_pl = pathloss_matrix.flatten()
            flat_pred = pred_matrix.flatten()
            flat_err = abs_error_matrix.flatten()
            
            # Prepare DataFrame
            df_out = pd.DataFrame({
                'i': flat_i,
                'j': flat_j,
                'height': flat_h,
                'path_loss': flat_pl,
                'predicted_path_loss': flat_pred,
                'abs_error': flat_err
            })
            
            # Write to CSV (Overwrite for this file)
            df_out.to_csv(pid_oid_csv_path, index=False)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute and Write Stats
    print("\nComputing global statistics...")
    if all_errors:
        all_errs = np.concatenate(all_errors)
        
        stats = {
            'max': float(np.max(all_errs)),
            'p99': float(np.percentile(all_errs, 99)),
            'p95': float(np.percentile(all_errs, 95)),
            'p75': float(np.percentile(all_errs, 75)),
            'p50': float(np.percentile(all_errs, 50)),
            'p20': float(np.percentile(all_errs, 20)),
            'min': float(np.min(all_errs)),
            'mean': float(np.mean(all_errs)),
            'std': float(np.std(all_errs)),
            'count': all_errs.size
        }
        
        with open(txt_output_path, 'a') as f:
            f.write(f"\nrun_timestamp: {datetime.now()}\n")
            f.write(f"pid_range: {args.pid_min}-{args.pid_max}\n")
            f.write(f"oid_range: {args.oid_min}-{args.oid_max}\n")
            f.write(f"files_processed: {processed_count}\n")
            f.write(f"total_processing_time: {total_time:.2f}s\n")
            if processed_count > 0:
                f.write(f"avg_time_per_file: {total_time/processed_count:.4f}s\n")
            f.write("Prediction Error (Predicted - Actual):\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std:  {stats['std']:.4f}\n")
            f.write(f"  Min:  {stats['min']:.4f}\n")
            f.write(f"  Max:  {stats['max']:.4f}\n")
            f.write(f"  20th percentile: {stats['p20']:.4f}\n")
            f.write(f"  50th percentile: {stats['p50']:.4f}\n")
            f.write(f"  75th percentile: {stats['p75']:.4f}\n")
            f.write(f"  95th percentile: {stats['p95']:.4f}\n")
            f.write(f"  99th percentile: {stats['p99']:.4f}\n")
            f.write("="*60 + "\n")
            
        print(f"Stats written to {txt_output_path}")

    print(f"Total time: {total_time:.2f}s")
    print(f"Results appended to {csv_output_path}")

if __name__ == '__main__':
    main()
