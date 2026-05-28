"""
Data preprocessing script for UNet path loss prediction.

This script:
1. Loads TIFF files containing height (band 2) and path loss (band 1) data
2. Crops matrices to 240x240 centered
3. Extracts metadata: frq, pol, tx_height, rx_height, pwr
4. Splits data by OID: oid <= 180 for training, oid > 180 for testing
5. Saves two pickle files: training_data.pkl and testing_data.pkl
"""

import rasterio
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from tqdm import tqdm


def crop_center(img, cropx, cropy):
    """Crop the center of a 2D array."""
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]


def process_tiff_files(data_dir, params_csv, catalog_csv, output_dir, crop_size=256):
    """
    Process TIFF files and create training/testing datasets.
    
    Args:
        data_dir: Directory containing TIFF files
        params_csv: Path to parameters.csv
        catalog_csv: Path to analysis_catalog.csv
        output_dir: Directory to save output pickle files
        crop_size: Size to crop matrices to (default 240x240)
    
    Returns:
        tuple: (training_data, testing_data) lists of dictionaries
    """
    # Load metadata
    df_params = pd.read_csv(params_csv)
    df_catalog = pd.read_csv(catalog_csv)
    
    # Map polarization to numeric
    pol_map = {'Horizontal': 0, 'Vertical': 1}
    df_params['pol'] = df_params['pol'].map(pol_map)
    
    # Merge parameters with catalog
    merged_data = pd.merge(df_catalog, df_params, on='PID', how='inner')
    
    training_data = []
    testing_data = []
    
    print(f"Processing {len(merged_data)} TIFF files...")
    
    for idx, row in tqdm(merged_data.iterrows(), total=len(merged_data)):
        filename = row['RID'] + '.tiff'
        tiff_path = os.path.join(data_dir, filename)
        
        # Check if file exists
        if not os.path.exists(tiff_path):
            print(f"Warning: File not found: {tiff_path}")
            continue
        
        try:
            # Open TIFF file
            with rasterio.open(tiff_path) as tiff:
                # Read band 1 (path loss) and band 2 (height)
                pathloss_full = tiff.read(1)
                height_full = tiff.read(2)
                
                # Check for invalid values
                mask = np.abs(pathloss_full) < 1e38
                if not mask.any():
                    print(f"Warning: All invalid values in {filename}")
                    continue
                
                # Replace invalid values with minimum valid value
                pathloss_full = np.where(
                    np.abs(pathloss_full) > 1e38,
                    np.min(pathloss_full[np.abs(pathloss_full) < 1e38]),
                    pathloss_full
                )
                height_full = np.where(
                    np.abs(height_full) > 1e38,
                    np.min(height_full[np.abs(height_full) < 1e38]),
                    height_full
                )
                
                # Check if matrices are large enough to crop
                if height_full.shape[0] < crop_size or height_full.shape[1] < crop_size:
                    print(f"Warning: Image too small to crop in {filename}: {height_full.shape}")
                    continue
                
                # Crop to center 240x240
                height_matrix = crop_center(height_full, crop_size, crop_size)
                pathloss_matrix = crop_center(pathloss_full, crop_size, crop_size)
                
                # Negate path loss (as in preprocessing_new.py)
                pathloss_matrix = -pathloss_matrix
                
                # Extract metadata
                oid = row['OID']
                frq = row['frq']
                pol = row['pol']
                tx_height = row['height']
                pwr = row['pwr']
                
                # Assume rx_height is 50 (as in test4.py line 382)
                rx_height = 50.0
                
                # Create data dictionary
                data_dict = {
                    'height_matrix': height_matrix.astype(np.float32),
                    'pathloss_matrix': pathloss_matrix.astype(np.float32),
                    'frq': float(frq),
                    'pol': int(pol),
                    'tx_height': float(tx_height),
                    'rx_height': float(rx_height),
                    'pwr': float(pwr),
                    'oid': int(oid),
                    'filename': filename
                }
                
                # Split by OID
                if oid <= 180:
                    training_data.append(data_dict)
                else:
                    testing_data.append(data_dict)
                    
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Training samples: {len(training_data)}")
    print(f"Testing samples: {len(testing_data)}")
    
    return training_data, testing_data


def main():
    # Configuration
    data_dir = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/'
    params_csv = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/parameters.csv'
    catalog_csv = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/analysis_catalog.csv'
    output_dir = '/scratch/tvs9by/ntia/cnn/unet_data/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process TIFF files
    training_data, testing_data = process_tiff_files(
        data_dir=data_dir,
        params_csv=params_csv,
        catalog_csv=catalog_csv,
        output_dir=output_dir,
        crop_size=256
    )
    
    # Save to pickle files
    training_path = os.path.join(output_dir, 'training_data.pkl')
    testing_path = os.path.join(output_dir, 'testing_data.pkl')
    
    print(f"\nSaving training data to {training_path}...")
    with open(training_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Saving testing data to {testing_path}...")
    with open(testing_path, 'wb') as f:
        pickle.dump(testing_data, f)
    
    print("\nData preprocessing complete!")
    print(f"Training data: {training_path} ({len(training_data)} samples)")
    print(f"Testing data: {testing_path} ({len(testing_data)} samples)")
    
    # Print sample statistics
    if training_data:
        sample = training_data[0]
        print(f"\nSample data structure:")
        print(f"  height_matrix shape: {sample['height_matrix'].shape}")
        print(f"  pathloss_matrix shape: {sample['pathloss_matrix'].shape}")
        print(f"  frq: {sample['frq']}")
        print(f"  pol: {sample['pol']}")
        print(f"  tx_height: {sample['tx_height']}")
        print(f"  rx_height: {sample['rx_height']}")
        print(f"  pwr: {sample['pwr']}")
        print(f"  oid: {sample['oid']}")


if __name__ == '__main__':
    main()
