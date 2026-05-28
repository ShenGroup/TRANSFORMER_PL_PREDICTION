#!/usr/bin/env python3
"""
Plot Combined CDF of absolute error for comparison between PathProfile and CNN results.
PIDs 1-10 in a 2x5 subplot grid.
IEEE Transactions style formatting.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob

# Fix Type 3 font issue: use TrueType fonts instead
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def load_pathprofile_data(pid):
    """Load from Source A: /scratch/tvs9by/ntia/pathprofile/inference_results4_256"""
    filepath = f'/scratch/tvs9by/ntia/pathprofile/inference_results4_256/prediction_results_OID182_PID{pid}.csv'
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        # Compute absolute error if not present, though 'error' column exists based on investigation
        if 'abs_error' not in df.columns:
            if 'error' in df.columns:
                return np.abs(df['error'].values)
            else:
                print(f"Warning: 'error' column missing in {filepath}")
                return None
        return df['abs_error'].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_cnn_data(pid):
    """Load from Source B: /scratch/tvs9by/ntia/cnn/inference_results"""
    # Pattern seems to be inference_pid{pid}_oid182.csv based on file listing
    filepath = f'/scratch/tvs9by/ntia/cnn/inference_results2/inference_pid{pid}_oid182.csv'
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if 'abs_error' not in df.columns:
             print(f"Warning: 'abs_error' column missing in {filepath}")
             return None
        return df['abs_error'].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    # IEEE Trans style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5
    })

    # Setup 2x5 grid
    # For square subplots with 2 rows and 5 columns, width/height ratio should be approx 5/2 = 2.5
    # (15, 6.5) provides roughly square subplots
    n_rows = 2
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6.5))
    axes = axes.flatten()
    
    output_dir = '/scratch/tvs9by/ntia/pathprofile/inference_results4_256'
    
    print("Starting combined plotting (IEEE style)...")
    
    for i, pid in enumerate(range(1, 11)):
        ax = axes[i]
        pid_str = str(pid)
        
        # Load data
        pp_data = load_pathprofile_data(pid_str)
        cnn_data = load_cnn_data(pid_str)
        
        has_plot = False
        
        mean_pp = None
        mean_cnn = None

        # Plot PathProfile (Source A) - Solid Blue
        if pp_data is not None and len(pp_data) > 0:
            sorted_data = np.sort(pp_data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, label='Transformer', color='blue') # Updated label as per likely context, but sticking to simple 'Ours'
            has_plot = True
            
            # Stats for Ours
            mean_pp = np.mean(pp_data)
            print(f"PID{pid} Ours: Mean={mean_pp:.2f}")

        # Plot CNN (Source B) - Dashed Red
        if cnn_data is not None and len(cnn_data) > 0:
            sorted_data = np.sort(cnn_data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, label='CNN_UNet', color='red', linestyle='--')
            has_plot = True
            
            # Stats for Baseline
            mean_cnn = np.mean(cnn_data)
            print(f"PID{pid} Base: Mean={mean_cnn:.2f}")
            
        # Add mean to title
        title_str = f'PID {pid}'
        stats_parts = []
        if mean_pp is not None:
             stats_parts.append(f"Trans={mean_pp:.2f}")
        if mean_cnn is not None:
             stats_parts.append(f"CNN={mean_cnn:.2f}")
        
        if stats_parts:
            title_str += "\n(" + ", ".join(stats_parts) + ")"

        ax.set_title(title_str, pad=5, fontsize=9)
        ax.set_xlim(left=0) 
        ax.set_ylim(0, 1.05)
        
        # Add labels only on edges
        if i >= 5: # Bottom row
            ax.set_xlabel('Abs. Error (dB)')
            
        if i % 5 == 0: # First column
            ax.set_ylabel('CDF')
            
        if has_plot:
            # Legend on PID 1 (Top Left)
            if i == 0:
                ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='gray')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=8)

    # Adjust layout to be appropriate
    plt.tight_layout()
    # Add a bit of space between subplots
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    output_path = os.path.join(output_dir, 'error_cdf_combined_ieee.pdf')
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches='tight')
    print(f"\nSaved combined comparison plot to: {output_path}")

if __name__ == "__main__":
    main()
