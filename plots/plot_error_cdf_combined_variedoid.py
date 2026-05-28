#!/usr/bin/env python3
"""
Plot Combined CDF of absolute error for comparison between PathProfile and CNN results.
OIDs 184-193 (using PID 2) in a 2x5 subplot grid.
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

NLOS_MODEL_TYPE = 'allinone_unified'
def load_pathprofile_data(oid):
    """Load from Source A: /scratch/tvs9by/ntia/pathprofile/inference_results4/prediction_results_OID{oid}_PID2.csv"""
    # Note: Source directory changed to inference_results4 as per planning
    filepath = f'/scratch/tvs9by/ntia/pathprofile/inference_results4_{NLOS_MODEL_TYPE}/prediction_results_OID{oid}_PID2.csv'
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
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

def load_cnn_data(oid):
    """Load from Source B: /scratch/tvs9by/ntia/cnn/inference_results/inference_pid2_oid{oid}.csv"""
    filepath = f'/scratch/tvs9by/ntia/cnn/inference_results/inference_pid2_oid{oid}.csv'
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
    n_rows = 2
    n_cols = 5
    # (15, 6.5) provides roughly square subplots nicely spaced
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6.5))
    axes = axes.flatten()
    
    # Save output in the same directory as Source A data or a general results dir
    output_dir = f'/scratch/tvs9by/ntia/pathprofile/inference_results4_{NLOS_MODEL_TYPE}' 
    if not os.path.exists(output_dir):
        # Fallback if dir doesn't exist, though it should based on file checks
        output_dir = '/scratch/tvs9by/ntia/pathprofile'
    
    print("Starting combined plotting for OIDs 184-193 (IEEE style)...")
    
    # OIDs 184 to 193
    oids = range(184, 194) 
    
    for i, oid in enumerate(oids):
        if i >= len(axes): break # Safety check
        
        ax = axes[i]
        oid_str = str(oid)
        
        # Load data
        pp_data = load_pathprofile_data(oid_str)
        cnn_data = load_cnn_data(oid_str)
        
        has_plot = False
        
        mean_pp = None
        mean_cnn = None

        # Plot PathProfile (Source A) - Solid Blue
        if pp_data is not None and len(pp_data) > 0:
            sorted_data = np.sort(pp_data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, label='Transformer', color='blue')
            has_plot = True
            
            # Stats for Ours
            mean_pp = np.mean(pp_data)
            print(f"OID{oid} Ours: Mean={mean_pp:.2f}")

        # Plot CNN (Source B) - Dashed Red
        if cnn_data is not None and len(cnn_data) > 0:
            sorted_data = np.sort(cnn_data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, label='CNN_UNet', color='red', linestyle='--')
            has_plot = True
            
            # Stats for Baseline
            mean_cnn = np.mean(cnn_data)
            print(f"OID{oid} Base: Mean={mean_cnn:.2f}")
            
        # Add mean to title
        title_str = f'OID {oid}'
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
            # Legend on OID 184 (Top Left)
            if i == 0:
                ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='gray')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    output_path = os.path.join(output_dir, 'error_cdf_combined_oid_ieee.pdf')
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches='tight')
    print(f"\nSaved combined OID comparison plot to: {output_path}")

if __name__ == "__main__":
    main()
