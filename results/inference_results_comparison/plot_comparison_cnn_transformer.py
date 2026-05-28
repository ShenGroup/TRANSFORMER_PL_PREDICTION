
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Fix Type 3 font issue: use TrueType fonts instead
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def load_grid(df, value_col):
    """Pivots DataFrame to create a 2D grid."""
    return df.pivot(index='i', columns='j', values=value_col).sort_index(axis=0).sort_index(axis=1)

def plot_grid(ax, grid, title, cbar_label, vmin=None, vmax=None, cmap='viridis'):
    """Helper to plot a single grid."""
    extent = [0, grid.shape[1], 0, grid.shape[0]]
    im = ax.imshow(grid.values, origin='lower', aspect='equal', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    return im

def main():
    transformer_path = '/scratch/tvs9by/ntia/pathprofile/inference_results4_plot_comparison/prediction_results_OID182_PID1.csv'
    cnn_path = '/scratch/tvs9by/ntia/pathprofile/inference_results4_plot_comparison/inference_pid1_oid182.csv'
    output_path = '/scratch/tvs9by/ntia/pathprofile/inference_results4_plot_comparison/comparison_plot_pid1_oid182.png'

    print(f"Loading Transformer results from: {transformer_path}")
    df_trans = pd.read_csv(transformer_path)
    
    print(f"Loading CNN results from: {cnn_path}")
    df_cnn = pd.read_csv(cnn_path)

    # --- Prepare Grids ---
    
    # Grid 1 (Top Left): Actual Pathloss (from Transformer CSV)
    actual_loss_grid = load_grid(df_trans, 'actual_pathloss')
    
    # Grid 2 (Bottom Left): Height (from Transformer CSV 'height_at_rx')
    height_grid = load_grid(df_trans, 'height_at_rx')

    # Grid 3 (Top Middle): Transformer Predicted Pathloss
    trans_pred_grid = load_grid(df_trans, 'predicted_pathloss')
    
    # Grid 4 (Bottom Middle): Transformer Error
    trans_error_grid = load_grid(df_trans, 'error')

    # Grid 5 (Top Right): CNN Predicted Pathloss
    cnn_pred_grid = load_grid(df_cnn, 'predicted_path_loss')
    
    # Grid 6 (Bottom Right): CNN Error (Calculate locally to ensure alignment or use 'abs_error' * sign?)
    # Using 'abs_error' from file might lose sign info, better to calc: Pred - Actual
    # We need to make sure the indices align. The CSVs should cover the same OID/PID region.
    # Let's align them by index (i, j) just to be safe, although load_grid handles separate pivoting.
    
    # For alignment, we can rely on the grids being sorted by load_grid.
    # Provided both CSVs cover exactly the same pixels, straightforward subtraction works.
    # However, let's look at df_cnn. 'path_loss' is actual.
    # let's recalculate error just to be consistent: Error = Predicted - Actual
    
    # CNN 'path_loss' is actual.
    cnn_actual_grid = load_grid(df_cnn, 'path_loss')
    cnn_error_calc_grid = cnn_pred_grid - cnn_actual_grid # Prediction - Actual

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Common color scale for Pathloss
    vmin_pl = min(actual_loss_grid.min().min(), trans_pred_grid.min().min(), cnn_pred_grid.min().min())
    vmax_pl = max(actual_loss_grid.max().max(), trans_pred_grid.max().max(), cnn_pred_grid.max().max())
    
    # Common color scale for Error
    vmin_err = min(trans_error_grid.min().min(), cnn_error_calc_grid.min().min())
    vmax_err = max(trans_error_grid.max().max(), cnn_error_calc_grid.max().max())
    
    # Make error scale symmetric around 0 roughly, or just fit to data
    abs_max_err = max(abs(vmin_err), abs(vmax_err))
    vmin_err = -abs_max_err
    vmax_err = abs_max_err

    # Col 1: Actual & Height
    plot_grid(axes[0, 0], actual_loss_grid, 'Actual Pathloss', 'dB', vmin=vmin_pl, vmax=vmax_pl)
    plot_grid(axes[1, 0], height_grid, 'Height at Rx', 'm') # Height has its own scale
    
    # Col 2: Transformer
    plot_grid(axes[0, 1], trans_pred_grid, 'Transformer Prediction', 'dB', vmin=vmin_pl, vmax=vmax_pl)
    plot_grid(axes[1, 1], trans_error_grid, 'Transformer Error', 'dB', vmin=vmin_err, vmax=vmax_err, cmap='coolwarm')
    
    # Col 3: CNN
    plot_grid(axes[0, 2], cnn_pred_grid, 'CNN Prediction', 'dB', vmin=vmin_pl, vmax=vmax_pl)
    plot_grid(axes[1, 2], cnn_error_calc_grid, 'CNN Error', 'dB', vmin=vmin_err, vmax=vmax_err, cmap='coolwarm')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
