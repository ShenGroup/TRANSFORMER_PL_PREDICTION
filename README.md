# Robust Pathloss Radio Map Prediction via Physics-Informed Transformer
The core idea is to cast path-loss (PL) prediction as a **sequence-learning
problem along the TX→RX great-circle path**, rather than as a 2-D
raster-to-raster map regression. A physics-informed transformer
(`CondSeqTransformer_EarlyFusion`) ingests:

* per-step physical features `x_i = [h_i, t_i, r_i]` — terrain height, distance
  from TX, distance to RX — sampled along the great-circle path, and
* per-link state features `(frq, pol, LOS/NLOS)` as **separate structured
  tokens**, one token per feature,

and predicts the PL at the RX from a learnable `[CLS]` readout. A shared model
handles both LOS and NLOS regimes via the LOS/NLOS conditioning token. Because
the model operates on the link-level 1-D ray, it captures long-range,
path-dependent effects (terrain blockages, knife-edge geometry) that purely
convolutional encoders over fixed 2-D crops under-represent. We compare the
transformer against a CNN–UNet baseline that learns the same mapping from 2-D height and radio parameters.

## Requirements

* Python 3.10+
* PyTorch ≥ 2.1 (CUDA build recommended)
* `numpy`, `pandas`, `tqdm`, `matplotlib`
* `rasterio` (UNet pipeline and transformer DSM inference)
* `wandb`

## Running training

Update the data and checkpoint paths near the bottom of each training script,
then run on a GPU node:

```bash
# Transformer — leave OID 182 out for PID-sweep evaluation
python transformer/training_variedpid.py

# Transformer — leave OIDs 184–193 out for OID-sweep evaluation
python transformer/training_variedoid.py

# UNet baseline
python cnn/prepare_unet_data.py    # builds training_data.pkl / testing_data.pkl
python cnn/train_unet.py
```

Both transformer training scripts share `model.py` (early-fusion model,
`d_model=512`, 8 layers, 16 heads, ~50 epochs of AdamW at `lr=2e-5`,
`batch_size=128`). Checkpoints and normalization stats are written to
`ckpt_dir` and named with the training timestamp.

## Running inference

Inference produces a per-grid-point prediction CSV
(`prediction_results_OID{oid}_PID{pid}.csv`) plus a summary plot
(`summary_plot_OID{oid}_PID{pid}.png`).

```bash
# Transformer — set checkpoint_path / norm_stats_file at the top of the script
python transformer/inference_variedpid.py     # OID 182 × PID 1..10
python transformer/inference_variedoid.py     # OIDs 184..193 × PID 2

# UNet baseline
python cnn/inference_unet.py --oid_min 181 --oid_max 193 --pid_min 2 --pid_max 2
```



## Reproducing the paper plots

The `results/` directory ships the inference CSVs and summary plots already
produced by the runs above. Two CDF/comparison scripts regenerate the figures
in the paper:

```bash
# Combined error-CDF figures (Fig. X — varied PID / varied OID)
python results/plots/plot_error_cdf_combined_variedpid.py
python results/plots/plot_error_cdf_combined_variedoid.py

# Side-by-side CNN vs Transformer maps (Fig. Y)
python results/inference_results_comparison/plot_comparison_cnn_transformer.py
python results/inference_results_comparison/plot_comparison_cnn_transformer2.py
```

`inference_statistics_log.txt` in
`results/cnn/inference_results_varied_pid/` records per-(OID,PID) MAE,
percentiles, and LOS/NLOS shares for the baseline.

## Citation

If you use this code or build on the results, please cite our DySPAN 2026
paper:

> Zihao Liang and Cong Shen, "Robust Pathloss Radio Map Prediction via
> Physics-Informed Transformer," in *Proc. IEEE International Symposium on
> Dynamic Spectrum Access Networks (DySPAN)*, 2026.

```bibtex
@inproceedings{liang2026robust,
  title     = {Robust Pathloss Radio Map Prediction via Physics-Informed Transformer},
  author    = {Liang, Zihao and Shen, Cong},
  booktitle = {Proc. IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)},
  year      = {2026},
  organization = {IEEE},
}
```
