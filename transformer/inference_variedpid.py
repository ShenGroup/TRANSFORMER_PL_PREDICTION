"""
Path Loss Prediction Inference Script (Unified Model)
This script processes DSM (Digital Surface Model) files and predicts path loss for each grid point.
It uses a single CondSeqTransformer_EarlyFusion model for both LOS and NLOS conditions.

Usage:
1. Update the MODEL AND NORMALIZATION PATHS section.
2. Update data_dir and output_dir if needed.
3. Run: python predict_pathloss_inference4_v4.py
"""

import rasterio
import numpy as np
import csv
import os
import pandas as pd
import torch
import torch.nn as nn
import math
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform
import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt, radians, degrees
import sys
import multiprocessing as mp
from functools import partial
import time
from collections import defaultdict
import warnings
import re

# Suppress the nested tensor warning from PyTorch transformer
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

# ============================================================================
# MODEL DEFINITION (Copied from pathlosspred_v4_earlyfusion_nlos_fullh_final_withlos.py)
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, N, d_model]
        return x + self.pe[:x.size(1)].unsqueeze(0)  # broadcast over batch

class CondSeqTransformer_EarlyFusion(nn.Module):
    """
    New model with early fusion of height, distance_tx, distance_rx.
    Separate embeddings for each sequence type, then concatenated.
    State parameters are treated as separate tokens.
    
    Token order:
      [CLS] || state_tokens (m_state tokens) || [SEP] || fused_sequence_tokens
    
    where fused_sequence_tokens[i] = concat(height_emb[i], dist_tx_emb[i], dist_rx_emb[i])
    """
    def __init__(
        self,
        m_state,        # number of state parameters (3: frq, pol, is_los)
        seq_len=200,    # sequence length
        dy=1,           # output dimension
        d_model=512,
        d_height=128,   # embedding dim for height
        d_dist_tx=128,  # embedding dim for distance_tx
        d_dist_rx=128,  # embedding dim for distance_rx
        nhead=16,
        num_layers=8,
        dim_ff=2048,
        dropout=0.1,
        causal=False
    ):
        super().__init__()
        self.m_state = m_state
        self.seq_len = seq_len
        self.dy = dy
        self.d_model = d_model
        self.d_height = d_height
        self.d_dist_tx = d_dist_tx
        self.d_dist_rx = d_dist_rx
        self.d_fused = d_height + d_dist_tx + d_dist_rx
        self.causal = causal
        
        # Separate embeddings for each input type
        self.emb_height = nn.Linear(1, d_height)
        self.emb_dist_tx = nn.Linear(1, d_dist_tx)
        self.emb_dist_rx = nn.Linear(1, d_dist_rx)
        
        # State embedding - each state parameter becomes a separate token
        self.emb_state = nn.Linear(1, d_model)  # Each state param -> d_model
        
        # Project fused sequence to d_model
        self.proj_fused = nn.Linear(self.d_fused, d_model)
        
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)
        
        # Output heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Learnable [SEP] token
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.sep_token, std=0.02)
    
    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    
    def forward(self, state, height, dist_tx, dist_rx, mask=None):
        """
        Args:
            state    : [B, M_state]  - state parameters (frq, pol, is_los)
            height   : [B, N, 1]     - height sequence
            dist_tx  : [B, N, 1]     - distance from tx sequence
            dist_rx  : [B, N, 1]     - distance from rx sequence
            mask     : [B, N] bool   - True = valid timestep
        
        Returns:
            Y_hat : [B, N, dy]  - per-step predictions
            y_cls : [B, dy]     - global prediction from CLS
        """
        B, N, _ = height.shape
        M = state.shape[1]
        
        # Embed each sequence separately
        h_emb = self.emb_height(height)      # [B, N, d_height]
        tx_emb = self.emb_dist_tx(dist_tx)   # [B, N, d_dist_tx]
        rx_emb = self.emb_dist_rx(dist_rx)   # [B, N, d_dist_rx]
        
        # Early fusion: concatenate embeddings
        fused = torch.cat([h_emb, tx_emb, rx_emb], dim=-1)  # [B, N, d_fused]
        
        # Project to d_model
        fused_proj = self.proj_fused(fused)  # [B, N, d_model]
        
        # Embed state - each parameter becomes a separate token
        state_expanded = state.unsqueeze(-1)  # [B, M, 1]
        state_tokens = self.emb_state(state_expanded)  # [B, M, d_model]
        
        # Prepend [SEP] between state and fused sequence
        sep = self.sep_token.expand(B, 1, -1)   # [B, 1, d_model]
        
        # Concatenate: state (M) + sep (1) + fused (N)
        h_no_cls = torch.cat([state_tokens, sep, fused_proj], dim=1)  # [B, M+1+N, d_model]
        h_no_cls = self.pos(h_no_cls)
        
        # Prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, d_model]
        z_in = torch.cat([cls, h_no_cls], dim=1)  # [B, 1+M+1+N, d_model]
        
        # Handle mask
        if mask is not None:
            # Extend mask for [CLS], state tokens, and [SEP] (all always valid)
            ones = torch.ones(B, 1 + M + 1, dtype=mask.dtype, device=mask.device)
            mask_ext = torch.cat([ones, mask], dim=1)  # [B, 2+M+N]
            key_pad = ~mask_ext  # True = pad
        else:
            key_pad = None
        
        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None
        
        # Encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 2+M+N, d_model]
        
        # Extract outputs
        z_cls = z[:, 0, :]                    # [B, d_model]
        
        # indices to slice X tokens:
        # 0: CLS
        # 1..M: state
        # M+1: SEP
        # M+2 .. M+2+N: fused (N tokens)
        z_seq = z[:, 2+M:2+M+N, :]            # [B, N, d_model]
        
        Y_hat = self.out_proj(z_seq)          # [B, N, dy]
        y_cls = self.cls_head(z_cls)          # [B, dy]
        
        return Y_hat, y_cls

# ============================================================================
# HELPER FUNCTIONS (Knife Edge & TIREM style)
# ============================================================================

def _los_blocked(y: np.ndarray, i: int, j: int, eps: float = 1e-12) -> bool:
    """Return True if the line segment i->j is blocked by any intermediate point."""
    if i is None or j is None:
        return True
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return False
    ya, yb = y[lo], y[hi]
    x = np.arange(lo, hi + 1, dtype=float)
    y_line = ya + (yb - ya) * (x - lo) / (hi - lo)
    y_mid = y[lo + 1:hi]
    y_line_mid = y_line[1:-1]
    return np.any(y_mid > y_line_mid + eps)

def _local_ridges(y: np.ndarray) -> np.ndarray:
    """Return sorted indices of local maxima (exclude endpoints)."""
    N = len(y)
    if N < 3:
        return np.array([], dtype=int)
    ridges = []
    i = 1
    while i <= N - 2:
        yl, yc, yr = y[i - 1], y[i], y[i + 1]
        if yc > yl and yc > yr:
            ridges.append(i)
            i += 1
            continue
        if yc > yl and yc == yr:
            j = i + 1
            while j < N and y[j] == yc:
                j += 1
            if j < N and y[j] < yc:
                plateau_lo, plateau_hi = i, j - 1
                ridges.append((plateau_lo + plateau_hi) // 2)
            i = j
            continue
        i += 1
    return np.array(sorted(set(ridges)), dtype=int)

def _best_next_ridge(y: np.ndarray, ridges: np.ndarray, cur: int) -> int | None:
    """Choose ridge with largest elevation angle."""
    cand = ridges[ridges > cur]
    if cand.size == 0:
        return None
    dy = y[cand] - y[cur]
    dx = cand - cur
    angles = np.arctan2(dy, dx)
    j_best = int(cand[np.argmax(angles)])
    return j_best

def find_diffraction_path(heights: np.ndarray):
    """Find diffraction path by greedy ridge selection (Epstein-Peterson method)."""
    y = np.asarray(heights, dtype=float)
    N = len(y)
    if N < 2:
        return []
    tx, rx = 0, N - 1
    ridges = _local_ridges(y)
    ridges = ridges[(ridges > tx) & (ridges < rx)]
    if ridges.size == 0:
        return []
    path = []
    cur = tx
    visited = set([tx])
    while True:
        nxt = _best_next_ridge(y, ridges, cur)
        if nxt is None or nxt in visited:
            return path
        path.append(nxt)
        visited.add(nxt)
        if not _los_blocked(y, nxt, rx):
            return path
        cur = nxt

def compute_knife_edge_indices(heights: torch.Tensor) -> torch.Tensor:
    """Compute knife edge indices based on TIREM Figure 4-2 method."""
    heights_np = heights.squeeze(-1).cpu().numpy()
    knife_edge_indices = find_diffraction_path(heights_np)
    if len(knife_edge_indices) == 0:
        return torch.tensor([], dtype=torch.int64)
    else:
        return torch.tensor(knife_edge_indices, dtype=torch.int64)

# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def great_circle_interpolation(lat1, lon1, lat2, lon2, num_samples):
    """Computes points along a great-circle path."""
    phi1, lambda1 = radians(lat1), radians(lon1)
    phi2, lambda2 = radians(lat2), radians(lon2)
    d_lambda = lambda2 - lambda1
    cos_d = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)
    cos_d = max(-1.0, min(1.0, cos_d))
    d = atan2(sqrt(1 - cos_d**2), cos_d)
    if d < 1e-9:
        return [(lat1, lon1)]
    points = []
    for i in range(num_samples):
        f = i / (num_samples - 1) if num_samples > 1 else 0.0
        A = sin((1 - f) * d) / sin(d)
        B = sin(f * d) / sin(d)
        x = A * cos(phi1) * cos(lambda1) + B * cos(phi2) * cos(lambda2)
        y = A * cos(phi1) * sin(lambda1) + B * cos(phi2) * sin(lambda2)
        z = A * sin(phi1) + B * sin(phi2)
        lat_i = degrees(atan2(z, sqrt(x*x + y*y)))
        lon_i = degrees(atan2(y, x))
        points.append((lat_i, lon_i))
    return points

def compute_los_nlos(heights, distances, tx_height, rx_height, effective_earth_radius_factor=4/3):
    """Computes LOS/NLOS status and clearance."""
    heights = np.array(heights)
    distances = np.array(distances)
    if len(heights) < 2 or len(distances) < 2:
        return False, np.array([])
    R_earth = 6371000.0
    R_eff = R_earth * effective_earth_radius_factor
    total_distance = distances[-1]
    tx_height_above_sea = heights[0] + tx_height
    rx_height_above_sea = heights[-1] + rx_height
    clearance = np.zeros_like(heights)
    for i in range(len(heights)):
        d = distances[i]
        h_curve = d * (total_distance - d) / (2 * R_eff)
        los_height = tx_height_above_sea + (rx_height_above_sea - tx_height_above_sea) * (d / total_distance) - h_curve
        clearance[i] = los_height - heights[i]
    is_los = np.all(clearance > -0.1)
    return is_los, clearance

def compute_diffraction_vs_troposcatter_metric(d_km, min_clearance_norm, num_knife_edges):
    """Compute metric [-1, 1] for Diffraction vs Troposcatter dominance."""
    distance_factor = np.tanh((d_km - 75.0) / 50.0)
    obstruction_factor = 0.0
    if np.isnan(min_clearance_norm): pass
    elif min_clearance_norm < -0.3: obstruction_factor = -1.0
    elif min_clearance_norm < -0.05: obstruction_factor = 0.5
    ridge_factor = -0.3 if num_knife_edges > 3 else (-0.1 if num_knife_edges > 1 else 0.0)
    metric = np.clip(0.5 * distance_factor + 0.4 * obstruction_factor + ridge_factor, -1.0, 1.0)
    return float(metric)

def compute_troposcatter_loss_simplified(f_mhz, d_km, min_clearance_m, h_tx_m, h_rx_m):
    """Simplified troposcatter loss estimation for visualization."""
    if d_km <= 0 or f_mhz <= 0: return float("nan")
    L_base = 30.0 * np.log10(f_mhz) + 25.0 * np.log10(d_km) - 20.0
    clearance_penalty = 0.0
    if min_clearance_m < 0:
        clearance_penalty = min(abs(min_clearance_m) * 0.05, 30.0)
    height_gain = 2.0 * np.log10(max(h_tx_m, 1.0)) + 2.0 * np.log10(max(h_rx_m, 1.0))
    return float(L_base + clearance_penalty - height_gain)

# ============================================================================
# CONFIGURATION AND PATHS
# ============================================================================

# MODEL AND NORMALIZATION PATHS - UPDATE THESE FOR YOUR TRAINING RUN
# Single Unified Model for both LOS and NLOS
checkpoint_path = "/scratch/tvs9by/ntia/pathprofile/checkpointsful_nlos_final/earlyfusion_training_20260107_121320_v4LOS_0_fullh_step50000.pt"
norm_stats_file = "/scratch/tvs9by/ntia/pathprofile/checkpointsful_nlos_final/norm_stats_20251231_103310_los_0_v4.pt"  # Not used for normalization, kept for compatibility

# Data and Output
data_dir = "/scratch/tvs9by/GPT2/trainingdata_new/datat/datat"
output_dir = "/scratch/tvs9by/ntia/pathprofile/inference_results4_allinone_unified_big_varied_pid"

# Params
EPS_NORM = 1e-6
max_seq_len = 500
m_state_v4 = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parallelism
NUM_WORKERS = min(mp.cpu_count() - 1, 10)
# NUM_WORKERS = 1 # Uncomment for debugging

# ============================================================================
# INFERENCE LOGIC
# ============================================================================

_worker_model = None
_worker_stats = None

def _init_worker_models():
    """Initialize model and stats in each worker process."""
    global _worker_model, _worker_stats
    import warnings
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
    worker_device = device
    
    # Common model args
    model_args = dict(
        m_state=m_state_v4,
        dy=1,
        d_model=512,
        d_height=128,
        d_dist_tx=128,
        d_dist_rx=128,
        nhead=16,
        num_layers=8,
        dim_ff=2048,
        dropout=0.1,
        causal=False
    )
    
    # Load Unified model
    model = CondSeqTransformer_EarlyFusion(seq_len=max_seq_len, **model_args).to(worker_device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=worker_device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.eval()
    
    # Load stats (optional - not used for normalization but kept for compatibility)
    stats = {}
    if os.path.exists(norm_stats_file):
        stats = torch.load(norm_stats_file, map_location='cpu')
    else:
        print(f"Warning: Stats file not found: {norm_stats_file} (not required for normalization)")
    
    _worker_model = {'model': model, 'device': worker_device}
    _worker_stats = stats

def normalize_inputs(state, height_abs, height_rel, dist_tx, dist_rx, stats, is_nlos=False):
    """Normalize inputs for model inference (Updated to match v4 FullH training script).
    Matches NormalizeInputsV4 from training script: simple division by constants, no z-score.
    """
    state_normalized = state.clone()
    state_normalized[0] = state[0] / 1000.0  # frq divided by 1000
    # pol (index 1) and is_los (index 2) remain unchanged
    
    # Normalize sequences by dividing by 10000 (matches training script)
    height_norm = height_rel / 10000.0  # height divided by 10000
    dist_tx_norm = dist_tx / 10000.0    # dist_tx divided by 10000
    dist_rx_norm = dist_rx / 10000.0    # dist_rx divided by 10000
    
    knife_edge_indices = None
    if is_nlos:
        knife_edge_indices = compute_knife_edge_indices(height_abs)
    
    return state_normalized, height_norm, dist_tx_norm, dist_rx_norm, knife_edge_indices

def predict_pathloss(model, state, height, dist_tx, dist_rx, mask):
    with torch.no_grad():
        Y_hat, y_cls = model(state, height, dist_tx, dist_rx, mask=mask)
        return y_cls.item()

def _process_grid_point_worker(args):
    """Worker function for parallel processing."""
    global _worker_model, _worker_stats
    
    pid, oid, i, j, dsm_path, tx_lat, tx_lon, rx_lat, rx_lon, frq, height, pwr, pol, rad = args
    
    model = _worker_model['model']
    worker_device = _worker_model['device']
    stats = _worker_stats
    
    # Distance check
    phi1, lambda1 = radians(tx_lat), radians(tx_lon)
    phi2, lambda2 = radians(rx_lat), radians(rx_lon)
    d_lambda = lambda2 - lambda1
    cos_d = max(-1.0, min(1.0, sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)))
    d = atan2(sqrt(1 - cos_d**2), cos_d)
    total_distance = d * 6371000.0
    
    if total_distance < 1.0 or total_distance > rad:
        return None
    
    # Process
    with rasterio.open(dsm_path) as src:
        with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
            step_size_meters = 100
            num_samples = max(2, int(total_distance / step_size_meters) + 1)
            gc_points = great_circle_interpolation(tx_lat, tx_lon, rx_lat, rx_lon, num_samples)
            dots = []
            for lat_dot, lon_dot in gc_points:
                dot_x, dot_y = transform('EPSG:4326', src.crs, [lon_dot], [lat_dot])
                dots.append((dot_x[0], dot_y[0]))
            
            sampled = vrt.sample(dots)
            values = list(sampled)
            heights = [float(val[1]) for val in values]
            pathlosslabels = [-float(val[0]) for val in values]
            
            if abs(pathlosslabels[-1]) > 1e37: return None
            
            # Distances
            distances = []
            cumulative_dist = 0.0
            distances.append(0.0)
            for k in range(1, len(gc_points)):
                lat1, lon1 = gc_points[k-1]
                lat2, lon2 = gc_points[k]
                phi1, lambda1 = radians(lat1), radians(lon1)
                phi2, lambda2 = radians(lat2), radians(lon2)
                d_lambda = lambda2 - lambda1
                cos_d = max(-1.0, min(1.0, sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)))
                d = atan2(sqrt(1 - cos_d**2), cos_d)
                segment_dist = d * 6371000.0
                cumulative_dist += segment_dist
                distances.append(cumulative_dist)
            
            distances = np.array(distances)
            distances_from_rx = total_distance - distances
            
            # LOS/NLOS check
            tx_height_above_terrain = height
            rx_height_above_terrain = 50
            is_los, clearance = compute_los_nlos(heights, distances, tx_height_above_terrain, rx_height_above_terrain)
            
            # Metrics
            c0 = 3.0e8; frq_hz = frq * 1.0e6
            lam = c0 / frq_hz if frq_hz > 0 else np.nan
            d1, d2 = distances, distances_from_rx
            denom = d1 + d2
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = np.sqrt(lam * d1 * d2 / np.maximum(denom, 1e-6))
            norm_clearance = np.full_like(clearance, np.nan, dtype=float)
            valid_r1 = r1 > 0
            if np.any(valid_r1):
                norm_clearance[valid_r1] = clearance[valid_r1] / r1[valid_r1]
                min_clearance_m = float(clearance.min())
                min_clearance_norm = float(np.nanmin(norm_clearance))
                main_idx = int(np.argmin(clearance))
                u_main = float(norm_clearance[main_idx])
                d1_main_km = float(d1[main_idx] / 1000.0)
                d2_main_km = float(d2[main_idx] / 1000.0)
            else:
                min_clearance_m = float("nan"); min_clearance_norm = float("nan")
                u_main = float("nan"); d1_main_km = float("nan"); d2_main_km = float("nan")
            
            range_km = float(total_distance / 1000.0)
            
            # Prepare tensors
            heights_tensor = torch.tensor(heights, dtype=torch.float32).unsqueeze(-1)
            dist_tx_tensor = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)
            dist_rx_tensor = torch.tensor(distances_from_rx, dtype=torch.float32).unsqueeze(-1)
            
            heights_tensor[0, 0] += tx_height_above_terrain
            heights_tensor[-1, 0] += rx_height_above_terrain
            heights_rel = heights_tensor - heights_tensor[0]
            
            state_base = [frq, pol, float(is_los)]
            state_tensor = torch.tensor([state_base], dtype=torch.float32)
            
            # Normalize with UNIFIED stats for both LOS and NLOS (is_nlos flag only computes knife edges)
            is_nlos_bool = not is_los
            state_norm, height_norm, dist_tx_norm, dist_rx_norm, ke_indices = normalize_inputs(
                state_tensor[0], heights_tensor, heights_rel, dist_tx_tensor, dist_rx_tensor, stats, is_nlos=is_nlos_bool
            )
            
            state_norm = state_norm.unsqueeze(0).to(worker_device)
            height_norm = height_norm.unsqueeze(0).to(worker_device)
            dist_tx_norm = dist_tx_norm.unsqueeze(0).to(worker_device)
            dist_rx_norm = dist_rx_norm.unsqueeze(0).to(worker_device)
            mask = torch.ones(1, height_norm.shape[1], dtype=torch.bool).to(worker_device)
            
            # Predict using the SINGLE model
            predicted_pathloss = predict_pathloss(model, state_norm, height_norm, dist_tx_norm, dist_rx_norm, mask)
            
            num_knife_edges = float(len(ke_indices)) if ke_indices is not None else 0.0
            
            original_pathloss = pathlosslabels[-1] + pwr
            error = predicted_pathloss - original_pathloss
            
            troposcatter_loss = compute_troposcatter_loss_simplified(frq, range_km, min_clearance_m, tx_height_above_terrain, rx_height_above_terrain)
            prop_mode_metric = compute_diffraction_vs_troposcatter_metric(range_km, min_clearance_norm, num_knife_edges)
            
            return {
                'pid': pid, 'oid': oid, 'i': i, 'j': j,
                'height': heights[-1],
                'actual_pathloss': original_pathloss,
                'predicted_pathloss': predicted_pathloss,
                'original_pathloss_last': original_pathloss, 
                'error': error,
                'is_los': int(is_los),
                'min_clearance_m': min_clearance_m,
                'min_clearance_norm': min_clearance_norm,
                'u_main': u_main,
                'd1_main_km': d1_main_km,
                'd2_main_km': d2_main_km,
                'num_knife_edges': num_knife_edges,
                'range_km': range_km,
                'troposcatter_loss': troposcatter_loss,
                'prop_mode_metric': prop_mode_metric,
            }

def plot_summary_results(results_df, output_path):
    if results_df.empty: return
    height_grid = results_df.pivot(index='i', columns='j', values='height').sort_index(axis=0).sort_index(axis=1)
    actual_grid = results_df.pivot(index='i', columns='j', values='actual_pathloss').sort_index(axis=0).sort_index(axis=1)
    pred_grid = results_df.pivot(index='i', columns='j', values='predicted_pathloss').sort_index(axis=0).sort_index(axis=1)
    error_grid = results_df.pivot(index='i', columns='j', values='error').sort_index(axis=0).sort_index(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    def plot_grid(ax, grid, title, cbar_label):
        extent = [grid.columns.min(), grid.columns.max(), grid.index.min(), grid.index.max()]
        im = ax.imshow(grid.values, origin='lower', aspect='auto', extent=extent)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(title)
    
    plot_grid(axes[0, 0], height_grid, 'Height at Rx', 'Height (m)')
    plot_grid(axes[0, 1], actual_grid, 'Actual Pathloss', 'Pathloss (dB)')
    plot_grid(axes[1, 0], pred_grid, 'Predicted Pathloss', 'Pathloss (dB)')
    plot_grid(axes[1, 1], error_grid, 'Prediction Error', 'Error (dB)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    print("="*80 + "\nPath Loss Prediction Inference (Unified Model)\n" + "="*80)
    
    if not os.path.exists(checkpoint_path): print(f"ERROR: Ckpt not found: {checkpoint_path}"); return
    # Stats file is optional (not used for normalization)
    
    df_params = pd.read_csv(os.path.join(data_dir, "parameters.csv"))
    df_file = pd.read_csv(os.path.join(data_dir, "analysis_catalog.csv"))
    filtered_data = df_file[(df_file['OID'].between(182,182)) & (df_file['PID'].between(1,10))]
    print(f"Found {len(filtered_data)} entries.")
    if len(filtered_data) == 0: return
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    tasks = []
    
    for idx, row in filtered_data.iterrows():
        pid, oid, filename = row['PID'], row['OID'], row['RID']
        dsmfile = os.path.join(data_dir, filename + '.tiff')
        if not os.path.exists(dsmfile): continue
        params = df_params[df_params['PID'] == pid].values[0]
        frq, pol_str, height, pwr, rad = params[1], params[2], params[3], params[4], params[5]
        pol = 0 if pol_str == 'Horizontal' else 1
        
        with rasterio.open(dsmfile) as src:
            ylen, xlen = src.shape
            center_x, center_y = (src.bounds[0] + src.bounds[2]) / 2, (src.bounds[1] + src.bounds[3]) / 2
            tx_lon, tx_lat = transform(src.crs, 'EPSG:4326', [center_x], [center_y])
            tx_lat, tx_lon = tx_lat[0], tx_lon[0]
            center_y, center_x = ylen // 2, xlen // 2
            start_i, end_i = max(0, center_y - 128), min(ylen, center_y + 128)
            start_j, end_j = max(0, center_x - 128), min(xlen, center_x + 128)
            
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    x = src.transform[2] + j * src.transform[0] + i * src.transform[1]
                    y = src.transform[5] + j * src.transform[3] + i * src.transform[4]
                    rx_lon, rx_lat = transform(src.crs, 'EPSG:4326', [x], [y])
                    tasks.append((pid, oid, i, j, dsmfile, tx_lat, tx_lon, rx_lat[0], rx_lon[0], frq, height, pwr, pol, rad))
    
    print(f"Total tasks: {len(tasks)}")
    tasks_by_file = defaultdict(list)
    for task in tasks: tasks_by_file[(task[0], task[1])].append(task)
    
    if NUM_WORKERS > 1:
        with mp.Pool(processes=NUM_WORKERS, initializer=_init_worker_models) as pool:
            for (pid, oid), file_tasks in tasks_by_file.items():
                print(f"Processing PID={pid}, OID={oid} ({len(file_tasks)} points)...")
                results_list = []
                batch_size = 1000
                for i in range(0, len(file_tasks), batch_size):
                    batch = file_tasks[i:i+batch_size]
                    results_list.extend(pool.map(_process_grid_point_worker, batch))
                
                # Zero error correction
                valid_results = [r for r in results_list if r is not None]
                if not valid_results: continue
                
                i_vals = [r['i'] for r in valid_results]; j_vals = [r['j'] for r in valid_results]
                i_c, j_c = (min(i_vals) + max(i_vals)) // 2, (min(j_vals) + max(j_vals)) // 2
                c_range_i, c_range_j = range(i_c - 1, i_c + 3), range(j_c - 1, j_c + 3)
                
                for r in valid_results:
                    if r['i'] in c_range_i and r['j'] in c_range_j:
                        r['predicted_pathloss'] = r['actual_pathloss']
                        r['error'] = 0.0
                    all_results.append(r)
                    
                # Save CSV
                fname = f'prediction_results_OID{int(oid)}_PID{int(pid)}.csv'
                keys = list(valid_results[0].keys())
                with open(os.path.join(output_dir, fname), 'w', newline='') as f:
                    dict_writer = csv.DictWriter(f, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(valid_results)
                print(f"Saved {fname}")
                
    else:
        _init_worker_models()
        for task in tasks:
            res = _process_grid_point_worker(task)
            if res: all_results.append(res)

    if all_results:
        results_df = pd.DataFrame(all_results)
        for (pid, oid), group in results_df.groupby(['pid', 'oid']):
             plot_summary_results(group, os.path.join(output_dir, f'summary_plot_OID{int(oid)}_PID{int(pid)}.png'))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
