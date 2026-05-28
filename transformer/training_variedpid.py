import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo
from model import SinusoidalPositionalEncoding
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math
from typing import Tuple
import numpy as np
import random
import glob
import re
import tqdm
from tqdm import tqdm
# Seeds and constants
EPS_NORM = 1e-6
EPS = 1e-6
seed = 1496
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

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


# Configuration
max_seq_len = 500  # Maximum sequence length expected
m_state_v4 = 3  # frq, pol, is_los (removed pwr, height, rad, receiver_height)
batch_size = 128
eval_every = 1000
wandbbbbb = 1  # Set to 0 to disable wandb
LOS = 0

# Paths
train_dir = "/scratch/tvs9by/ntia/pathprofile/trainingdata/trainingdata/training"
test_dir = "/scratch/tvs9by/ntia/pathprofile/testingdata/testingdata/testing"
ckpt_dir = "/scratch/tvs9by/ntia/pathprofile/checkpointsful_nlos_final"

# Run metadata
pt_now_v4 = datetime.now(ZoneInfo("America/Los_Angeles"))
ts_v4 = pt_now_v4.strftime("%Y%m%d_%H%M%S")
run_name_v4 = f"earlyfusion_training_{ts_v4}_v4LOS_{LOS}_fullh"

# ---------- Knife Edge Detection Functions (for NLOS TIREM computation, Figure 4-2) ----------
def _los_blocked(y: np.ndarray, i: int, j: int, eps: float = 1e-12) -> bool:
    """
    Return True if the line segment i->j is blocked by any intermediate point.
    """
    if i is None or j is None:
        return True
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return False  # adjacent or same => nothing in between

    ya, yb = y[lo], y[hi]
    x = np.arange(lo, hi + 1, dtype=float)
    y_line = ya + (yb - ya) * (x - lo) / (hi - lo)

    # strictly above the straight line means blocking
    y_mid = y[lo + 1:hi]
    y_line_mid = y_line[1:-1]
    return np.any(y_mid > y_line_mid + eps)

def _local_ridges(y: np.ndarray) -> np.ndarray:
    """
    Return sorted indices of local maxima (exclude endpoints).
    Plateau tops are represented by the plateau center index.
    Only maxima strictly higher than neighbors are kept.
    
    Args:
        y: 1D array of heights
        
    Returns:
        np.ndarray of ridge indices (sorted, excluding endpoints)
    """
    N = len(y)
    if N < 3:
        return np.array([], dtype=int)

    ridges = []
    i = 1
    while i <= N - 2:
        yl, yc, yr = y[i - 1], y[i], y[i + 1]
        # strict peak
        if yc > yl and yc > yr:
            ridges.append(i)
            i += 1
            continue
        # plateau: rise into flat then fall
        if yc > yl and yc == yr:
            j = i + 1
            while j < N and y[j] == yc:
                j += 1
            # if it falls after the flat => true flat-topped local max
            if j < N and y[j] < yc:
                plateau_lo, plateau_hi = i, j - 1
                ridges.append((plateau_lo + plateau_hi) // 2)
            i = j
            continue
        i += 1

    return np.array(sorted(set(ridges)), dtype=int)

def _best_next_ridge(y: np.ndarray, ridges: np.ndarray, cur: int) -> int | None:
    """
    Among ridge indices > cur, choose the one with the largest elevation angle
    as seen from 'cur': angle = atan2(y[j] - y[cur], j - cur).
    If all candidates are behind 'cur' or there are none, return None.
    """
    cand = ridges[ridges > cur]
    if cand.size == 0:
        return None
    dy = y[cand] - y[cur]
    dx = cand - cur
    angles = np.arctan2(dy, dx)  # negative if below, positive if above
    j_best = int(cand[np.argmax(angles)])
    return j_best

def find_diffraction_path(heights: np.ndarray):
    """
    Find diffraction path by greedy ridge selection (Epstein-Peterson method).
    All inputs are NLOS, so we always expect a diffraction path.
    Based on TIREM Figure 4-2.
    
    Args:
        heights: 1-D array along TX->RX (absolute heights with tx_height and rx_height added)
        
    Returns:
        knife_edge_indices: list of knife edge indices (excludes TX/RX indices)
        Returns empty list if no valid path found (shouldn't happen for NLOS)
    """
    y = np.asarray(heights, dtype=float)
    N = len(y)
    if N < 2:
        return []  # trivial/invalid

    tx, rx = 0, N - 1

    # Precompute ridge set (exclude endpoints)
    ridges = _local_ridges(y)
    # keep only interior ridges between TX and RX
    ridges = ridges[(ridges > tx) & (ridges < rx)]
    if ridges.size == 0:
        return []  # No ridges found (unexpected for NLOS)

    # Greedy climb by elevation angle (Epstein-Peterson method)
    path = []
    cur = tx
    visited = set([tx])
    while True:
        nxt = _best_next_ridge(y, ridges, cur)
        if nxt is None or nxt in visited:
            # no unseen ridge ahead
            return path  # Return what we have (may be incomplete)

        path.append(nxt)
        visited.add(nxt)

        # Can this ridge "see" RX?
        if not _los_blocked(y, nxt, rx):
            # success: we have a diffraction chain
            return path

        # Otherwise continue jumping ridge-to-ridge
        cur = nxt

def compute_knife_edge_indices(heights: torch.Tensor) -> torch.Tensor:
    """
    Compute knife edge indices based on TIREM Figure 4-2 method.
    Knife edges are the ridges that form the diffraction path (Epstein-Peterson construction).
    All inputs are NLOS, so we always expect knife edges.
    
    Args:
        heights: [N, 1] tensor of heights (already with tx_height and rx_height added)
        
    Returns:
        torch.Tensor of knife edge indices (1D, dtype=int64)
    """
    # Convert to numpy for knife edge detection
    heights_np = heights.squeeze(-1).cpu().numpy()  # [N]
    
    # Find diffraction path using Epstein-Peterson method (all NLOS)
    knife_edge_indices = find_diffraction_path(heights_np)
    
    # Convert back to torch tensor
    if len(knife_edge_indices) == 0:
        return torch.tensor([], dtype=torch.int64)
    else:
        return torch.tensor(knife_edge_indices, dtype=torch.int64)

# ---------- Dataset ----------
class VarLenDatasetV4(Dataset):
    """
    Dataset loader for training_v4 CSV structure with variable-length sequences:
    - Optimized loading with grouping by (oid, los) and caching chunks.
    - height_profile (variable length) - tx_height added to heights[0], rx_height added to heights[-1]
    - distance_from_tx_profile (variable length)
    - distance_from_rx_profile (variable length)
    - pathlosslabel_profile (variable length) - pwr added to pathloss[-1]
    - State parameters (3): frq, pol, is_los (treated as separate tokens)
    - Returns full sequences + knife edge indices.
    """
    def __init__(self, filelist, max_seq_len=500, cache_mode=False, cache_dir=None, filter_oid=None, filter_pid=None, apply_oid_filter=False):
        """
        Args:
            filelist: List of file paths
            max_seq_len: Maximum sequence length
            cache_mode: If True, enables granular caching/loading
            cache_dir: Directory to save/load cache chunks (required if cache_mode=True)
            filter_oid: If not None, only include rows where oid == filter_oid
            filter_pid: If not None, only include rows where pid == filter_pid
            apply_oid_filter: If True, skip files where oid >= 11 (for training only)
        """
        self.max_seq_len = max_seq_len
        self.items = []
        
        # Group files by (oid, k) where k is los/nlos indicator
        # File format expected: ..._oid_{i}_{j}_{k}.csv
        file_groups = {}
        
        print(f"Organizing {len(filelist)} files into groups...")
        for filename in filelist:
            basename = os.path.basename(filename)
            try:
                # Match pattern: ..._oid_{i}_{j}_{k}.csv OR ..._oid_{i}_los_{k}.csv
                # We need i (oid) and k (los/nlos)
                
                # First try the standard training pattern: _oid_(\d+)_(\d+)_(\d+)
                match1 = re.search(r'_oid_(\d+)_(\d+)_(\d+)\.csv$', basename)
                
                # Second try the evaluation pattern: _oid_(\d+)_los_(\d+)
                match2 = re.search(r'_oid_(\d+)_los_(\d+)\.csv$', basename)

                # Third try the simplest pattern: oid_{i}_{k}.csv
                match3 = re.search(r'oid_(\d+)_(\d+)\.csv$', basename)
                
                if match1:
                    oid = int(match1.group(1))
                    # j = int(match1.group(2)) # height len
                    k = int(match1.group(3))   # los type
                    
                    if apply_oid_filter and oid >= 11:
                         continue
                         
                    key = (oid, k)
                    if key not in file_groups:
                        file_groups[key] = []
                    file_groups[key].append(filename)

                elif match2:
                    oid = int(match2.group(1))
                    k = int(match2.group(2))   # los type
                    
                    if apply_oid_filter and oid >= 11:
                         continue
                         
                    key = (oid, k)
                    if key not in file_groups:
                        file_groups[key] = []
                    file_groups[key].append(filename)

                elif match3:
                    oid = int(match3.group(1))
                    k = int(match3.group(2))   # los type
                    
                    if apply_oid_filter and oid >= 11:
                         continue
                         
                    key = (oid, k)
                    if key not in file_groups:
                        file_groups[key] = []
                    file_groups[key].append(filename)
                else:
                    print(f"Warning: Could not parse oid/los from filename {basename}, skipping grouping optimization for this file.")
                    # Fallback or just ignore? For now let's skip to be safe as per user request structure
            except Exception as e:
                print(f"Error parsing filename {basename}: {e}")

        # Sort keys for deterministic order
        sorted_keys = sorted(file_groups.keys())
        print(f"Found {len(sorted_keys)} distinct (oid, los) groups.")
        
        for key in tqdm(sorted_keys, desc="Processing Groups"):
            oid, k = key
            group_files = file_groups[key]
            
            # Define cache filename for this group
            if cache_mode and cache_dir:
                cache_name = f"chunk_oid_{oid}_los_{k}.pt"
                cache_path = os.path.join(cache_dir, cache_name)
                
                if os.path.exists(cache_path):
                    # print(f"Loading cached chunk: {cache_name}")
                    try:
                        chunk_items = torch.load(cache_path)
                        self.items.extend(chunk_items)
                        continue
                    except Exception as e:
                        print(f"Failed to load cache {cache_path}: {e}. Reprocessing.")
            
            # Process files in this group
            group_items = []
            for filename in group_files:
                try:
                    df_in = pd.read_csv(filename) # Read full file
                    
                    for ind in range(len(df_in)):
                        df_row = df_in.iloc[ind]
                        
                        frq = float(df_row['frq'])
                        tx_height = float(df_row['height'])
                        pwr = float(df_row['pwr'])
                        pol = float(df_row['pol'])
                        rx_height = float(df_row['receiver_height'])
                        is_los = float(df_row['is_los'])
                        
                        # State only contains: frq, pol, is_los
                        state = torch.tensor([frq, pol, is_los], dtype=torch.float32)
                        
                        # Parse height profile
                        heights_str = df_row['height_profile']
                        heights = [float(tok.strip(" '\""))
                                   for tok in heights_str.strip()[1:-1].split(';')
                                   if tok.strip()]
                        if len(heights) == 0:
                            continue
                        
                        # Parse distance from tx
                        dist_tx_str = df_row['distance_from_tx_profile']
                        dist_tx = [float(tok.strip(" '\""))
                                   for tok in dist_tx_str.strip()[1:-1].split(';')
                                   if tok.strip()]
                        
                        # Parse distance from rx
                        dist_rx_str = df_row['distance_from_rx_profile']
                        dist_rx = [float(tok.strip(" '\""))
                                   for tok in dist_rx_str.strip()[1:-1].split(';')
                                   if tok.strip()]
                        
                        # Parse pathloss labels
                        pathloss_str = df_row['pathlosslabel_profile']
                        pathloss = [float(tok.strip(" '\""))
                                    for tok in pathloss_str.strip()[1:-1].split(';')
                                    if tok.strip()]

                        # Convert to tensors
                        heights = torch.tensor(heights, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
                        dist_tx = torch.tensor(dist_tx, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
                        dist_rx = torch.tensor(dist_rx, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
                        pathloss = torch.tensor(pathloss, dtype=torch.float32).unsqueeze(-1) # [N, 1]

                        # Add tx_height to heights[0] and rx_height to heights[-1]
                        heights[0, 0] = heights[0, 0] + tx_height
                        heights[-1, 0] = heights[-1, 0] + rx_height

                        # Add pwr to pathloss[-1]
                        pathloss[-1, 0] = pathloss[-1, 0] + pwr

                        # Compute knife edge indices (using absolute heights)
                        knife_edge_indices = compute_knife_edge_indices(heights)

                        # Compute relative height: heights - heights[0]
                        heights = heights - heights[0]

                        # Ensure all sequences have the same length and are within max_seq_len
                        seq_len = heights.shape[0]
                        if (heights.shape[0] != dist_tx.shape[0] or 
                            heights.shape[0] != dist_rx.shape[0] or 
                            heights.shape[0] != pathloss.shape[0]):
                            continue
                        if seq_len > max_seq_len:
                            continue
                        if seq_len == 0:
                            continue
                        
                        # Store: state, heights, dist_tx, dist_rx, pathloss, knife_edge_indices, tx_height, rx_height
                        # This matches the tuple expectation of the target file's NormalizeInputsV4 and existing pipeline
                        group_items.append((state, heights, dist_tx, dist_rx, pathloss, knife_edge_indices, tx_height, rx_height))

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
            
            # Save chunk if caching is enabled
            if cache_mode and cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(group_items, cache_path)
            
            self.items.extend(group_items)
            
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]

# ---------- Normalization ----------
def fit_input_stats_v4(ds):
    """
    Compute per-dimension mean/std for height over TRAIN SET ONLY.
    State parameters don't need normalization (frq is divided by 1000, pol and is_los unchanged).
    Distance_tx and distance_rx are normalized by dividing by 10000 (no stats needed).
    """
    # No stats needed for state parameters, dist_tx, or dist_rx
    
    height_sum = torch.zeros(1, dtype=torch.float64)
    height_sq = torch.zeros(1, dtype=torch.float64)
    height_cnt = 0
    
    for state, height, dist_tx, dist_rx, pathloss, knife_edge_indices, tx_height, rx_height in ds:
        # Height stats
        height = height.to(dtype=torch.float64)
        height_sum += height.sum(dim=0)
        height_sq += (height * height).sum(dim=0)
        height_cnt += height.size(0)
    
    height_mean = (height_sum / max(height_cnt, 1)).to(torch.float32)
    height_var = (height_sq / max(height_cnt, 1)) - height_mean.double()**2
    height_std = torch.sqrt(torch.clamp(height_var.to(torch.float32), min=EPS_NORM))
    
    return {
        "height_mean": height_mean,
        "height_std": height_std
    }

class NormalizeInputsV4(Dataset):
    """
    Wraps V4 dataset; applies normalization to state, height, dist_tx, dist_rx.
    - frq: divide by 1000
    - pol, is_los: no normalization (unchanged)
    - height: z-score normalization
    - dist_tx, dist_rx: divide by 10000
    Pathloss is returned unchanged.
    """
    def __init__(self, base_ds, stats):
        self.base = base_ds
        self.height_mean = stats["height_mean"]
        self.height_std = stats["height_std"]
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, i):
        state, height, dist_tx, dist_rx, pathloss, knife_edge_indices, tx_height, rx_height = self.base[i]
        
        # Normalize state: frq/1000, pol and is_los unchanged
        state_normalized = state.clone()
        state_normalized[0] = state[0] / 1000.0  # frq divided by 1000
        # pol (index 1) and is_los (index 2) remain unchanged
        
        # Normalize sequences
        height = (height - self.height_mean) / self.height_std
        dist_tx = dist_tx / 10000.0  # dist_tx divided by 10000
        dist_rx = dist_rx / 10000.0  # dist_rx divided by 10000
        
        # Return full sequences (masking knofe_edge_indices and heights which are not used by the model here)
        return state_normalized, height, dist_tx, dist_rx, pathloss

# ---------- Collate function ----------
def collate_varlen_v4(batch):
    """Collate function for VarLenDatasetV4"""
    state_list, height_list, dist_tx_list, dist_rx_list, pathloss_list = zip(*batch)
    state_batch = torch.stack(state_list, dim=0)                    # [B, M_state]
    height_pad = pad_sequence(height_list, batch_first=True)        # [B, N_max, 1]
    dist_tx_pad = pad_sequence(dist_tx_list, batch_first=True)      # [B, N_max, 1]
    dist_rx_pad = pad_sequence(dist_rx_list, batch_first=True)      # [B, N_max, 1]
    pathloss_pad = pad_sequence(pathloss_list, batch_first=True)    # [B, N_max, 1]
    lengths = torch.tensor([h.shape[0] for h in height_list])
    N_max = height_pad.size(1)
    mask = (torch.arange(N_max).unsqueeze(0) < lengths.unsqueeze(1))  # [B, N_max] bool
    return state_batch, height_pad, dist_tx_pad, dist_rx_pad, pathloss_pad, mask

# ---------- Loss functions ----------
def masked_mse(pred, target, mask, reduction="mean"):
    m = mask.unsqueeze(-1).to(pred.dtype)
    diff2 = (pred - target)**2 * m
    if reduction == "sum":
        return diff2.sum()
    elif reduction == "mean":
        denom = m.sum().clamp_min(1.0)
        return diff2.sum() / denom
    elif reduction == "none":
        return diff2
    else:
        raise ValueError("reduction must be 'mean'|'sum'|'none'")

# ---------- Checkpoint functions ----------
@torch.no_grad()
def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)

@torch.no_grad()
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    blob = torch.load(path, map_location=map_location)
    model.load_state_dict(blob["model_state"])
    if optimizer is not None and "optimizer_state" in blob:
        optimizer.load_state_dict(blob["optimizer_state"])
    return blob.get("step", None)

# ---------- Evaluation functions ----------
@torch.no_grad()
def evaluate_v4(model, loader, device):
    """Evaluation function for V4 model - uses last valid position from mask"""
    model.eval()
    total_loss, total_mae, total_count = 0.0, 0.0, 0.0
    for state, height, dist_tx, dist_rx, pathloss, mask in loader:
        state = state.to(device)
        height = height.to(device)
        dist_tx = dist_tx.to(device)
        dist_rx = dist_rx.to(device)
        pathloss = pathloss.to(device)
        mask = mask.to(device)
        
        Y_hat, y_cls = model(state, height, dist_tx, dist_rx, mask=mask)
        
        # Get last valid position for each sample in the batch
        # mask is [B, N_max] where True = valid, False = padding
        lengths = mask.sum(dim=1)  # [B] - actual length of each sequence
        last_indices = (lengths - 1).clamp(min=0)  # [B] - index of last valid position
        
        # Extract last valid pathloss value for each sample (vectorized)
        batch_size = state.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        target_last = pathloss[batch_indices, last_indices, :]  # [B, 1]
        
        loss = torch.nn.functional.mse_loss(y_cls, target_last)
        mae = torch.nn.functional.l1_loss(y_cls, target_last)
        
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_count += batch_size
    
    if total_count == 0:
        return float("nan"), float("nan")
    return total_loss / total_count, total_mae / total_count

@torch.no_grad()
def evaluate_worstcase_v4(model, loader, device):
    """Worst-case evaluation for V4 model - returns distribution of all individual errors"""
    model.eval()
    error_list = []
    
    for state, height, dist_tx, dist_rx, pathloss, mask in loader:
        state = state.to(device)
        height = height.to(device)
        dist_tx = dist_tx.to(device)
        dist_rx = dist_rx.to(device)
        pathloss = pathloss.to(device)
        mask = mask.to(device)
        
        Y_hat, y_cls = model(state, height, dist_tx, dist_rx, mask=mask)
        
        # Get last valid position for each sample in the batch
        lengths = mask.sum(dim=1)  # [B] - actual length of each sequence
        last_indices = (lengths - 1).clamp(min=0)  # [B] - index of last valid position
        
        # Extract last valid pathloss value for each sample (vectorized)
        batch_size = state.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        target_last = pathloss[batch_indices, last_indices, :]  # [B, 1]
        
        err = (y_cls - target_last).abs()  # [B, 1]
        
        # Collect all individual errors (not batch max)
        err_flat = err.squeeze(-1)  # [B]
        error_list.append(err_flat.detach().cpu())
    
    if not error_list:
        return {"max": float("nan"), "p95": float("nan"), "p75": float("nan"),
                "p50": float("nan"), "p20": float("nan"), "min": float("nan"),
                "count": 0}, torch.empty(0)
    
    # Concatenate all errors into a single tensor
    v = torch.cat(error_list, dim=0)  # [total_samples]
    
    stats = {
        "max": v.max().item(),
        "p95": torch.quantile(v, 0.95).item(),
        "p75": torch.quantile(v, 0.75).item(),
        "p50": torch.quantile(v, 0.50).item(),
        "p20": torch.quantile(v, 0.20).item(),
        "min": v.min().item(),
        "mean": v.mean().item(),
        "std": v.std().item(),
        "count": v.numel(),
    }
    return stats, v

# ---------- Main training ----------
if __name__ == "__main__":
    los_label = "LOS" if LOS == 1 else "NLOS"
    print("\n" + "="*80)
    print(f"Training EarlyFusion model on V4 dataset ({los_label} only, variable length)")
    print("="*80 + "\n")
    
    # Find all files matching LOS value (_0.csv or _1.csv) in training and testing directories
    # Data directories
    train_dir1 = "/scratch/tvs9by/ntia/pathprofile/trainingdata/trainingdata4/training"
    train_dir2 = "/scratch/tvs9by/ntia/pathprofile/trainingdata/trainingdata5/training"

    # Find all files matching in both training directories
    train_files1 = glob.glob(os.path.join(train_dir1, "*_oid_*_*.csv"))
    train_files2 = glob.glob(os.path.join(train_dir2, "*_oid_*_*.csv"))
    train_files = sorted(train_files2)
    # Specific evaluation files from reference
    test_files = [
        "/scratch/tvs9by/ntia/pathprofile/trainingdata/testing_v4_bilinear_pid_2_oid_190_los_0.csv"
    ]
    
    print(f"Found {len(train_files)} training {los_label} files")
    print(f"Will evaluate on {len(test_files)} specific CSV files:")
    for f in test_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (NOT FOUND)")
    
    # Create cache directory based on LOS value
    cache_dir = os.path.join(ckpt_dir, "cache_chunksvariedpid")
    
    # Load datasets
    print("\nLoading datasets...")
    
    train_ds_v4_raw = VarLenDatasetV4(filelist=train_files, max_seq_len=max_seq_len, cache_mode=True, cache_dir=cache_dir, apply_oid_filter=True)
    test_ds_v4_raw = VarLenDatasetV4(filelist=test_files, max_seq_len=max_seq_len, cache_mode=True, cache_dir=cache_dir, apply_oid_filter=False)
    
    print(f"Train dataset size: {len(train_ds_v4_raw)}")
    print(f"Test dataset size: {len(test_ds_v4_raw)}")
    
    # Fit normalization stats on train set only
    print("\nComputing normalization statistics...")
    input_stats_v4 = fit_input_stats_v4(train_ds_v4_raw)
    print(f"Height mean: {input_stats_v4['height_mean'].item():.4f}, std: {input_stats_v4['height_std'].item():.4f}")
    
    # Save normalization stats with timestamp for inference
    norm_stats_path = os.path.join(ckpt_dir, f"norm_stats_{ts_v4}_los_{LOS}_v4.pt")
    torch.save(input_stats_v4, norm_stats_path)
    print(f"Saved normalization stats to {norm_stats_path}")
    
    # Apply normalization
    train_ds_v4 = NormalizeInputsV4(train_ds_v4_raw, input_stats_v4)
    test_ds_v4 = NormalizeInputsV4(test_ds_v4_raw, input_stats_v4)
    
    # Create data loaders
    print("Creating data loaders...")
    loader_v4 = DataLoader(train_ds_v4, batch_size=batch_size, shuffle=True, collate_fn=collate_varlen_v4)
    test_loader_v4 = DataLoader(test_ds_v4, batch_size=batch_size, shuffle=False, collate_fn=collate_varlen_v4)
    
    # Initialize model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use max_seq_len for model initialization (though it handles variable lengths)
    model_v4 = CondSeqTransformer_EarlyFusion(
        m_state=m_state_v4,
        seq_len=max_seq_len,
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
    ).to(device)
    
    opt_v4 = torch.optim.AdamW(model_v4.parameters(), lr=2e-5, weight_decay=1e-6)
    
    # Initialize wandb
    if wandbbbbb == 1:
        wandb.login(key='458bc70094e51a25582798845f13b9208bb80cc4')
        cfg_v4 = dict(
            m_state=m_state_v4, 
            max_seq_len=max_seq_len, 
            batch_size=batch_size, 
            lr=2e-4, 
            weight_decay=1e-5, 
            eval_every=eval_every,
            train_files=len(train_files),
            test_files=len(test_files)
        )
        wandb.init(project="my-gpt2-project", config=cfg_v4, name=run_name_v4)
        wandb.watch(model_v4, log="all", log_freq=eval_every)
    
    # Training loop
    print("\nStarting training...")
    model_v4.train()
    step_global_v4 = 0
    
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}/50")
        for step, (state, height, dist_tx, dist_rx, pathloss, mask) in enumerate(loader_v4, start=1):
            step_global_v4 += 1
            
            state = state.to(device)
            height = height.to(device)
            dist_tx = dist_tx.to(device)
            dist_rx = dist_rx.to(device)
            pathloss = pathloss.to(device)
            mask = mask.to(device)
            
            opt_v4.zero_grad()
            Y_hat, y_cls = model_v4(state, height, dist_tx, dist_rx, mask=mask)
            
            # Get last valid position for each sample in the batch
            lengths = mask.sum(dim=1)  # [B] - actual length of each sequence
            last_indices = (lengths - 1).clamp(min=0)  # [B] - index of last valid position
            
            # Extract last valid pathloss value for each sample (vectorized)
            batch_size = state.size(0)
            batch_indices = torch.arange(batch_size, device=device)
            target_last = pathloss[batch_indices, last_indices, :]  # [B, 1]
            
            # Loss: only CLS output vs last pathloss value
            loss = torch.nn.functional.mse_loss(y_cls, target_last)
            loss.backward()
            opt_v4.step()
            
            if wandbbbbb == 1:
                wandb.log({"train/loss": loss.item()}, step=step_global_v4)
            
            if step_global_v4 % eval_every == 0:
                val_loss, val_mae = evaluate_v4(model_v4, test_loader_v4, device)
                if wandbbbbb == 1:
                    wandb.log({"test/loss": val_loss, "test/mae": val_mae}, step=step_global_v4)
                print(f"step {step_global_v4:04d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | val_mae {val_mae:.4f}")
                
                stats, _ = evaluate_worstcase_v4(model_v4, test_loader_v4, device)
                print(
                    f"[error distribution] "
                    f"max={stats['max']:.4f} | p95={stats['p95']:.4f} | p75={stats['p75']:.4f} | "
                    f"p50={stats['p50']:.4f} | p20={stats['p20']:.4f} | min={stats['min']:.4f} | "
                    f"mean={stats['mean']:.4f} | std={stats['std']:.4f} | n={stats['count']}"
                )
                
                if wandbbbbb == 1:
                    wandb.log({
                        "test/error_max": stats["max"],
                        "test/error_p95": stats["p95"],
                        "test/error_p75": stats["p75"],
                        "test/error_p50": stats["p50"],
                        "test/error_p20": stats["p20"],
                        "test/error_min": stats["min"],
                        "test/error_mean": stats["mean"],
                        "test/error_std": stats["std"],
                        "test/examples": stats["count"],
                    }, step=step_global_v4)
                
                save_checkpoint(model_v4, opt_v4, step_global_v4, 
                              os.path.join(ckpt_dir, f"{run_name_v4}_step{step_global_v4}.pt"))
                
                model_v4.train()  # Back to training mode
    
    # Final evaluation
    print("\nFinal evaluation...")
    val_loss, val_mae = evaluate_v4(model_v4, test_loader_v4, device)
    print(f"FINAL | val_loss {val_loss:.4f} | val_mae {val_mae:.4f}")
    if wandbbbbb == 1:
        wandb.log({"test/final_loss": val_loss, "test/final_mae": val_mae}, step=step_global_v4+1)
    
    stats, _ = evaluate_worstcase_v4(model_v4, test_loader_v4, device)
    print(
        f"[error distribution] "
        f"max={stats['max']:.4f} | p95={stats['p95']:.4f} | p75={stats['p75']:.4f} | "
        f"p50={stats['p50']:.4f} | p20={stats['p20']:.4f} | min={stats['min']:.4f} | "
        f"mean={stats['mean']:.4f} | std={stats['std']:.4f} | n={stats['count']}"
    )
    
    if wandbbbbb == 1:
        wandb.log({
            "test/error_max": stats["max"],
            "test/error_p95": stats["p95"],
            "test/error_p75": stats["p75"],
            "test/error_p50": stats["p50"],
            "test/error_p20": stats["p20"],
            "test/error_min": stats["min"],
            "test/error_mean": stats["mean"],
            "test/error_std": stats["std"],
            "test/examples": stats["count"],
        }, step=step_global_v4)
    
    save_checkpoint(model_v4, opt_v4, step_global_v4, 
                  os.path.join(ckpt_dir, f"{run_name_v4}_final_step{step_global_v4}.pt"))
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

