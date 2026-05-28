import torch
import torch.nn as nn
import math
from typing import Tuple

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

class CondSeqTransformerMaskedCLS_withouts(nn.Module):
    """
    Seq-to-seq with a learnable [CLS] token (no separate `s` input).
    Inputs:
      X    : [B, N, dx]       (you can pre-concat system params as tokens)
      mask : [B, N]  bool     (True = valid timestep)
    Returns:
      Y_hat : [B, N, dy]      (per-step outputs)
      y_cls : [B, dy]         (global readout from [CLS])
    """
    def __init__(self, dx, dy, d_model=256, nhead=4, num_layers=4,
                 dim_ff=512, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.dx, self.dy = dx, dy

        self.in_proj = nn.Linear(dx, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, X, mask=None):
        """
        X    : [B, N, dx]
        mask : [B, N] bool  (True = valid timestep)
        """
        B, N, _ = X.shape

        # project + positional encodings
        h = self.in_proj(X)                 # [B,N,d_model]
        h = self.pos(h)                     # [B,N,d_model]

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)   # [B,1,d_model]
        z_in = torch.cat([cls, h], dim=1)       # [B,N+1,d_model]

        # extend masks: key_padding_mask expects True=PAD; our mask=True means VALID → invert
        if mask is not None:
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS always valid
            mask_ext = torch.cat([one, mask], dim=1)                       # [B,N+1] (True=valid)
            key_pad = ~mask_ext                                            # [B,N+1] (True=pad)
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B,N+1,d_model]

        # outputs
        z_cls = z[:, 0, :]          # [B,d_model]
        z_steps = z[:, 1:, :]       # [B,N,d_model]
        Y_hat = self.out_proj(z_steps)   # [B,N,dy]
        y_cls = self.cls_head(z_cls)     # [B,dy]
        return Y_hat, y_cls

class CondSeqTransformerMasked(nn.Module):
    """
    Order-aware conditional seq-to-seq:
      inputs:  s [B,M], X [B,N,dx], mask [B,N] (True=valid)
      outputs: Y [B,N,dy]
    - Adds positional encodings (order sensitivity)
    - Conditions on s via a learned bias added to all tokens
    - Handles variable N with key_padding_mask
    - Optional causal attention (set causal=True)
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=4, num_layers=4, dim_ff=512, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.in_proj  = nn.Linear(dx + m, d_model)
        # self.cond_proj = nn.Linear(m, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, dy))

    def _causal_mask(self, N, device):
        # True where attention is NOT allowed (upper triangle)
        return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, mask=None):
        # X: [B,N,dx], s: [B,M], mask: [B,N] (True=valid)
        B, N, _ = X.shape
        s_exp = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)                        # [B,N,d_model]
        Xc = torch.cat([X, s_exp], dim=-1)
        h = self.in_proj(Xc)
        h = self.pos(h)                              # add positional encodings

        # key_padding_mask expects True for PAD positions; our mask=True means VALID → invert it
        key_pad = None if mask is None else ~mask    # [B,N], True=pad
        attn_mask = self._causal_mask(N, X.device) if self.causal else None  # [N,N] or None

        z = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_pad)    # [B,N,d_model]
        y = self.out_proj(z)                           # [B,N,dy]
        return y

class CondSeqTransformerMasked2(nn.Module):
    """
    Order-aware conditional seq-to-seq:
      inputs:  s [B,M], X [B,N,dx], mask [B,N] (True=valid)
      outputs: Y [B,N,dy]
    - Adds positional encodings (order sensitivity)
    - Conditions on s via a learned bias added to all tokens
    - Handles variable N with key_padding_mask
    - Optional causal attention (set causal=True)
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=4, num_layers=4, dim_ff=512, dropout=0.1, causal=False, prefix_k =4):
        super().__init__()
        self.causal = causal
        self.in_proj  = nn.Linear(dx + m, d_model)
        # self.cond_proj = nn.Linear(m, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.prefix_k = prefix_k
        self.out_proj = nn.Sequential(nn.Linear(d_model + self.prefix_k, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, dy))
        

    def _causal_mask(self, N, device):
        # True where attention is NOT allowed (upper triangle)
        return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, mask=None, y_prefix = None):
        # X: [B,N,dx], s: [B,M], mask: [B,N] (True=valid)
        B, N, _ = X.shape
        s_exp = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)                        # [B,N,d_model]
        Xc = torch.cat([X, s_exp], dim=-1)
        h = self.in_proj(Xc)
        h = self.pos(h)                              # add positional encodings

        # key_padding_mask expects True for PAD positions; our mask=True means VALID → invert it
        key_pad = None if mask is None else ~mask    # [B,N], True=pad
        attn_mask = self._causal_mask(N, X.device) if self.causal else None  # [N,N] or None

        z = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_pad)    # [B,N,d_model]

        # flatten first K*dy, pad to fixed width (K=self.prefix_k)
        v4 = y_prefix.to(z.dtype).to(z.device)
        v4_exp = v4.unsqueeze(1).expand(B, N, self.prefix_k)
        z = torch.cat([z, v4_exp], dim=-1)
        y = self.out_proj(z)                           # [B,N,dy]
        return y

class CondSeqTransformerMasked3(nn.Module):
    """
    Order-aware conditional seq-to-seq:
      inputs:  s [B,M], X [B,N,dx], mask [B,N] (True=valid)
      outputs: Y [B,N,dy]
    - Adds positional encodings (order sensitivity)
    - Conditions on s via a learned bias added to all tokens
    - Handles variable N with key_padding_mask
    - Optional causal attention (set causal=True)
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=4, num_layers=4, dim_ff=512, dropout=0.1, causal=False, prefix_k =4):
        super().__init__()
        self.causal = causal
        self.in_proj  = nn.Linear(dx + m, d_model)
        # self.cond_proj = nn.Linear(m, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.prefix_k = prefix_k
        self.out_proj = nn.Sequential(nn.Linear(d_model + self.prefix_k, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, dy))
        

    def _causal_mask(self, N, device):
        # True where attention is NOT allowed (upper triangle)
        return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, mask=None, y_prefix = None):
        # X: [B,N,dx], s: [B,M], mask: [B,N] (True=valid)
        B, N, _ = X.shape
        s_exp = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)                        # [B,N,d_model]
        Xc = torch.cat([X, s_exp], dim=-1)
        h = self.in_proj(Xc)
        h = self.pos(h)                              # add positional encodings

        # key_padding_mask expects True for PAD positions; our mask=True means VALID → invert it
        key_pad = None if mask is None else ~mask    # [B,N], True=pad
        attn_mask = self._causal_mask(N, X.device) if self.causal else None  # [N,N] or None

        z = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_pad)    # [B,N,d_model]

        # flatten first K*dy, pad to fixed width (K=self.prefix_k)
        v4 = y_prefix.to(z.dtype).to(z.device)
        v4_exp = v4.unsqueeze(1).expand(B, N, self.prefix_k)
        z = torch.cat([z, v4_exp], dim=-1)
        y = self.out_proj(z)                           # [B,N,dy]
        return y

import torch
import torch.nn as nn

class CondSeqTransformerEncDec(nn.Module):
    """
    Encoder–decoder transformer for continuous seq2seq regression.

    Inputs:
      s        : [B, M]           (conditioning vector)
      X        : [B, N, dx]       (source sequence)
      Y_inp    : [B, N, dy]       (teacher-forced target inputs; BOS + Y[:-1])
      src_mask : [B, N] bool      (True = valid)
      tgt_mask : [B, N] bool      (True = valid)

    Output:
      Y_hat    : [B, N, dy]
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=8, num_layers=8, dim_ff=512, dropout=0.1):
        super().__init__()
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model

        # project [token || s] -> d_model for src/tgt
        self.in_proj_src = nn.Linear(dx + m, d_model)
        self.in_proj_tgt = nn.Linear(dy + m, d_model)

        self.pos_src = SinusoidalPositionalEncoding(d_model)
        self.pos_tgt = SinusoidalPositionalEncoding(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        dec = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )

        # Learned BOS for continuous targets
        self.bos = nn.Parameter(torch.zeros(1, 1, dy))

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, Y_inp, src_mask=None, tgt_mask=None):
        """
        s: [B,M], X: [B,N,dx], Y_inp: [B,N,dy]
        src_mask/tgt_mask: True = valid (we invert for key_padding_mask)
        """
        B, N, _ = X.shape
        device = X.device

        # ----- Encoder -----
        s_src = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)     # [B,N,M]
        src = torch.cat([X, s_src], dim=-1)                     # [B,N,dx+M]
        h_src = self.pos_src(self.in_proj_src(src))
        src_key_pad = None if src_mask is None else ~src_mask   # True=pad
        memory = self.encoder(h_src, mask=None, src_key_padding_mask=src_key_pad)

        # ----- Decoder (causal) -----
        s_tgt = s.to(Y_inp.dtype).unsqueeze(1).expand(B, N, -1) # [B,N,M]
        tgt = torch.cat([Y_inp, s_tgt], dim=-1)                 # [B,N,dy+M]
        h_tgt = self.pos_tgt(self.in_proj_tgt(tgt))
        tgt_key_pad = None if tgt_mask is None else ~tgt_mask   # True=pad
        dec_mask = self._causal_mask(N, device)

        z = self.decoder(
            h_tgt, memory,
            tgt_mask=dec_mask,
            tgt_key_padding_mask=tgt_key_pad,
            memory_key_padding_mask=src_key_pad
        )
        return self.out_proj(z)                                  # [B,N,dy]

    @torch.no_grad()
    def predict_autoregressive(self, s, X, N, src_mask=None):
        """
        Greedy generation of length N (no teacher forcing).
        """
        self.eval()
        B = X.size(0); device = X.device

        # Encode once
        s_src = s.to(X.dtype).unsqueeze(1).expand(B, X.size(1), -1)
        h_src = self.pos_src(self.in_proj_src(torch.cat([X, s_src], dim=-1)))
        src_key_pad = None if src_mask is None else ~src_mask
        memory = self.encoder(h_src, src_key_padding_mask=src_key_pad)

        y_seq = self.bos.expand(B, 1, self.dy).to(device)  # [B,1,dy]
        for _ in range(N):
            s_t = s.to(y_seq.dtype).unsqueeze(1).expand(B, y_seq.size(1), -1)
            h_tgt = self.pos_tgt(self.in_proj_tgt(torch.cat([y_seq, s_t], dim=-1)))
            dec_mask = self._causal_mask(h_tgt.size(1), device)
            z = self.decoder(h_tgt, memory, tgt_mask=dec_mask, memory_key_padding_mask=src_key_pad)
            y_next = self.out_proj(z)[:, -1:, :]           # last step
            y_seq = torch.cat([y_seq, y_next], dim=1)
        return y_seq[:, 1:, :]  # drop BOS

    @torch.no_grad()
    def predict_autoregressive_prefix(self, s, X, N, src_mask=None, Y_prefix=None):
        """
        Greedy generation of length N.

        Args:
        s         : [B, M]
        X         : [B, Nx, dx]
        N         : int, number of target steps to produce
        src_mask  : [B, Nx] bool, True = valid (same as train)
        Y_prefix  : Optional known labels to seed the decoder.
                    Shape [B, K, dy] or [B, dy] (K>=1). If provided, the first K
                    outputs will be exactly this prefix, and decoding continues
                    from step K.

        Returns:
        Y_gen     : [B, N, dy]
                    - If Y_prefix is given, Y_gen[:, :K, :] == Y_prefix.
                    - If Y_prefix is None, standard BOS-start decoding.
        """
        self.eval()
        B, Nx, _ = X.shape
        device = X.device

        # ---------- Encode source once ----------
        s_src = s.to(X.dtype).unsqueeze(1).expand(B, Nx, -1)
        h_src = self.pos_src(self.in_proj_src(torch.cat([X, s_src], dim=-1)))
        src_key_pad = None if src_mask is None else ~src_mask  # True=pad
        memory = self.encoder(h_src, src_key_padding_mask=src_key_pad)

        # ---------- Initialize decoder sequence ----------
        if Y_prefix is not None:
            # Accept [B, dy] or [B, K, dy]
            if Y_prefix.dim() == 2:
                Y_prefix = Y_prefix.unsqueeze(1)  # -> [B,1,dy]
            y_seq = Y_prefix.to(device).clone()    # [B,K,dy]
            # If prefix already meets/exceeds N, just truncate and return.
            if y_seq.size(1) >= N:
                return y_seq[:, :N, :]
            target_len = N                    # grow from K -> N
            drop_first_token = False          # we keep the whole sequence
        else:
            # No prefix: start from BOS, then drop it at the end (match old behavior)
            y_seq = self.bos.expand(B, 1, self.dy).to(device)  # [B,1,dy]
            target_len = N + 1              # grow from 1 -> N+1, then strip BOS
            drop_first_token = True

        # ---------- Autoregressive loop ----------
        while y_seq.size(1) < target_len:
            Tt = y_seq.size(1)
            s_t = s.to(y_seq.dtype).unsqueeze(1).expand(B, Tt, -1)
            h_tgt = self.pos_tgt(self.in_proj_tgt(torch.cat([y_seq, s_t], dim=-1)))
            dec_mask = self._causal_mask(Tt, device)
            z = self.decoder(
                h_tgt, memory,
                tgt_mask=dec_mask,
                memory_key_padding_mask=src_key_pad
            )
            y_next = self.out_proj(z)[:, -1:, :]   # last step prediction
            y_seq = torch.cat([y_seq, y_next], dim=1)

        # ---------- Return ----------
        if drop_first_token:
            # started from BOS → drop it
            return y_seq[:, 1:, :]    # [B, N, dy]
        else:
            # started from prefix → keep it
            return y_seq[:, :N, :]    # [B, N, dy]

class CondSeqTransformerMaskedCLS(nn.Module):
    """
    Like CondSeqTransformerMasked but with a learnable [CLS] token.
    Returns:
      Y_hat : [B, N, dy]   (per-step outputs, same as before)
      y_cls : [B, dy]      (global readout from [CLS])
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=4, num_layers=4,
                 dim_ff=512, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy

        self.in_proj = nn.Linear(dx + m, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, mask=None):
        """
        s    : [B, M]
        X    : [B, N, dx]
        mask : [B, N] bool  (True = valid timestep)
        """
        B, N, _ = X.shape

        # concat conditioning to each token, then project + add positional encodings
        s_exp = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)         # [B,N,M]
        h = self.in_proj(torch.cat([X, s_exp], dim=-1))             # [B,N,d]
        h = self.pos(h)                                             # [B,N,d]

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)                       # [B,1,d]
        z_in = torch.cat([cls, h], dim=1)                           # [B,N+1,d]

        # extend masks: key_padding_mask expects True=PAD; your mask=True means VALID → invert
        if mask is not None:
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS is always valid
            mask_ext = torch.cat([one, mask], dim=1)                       # [B,N+1] (True=valid)
            key_pad = ~mask_ext                                            # [B,N+1] (True=pad)
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), X.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B,N+1,d]

        # outputs
        z_cls = z[:, 0, :]                         # [B,d]
        z_steps = z[:, 1:, :]                      # [B,N,d]
        Y_hat = self.out_proj(z_steps)             # [B,N,dy]
        y_cls = self.cls_head(z_cls)               # [B,dy]
        return Y_hat, y_cls

class SwitchMoE(nn.Module):
    """
    Mixture-of-Experts FFN with top-1 routing (Switch Transformers).
    - Experts: simple MLPs (Linear -> GELU -> Linear)
    - Gating: Linear(d_model -> n_experts), softmax -> pick argmax per token
    - Aux loss: load-balancing per paper (encourage even assignment & importance)

    Args:
      d_model, d_hidden: FFN sizes
      n_experts: number of experts
      capacity_factor: optional token capacity per expert (relative). If None, no cap.
      dropout: dropout inside experts
    """
    def __init__(self, d_model: int, d_hidden: int, n_experts: int,
                 capacity_factor: float = None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model),
            )
            for _ in range(n_experts)
        ])

    @torch.no_grad()
    def _compute_aux_loss(self, gates_softmax: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        """
        gates_softmax: [B*T, E] softmax probs
        expert_idx:    [B*T]    chosen expert ids (hard)
        L_aux = E * sum( importance * load ), where:
          importance_e = mean(prob_e)
          load_e       = mean(1[expert_idx == e])
        """
        E = gates_softmax.size(-1)
        importance = gates_softmax.mean(dim=0)              # [E]
        # histogram of assignments:
        load = torch.bincount(expert_idx, minlength=E).to(gates_softmax.dtype) / gates_softmax.size(0)
        aux = E * torch.sum(importance * load)              # scalar
        return aux

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x          : [B, T, d_model]
        valid_mask : [B, T] bool  (True = valid token; False = pad). CLS should be True.

        Returns:
          y:      [B, T, d_model]
          aux_ls: scalar load-balancing loss
        """
        B, T, D = x.shape
        device = x.device
        E = self.n_experts

        # Flatten valid tokens; keep pad positions to zero out later
        x_flat = x.reshape(B * T, D)                        # [BT, D]
        v_flat = valid_mask.reshape(B * T)                  # [BT]

        # Gating
        logits = self.gate(x_flat)                          # [BT, E]
        probs  = torch.softmax(logits, dim=-1)              # [BT, E]
        top1   = torch.argmax(probs, dim=-1)                # [BT]
        aux_loss = self._compute_aux_loss(probs[v_flat], top1[v_flat]) if v_flat.any() else x_flat.new_zeros([])

        # Optional capacity (drop overflow) — simple dynamic cap per batch
        if self.capacity_factor is not None and v_flat.any():
            # tokens per expert (valid only)
            counts = torch.bincount(top1[v_flat], minlength=E)
            cap = torch.ceil(counts.float() * self.capacity_factor).to(torch.int64)  # [E]
            # build per-expert running counters and a keep mask
            keep = torch.zeros_like(v_flat, dtype=torch.bool)
            counters = torch.zeros(E, dtype=torch.int64, device=device)
            # Iterate valid positions and mark keep if under capacity (O(N))
            idxs = torch.nonzero(v_flat, as_tuple=False).squeeze(1)
            for i in idxs:
                e = top1[i].item()
                if counters[e] < cap[e]:
                    keep[i] = True
                    counters[e] += 1
            route_mask = keep
        else:
            route_mask = v_flat  # keep all valid tokens

        # Prepare output buffer
        y_flat = torch.zeros_like(x_flat)                   # [BT, D]

        # Dispatch per expert (process only routed tokens)
        for e, expert in enumerate(self.experts):
            token_idx = torch.nonzero((top1 == e) & route_mask, as_tuple=False).squeeze(1)
            if token_idx.numel() == 0:
                continue
            y_flat[token_idx] = expert(x_flat[token_idx])

        # Reshape back and zero out pad positions (and any dropped ones)
        y = y_flat.view(B, T, D)
        y = y * valid_mask.unsqueeze(-1)                    # zero padded / dropped
        return y, aux_loss

# ---------- A Transformer block with MoE FFN (pre-norm, batch_first) ----------
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, nhead, d_hidden, dropout=0.1,
                 n_experts=4, capacity_factor=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe   = SwitchMoE(d_model, d_hidden, n_experts, capacity_factor, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, valid_mask=None):
        # x: [B,T,D], key_padding_mask: [B,T] True=PAD, valid_mask: [B,T] True=valid
        # Self-attention (pre-norm)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(a)

        # MoE FFN (pre-norm)
        h = self.norm2(x)
        y, aux = self.moe(h, valid_mask=valid_mask if valid_mask is not None else ~key_padding_mask)
        x = x + self.drop2(y)
        return x, aux

# ------------- Full model: CLS + Transformer(MoE FFN) encoder -----------------
class CondSeqTransformerMaskedCLS_SwitchMoE(nn.Module):
    """
    Encoder with learnable [CLS] + MoE-FFN blocks (Switch-Transformer style).

    Returns:
      Y_hat : [B, N, dy]  per-step outputs
      y_cls : [B, dy]     CLS readout (train vs Y[:, -1, :])
      aux_moe_loss: scalar (add to main loss with a small coefficient)
    """
    def __init__(self, m, dx, dy, d_model=256, nhead=4, num_layers=4,
                 dim_ff=512, dropout=0.1, causal=False,
                 n_experts=4, capacity_factor=None):
        super().__init__()
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model
        self.causal = causal

        # token projection & positional enc
        self.in_proj = nn.Linear(dx + m, d_model)
        self.pos     = SinusoidalPositionalEncoding(d_model)

        # CLS token & heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Stack of MoE blocks
        self.blocks = nn.ModuleList([
            TransformerBlockWithMoE(
                d_model=d_model, nhead=nhead, d_hidden=dim_ff,
                dropout=dropout, n_experts=n_experts, capacity_factor=capacity_factor
            )
            for _ in range(num_layers)
        ])

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, mask=None):
        """
        s    : [B, M]         conditioning vector
        X    : [B, N, dx]     source seq
        mask : [B, N] bool    True = valid timestep (padding=False)
        """
        B, N, _ = X.shape
        s_exp = s.to(X.dtype).unsqueeze(1).expand(B, N, -1)    # [B,N,M]
        h = self.in_proj(torch.cat([X, s_exp], dim=-1))        # [B,N,d]
        h = self.pos(h)                                        # [B,N,d]

        # prepend CLS
        cls = self.cls_token.expand(B, 1, -1)                  # [B,1,d]
        z = torch.cat([cls, h], dim=1)                         # [B,N+1,d]

        # masks
        if mask is not None:
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS valid
            mask_ext = torch.cat([one, mask], dim=1)            # [B,N+1] True=valid
            key_pad   = ~mask_ext                                # [B,N+1] True=pad
            valid     = mask_ext
        else:
            key_pad = None
            valid   = torch.ones(B, z.size(1), dtype=torch.bool, device=z.device)
        attn_mask = self._causal_mask(z.size(1), z.device) if self.causal else None

        # run blocks & accumulate MoE aux loss
        aux_total = z.new_zeros([])
        for blk in self.blocks:
            z, aux = blk(z, attn_mask=attn_mask, key_padding_mask=key_pad, valid_mask=valid)
            aux_total = aux_total + aux

        # outputs
        z_cls   = z[:, 0, :]           # [B,d]
        z_steps = z[:, 1:, :]          # [B,N,d]
        Y_hat   = self.out_proj(z_steps)      # [B,N,dy]
        y_cls   = self.cls_head(z_cls)        # [B,dy]
        return Y_hat, y_cls, aux_total

import torch
import torch.nn as nn

class CondSeqTransformerMaskedCLS_withsxttoken(nn.Module):
    """
    Like before, but now s, X, and Y_inp each have their own embedding layers.
    Forward expects mask over [s_tokens || X || Y_inp] (True = valid).
    Returns:
      Y_hat : [B, N, dy]   (preds aligned to X positions)
      y_cls : [B, dy]      (global readout from [CLS])
    """
    def __init__(
        self,
        m, dx, dy,
        d_model=256, nhead=4, num_layers=4,
        dim_ff=512, dropout=0.1, causal=False,
        t_s=4  # number of tokens to generate from s (matches mask[:,:4])
    ):
        super().__init__()
        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model
        self.t_s = t_s

        # Separate projections
        #   s -> [B, t_s, d]
        self.in_proj_s = nn.Linear(m, t_s * d_model)
        #   X -> [B, N, d]
        self.in_proj_x = nn.Linear(dx, d_model)
        #   Y_inp (prefix) -> [B, k_in, d]
        self.in_proj_y = nn.Linear(dy, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, Y_inp, mask=None):
        """
        s     : [B, M]
        X     : [B, N, dx]
        Y_inp : [B, k_in, dy]   (prefix of Y you supply)
        mask  : [B, t_s + N + k_in] bool (True = valid)
                Corresponds to [s_tokens || X || Y_inp]
        """
        B, N, _ = X.shape
        k_in = Y_inp.size(1)

        # --- Embed each stream ---
        # s -> t_s tokens
        s_tok = self.in_proj_s(s).view(B, self.t_s, self.d_model)     # [B, t_s, d]
        x_tok = self.in_proj_x(X)                                     # [B, N, d]
        y_tok = self.in_proj_y(Y_inp)                                 # [B, k_in, d]

        # concat (without CLS), then add positional encodings
        h_no_cls = torch.cat([s_tok, x_tok, y_tok], dim=1)            # [B, t_s+N+k_in, d]
        h_no_cls = self.pos(h_no_cls)

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)                         # [B,1,d]
        z_in = torch.cat([cls, h_no_cls], dim=1)                      # [B, 1+t_s+N+k_in, d]

        # extend masks: key_padding_mask expects True=PAD; your mask=True means VALID → invert
        if mask is not None:
            assert mask.shape[1] == self.t_s + N + k_in, \
                f"mask length {mask.shape[1]} must equal t_s({self.t_s})+N({N})+k_in({k_in})"
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS is always valid
            mask_ext = torch.cat([one, mask], dim=1)                       # [B, 1+t_s+N+k_in] (True=valid)
            key_pad = ~mask_ext                                            # [B, 1+t_s+N+k_in] (True=pad)
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 1+t_s+N+k_in, d]

        # indices:
        #  0                 -> CLS
        #  1 .. t_s          -> s tokens
        #  t_s+1 .. t_s+N    -> X tokens (we predict per-step here)
        #  t_s+N+1 .. end    -> Y_inp tokens
        start_x = 1 + self.t_s
        end_x   = start_x + N

        z_cls   = z[:, 0, :]                         # [B, d]
        z_x     = z[:, start_x:end_x, :]             # [B, N, d]

        Y_hat = self.out_proj(z_x)                   # [B, N, dy]
        y_cls = self.cls_head(z_cls)                 # [B, dy]
        return Y_hat, y_cls

class CondSeqTransformerMaskedCLS_withsxttoken_withdiff(nn.Module):
    """
    Like before, but now s, X, and Y_inp each have their own embedding layers.
    Forward expects mask over [s_tokens || X || Y_inp] (True = valid).
    Returns:
      Y_hat : [B, N, dy]   (preds aligned to X positions)
      y_cls : [B, dy]      (global readout from [CLS])
    """
    def __init__(
        self,
        m, dx, dy,
        d_model=256, nhead=4, num_layers=4,
        dim_ff=512, dropout=0.1, causal=False,
        t_s=4  # number of tokens to generate from s (matches mask[:,:4])
    ):
        super().__init__()
        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model
        self.t_s = t_s

        # Separate projections
        #   s -> [B, t_s, d]
        self.in_proj_s = nn.Linear(m, t_s * d_model)
        #   X -> [B, N, d]
        self.in_proj_x = nn.Linear(dx, d_model)
        #   Y_inp (prefix) -> [B, k_in, d]
        self.in_proj_y = nn.Linear(dy, d_model)
        self.in_proj_x_diff = nn.Linear(dx, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, Y_inp, X_diff, mask=None):
        """
        s     : [B, M]
        X     : [B, N, dx]
        Y_inp : [B, k_in, dy]   (prefix of Y you supply)
        mask  : [B, t_s + N + k_in] bool (True = valid)
                Corresponds to [s_tokens || X || Y_inp]
        """
        B, N, _ = X.shape
        k_in = Y_inp.size(1)

        # --- Embed each stream ---
        # s -> t_s tokens
        s_tok = self.in_proj_s(s).view(B, self.t_s, self.d_model)     # [B, t_s, d]
        x_tok = self.in_proj_x(X)                                     # [B, N, d]
        y_tok = self.in_proj_y(Y_inp)                                 # [B, k_in, d]
        x_diff_tok = self.in_proj_x_diff(X_diff)

        # concat (without CLS), then add positional encodings
        h_no_cls = torch.cat([s_tok, x_tok, x_diff_tok, y_tok], dim=1)            # [B, t_s+N+k_in, d]
        h_no_cls = self.pos(h_no_cls)

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)                         # [B,1,d]
        z_in = torch.cat([cls, h_no_cls], dim=1)                      # [B, 1+t_s+N+k_in, d]

        # extend masks: key_padding_mask expects True=PAD; your mask=True means VALID → invert
        if mask is not None:
            # assert mask.shape[1] == self.t_s + N + k_in + 99, \
            #     f"mask length {mask.shape[1]} must equal t_s({self.t_s})+N({N})+k_in({k_in})"
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS is always valid
            mask_ext = torch.cat([one, mask], dim=1)                       # [B, 1+t_s+N+k_in] (True=valid)
            key_pad = ~mask_ext                                            # [B, 1+t_s+N+k_in] (True=pad)
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 1+t_s+N+k_in, d]

        # indices:
        #  0                 -> CLS
        #  1 .. t_s          -> s tokens
        #  t_s+1 .. t_s+N    -> X tokens (we predict per-step here)
        #  t_s+N+1 .. end    -> Y_inp tokens
        start_x = 1 + self.t_s
        end_x   = start_x + N

        z_cls   = z[:, 0, :]                         # [B, d]
        z_x     = z[:, start_x:end_x, :]             # [B, N, d]

        Y_hat = self.out_proj(z_x)                   # [B, N, dy]
        y_cls = self.cls_head(z_cls)                 # [B, dy]
        return Y_hat, y_cls

class CondSeqTransformerMaskedCLS_withsxttoken_withdiff1(nn.Module):
    """
    Like before, but now s, X, and Y_inp each have their own embedding layers.
    Forward expects mask over [s_tokens || X || Y_inp] (True = valid).
    Returns:
      Y_hat : [B, N, dy]   (preds aligned to X positions)
      y_cls : [B, dy]      (global readout from [CLS])
    """
    def __init__(
        self,
        m, dx, dy,
        d_model=256, nhead=4, num_layers=4,
        dim_ff=512, dropout=0.1, causal=False,
        t_s=4  # number of tokens to generate from s (matches mask[:,:4])
    ):
        super().__init__()
        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model
        self.t_s = t_s

        # Separate projections
        #   s -> [B, t_s, d]
        self.in_proj_s = nn.Linear(m, t_s * d_model)
        #   X -> [B, N, d]
        self.in_proj_x = nn.Linear(dx, d_model)
        #   Y_inp (prefix) -> [B, k_in, d]
        self.in_proj_y = nn.Linear(dy, d_model)
        self.in_proj_x_diff = nn.Linear(dx, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, Y_inp, X_diff, mask=None):
        """
        s     : [B, M]
        X     : [B, N, dx]
        Y_inp : [B, k_in, dy]   (prefix of Y you supply)
        mask  : [B, t_s + N + k_in] bool (True = valid)
                Corresponds to [s_tokens || X || Y_inp]
        """
        B, N, _ = X.shape
        k_in = Y_inp.size(1)

        # --- Embed each stream ---
        # s -> t_s tokens
        s_tok = self.in_proj_s(s).view(B, self.t_s, self.d_model)     # [B, t_s, d]
        x_tok = self.in_proj_x(X)                                     # [B, N, d]
        # y_tok = self.in_proj_y(Y_inp)                                 # [B, k_in, d]
        x_diff_tok = self.in_proj_x_diff(X_diff)
        x_pair = torch.stack([x_tok, x_diff_tok], dim=2).reshape(B, 2*N, self.d_model)
        h_no_cls = torch.cat([s_tok, x_pair], dim=1)
        # concat (without CLS), then add positional encodings
        # h_no_cls = torch.cat([s_tok, x_tok, x_diff_tok, y_tok], dim=1)            # [B, t_s+N+k_in, d]
        h_no_cls = self.pos(h_no_cls)

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)                         # [B,1,d]
        sep = self.sep_token.expand(B, 1, -1)
        z_in = torch.cat([cls, s_tok, sep, x_pair], dim=1)                      # [B, 1+t_s+N+k_in, d]

        # extend masks: key_padding_mask expects True=PAD; your mask=True means VALID → invert
        if mask is not None:
            # assert mask.shape[1] == self.t_s + N + k_in + 99, \
            #     f"mask length {mask.shape[1]} must equal t_s({self.t_s})+N({N})+k_in({k_in})"
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)  # CLS is always valid
            mask_ext = torch.cat([one, mask], dim=1)                       # [B, 1+t_s+N+k_in] (True=valid)
            key_pad = ~mask_ext                                            # [B, 1+t_s+N+k_in] (True=pad)
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 1+t_s+N+k_in, d]

        # indices:
        #  0                 -> CLS
        #  1 .. t_s          -> s tokens
        #  t_s+1 .. t_s+N    -> X tokens (we predict per-step here)
        #  t_s+N+1 .. end    -> Y_inp tokens
        start_x = 1 + self.t_s
        end_x   = start_x + N

        z_cls   = z[:, 0, :]                         # [B, d]
        z_x     = z[:, start_x:end_x, :]             # [B, N, d]

        Y_hat = self.out_proj(z_x)                   # [B, N, dy]
        y_cls = self.cls_head(z_cls)                 # [B, dy]
        return Y_hat, y_cls

class CondSeqTransformerMaskedCLS_ConcatPerStep(nn.Module):
    """
    Each per-step token = concat([s_chunk0, s_chunk1, s_chunk2, s_chunk3, x_i, xdiff_i])
    Sizes: 64 + 64 + 64 + 64 + 128 + 128 = 512 (= d_model)
    Also supports Y_inp prefix tokens projected to d_model.
    Returns:
      Y_hat : [B, N, dy]  (aligned to the N per-step tokens)
      y_cls : [B, dy]
    """
    def __init__(self, m, dx, dy,
                 d_model=512, nhead=8, num_layers=6,
                 dim_ff=1024, dropout=0.1, causal=False,
                 t_s=4, s_chunk_dim=64, x_dim=128, xdiff_dim=128):
        super().__init__()
        assert t_s * s_chunk_dim + x_dim + xdiff_dim == d_model, \
            f"d_model must equal t_s*s_chunk_dim + x_dim + xdiff_dim, got {d_model} vs {t_s*s_chunk_dim + x_dim + xdiff_dim}"

        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy
        self.t_s = t_s
        self.s_chunk_dim = s_chunk_dim
        self.x_dim = x_dim
        self.xdiff_dim = xdiff_dim
        self.d_model = d_model

        # s -> [B, t_s*s_chunk_dim] then view to [B, t_s, s_chunk_dim]
        self.in_proj_s = nn.Linear(t_s, t_s * s_chunk_dim)

        # X and X_diff -> 128 each (configurable)
        self.in_proj_x      = nn.Linear(dx, x_dim)
        self.in_proj_x_diff = nn.Linear(dx, xdiff_dim)

        # Y prefix tokens -> d_model directly
        self.in_proj_y = nn.Linear(dy, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pos = SinusoidalPositionalEncoding(d_model)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, Y_inp, X_diff, mask=None):
        """
        s      : [B, M]
        X      : [B, N, dx]
        Y_inp  : [B, k_in, dy]
        X_diff : [B, N, dx]
        mask   :
          - preferred: [B, N + k_in] over [X || Y_inp], True=valid
          - legacy   : [B, t_s + N + N + k_in] over [s || X || X_diff || Y_inp]; we’ll fold to [N + k_in]
        """
        B, N, _ = X.shape
        k_in = Y_inp.size(1)

        # --- build per-step tokens by concatenation ---
        # s -> [B, t_s, s_chunk_dim] then flatten to [B, t_s*s_chunk_dim] and broadcast to N
        s_chunks = self.in_proj_s(s).view(B, self.t_s, self.s_chunk_dim)  # [B, 4, 64]
        s_block  = s_chunks.reshape(B, self.t_s * self.s_chunk_dim)       # [B, 256]
        s_rep    = s_block.unsqueeze(1).expand(B, N, -1)                   # [B, N, 256]

        # X and X_diff parts
        x_feat     = self.in_proj_x(X)                                     # [B, N, 128]
        xdiff_feat = self.in_proj_x_diff(X_diff)                           # [B, N, 128]

        # concat -> [B, N, 512]
        step_tokens = torch.cat([s_rep, x_feat, xdiff_feat], dim=-1)       # [B, N, d_model]

        # Y prefix tokens directly to d_model
        # y_tok = self.in_proj_y(Y_inp)                                      # [B, k_in, d_model]

        # assemble sequence: [CLS] || step_tokens (N) || y_tok (k_in)
        # seq_no_cls = torch.cat([step_tokens, y_tok], dim=1)                # [B, N+k_in, d_model]
        seq_no_cls = self.pos(step_tokens)

        cls = self.cls_token.expand(B, 1, -1)                              # [B,1,d_model]
        z_in = torch.cat([cls, seq_no_cls], dim=1)                         # [B, 1+N+k_in, d_model]

        # --- masks ---
        key_pad = None
        if mask is not None:
            # Accept either compact or legacy layout
            if mask.size(1) == N + k_in:
                m_step = mask[:, :N]                                       # [B,N]
                m_y    = mask[:, N:N+k_in]                                 # [B,k_in]
            else:
                # legacy: [s(4) || X || X_diff || Y]
                assert mask.size(1) >= self.t_s + N + N + k_in, \
                    f"mask length {mask.size(1)} doesn't match legacy layout t_s+N+N+k_in"
                m_step = mask[:, self.t_s:self.t_s+N]                      # take X as step validity
                # (optional) combine with X_diff: m_step = m_step & mask[:, self.t_s+N:self.t_s+2*N]
                m_y    = mask[:, self.t_s+2*N:self.t_s+2*N+k_in]
            mask_compact = torch.cat([m_step, m_y], dim=1)                  # [B, N+k_in]
            ones = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)   # CLS valid
            mask_ext = torch.cat([ones, mask_compact], dim=1)               # [B, 1+N+k_in]
            key_pad = ~mask_ext                                             # True = pad

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # --- encode ---
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 1+N+k_in, d_model]

        # outputs: take the N per-step positions right after CLS
        z_cls = z[:, 0, :]                                                  # [B, d_model]
        z_steps = z[:, 1:1+N, :]                                            # [B, N, d_model]

        Y_hat = self.out_proj(z_steps)                                       # [B, N, dy]
        y_cls = self.cls_head(z_cls)                                         # [B, dy]
        return Y_hat, y_cls


class CondSeqTransformerMaskedCLS_withsxttoken_withpathloss(nn.Module):
    """
    Now uses: s, X, pathloss
      - s produces t_s tokens
      - X has length N
      - pathloss has length N-1 (one shorter than X)
    Token order fed to encoder:
      [ s_tokens,  X1,PL1, X2,PL2, ... , X(N-1),PL(N-1), XN ]
    Returns:
      Y_hat : [B, N, dy]   (per-step outputs aligned to X positions)
    """
    def __init__(
        self,
        m, dx, dy,
        d_model=256, nhead=4, num_layers=4,
        dim_ff=512, dropout=0.1, causal=False,
        t_s=4,          # number of s-tokens
        d_pl=1          # pathloss feature dim (often 1)
    ):
        super().__init__()
        self.causal = causal
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model = d_model
        self.t_s = t_s

        # Projections
        self.in_proj_s  = nn.Linear(m, t_s * d_model)   # s -> [B, t_s, d]
        self.in_proj_x  = nn.Linear(dx, d_model)        # X -> [B, N, d]
        self.in_proj_pl = nn.Linear(d_pl, d_model)      # pathloss -> [B, N-1, d]

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Head for per-X-step predictions
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )

    @staticmethod
    def _causal_mask(T, device):
        # True where attention is disallowed (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X, pathloss, mask=None):
        """
        s        : [B, M]
        X        : [B, N, dx]
        pathloss : [B, N-1, d_pl]  (one shorter than X)
        mask     : [B, t_s + (2*(N-1) + 1)] bool (True = valid)
                   corresponds to [ s_tokens, X1,PL1, X2,PL2, ..., X_{N-1},PL_{N-1}, XN ]
        """
        B, N, _ = X.shape
        assert pathloss.size(1) == N - 1, "pathloss must be one shorter than X (N-1)."

        # --- Embed each stream ---
        s_tok   = self.in_proj_s(s).view(B, self.t_s, self.d_model)      # [B, t_s, d]
        x_tok   = self.in_proj_x(X)                                      # [B, N, d]
        pl_tok  = self.in_proj_pl(pathloss)                              # [B, N-1, d]

        # Interleave [X1,PL1, X2,PL2, ..., X_{N-1},PL_{N-1}] then append XN
        # Make a [B, 2*(N-1), d] by stacking and reshaping
        if N > 1:
            inter = torch.stack([x_tok[:, :N-1, :], pl_tok], dim=2).reshape(B, 2*(N-1), self.d_model)
            h_no_cls = torch.cat([s_tok, inter, x_tok[:, N-1:N, :]], dim=1)  # [B, t_s + 2*(N-1) + 1, d]
        else:
            # Degenerate case N=1 (no pathloss steps): just s_tokens + X1
            h_no_cls = torch.cat([s_tok, x_tok], dim=1)  # [B, t_s + 1, d]

        # Positional encodings
        h_no_cls = self.pos(h_no_cls)  # [B, T, d]
        T = h_no_cls.size(1)

        # key_padding_mask expects True=PAD; your mask=True means VALID → invert
        key_pad = (~mask) if (mask is not None) else None
        attn_mask = self._causal_mask(T, h_no_cls.device) if self.causal else None

        # Encode
        z = self.encoder(h_no_cls, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, T, d]

        # Extract the positions corresponding to X steps:
        # Layout: [ s(0..t_s-1) , inter(0..2*(N-1)-1) , last_x ]
        # In inter, X tokens are at even positions 0,2,4,..., so:
        if N > 1:
            start_inter = self.t_s
            end_inter   = self.t_s + 2*(N-1)
            z_inter     = z[:, start_inter:end_inter, :]     # [B, 2*(N-1), d]
            z_x_inter   = z_inter[:, 0::2, :]                # [B, N-1, d] (X1..X_{N-1})
            z_x_last    = z[:, end_inter:end_inter+1, :]     # [B, 1, d]   (XN)
            z_x = torch.cat([z_x_inter, z_x_last], dim=1)    # [B, N, d]
        else:
            z_x = z[:, self.t_s:self.t_s+1, :]               # [B, 1, d]

        Y_hat = self.out_proj(z_x)                           # [B, N, dy]
        return Y_hat

class SimpleMLPCLS(nn.Module):
    """
    Drop-in replacement for CondSeqTransformerMaskedCLS_withsxttoken_withdiff
    that keeps the same forward(s, X, Y_inp, X_diff, mask=None) -> (Y_hat, y_cls).

    - y_cls is produced from pooled features of s, X, X_diff, and Y_inp
    - Y_hat is a cheap per-step linear head (not used by your loss, but returned for API parity)
    """
    def __init__(self, m, dx, dy, d_hidden=512):
        super().__init__()
        self.proj_x     = nn.Linear(dx, d_hidden)
        self.proj_xdiff = nn.Linear(dx, d_hidden)
        self.proj_y     = nn.Linear(dy, d_hidden)

        # y_cls head (global prediction)
        self.cls_head = nn.Sequential(
            nn.Linear(m + 3*d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, dy)
        )

        # per-step head (kept for API compatibility)
        self.step_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, dy)
        )

    def forward(self, s, X, Y_inp, X_diff, mask=None):
        # Mean-pool token features (respect mask if provided)
        if mask is not None:
            # mask only applies to X/X_diff length dims; clamp to their T
            T = X.size(1)
            m = mask[:, :T].unsqueeze(-1).to(X.dtype)  # [B,T,1]
            denom = m.sum(dim=1).clamp_min(1.0)        # [B,1]
            hx  = (self.proj_x(X)     * m).sum(dim=1) / denom     # [B,H]
            hxd = (self.proj_xdiff(X_diff) * m).sum(dim=1) / denom # [B,H]
        else:
            hx  = self.proj_x(X).mean(dim=1)            # [B,H]
            hxd = self.proj_xdiff(X_diff).mean(dim=1)   # [B,H]

        hy = self.proj_y(Y_inp).mean(dim=1)             # [B,H]

        # Global readout
        h_global = torch.cat([s, hx, hxd, hy], dim=-1)  # [B, m+3H]
        y_cls = self.cls_head(h_global)                 # [B, dy]

        # Per-step (not used by your loss but returned)
        y_steps = self.step_head(self.proj_x(X))        # [B, T, dy]
        return y_steps, y_cls

class CondSeqTransformer_LDMRidge(nn.Module):
    """
    Token order:
      [CLS] || s_tokens(t_s) || [SEP] || X_100 (100) || [SEP] || LDM(1 token) || [SEP] || RIDGE(10 tokens)
    Predict:
      Y_hat over the 100 X tokens, and y_cls from [CLS].
    """
    def __init__(
        self,
        m, dx, dy,
        d_model=512, nhead=16, num_layers=8,
        dim_ff=2048, dropout=0.1, causal=False,
        t_s=4,                    # number of tokens generated from s
        ridge_vocab=128           # embed indices 0..(ridge_vocab-1); 0/99 used by you
    ):
        super().__init__()
        self.m, self.dx, self.dy = m, dx, dy
        self.d_model, self.t_s = d_model, t_s
        self.causal = causal

        # Projections/embeddings
        self.in_proj_s   = nn.Linear(m, t_s * d_model)     # -> [B,t_s,d]
        self.in_proj_x   = nn.Linear(dx, d_model)          # -> [B,100,d]
        self.in_proj_ldm = nn.Linear(3, d_model)           # -> [B,1,d]  (los,diff,too_many)
        self.emb_ridge   = nn.Embedding(ridge_vocab, d_model)  # -> [B,10,d]

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos     = SinusoidalPositionalEncoding(d_model)

        # Heads
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dy)
        )
        self.cls_head = nn.Linear(d_model, dy)

        # Learnable special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, s, X_100, ldm, ridge_idx, mask=None):
        """
        s        : [B, M]
        X_100    : [B, 100, dx]
        ldm      : [B, 3]  (float: los, diff, too_many; 0/1 is fine)
        ridge_idx: [B, 10] (long indices, e.g., in [0..99])
        mask     : optional [B, t_s + 1 + 100 + 1 + 1 + 1 + 10] (valid=True). If None, assume all valid.
        """
        B = s.size(0)

        # --- Build tokens ---
        s_tok   = self.in_proj_s(s).view(B, self.t_s, self.d_model)        # [B,t_s,d]
        x_tok   = self.in_proj_x(X_100)                                    # [B,100,d]
        ldm_tok = self.in_proj_ldm(ldm).unsqueeze(1)                        # [B,1,d]
        ridge_tok = self.emb_ridge(ridge_idx)                               # [B,10,d]
        sep = self.sep_token.expand(B, 1, -1)                               # [B,1,d]
        cls = self.cls_token.expand(B, 1, -1)                               # [B,1,d]

        # concat WITHOUT CLS (order you requested)
        h_no_cls = torch.cat([s_tok, sep, x_tok, sep, ldm_tok, sep, ridge_tok], dim=1)
        h_no_cls = self.pos(h_no_cls)

        # prepend [CLS]
        z_in = torch.cat([cls, h_no_cls], dim=1)                            # [B, 1 + t_s+1+100+1+1+1+10, d]

        # masks: key_padding_mask expects True=PAD; our mask=True means VALID → invert
        if mask is not None:
            one = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)    # CLS always valid
            mask_ext = torch.cat([one, mask], dim=1)                        # [B, T]
            key_pad = ~mask_ext
        else:
            key_pad = None

        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None

        # encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B,T,d]

        # indices to slice X tokens:
        #   CLS: 0
        #   s:   1 .. t_s
        #   sep: t_s+1
        #   X:   t_s+2 .. t_s+101     (100 tokens)
        start_x = 1 + self.t_s + 1
        end_x   = start_x + 100

        z_cls = z[:, 0, :]                        # [B,d]
        z_x   = z[:, start_x:end_x, :]            # [B,100,d]

        Y_hat = self.out_proj(z_x)                # [B,100,dy]
        y_cls = self.cls_head(z_cls)              # [B,dy]
        return Y_hat, y_cls

class CondSeqTransformer_EarlyFusion(nn.Module):
    """
    New model with early fusion of height, distance_tx, distance_rx.
    Separate embeddings for each sequence type, then concatenated.
    State parameters are treated as separate tokens.
    
    Token order:
      [CLS] || state_tokens (m_state tokens) || fused_sequence_tokens
    
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
        
        # Concatenate state tokens + fused sequences
        h_no_cls = torch.cat([state_tokens, fused_proj], dim=1)  # [B, M+N, d_model]
        h_no_cls = self.pos(h_no_cls)
        
        # Prepend [CLS]
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, d_model]
        z_in = torch.cat([cls, h_no_cls], dim=1)  # [B, 1+M+N, d_model]
        
        # Handle mask
        if mask is not None:
            # Extend mask for [CLS] and state tokens (all always valid)
            ones = torch.ones(B, 1 + M, dtype=mask.dtype, device=mask.device)
            mask_ext = torch.cat([ones, mask], dim=1)  # [B, 1+M+N]
            key_pad = ~mask_ext  # True = pad
        else:
            key_pad = None
        
        attn_mask = self._causal_mask(z_in.size(1), z_in.device) if self.causal else None
        
        # Encode
        z = self.encoder(z_in, mask=attn_mask, src_key_padding_mask=key_pad)  # [B, 1+M+N, d_model]
        
        # Extract outputs
        z_cls = z[:, 0, :]                    # [B, d_model]
        z_seq = z[:, 1+M:1+M+N, :]            # [B, N, d_model]  (skip CLS and state tokens)
        
        Y_hat = self.out_proj(z_seq)          # [B, N, dy]
        y_cls = self.cls_head(z_cls)          # [B, dy]
        
        return Y_hat, y_cls
