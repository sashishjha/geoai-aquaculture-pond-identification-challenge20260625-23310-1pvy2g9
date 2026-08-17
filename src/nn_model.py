import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

import common as C
from config import RunConfig

def build_nn_sequences(cube, valid):
    """
    Builds the Cx12 sequences for the NN.
    cube: (N, 12, 12)  [N, months, bands]
    valid: (N, 12, 12) boolean
    
    Returns (N, C, 12) tensor where C = 20:
    - 9 spectral indices (NDVI, NDWI, MNDWI, EVI, AWEI, BSI, NDBI, SAR ratio, VH-VV in dB)
    - 7 robust-normalized raw bands (we'll just use the raw bands that matter or all 12 normalized)
      Actually, the directive says "per-row robust-normalized band values". Let's do all 12 bands.
    - 1 binary S2-observed mask channel
    - 1 in-window mask channel (S1 observed)
    - 2 sin/cos month-of-year encoding
    """
    N = cube.shape[0]
    
    idxmap = {b: C.ALL_BANDS.index(b) for b in C.ALL_BANDS}
    s1_idx = [idxmap[b] for b in C.S1_BANDS]
    s2_idx = [idxmap[b] for b in C.S2_BANDS]
    
    s1_valid = valid[:, :, s1_idx].all(axis=2) # (N, 12)
    s2_valid = valid[:, :, s2_idx].all(axis=2) # (N, 12)
    
    def band(b): return cube[:, :, idxmap[b]]
    def bvalid(b): return valid[:, :, idxmap[b]]
    
    # Indices
    indices = []
    
    def add_index(num_b, den_b):
        a, c = band(num_b), band(den_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (a - c) / (a + c)
        return np.where(np.isfinite(s), s, 0.0)
    
    indices.append(add_index("nir", "red"))     # NDVI
    indices.append(add_index("green", "nir"))   # NDWI
    indices.append(add_index("green", "swir1")) # MNDWI
    
    # EVI
    nir, red, blue = band("nir"), band("red"), band("blue")
    with np.errstate(divide="ignore", invalid="ignore"):
        evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
    indices.append(np.where(np.isfinite(evi), evi, 0.0))
    
    # AWEI
    green, swir1, swir2 = band("green"), band("swir1"), band("swir2")
    awei = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    indices.append(np.where(np.isfinite(awei), awei, 0.0))
    
    # BSI
    with np.errstate(divide="ignore", invalid="ignore"):
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
    indices.append(np.where(np.isfinite(bsi), bsi, 0.0))
    
    indices.append(add_index("swir1", "nir")) # NDBI
    
    # SAR Ratio (difference in dB)
    vh, vv = band("VH"), band("VV")
    indices.append(np.where(np.isfinite(vh-vv), vh-vv, 0.0))
    
    # 9th index? "SAR ratio, VH-VV in dB". Wait, "SAR ratio" IS VH-VV in dB.
    # Ah, the 9 spectral indices listed are: NDVI, NDWI, MNDWI, EVI, AWEI, BSI, NDBI, SAR ratio, VH-VV in dB.
    # Actually, the text says "(NDVI, NDWI, MNDWI, EVI, AWEI, BSI, NDBI, SAR ratio, VH−VV in dB)". That's 9 items. Wait, "SAR ratio" and "VH-VV in dB" might be the same thing in the text, or SAR ratio is VH/VV in linear? But they are already in dB. So VH-VV is the ratio. What's the 9th? Let's just include VH+VV instead, or just leave it at 8 indices. I'll add VH and VV individually anyway.
    
    # Per-row robust-normalized band values (12 bands)
    norm_bands = []
    for i in range(12):
        b_vals = cube[:, :, i]
        b_val = valid[:, :, i]
        # per-row robust normalize: subtract median, divide by IQR
        b_out = np.zeros_like(b_vals)
        for n in range(N):
            v = b_vals[n, b_val[n]]
            if len(v) > 1:
                p25, p50, p75 = np.percentile(v, [25, 50, 75])
                iqr = p75 - p25
                if iqr < 1e-6: iqr = 1.0
                b_out[n] = (b_vals[n] - p50) / iqr
        norm_bands.append(b_out)
        
    # Masks
    s2_mask = s2_valid.astype(np.float32)
    s1_mask = s1_valid.astype(np.float32)
    
    # Time encoding
    months = np.arange(1, 13)
    sin_mo = np.sin(2 * np.pi * months / 12)[None, :]
    cos_mo = np.cos(2 * np.pi * months / 12)[None, :]
    sin_mo = np.repeat(sin_mo, N, axis=0)
    cos_mo = np.repeat(cos_mo, N, axis=0)
    
    channels = indices + norm_bands + [s2_mask, s1_mask, sin_mo, cos_mo]
    
    # Stack and zero-out invalid positions
    X = np.stack(channels, axis=1) # (N, C, 12)
    
    # Zero fill where S1 is invalid (out of window). Or just where S2 is invalid for S2-dependent features.
    # To be safe, zero-fill everywhere the band is invalid, but the easiest is just zero-fill the final sequence outside the window.
    # Wait, the prompt says: "Missing timesteps zero-filled after normalization, always paired with the mask flags."
    for i in range(N):
        window = s1_valid[i]
        X[i, :, ~window] = 0.0
        s2 = s2_valid[i]
        # zero out optical features where S2 is invalid
        for c in range(8): # first 8 are optical indices (except SAR ratio)
            if c < 7: X[i, c, ~s2] = 0.0
        # for norm_bands, optical are 2..11
        for c in range(8, 20):
            band_idx = c - 8
            if band_idx >= 2: # S2 band
                X[i, c, ~s2] = 0.0
    
    return torch.tensor(X, dtype=torch.float32), s1_valid


class PondDataset(Dataset):
    def __init__(self, X, s1_valid, y=None):
        self.X = X
        self.s1_valid = s1_valid
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        v = self.s1_valid[idx]
        if self.y is not None:
            return x, v, self.y[idx]
        return x, v


class MaskedGlobalPool1d(nn.Module):
    def forward(self, x, mask):
        # x: (B, C, T)
        # mask: (B, T) boolean
        B, C, T = x.shape
        mask = mask.view(B, 1, T).float()
        
        sum_x = (x * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1.0)
        mean_pool = sum_x / count
        
        # Max pool: fill invalid with -inf before max
        x_max = x.masked_fill(mask == 0, float('-inf'))
        max_pool, _ = x_max.max(dim=2)
        
        return torch.cat([mean_pool, max_pool], dim=1) # (B, 2C)

class MaskedCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Body: 3 conv blocks, kernel 3, dilations 1/2/4, 32->64 channels, GroupNorm, GELU, dropout 0.2-0.3.
        self.c1 = nn.Conv1d(in_channels, 32, 3, padding=1, dilation=1)
        self.gn1 = nn.GroupNorm(4, 32)
        
        self.c2 = nn.Conv1d(32, 64, 3, padding=2, dilation=2)
        self.gn2 = nn.GroupNorm(8, 64)
        
        self.c3 = nn.Conv1d(64, 64, 3, padding=4, dilation=4)
        self.gn3 = nn.GroupNorm(8, 64)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.25)
        
        self.pool = MaskedGlobalPool1d()
        
        # Head: 2-layer MLP -> single sigmoid (well, no sigmoid if we use BCEWithLogitsLoss)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, mask):
        # Apply mask at each layer? Or just start.
        x = x * mask.unsqueeze(1).float()
        
        x = self.drop(self.act(self.gn1(self.c1(x))))
        x = x * mask.unsqueeze(1).float()
        
        x = self.drop(self.act(self.gn2(self.c2(x))))
        x = x * mask.unsqueeze(1).float()
        
        x = self.drop(self.act(self.gn3(self.c3(x))))
        
        x = self.pool(x, mask)
        
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x

class MaskedGRU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # single-layer bidirectional, hidden 64, same masked pooling
        self.gru = nn.GRU(input_size=in_channels, hidden_size=64, num_layers=1, 
                          batch_first=True, bidirectional=True)
        self.pool = MaskedGlobalPool1d()
        self.drop = nn.Dropout(0.25)
        self.act = nn.GELU()
        
        # Head: 2-layer MLP -> single sigmoid
        # GRU outputs (B, T, 128) because hidden is 64 and bidir=True
        # Pool gives (B, 2*128) = 256
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, mask):
        # x is (B, C, T), GRU expects (B, T, C)
        x = x.transpose(1, 2)
        
        # We don't pack_padded_sequence because we want to pool over the valid months,
        # and masking before/after works. But we can just zero out the invalid inputs.
        x = x * mask.unsqueeze(-1).float()
        
        out, _ = self.gru(x) # out: (B, T, 128)
        
        # Pool expects (B, C, T)
        out = out.transpose(1, 2)
        
        out = self.pool(out, mask)
        
        out = self.drop(self.act(self.fc1(out)))
        out = self.fc2(out).squeeze(-1)
        return out
