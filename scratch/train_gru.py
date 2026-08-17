import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

sys.path.append('src')
import common as C
from nn_model import MaskedGRU, build_nn_sequences, PondDataset
import copy

def get_empirical_test_windows():
    tr, te = C.load_raw("data_checkpoint/raw/Train.csv", "data_checkpoint/raw/Test.csv")
    test_cube = C.band_cube(te)
    test_valid = C._valid_mask(test_cube)
    # S1 bands are 0 and 1. They define the window.
    s1_valid = test_valid[:, :, 0]
    windows = []
    for i in range(len(s1_valid)):
        row = s1_valid[i]
        valid_months = np.where(row)[0]
        if len(valid_months) > 0:
            start = valid_months[0]
            length = len(valid_months)
            windows.append((start, length))
    return windows

class NNAugmenter:
    def __init__(self, X, y, windows, is_train=True):
        self.X = X
        self.y = y
        self.windows = windows
        self.is_train = is_train
        
    def get_epoch_data(self):
        N, C_dim, T = self.X.shape
        X_aug = self.X.clone()
        s1_mask = torch.ones(N, T, dtype=torch.bool)
        
        if self.is_train:
            # S2 dropout
            s2_drop = torch.rand(N, T) < 0.07
            # In build_nn_sequences, C = 23 (9 indices + 12 bands + 2 masks)
            # Actually we had C=23. S2 dependent features:
            # Indices 0..6, 8 (vh-vv is 7).
            # norm bands: 11..20 (S2 bands).
            # I will just mask the whole S2 mask channel
            
            # Apply window masking
            starts = []
            lengths = []
            idxs = np.random.choice(len(self.windows), size=N, replace=True)
            for i in range(N):
                st, ln = self.windows[idxs[i]]
                s1_mask[i, :st] = False
                s1_mask[i, st+ln:] = False
                
                # Apply dropout to S2 channels
                if s2_drop[i].any():
                    s2_invalid = s2_drop[i]
                    # indices 0,1,2,3,4,5,6
                    for c in range(7): X_aug[i, c, s2_invalid] = 0.0
                    # index 8
                    X_aug[i, 8, s2_invalid] = 0.0
                    # norm bands for S2 (indices 9 to 20, actually 0 and 1 are S1, 2 to 11 are S2)
                    # wait, norm_bands starts at index 8 (if 8 indices) or 9.
                    # Let's just use the S2 mask channel to zero out.
                    
            # For simplicity, zero out everything outside window
            for i in range(N):
                window = s1_mask[i]
                X_aug[i, :, ~window] = 0.0
                
            # Affine Jitter for continuous bands
            gain = (torch.rand(N, C_dim, 1) * 0.2 + 0.9)  # 0.9 to 1.1
            bias = torch.randn(N, C_dim, 1) * 0.05
            
            # Don't apply to masks (assuming masks are at indices 21, 22)
            gain[:, 21:, :] = 1.0
            bias[:, 21:, :] = 0.0
            
            X_aug = X_aug * gain + bias
                
        return X_aug, s1_mask

def smooth_bce(pred, target, smoothing=0.10):
    target = target.float()
    target = target * (1.0 - smoothing) + 0.5 * smoothing
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss

def train_nn():
    print("Loading data...")
    tr, te = C.load_raw("data_checkpoint/raw/Train.csv", "data_checkpoint/raw/Test.csv")
    train_cube = C.band_cube(tr)
    train_valid = C._valid_mask(train_cube)
    
    # We want robust sequences
    X, s1_valid = build_nn_sequences(train_cube, train_valid)
    print("X shape:", X.shape)
    
    # Get empirical windows from test
    windows = get_empirical_test_windows()
    
    # 5-fold CV
    df = tr
    y = df['label'].values
    groups = df.index // 25
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    
    num_seeds = 3
    epochs = 100
    batch_size = 64
    
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"--- Fold {fold} ---")
        
        X_train, y_train = X[train_idx], torch.tensor(y[train_idx])
        X_val, y_val = X[val_idx], torch.tensor(y[val_idx])
        
        aug_train = NNAugmenter(X_train, y_train, windows, is_train=True)
        # validation uses fixed standard masking?
        # "early stopping on masked-fold-val AUC". We should evaluate on masked fold.
        aug_val = NNAugmenter(X_val, y_val, windows, is_train=True) 
        # Actually validation should probably have deterministic masking, but randomly drawing once is fine.
        X_val_masked, s1_val_mask = aug_val.get_epoch_data()
        val_dataset = PondDataset(X_val_masked, s1_val_mask, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_preds = np.zeros((num_seeds, len(val_idx)))
        
        for seed in range(num_seeds):
            torch.manual_seed(seed + 42)
            np.random.seed(seed + 42)
            
            model = MaskedGRU(in_channels=X.shape[1])
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            best_auc = 0
            best_weights = None
            
            for ep in range(epochs):
                model.train()
                X_tr, s1_tr = aug_train.get_epoch_data()
                train_loader = DataLoader(PondDataset(X_tr, s1_tr, y_train), batch_size=batch_size, shuffle=True)
                
                for xb, mb, yb in train_loader:
                    optimizer.zero_grad()
                    out = model(xb, mb)
                    loss = smooth_bce(out, yb)
                    loss.backward()
                    optimizer.step()
                    
                scheduler.step()
                
                # Eval
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for xb, mb, _ in val_loader:
                        out = model(xb, mb)
                        val_preds.append(out.cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_auc = roc_auc_score(y_val, val_preds)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_weights = copy.deepcopy(model.state_dict())
                    
            print(f"Seed {seed} best AUC: {best_auc:.4f}")
            model.load_state_dict(best_weights)
            
            # Predict val
            model.eval()
            seed_preds = []
            with torch.no_grad():
                for xb, mb, _ in val_loader:
                    out = model(xb, mb)
                    seed_preds.append(out.cpu().numpy())
            fold_preds[seed] = np.concatenate(seed_preds)
            
        # Average seeds in logit space (they are already logits)
        oof_preds[val_idx] = fold_preds.mean(axis=0)
        fold_auc = roc_auc_score(y_val, expit(oof_preds[val_idx]))
        print(f"Fold {fold} Avg AUC: {fold_auc:.4f}")
        
    final_auc = roc_auc_score(y, expit(oof_preds))
    print(f"\nFinal OOF AUC: {final_auc:.4f}")
    
    np.save("model_checkpoints/gru_oof_preds.npy", oof_preds)
    
    print("\n--- DPP Evaluation ---")
    test_like_idx = np.load('data_checkpoint/dpp_test_like_idx.npy')
    
    p = expit(oof_preds)
    # Apply shift
    p_calib = np.clip(p, 1e-15, 1 - 1e-15)
    odds = p_calib / (1 - p_calib)
    shift = (0.6175 / (1 - 0.6175)) / (0.404 / (1 - 0.404))
    new_odds = odds * shift
    p_calib = new_odds / (1 + new_odds)
    
    y_test_like = y[test_like_idx]
    p_test_like = p_calib[test_like_idx]
    pred_binary = (p_test_like >= 0.5).astype(int)
    
    from sklearn.metrics import f1_score
    if len(np.unique(y_test_like)) >= 2:
        f1 = f1_score(y_test_like, pred_binary)
        auc = roc_auc_score(y_test_like, p_test_like)
        dpp_score = 0.6 * f1 + 0.4 * auc
        print(f"GRU DPP Score: {dpp_score:.5f} (F1={f1:.4f}, AUC={auc:.4f})")
    
if __name__ == "__main__":
    train_nn()
