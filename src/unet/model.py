import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime
from pathlib import Path
from src.finetuning.metrics import MSELoss, get_metrics, get_profiles
from src.finetuning.bands11geo_vars3cs import CS_VARS

class Cloud3DFullSceneModel(pl.LightningModule):
    def __init__(self, in_channels=17, variables=3, z_bins=80, encoder_name="resnet34", lr=1e-4, wd=1e-2, T_max=100, eta_min=1e-6):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet", 
            in_channels=in_channels,
            classes=variables * z_bins, 
            activation=None,            
        )

    def forward(self, x):
        pred = self.net(x) 
        return pred.view(-1, self.hparams.variables, self.hparams.z_bins, x.shape[2], x.shape[3])

    def get_aligned_profiles(self, batch):
        cs_target = batch["cloudsat"]         
        overpass_mask = batch["overpass_mask"] 
        pred_volume = self(batch["image"])     

        batch_size = cs_target.shape[0] if cs_target.ndim > 3 else 1
        # Handle Lightning's potential unbatched input during certain steps
        if cs_target.ndim == 3:
            cs_target = cs_target.unsqueeze(0)
            overpass_mask = overpass_mask.unsqueeze(0)
            pred_volume = pred_volume.unsqueeze(0)

        aligned_preds = []
        aligned_targets = []

        for i in range(batch_size):
            coords = torch.nonzero(overpass_mask[i])
            if coords.shape[0] == 0: continue
            
            sorted_idx = torch.argsort(coords[:, 0]) 
            coords = coords[sorted_idx]
            y, x = coords[:, 0], coords[:, 1]

            p_profile = pred_volume[i, :, :, y, x] 
            t_profile = cs_target[i].permute(0, 2, 1)
            
            finite_mask = torch.isfinite(t_profile[0, 0, :])
            t_profile = t_profile[:, :, finite_mask]

            min_len = min(p_profile.shape[2], t_profile.shape[2])
            if min_len < 2: continue

            aligned_preds.append(p_profile[:, :, :min_len])
            aligned_targets.append(t_profile[:, :, :min_len])

        return aligned_preds, aligned_targets

    def training_step(self, batch, batch_idx):
        preds, targets = self.get_aligned_profiles(batch)
        if not preds: return None

        mse_list = []
        for p, t in zip(preds, targets):
            mse_list.append(F.mse_loss(p, t) * 100.0)
        
        loss = torch.stack(mse_list).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, targets = self.get_aligned_profiles(batch)
        if not preds: return None

        mse_list = []
        for p, t in zip(preds, targets):
            mse_list.append(F.mse_loss(p, t) * 100.0)
        
        loss = torch.stack(mse_list).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd) #1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }