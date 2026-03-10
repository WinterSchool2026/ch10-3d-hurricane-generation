import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime
from pathlib import Path
from src.finetuning.metrics import MSELoss, get_metrics, get_profiles
from src.finetuning.bands11geo_vars3cs import CS_VARS

class UNetAutoencoder(pl.LightningModule):
    """
    Autoencoder-like model using segmentation_models_pytorch.Unet.
    Takes 17-channel input, reconstructs only the first 11 channels.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        log_images_every_n_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_images_every_n_steps = log_images_every_n_steps
        self.criterion = MSELoss()

        # Unet returns `classes` channels. We'll make it output only recon_channels.
        self.net = UNet2Dto3D_Mixed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 17, H, W] -> yhat: [B, 11, H, W]
        return self.net(x)

    def compute_loss(self, batch, batch_idx=None, stage=None):
    
        losses = []
        metrics = []

        overpass_mask = batch["overpass_mask"]
        img = batch["image"]
        x_pred = self(img)  
        cs = batch["cloudsat"]

        # loop through vars radar reflectivity, ice-water-content, effective radisu
        for i in range(batch["cloudsat"].shape[1]):
            cs_i = cs[:,i,]
            cs_p_i = x_pred[:, i, ...]
            # Reorder from [B, L, H] to [B, H, L]
            cs_i = cs_i.permute(0, 2, 1)
            # Change dtype to match x_pred
            cs_i = cs_i.to(dtype=x_pred.dtype)
            losses.append(self.criterion(cs=cs_i, cs_p=cs_p_i, overpass_mask=overpass_mask))

            metrics.append(get_metrics(cs=cs_i,cs_p=cs_p_i,overpass_mask=overpass_mask,))

        mse, rmse, psnr = (torch.mean(torch.stack(metric)) for metric in zip(*metrics))

        # log metrics 
        self.log(f"{stage}_mse", mse,on_step=False,on_epoch=True,prog_bar=True,logger=True,)
        self.log(f"{stage}_rmse", rmse,on_step=False,on_epoch=True,prog_bar=True,logger=True,)
        self.log(f"{stage}_psnr", psnr,on_step=False,on_epoch=True,prog_bar=True,logger=True,)

        loss =torch.mean(torch.stack(losses))
        return loss

    def training_step(self, batch, batch_idx):
        
        loss = self.compute_loss(batch, batch_idx, stage="train")
        
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch  # For image logging
        self.val_idx = batch_idx
        
        loss = self.compute_loss(batch, batch_idx, stage="val")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx, stage="test")
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
    
    

    def plot_multi_profiles(self, batch, output, batch_idx):
        """
        Function to plot the true and predicted cloudsat profiles and upload them to wandb for multiple output variables

        Inputs:
            x: batch of input images (modis or msg) [batch_size x n_channels x patch_size_x x patch_size_y]
            cs: batch of true 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
            cs_p: batch of predicted 3d cloudsat profiles [batch_size x vars x height x patch_size_x x patch_size_y]
            overpass_mask: batch of overpass masks [batch_size x patch_size_x x patch_size_y] # TODO check size
            current_epoch: the current epoch, used for labelling the plot when uploaded to wandb
            log_image_samples: number of images to plot
            experiment: wandb experiment
            plot_channel: the input image channel that is supposed to be plotted
        """
        x = batch["image"]
        cs = batch["cloudsat"]
        cs_p = output.detach()  
        overpass_mask= batch["overpass_mask"]
        current_epoch=self.current_epoch
        plot_channel=5
        satellite = batch["satellite"]


        # check that we are not trying to plot more samples than we have in a batch
        batch_size = x.shape[0]
        max_images = 8
        if max_images > batch_size:
            max_images = batch_size

        n_variables = cs.shape[1]
        figs_wide = 2 * n_variables + 1

        # Check if output prediction shape is wrong for single variable prediction, unsqueeze for compatibility:
        if n_variables == 1 and len(cs_p.shape) < 5:
            cs_p = cs_p.unsqueeze(1)

        # plot masked input, prediction, and original image
        fig, axes = plt.subplots(
            max_images, figs_wide, figsize=(3.2 * figs_wide, max_images * 2.4)
        )

        # set colormaps
        overpass_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "", [(0, 0, 0, 0), (1, 0, 0, 1)]
        )  # transparent to red

        image_cmap = "Blues"

        cmap_dict = {
            "Radar_Reflectivity": "BuGn",
            "Ice_Water_Content": "PuBu",
            "Effective_Radius": "RdPu",
        }

        cmap = lambda var: cmap_dict.get(var, "viridis")

        for i in range(max_images):
            x_i = x[i].cpu().numpy() if x[i].is_cuda else x[i]
            overpass_mask_i = (
                overpass_mask[i].cpu().numpy()
                if overpass_mask[i].is_cuda
                else overpass_mask[i]
            )

            vmin = np.nanmin(x_i[plot_channel])
            vmax = np.nanmax(x_i[plot_channel])

            # plot one of the modis input channels
            axes[i, 0].imshow(
                x_i[plot_channel],
                cmap=image_cmap,
                vmin=vmin,  # the input images are normalized to [-1, 1]
                vmax=vmax,  # lock the colorbar to the same range for all images
                interpolation=None,
            )

            # binarise overpass mask
            overpass_i = (overpass_mask_i > 0).squeeze()

            # plot as red overlay
            axes[i, 0].imshow(
                overpass_i, vmin=0, vmax=1, cmap=overpass_cmap, interpolation="nearest"
            )

        # lock the colorbar to the same range
        vmin = -1  # np.nanmin([min_cs_p_i, min_cs_i])
        vmax = 1  # np.nanmax([max_cs_p_i, max_cs_i])

        # Now loop over cloudsat vars
        for k, (j, var) in enumerate(zip(range(1, 2 * n_variables, 2), CS_VARS)):
            cs_j, cs_pj = get_profiles(cs[:,k].transpose(-2, -1), cs_p[:, k], overpass_mask)
            # plot profiles and add shared colorbar
            for i in range(max_images):

                axes[i, j].imshow(
                    cs_pj[i].cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap(var)
                )
                im_cs = axes[i, j + 1].imshow(
                    cs_j[i].cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap(var)
                )
                plt.colorbar(
                    im_cs,
                    ax=axes[i, j : j + 2],
                    shrink=0.6,
                    pad=0.2,
                    orientation="horizontal",
                    label=var,
                )
            axes[0, j].set_title(f"Predicted profile")
            axes[0, j + 1].set_title(f"Target profile")

        # # set titles for clarity
        if satellite is not None:
            axes[0, 0].set_title(f"{satellite[0].split('_')[0]} channel {plot_channel}")
        else:
            axes[0, 0].set_title(f"Input channel {plot_channel}")

        # add general title with epoch number 
        # paste strings together to print satellite name
        sats = ";".join(satellite)
        fig.suptitle(f"Epoch {current_epoch:02d} - Batch {batch_idx} - Sats: {sats}")
        return fig

    # write a on_validation_epoch_end to log some predicted vs original images at 
    # the end of each epoch and save them to the log directory
    def on_validation_epoch_end(self):
        
        batch = self.val_batch 
        batch_idx = self.val_idx
        output = self(batch["image"])
        fig = self.plot_multi_profiles(batch, output, batch_idx)

        save_dir = Path(self.logger.log_dir) / "plots"
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"val_epoch_{self.current_epoch}.png")
        plt.close(fig)


    

class UNet2Dto3D_Mixed(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights=None, levels=80, out_ch=3, in_ch=17):
        super().__init__()
        self.levels = levels
        self.out_ch = out_ch
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=levels * out_ch,
            activation=None,
        )
        self.vmix = VerticalMix(out_ch=out_ch, levels=levels, k=5)

    def forward(self, x):
        y = self.net(x)                 # (B, 240, H, W)
        b, c, h, w = y.shape
        y = y.view(b, self.out_ch, self.levels, h, w)  # (B,3,80,H,W)
        y = self.vmix(y)                # enforce vertical smoothness/coupling
        return y


class VerticalMix(nn.Module):
    def __init__(self, out_ch=3, levels=80, k=5):
        super().__init__()
        # Mix along the levels dimension, applied per channel
        self.mix = nn.Conv3d(out_ch, out_ch, kernel_size=(k, 1, 1), padding=(k//2, 0, 0), groups=out_ch)

    def forward(self, y):  # y: (B,3,L,H,W)
        return self.mix(y)

