import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path


class UNetReconstructor(nn.Module):
    """
    Takes 17-channel input, reconstructs first 11 channels.
    Assumes:
      - x[:, :11] already normalized to [-1, 1]
      - x[:, 11:] already normalized to [0, 1]
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,  # None recommended for non-RGB
        in_channels: int = 17,
        out_channels: int = 11,
    ):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # keep logits/continuous output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,17,H,W) -> yhat: (B,11,H,W)
        return self.net(x)


class ReconLoss(nn.Module):
    """Reconstruction loss on the first 11 channels only."""
    def __init__(self, kind: str = "huber", huber_delta: float = 0.1):
        super().__init__()
        kind = kind.lower()
        if kind == "mse":
            self.loss = nn.MSELoss()
        elif kind == "l1":
            self.loss = nn.L1Loss()
        elif kind == "huber":
            self.loss = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError("kind must be one of: 'mse', 'l1', 'huber'")

    def forward(self, pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :pred.shape[1]]  # assumes pred has 11 channels
        return self.loss(pred, target)

class UNetAutoencoder(pl.LightningModule):
    """
    Autoencoder-like model using segmentation_models_pytorch.Unet.
    Takes 17-channel input, reconstructs only the first 11 channels.
    """

    def __init__(
        self,
        in_channels: int = 17,
        recon_channels: int = 11,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,  # None recommended for non-RGB inputs
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mse",  # "mse" or "l1"
        log_images_every_n_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.recon_channels = recon_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss.lower()
        self.log_images_every_n_steps = log_images_every_n_steps

        # Unet returns `classes` channels. We'll make it output only recon_channels.
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=recon_channels,
            activation=None,  # we don't force tanh since your normalization is already done
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 17, H, W] -> yhat: [B, 11, H, W]
        return self.net(x)

    def _recon_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return F.mse_loss(y_hat, y_true)
        if self.loss_name in ("l1", "mae"):
            return F.l1_loss(y_hat, y_true)
        raise ValueError(f"Unknown loss='{self.loss_name}'")
    
    # put all together in aplotting functionthat takes in batch, output, and plot_channel as parameters
    def plot_predictions(self, batch, output, plot_channel=5, title=""):
        images = batch["image"].cpu().detach().numpy()
        pred_image = output.cpu().detach().numpy()
        timestamps = batch["date"]
        timestamps = pd.to_datetime(timestamps, format="mixed")


        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="black")

        num_batches = batch["image"].shape[0]
        num_cols = 3
        fig, axes = plt.subplots(num_batches,num_cols,figsize=(10, num_batches * num_cols),)       

        for i in range(num_batches):
            if timestamps is not None:
                date_array = timestamps[i]
                dt = datetime.fromtimestamp(date_array.timestamp())
                time_label = dt.strftime("(%Y-%m-%d %H:%M:%S)")
            else:
                time_label = ""

            # set vmin and vmax based on the original image
            vmin = np.nanmin(images[i][plot_channel])
            vmax = np.nanmax(images[i][plot_channel])

            # plot masked input, prediction, and original image

            axes[i, 0].imshow(pred_image[i][plot_channel],vmin=vmin,vmax=vmax,cmap=cmap,)


            ax = axes[i, 1]
            ax.axis("off")

            # format however you like
            txt = batch['satellite'][i]+"\n"+ time_label
            ax.text(0.5, 0.5, txt,ha="center", va="center",fontsize=12,transform=ax.transAxes,)

            axes[i, 2].imshow(images[i][plot_channel],vmin=vmin,vmax=vmax,cmap=cmap,)

            # set titles for clarity
            axes[0, 0].set_title("Predicted")
            axes[0, 2].set_title(f"Original ")

        fig.suptitle(f"{title} channel {plot_channel}", y=0.92)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0, hspace=0.4) 
        return fig       
                
                
                
            

    def training_step(self, batch, batch_idx):
        x = batch["image"]  # [B, 17, 128, 128]

        # target = first 11 channels
        y = x[:, : self.recon_channels, :, :]

        y_hat = self(x)
        loss = self._recon_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch  # For image logging
        x = batch["image"]
        y = x[:, : self.recon_channels, :, :]
        y_hat = self(x)
        loss = self._recon_loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = x[:, : self.recon_channels, :, :]
        y_hat = self(x)
        loss = self._recon_loss(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            #weight_decay=self.weight_decay,
        )
        return optimizer

    # write a on_validation_epoch_end to log some predicted vs original images at 
    # the end of each epoch and save them to the log directory
    def on_validation_epoch_end(self):
        batch = self.val_batch 
        output = self(batch["image"])
        fig = self.plot_predictions(batch, output, plot_channel=5, title=f"Epoch {self.current_epoch}")
        

        save_dir = Path(self.logger.log_dir) / "plots"
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"val_epoch_{self.current_epoch}.png")
        plt.close(fig)

