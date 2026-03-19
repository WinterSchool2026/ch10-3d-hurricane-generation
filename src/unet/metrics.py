import torch
import torch.nn as nn

class MSELoss(nn.Module):
    """
    MSE loss for 2D profiles
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        mse_vals = []
        for i in range(
            len(cs)
        ):  # need to loop through since patches are not the same size
            mse_vals.append(self.mse(cs[i], cs_p[i]))
        mse = torch.mean(torch.stack(mse_vals))
        return mse

class PSNRMetric(nn.Module):
    """
    PSNRMetric: Computes PSNR to be provided to wandb as a metric.
    """

    def __init__(self):
        super().__init__()
        self.pixel_max = 2  # Assuming the pixel values are normalized between -1 and 1

    def psnr(self, x_pred, targets):
        mse = torch.mean((targets - x_pred) ** 2)
        if mse == 0:
            return float("inf")
        PIXEL_MAX = self.pixel_max
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        psnr_vals = []
        for i in range(len(cs)):
            cs_i = cs[i].unsqueeze(0).unsqueeze(0)
            cs_p_i = cs_p[i].unsqueeze(0).unsqueeze(0)
            psnr_vals.append(self.psnr(cs_p_i, cs_i))

        return torch.mean(torch.stack(psnr_vals))

def psnr_metric(x_pred, targets, overpass_mask):
    PSNR = PSNRMetric().to("cuda")
    psnr_cs = PSNR(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return psnr_cs

def get_metrics(
    cs,
    cs_p,
    overpass_mask,
  
):
    """
    Function to calculate metrics.
    cs: true cloudsat profile
    cs_p: predicted cloudsat profile
    overpass_mask: overpass mask
    """
    mse = MSELoss().to("cuda")
    mse_cs = mse(cs, cs_p, overpass_mask)

    # Normalized RMSE
    rmse_cs = torch.sqrt(mse_cs)

    # Log PSNR
    psnr_cs = psnr_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)

    
    return (
        mse_cs,
        rmse_cs,
        psnr_cs,
    )



def get_profiles(cs, cs_p, overpass_mask):
    """
    Extracts profiles from the Clouds and Clouds Prediction tensors based on the overpass mask.

    Args:
        cs (torch.Tensor): Clouds tensor of shape (batch_size, height, padded_length).
        cs_p (torch.Tensor): Clouds Prediction tensor of shape (batch_size, height, width, length).
        overpass_mask (torch.Tensor): Overpass mask tensor of shape (batch_size, width, length).
    """
    batch_size, height, length = cs.shape
    cs_profiles = []
    cs_p_profiles = []
    for item in range(batch_size):
        cs_i = cs[item, :, :]
        cs_p_i = cs_p[item, :, :, :]
        overpass_mask_i = overpass_mask[item, :, :]

        binary_overpass_mask_i = overpass_mask_i > 0
        binary_overpass_mask_i = binary_overpass_mask_i.expand(
            height, -1, -1
        )  # (90, 256, 256)

        cs_profile_i = cs_i[~torch.isnan(cs_i)].reshape([height, -1])
        cs_p_profile_i = cs_p_i[binary_overpass_mask_i].reshape([height, -1])

        cs_profiles.append(cs_profile_i)
        cs_p_profiles.append(cs_p_profile_i)
    return cs_profiles, cs_p_profiles