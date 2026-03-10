import autoroot
import tacoreader
from shapely import wkb
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import LightningDataModule
from lightning.pytorch import seed_everything
from loguru import logger

from src.finetuning.dataloader import Cloud3DDataModule
from src.finetuning.transforms import GeoSatTransform
from src.finetuning.model import UNetAutoencoder


SPLITS_DICT = {
    "train": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(2, 23).tolist(),
    },
    "val": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(24, 27).tolist(),
    },
    "test": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(28, 32).tolist(),
    },
}

def add_split_column(df: pd.DataFrame, date_col: str = "date", split_col: str = "split") -> pd.DataFrame:
    
    # ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    y = df[date_col].dt.year
    m = df[date_col].dt.month
    d = df[date_col].dt.day

    # start with NaN / unknown
    split = np.full((len(df),), np.nan, dtype=object)
    #split = pd.Series(pd.NA, index=df.index, dtype="string")

    for name, spec in SPLITS_DICT.items():
        mask = (
            y.isin(spec["years"]) &
            m.isin(spec["months"]) &
            d.isin(spec["days"])
        )
        #split.loc[mask] = name
        split[mask] = name

    df[split_col] = split
    return df


def main(dataset_dir, batch_size, patch_size, lr, wd, 
         dir_save, exp_nm, file_nm, save_top_k, 
         epochs, limit_train_batches, limit_val_batches, limit_test_batches, seed):
    
    
    
    # read taco dataset 
    tacoreader.use("pandas")
    goes = tacoreader.load(dataset_dir+"goes/")
    himawari = tacoreader.load(dataset_dir+"himawari/")
    msg = tacoreader.load(dataset_dir+"msi/")

    # Concat
    full_dataset = tacoreader.concat([goes, himawari, msg])
    dataset = full_dataset.data

    
    # add trial, val, test split column
    dataset = add_split_column(dataset, date_col="stac:time_start")
    
    transform = GeoSatTransform(patch_size=patch_size)
    
    
    # for reproducibility
    seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Avoid silent TF32 numeric changes on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    dm = Cloud3DDataModule(dataset, transforms=transform, batch_size=batch_size, num_workers=2)
    it = iter(dm.train_sampler)
    print([next(it) for _ in range(20)])
    
    model = UNetAutoencoder(lr=lr, weight_decay=wd)
    
    
    # checkpoint and logging
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min",save_top_k=save_top_k,
                                    save_last=True, filename=file_nm,)
    logger = CSVLogger(save_dir=dir_save, name=exp_nm)

    # declare trainining
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, 
                     limit_train_batches=limit_train_batches,
                     limit_val_batches=limit_val_batches,
                    limit_test_batches=limit_test_batches,
                        callbacks=[checkpoint_cb],
                        logger=logger)
    trainer.fit(model, datamodule=dm)
    best_model = UNetAutoencoder.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(model=best_model, datamodule=dm)


if __name__ == "__main__":
    seed = 42
    dataset_dir = "/data/databases/CLOUD_3D/pretraining/tacos/finetune/"
    batch_size = 4
    patch_size = [256, 256]
    lr=1e-5
    wd=1e-4
    dir_save = "/data/users/emiliano/3DcloudsData/output/finetuning/"
    file_nm = "ae-{epoch:03d}-{val_loss:.4f}"
    exp_nm = "cloud_ae"
    save_top_k = 3
    epochs=100 #3
    l_tr_b=0.25 #0.25 #0.001
    l_va_b=0.5 #0.5 #0.005 
    l_te_b=1.0 #1.0 #0.005
    

    
    main(dataset_dir, batch_size, patch_size, lr, wd, 
         dir_save, exp_nm, file_nm,save_top_k, 
         epochs, l_tr_b, l_va_b, l_te_b, seed)  