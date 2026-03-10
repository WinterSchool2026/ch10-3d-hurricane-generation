import autoroot
import tacoreader
from shapely import wkb
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything

from src.pretraining.dataloader import Cloud3DDataModule
from src.pretraining.transforms import GeoSatTransform
from src.pretraining.model import UNetAutoencoder

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
    df = df.copy()

    # ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    y = df[date_col].dt.year
    m = df[date_col].dt.month
    d = df[date_col].dt.day

    # start with NaN / unknown
    split = pd.Series(pd.NA, index=df.index, dtype="string")

    for name, spec in SPLITS_DICT.items():
        mask = (
            y.isin(spec["years"]) &
            m.isin(spec["months"]) &
            d.isin(spec["days"])
        )
        split.loc[mask] = name

    df[split_col] = split
    return df


def main(dataset_dir, batch_size, patch_size, radius, in_channels, recon_channels, lr, wd, 
         dir_save, exp_nm, file_nm, save_top_k, 
         epochs, limit_train_batches, limit_val_batches, limit_test_batches, seed):
    
    
    
    # read taco dataset 
    tacoreader.use("pandas")
    goes = tacoreader.load(dataset_dir+"goes/")
    himawari = tacoreader.load(dataset_dir+"himawari/")
    msg = tacoreader.load(dataset_dir+"msg/")

    # User columns + navigation columns
    user_cols = ['"cloud3d:satellite" AS satellite', '"stac:time_start" AS date','"stac:centroid" AS centroid']
    nav_cols = [f'"{c}"' for c in goes.navigation_columns()]
    cols = ', '.join(user_cols + nav_cols)

    # Filter each dataset
    goes_filtered = goes.sql(f"SELECT {cols} FROM data")
    himawari_filtered = himawari.sql(f"SELECT {cols} FROM data")
    msg_filtered = msg.sql(f"SELECT {cols} FROM data")
    # Concat
    full_dataset = tacoreader.concat([goes_filtered, himawari_filtered, msg_filtered])
    dataset = full_dataset.data.to_pandas()

    # add coordinates as geometry 
    dataset["geometry"] = dataset.centroid.apply(lambda v: wkb.loads(bytes(v)) if v is not None else None)
    dataset = gpd.GeoDataFrame(dataset, geometry="geometry", crs="EPSG:4326")
    dataset["lon"] = dataset.geometry.x
    dataset["lat"] = dataset.geometry.y
    
    # add trial, val, test split column
    dataset = add_split_column(dataset, date_col="date")
    
    transform = GeoSatTransform(patch_size=patch_size, center_crop=True, radius=radius)
    
    
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
    
    model = UNetAutoencoder(in_channels=in_channels,recon_channels=recon_channels,
                            encoder_name="resnet34",encoder_weights=None,lr=lr,weight_decay=wd)
    
    
    
    
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
    dataset_dir = "/data/databases/CLOUD_3D/pretraining/tacos/pretraining/"
    batch_size = 4
    patch_size = [128, 128]
    radius = 32
    in_channels = 17
    recon_channels = 11
    lr=1e-5
    wd=1e-4
    dir_save = "/data/users/emiliano/3DcloudsData/output/pretraining/"
    file_nm = "ae-{epoch:03d}-{val_loss:.4f}"
    exp_nm = "cloud_ae"
    save_top_k = 3
    epochs=100 #3
    l_tr_b=0.05 #0.25 #0.001
    l_va_b=0.1 #0.5 #0.005 
    l_te_b=1.0 #1.0 #0.005
    

    
    main(dataset_dir, batch_size, patch_size, radius, in_channels, recon_channels, lr, wd, 
         dir_save, exp_nm, file_nm,save_top_k, 
         epochs, l_tr_b, l_va_b, l_te_b, seed)  