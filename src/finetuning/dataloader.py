import autoroot
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.finetuning.bands11geo_vars3cs import BAND_MAPPING, BAND_NAMES, BAND_TYPES, CS_BAND_MAPPING, CS_VARS

from lightning.pytorch import LightningDataModule
from loguru import logger

class Cloud3DDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_df,
        transforms=None,
        batch_size: int = 4,
        num_workers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.dataset_df = dataset_df
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        logger.info(f"There are {len(self.dataset_df)} files in taco dataset")

        # split filenames based on train/test/val criteria
        train_df = dataset_df.loc[dataset_df['split']=='train']
        test_df = dataset_df.loc[dataset_df['split']=='test']
        val_df = dataset_df.loc[dataset_df['split']=='val']
        
        train_idxs = train_df.index.tolist()
        test_idxs = test_df.index.tolist()
        val_idxs = val_df.index.tolist()
        
        
        self.train_dataset = Cloud3DDataset(dataset_df, train_idxs, self.transforms)
        self.test_dataset = Cloud3DDataset(dataset_df, test_idxs, self.transforms)
        self.val_dataset = Cloud3DDataset(dataset_df, val_idxs, self.transforms)
        
        #train
        ws = train_df['cloud3d:satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in train_df['cloud3d:satellite']]
        self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # test
        ws = test_df['cloud3d:satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in test_df['cloud3d:satellite']]
        self.test_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # val
        ws = val_df['cloud3d:satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in val_df['cloud3d:satellite']]
        self.val_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        logger.info("MSG DataModule initialized ...")
        logger.info(f"Length of train dataset: {len(self.train_dataset)}")
        logger.info(f"Length of test dataset: {len(self.test_dataset)}")
        logger.info(f"Length of val dataset: {len(self.val_dataset)}")

    def prepare_data(self):
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)



    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
            #prefetch_factor=self.hparams.prefetch_factor,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
            #prefetch_factor=self.hparams.prefetch_factor,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
            #prefetch_factor=self.hparams.prefetch_factor,
            sampler=self.test_sampler,
        )

class Cloud3DDataset(Dataset):
    """MAE pretraining dataset with homogenized bands."""

    def __init__(self, tacoreader_df,  idxs,transforms=None):
        #self.ds = tacoreader_ds
        #user_cols = ['"cloud3d:satellite" AS satellite','"cloud3d:satellite" AS satellite', '"stac:time_start" AS date','"stac:centroid" AS centroid']
        #nav_cols = [f'"{c}"' for c in goes.navigation_columns()]
        #cols = ', '.join(user_cols + nav_cols)
        self.df = tacoreader_df #full_dataset.sql(f"SELECT {cols} FROM data").data.to_pandas()
        self.idxs = idxs
        self.transforms = transforms

    def __len__(self):
        #return len(self.df)
        return len(self.idxs)
    
    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, idx):
        #row = self.df.iloc[idx]
        row = self.df.iloc[self.idxs[idx]]
        satellite = row["cloud3d:satellite"]
        img_cs = self.df.read(self.idxs[idx]).to_pandas()
        
        vsi_path_img = img_cs.iloc[0]["internal:gdal_vsi"]
        vsi_path_cs = img_cs.iloc[1]["internal:gdal_vsi"]


    
        # Read image
        with rasterio.open(vsi_path_img) as src:
            # Read ONLY the bands we need (1-indexed for rasterio)
            band_indices = BAND_MAPPING[satellite]
            bands_1indexed = [b + 1 for b in band_indices]
            img = src.read(bands_1indexed)  # shape: (15, H, W)

        # Read cloudsat vertical profile
        with rasterio.open(vsi_path_cs) as src:
            # Read ONLY the bands we need (1-indexed for rasterio)
            band_indices = CS_BAND_MAPPING
            bands_1indexed = [b + 1 for b in band_indices]
            cs = src.read(bands_1indexed)  # shape: (15, H, W)
        
        idxs = list(range(15))
        data_dict  ={
            "image": img[idxs,],
            "cloudsat": cs,
            "overpass_mask": img[15,],
            "satellite": satellite,
            "date": row["stac:time_start"],
            "id": row["id"]
        }
        
        # Apply transformations
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        
        data_dict["image"] = data_dict["image"].astype(np.float32)

        return data_dict