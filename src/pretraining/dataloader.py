import autoroot
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.pretraining.bands11 import BAND_MAPPING, BAND_NAMES, BAND_TYPES

from lightning.pytorch import LightningDataModule
from loguru import logger


class Cloud3DDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_df,
        transforms=None,
        batch_size: int = 4,
        num_workers: int = 1,
        #patch_size: list = None,  # whether to crop the data to a smaller patch size (e.g. [128, 128])
        #center_crop: bool = False,  # if True, will crop to the center of the image
        #radius: int = 0,  # radius for cropping, if center_crop is True
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.dataset_df = dataset_df
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        #self.patch_size = patch_size
        #self.center_crop = center_crop
        #self.radius = radius

        
        logger.info(f"There are {self.dataset_df.shape[0]} files in taco dataset")

        # split filenames based on train/test/val criteria
        train_df = dataset_df.loc[dataset_df['split']=='train']
        test_df = dataset_df.loc[dataset_df['split']=='test']
        val_df = dataset_df.loc[dataset_df['split']=='val']

        
        self.train_dataset = Cloud3DDataset(train_df, self.transforms)
        self.test_dataset = Cloud3DDataset(test_df, self.transforms)
        self.val_dataset = Cloud3DDataset(val_df, self.transforms)
        
        #train
        ws = train_df['satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in train_df.satellite]
        self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # test
        ws = test_df['satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in test_df.satellite]
        self.test_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # val
        ws = val_df['satellite'].value_counts(normalize=True)
        sample_weights = [1/ws.to_dict()[sat] for sat in val_df.satellite]
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

    def __init__(self, tacoreader_df, transforms=None):
        self.df = tacoreader_df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vsi_path = row["internal:gdal_vsi"]
        satellite = row["satellite"]

        # Read image
        with rasterio.open(vsi_path) as src:
            # Read ONLY the bands we need (1-indexed for rasterio)
            band_indices = BAND_MAPPING[satellite]
            bands_1indexed = [b + 1 for b in band_indices]
            img = src.read(bands_1indexed)  # shape: (15, H, W)

        
        data_dict  ={
            "image": img,
            "satellite": satellite,
            "date": row["date"],
            "id": row["id"]
        }
        
        # Apply transformations
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        
        data_dict["image"] = data_dict["image"].astype(np.float32)

        return data_dict