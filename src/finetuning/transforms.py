from random import randint
import numpy as np
from torchvision.transforms import Compose
from src.finetuning.bands11geo_vars3cs import BAND_MAPPING, BAND_NAMES, BAND_TYPES, CS_BAND_MAPPING, CS_VARS, CLOUDSAT_NAN_FILL_VALUES

class GeoSatTransform:
    """Crop and normalize"""

    def __init__(
        self,
        patch_size: list
        | None = None,  # Whether to crop the data to a smaller patch size (e.g. [128, 128] for pre-training)
        #center_crop: bool = False,
        #radius: int = 0,  # radius for the center_crop in pixels
    ):
        transform_list = []


        # normalize REF & BT bands & angles
        transform_list += [MinMaxNormaliseTransform()]
        
        
        # time to fractional day and year and then 2d-field
        transform_list += [TimeTo2DTransform(height=patch_size[0], width=patch_size[1])]
        
        # Fill nans
        transform_list += [NanDictTransform()]
        
        # date to string
        transform_list += [date_to_str()]
        
        
        # CloudSat Transforms
        
        transform_list += [CloudSatReplaceNansTransform()]
        transform_list += [CloudSatPadAlongTrackTransform()]
                           
        transform_list += [CloudSatLinearNormaliseTransform(var="Radar_Reflectivity", min=-30, max=20)]
        transform_list += [CloudSatLogNormaliseTransform(var="Ice_Water_Content", min=1e-5, max=10)]
        transform_list += [CloudSatLinearNormaliseTransform(var="Effective_Radius", min=0, max=160)]
        transform_list += [CropHeightTransform(bottom_cutoff=20, top_cutoff=25),]

        self.transform = Compose(transform_list)

    def __call__(self, sample):
        s = self.transform(sample)
        return s

class MinMaxNormaliseTransform:
    """
    Normalises data to a range of [-1, 1] using min-max scaling.
    """

    def __init__(self, bt_min=180, bt_max=350, nr_min=0, nr_max=100, norm_angles=True):
        self.bt_min = bt_min
        self.bt_max = bt_max
        self.nr_min = nr_min
        self.nr_max = nr_max
        self.norm_angles = norm_angles
        
    def convert_angle(self, data, min, max):
        """
        Convert angles in degrees to radians and scale to [0, 2*pi].
        """
        # convert to radians and scale to [0, 2*pi]
        val_radians = 2 * np.pi * (data - min) / (max - min)
        return val_radians

    def convert_half_angle(self, data, min, max):
        """
        Convert angles in degrees to radians and scale to [0, pi].
        """
        # convert to radians and scale to [0, pi]
        val_radians = np.pi * (data - min) / (max - min)
        return val_radians

    def __call__(self, data_dict, **kwargs):
        arr = data_dict["image"]
        for i, sensor_type in enumerate(BAND_TYPES):
            
            if sensor_type == "bt":
                arr[i] = np.clip(
                    arr[i], self.bt_min, self.bt_max
                )
                # Apply min-max scaling to [-1, 1]
                arr[i] = (
                    (arr[i] - self.bt_min)
                    / (self.bt_max - self.bt_min)
                    * 2
                ) - 1
            if sensor_type == "nr":
                arr[i] = np.clip(
                    arr[i], self.nr_min, self.nr_max
                )
                # Apply min-max scaling to [-1, 1]
                arr[i] = (
                    (arr[i] - self.nr_min)
                    / (self.nr_max - self.nr_min)
                    * 2
                ) - 1
                
            if sensor_type == "angle_azi":
                arr[i] = self.convert_angle(
                    arr[i], 0, 360
                )
                
            if sensor_type == "angle_zen":    
                arr[i] = self.convert_half_angle(
                    arr[i], 0, 180
                )
            if (sensor_type in ["angle_azi", "angle_zen"]) & self.norm_angles:  # normalize between [0, 1] if self.norm
                    arr[i] = arr[i] / (2 * np.pi)
                
            data_dict["image"] = arr
        return data_dict
    
    
class TimeTo2DTransform:
    """
    Copy 1D arrays to 2D arrays of shape (1, H, W).
    """

    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width

    def __call__(self, data_dict):
        
        #add fractional day and year conversion here
        
        datetime_obj = data_dict["date"].to_pydatetime()
        fraction_of_year = np.clip(int(datetime_obj.strftime("%j")) / 365, 0, 1)
        fraction_of_day = np.clip(datetime_obj.hour / 24 + datetime_obj.minute / 24 / 60+ datetime_obj.second / 24 / 60 / 60,
            0,
            1,
         )
        time = np.array([fraction_of_year, fraction_of_day], dtype=np.float32)
        
        length = time.shape[0]
        data_2d = np.zeros(
                (length, self.height, self.width), dtype=time.dtype
            )
        for i in range(length):
            data_2d[i, :, :] = np.tile(
                    time[i], (1, self.height, self.width)
                )
        # NOTE: saving with a new key to avoid overwriting original times
        data_dict["image"] = np.concatenate([data_dict["image"], data_2d], axis=0)
        return data_dict
    
class NanDictTransform:
    # TODO not yet tested
    """
    Removes NaN values from data dictionary.
    Can also be used to replace NaN values of coordinates to remove off limb data.
    """

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, data_dict, **kwargs):
        data = data_dict["image"]
        # Replace NaN values
        data = np.nan_to_num(data, nan=self.fill_value)
        # Update dictionary
        data_dict["image"] = data
        return data_dict
    
class date_to_str:
    
    """
    Transforms date to string
    """

    def __init__(self,):
        pass

    def __call__(self, data_dict):
        data_dict["date"] = str(data_dict["date"])
        return data_dict
    
class CropHeightTransform:
    """
    Crop height levels, e.g. for Cloudsat data.
    """

    def __init__(self, bottom_cutoff: int, top_cutoff: int, key="cloudsat"):
        self.bottom_cutoff = bottom_cutoff
        self.top_cutoff = top_cutoff
        self.key = key

    def __call__(self, data_dict, **kwargs):
        # Crop height levels
        # CloudSat profiles go top to bottom
        data_dict[self.key] = data_dict[self.key][...,self.top_cutoff : -self.bottom_cutoff,  ]
        return data_dict
    
class CloudSatLinearNormaliseTransform:
    """
    Normalizes variables to [-1, 1].
    """

    def __init__(
        self,
        var: str,
        min: float,
        max: float,
        key: str = "cloudsat",
    ):
        """
        Args:
            var (list): The radar product to select
            key (str): Key in dictionary to apply transformation
        """
        self.var = var
        self.key = key
        self.min = min
        self.max = max
        self.idxvar = CS_VARS.index(var)

    def __call__(self, data_dict, **kwargs):
        # Get data
        
        data = data_dict[self.key]
        data = data[self.idxvar, ...]
        # clip extreme values
        data = np.clip(data, self.min, self.max)
        # apply normalization to valid data
        normalized_data = np.empty_like(data, dtype=float)
        normalized_data = (2 * (data - self.min) / (self.max - self.min)) - 1
        data_dict[self.key][self.idxvar, ...] = normalized_data
        return data_dict
    
class CloudSatLogNormaliseTransform:
    """
    Log normalizes CloudSat variables and scales to [-1, 1].
    """

    def __init__(
        self,
        var: str,
        key: str = "cloudsat",
        min=1e-4,
        max=100,
    ):
        """
        Args:
            var (list): The radar product to select
            key (str): Key in dictionary to apply transformation
        """
        self.var = var
        self.key = key
        self.min = min
        self.max = max
        self.idxvar = CS_VARS.index(var)

    def __call__(self, data_dict, **kwargs):
        # Get data
        data = data_dict[self.key][self.idxvar,...]
        # clip extreme values and take log
        data = np.log10(np.clip(data, self.min, self.max))
        # apply normalization to valid data
        normalized_data = np.empty_like(data)
        normalized_data = (
            2
            * ((data - np.log10(self.min)) / (np.log10(self.max) - np.log10(self.min)))
            - 1
        ).astype(data.dtype)
        data_dict[self.key][self.idxvar,...] = normalized_data
        return data_dict
    
class CloudSatReplaceNansTransform:
    """
    # Replace Nans with appropriate fill values
    """

    def __init__(
        self,
    ):
        """
        Args:
        
        """
        self.vars = CS_VARS
        

    def __call__(self, data_dict, **kwargs):
        # Get data
        cloudsat = data_dict["cloudsat"]
        for var in self.vars:
            idx = CS_VARS.index(var)
            cloudsat[idx,...] = np.where(np.isnan(cloudsat[idx,...]), CLOUDSAT_NAN_FILL_VALUES[var], cloudsat[idx,...])
        data_dict["cloudsat"] = cloudsat
        return data_dict

class CloudSatPadAlongTrackTransform:
    """
    
    """

    def __init__(
        self,
        target_length: int = 512,
    ):
        """
        Args:
        
        """
        self.target_length = target_length
        

    def __call__(self, data_dict, **kwargs):
        # Get data
        cloudsat = data_dict["cloudsat"]
        # Extra dimensions (C: vars ,H: length, W: height of the track)
        C,H, W = cloudsat.shape
        
        # Add extra padding along the length dimension
        padded = np.full((C, self.target_length, W), np.nan, dtype=cloudsat.dtype)
        padded[:,:H, :] = cloudsat        
        data_dict["cloudsat"] = padded
        return data_dict

    
    