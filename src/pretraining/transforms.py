from random import randint
import numpy as np
from torchvision.transforms import Compose
from src.pretraining.bands11 import BAND_MAPPING, BAND_NAMES, BAND_TYPES


class GeoSatTransform:
    """Crop and normalize"""

    def __init__(
        self,
        patch_size: list
        | None = None,  # Whether to crop the data to a smaller patch size (e.g. [128, 128] for pre-training)
        center_crop: bool = False,
        radius: int = 0,  # radius for the center_crop in pixels
    ):
        transform_list = []

        #crop
        if patch_size is not None:
            transform_list += [
                RandomCropTransform(
                    center_crop=center_crop,
                    patch_size=patch_size,
                    radius=radius,
                )
            ]

        # normalize REF & BT bands & angles
        transform_list += [MinMaxNormaliseTransform()]
        
        
        # time to fractional day and year and then 2d-field
        transform_list += [TimeTo2DTransform(height=patch_size[0], width=patch_size[1])]
        
        # Fill nans
        transform_list += [NanDictTransform()]
        
        # date to string
        transform_list += [date_to_str()]

        self.transform = Compose(transform_list)

    def __call__(self, sample):
        s = self.transform(sample)
        return s

class RandomCropTransform:
    
    def __init__(
        self,
        patch_size,
        center_crop=False,
        radius=0,  # Defined in pixels
    ):
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.radius = radius

    def __call__(self, data_dict):
        arr = data_dict["image"]
        assert arr.shape[1] >= self.patch_size[0], "Invalid shape to crop"
        assert arr.shape[2] >= self.patch_size[1], "Invalid shape to crop"
        if not self.center_crop:
            xmin = randint(0, arr.shape[1] - self.patch_size[0])
            ymin = randint(0, arr.shape[2] - self.patch_size[1])
            cropped_arr = arr[
                :, xmin : xmin + self.patch_size[0], ymin : ymin + self.patch_size[1]
            ]
        else:
            central_idxx, central_idxy = arr.shape[1] // 2, arr.shape[2] // 2
            central_x = randint(central_idxx - self.radius, central_idxx + self.radius)
            central_y = randint(central_idxy - self.radius, central_idxy + self.radius)
            cropped_arr = arr[
                :,
                central_x
                - self.patch_size[0] // 2 : central_x
                + self.patch_size[0] // 2,
                central_y
                - self.patch_size[1] // 2 : central_y
                + self.patch_size[1] // 2,
            ]
        data_dict["image"] = cropped_arr
        return data_dict
    
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