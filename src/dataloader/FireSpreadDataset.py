from pathlib import Path
from typing import List, Optional
import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import warnings
from .utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):
    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None,
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False):
        super().__init__()

        self.stats_years = stats_years
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir

        self.validate_inputs()

        self.skip_initial_samples = (
            self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.n_leading_observations_test_adjustment is not None
            else 0
        )
        if self.skip_initial_samples < 0:
            raise ValueError(
                f"n_leading_observations_test_adjustment must be >= n_leading_observations."
            )

        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum(
            sum(self.datapoints_per_fire[fire_year].values())
            for fire_year in self.datapoints_per_fire
        )

        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()
    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(f"Index {target_id} out of range.")

        first_id_in_current_fire = 0
        found_fire_year, found_fire_name = None, None

        for fire_year in self.datapoints_per_fire:
            for fire_name, count in self.datapoints_per_fire[fire_year].items():
                if target_id - first_id_in_current_fire < count:
                    found_fire_year, found_fire_name = fire_year, fire_name
                    break
                first_id_in_current_fire += count

        in_fire_index = target_id - first_id_in_current_fire
        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        in_fire_index += self.skip_initial_samples
        end_index = in_fire_index + self.n_leading_observations + 1

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:end_index - 1]
                    doys = self.img_dates_to_doys(doys)
                    return np.split(imgs, [-1], axis=0)[0], imgs[-1, -1], torch.Tensor(doys)
        else:
            img_paths = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = [rasterio.open(img).read() for img in img_paths]
            return np.stack(imgs[:-1], axis=0), imgs[-1][-1]

    def read_list_of_images(self):
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                for fire_dir in fires:
                    fire_name = fire_dir.split("/")[-2]
                    imgs_per_fire[fire_year][fire_name] = sorted(glob.glob(f"{fire_dir}/*.tif"))
            else:
                files = glob.glob(f"{self.data_dir}/{fire_year}/*.hdf5")
                for fire_hdf5 in files:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire
    def __getitem__(self, index):
        year, fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)
        loaded_imgs = self.load_imgs(year, fire_name, in_fire_index)

        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(torch.Tensor(x), torch.Tensor(y))

        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)
        elif self.features_to_keep is not None:
            x = x[:, self.features_to_keep, ...]

        return (x, y, doys) if self.return_doy else (x, y)

    def __len__(self):
        return self.length
    def preprocess_and_augment(self, x, y):
        y = (y > 0).long()

        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        binary_af_mask = (x[:, -1:, ...] > 0).float()
        x = self.standardize_features(x)
        x = torch.cat([x, binary_af_mask], dim=1)

        landcover_classes = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes].reshape(
            x.shape[0], x.shape[2], x.shape[3], -1
        ).permute(0, 3, 1, 2)

        return torch.cat([x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1), y

    def augment(self, x, y):
        crop_side = self.crop_side_length
        best_crop = max(
            [
                (
                    TF.crop(x, t, l, crop_side, crop_side),
                    TF.crop(y, t, l, crop_side, crop_side),
                )
                for t in range(10)
                for l in range(10)
            ],
            key=lambda c: c[1].float().mean(),
        )
        x, y = best_crop
        return x, y
