from torch.utils.data import Dataset
from pytz import timezone
from bisect import bisect_right
import cv2
import os
import h5py
import numpy as np
import datetime
from pysolar.solar import get_azimuth, get_altitude

from cloudseg.utils.constants import *
from cloudseg.datasets.preprocessing import apply_clear_sky


class HDF5Dataset(Dataset):
    """
    PyTorch dataset module for loading data from daily H5 files.

    Inspired by https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    """

    def __init__(self, dataset_root, split, transform=None, use_clear_sky=False, use_sun_mask=True, sun_radius=40):
        super().__init__()
        assert split in ["train", "val", "test"], "Invalid split {}".format(split)

        self.use_clear_sky = use_clear_sky
        self.transform = transform

        self.dataset_root = dataset_root
        self.split = split
        self.days = np.loadtxt(os.path.join(dataset_root, split + ".txt"), dtype="str")

        self.data_info = []
        self.length = 0
        self.offsets = []

        self.use_sun_mask = use_sun_mask
        self.sun_radius = sun_radius

        self.files = [os.path.join(os.path.join(dataset_root, day + ".h5")) for day in self.days]
        for h5dataset_fp in self.files:
            self._add_data_infos(h5dataset_fp)

    def __getitem__(self, index):
        timestamp, irc_raw, label, clear_sky, _ = self.get_data(index)
        if type(timestamp) == bytes:
            timestamp = timestamp.decode("utf-8")

        # TODO when to apply mask and fill in nans, here or after clear sky modifications
        if self.use_sun_mask:
            sun = get_sun_position(timestamp)
            sun_mask = create_sun_mask(sun, self.sun_radius)
            irc_raw[sun_mask == 1] = 0
            label[sun_mask == 1] = -1

        np.nan_to_num(label, copy=False, nan=255.0)
        np.nan_to_num(irc_raw, copy=False, nan=255.0)
        label = label.astype(np.long)

        if self.use_clear_sky:
            # Subtract clear sky, scale to [0, 1]
            irc = apply_clear_sky(irc_raw, clear_sky)
        else:
            # Scale to [0,1]
            irc = irc_raw / 255.0
            irc[irc_raw == 255] = 0.0

        if self.transform:
            irc, label = self.transform((irc, label))

        return {"index": index, "timestamp": timestamp, "irc": irc, "label": label}

    def __len__(self):
        return self.length

    def _add_data_infos(self, file_path):
        with h5py.File(file_path, "r") as h5_file:
            size = h5_file["timestamp"].shape[0]
            self.offsets.append(self.length)
            self.length += size

    def get_data_infos(self, type):
        data_info_type = [di for di in self.data_info if di["type"] == type]
        return data_info_type

    def get_data(self, index):
        file_index = bisect_right(self.offsets, index) - 1
        file_offset = self.offsets[file_index]
        i = index - file_offset
        with h5py.File(self.files[file_index], "r") as h5_file:
            return (
                h5_file["timestamp"][i],
                h5_file["irc"][i],
                h5_file["selected_label"][i],
                h5_file["clear_sky"][i] if self.use_clear_sky else None,
                h5_file["sun_mask"][i] if self.use_sun_mask else None,
            )


def create_sun_mask(position, radius, img_size=(420, 420)):
    img = np.zeros(img_size).astype(np.uint8)
    cv2.circle(img, position, radius, 1, -1)
    return img


def get_sun_position(ts):
    # Date
    date = datetime.datetime.strptime(ts, "%Y%m%d%H%M%S")
    tz = timezone("UTC")
    date = tz.localize(date)

    # Sun angles
    location = (LOCATION.latitude, LOCATION.longitude)
    azimuth = get_azimuth(*location, date)
    zenith = 90 - get_altitude(*location, date)

    # Position in image
    center = (220, 220)
    radius = 220
    alpha = 90
    sun_x = int(center[0] - radius * np.sin(np.radians(azimuth)) * zenith / alpha)
    sun_y = int(center[1] + radius * np.cos(np.radians(azimuth)) * zenith / alpha)
    sun = (sun_x, sun_y)
    return sun


if __name__ == "__main__":
    dataset = HDF5Dataset("../../data/datasets/main_3", "test", use_sun_mask=True)
    print(len(dataset))
    print(dataset[14]["timestamp"])

    dataset = HDF5Dataset("../../data/datasets/optimized_3", "test")
    print(len(dataset))
    print(dataset[23]["timestamp"])
