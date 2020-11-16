from torch.utils.data import Dataset
import cv2
import os
import h5py
import numpy as np
from bisect import bisect_right
import math

"""
Implemented the dataset from daily H5 files

Inspiration https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
"""


class CloudDataset(Dataset):
    def __init__(self, dataset_root, split, transform=None, nth_sample=1):
        assert split in ["train", "val", "test"], "Invalid split {}".format(split)

        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.days = np.loadtxt(os.path.join(dataset_root, split + ".txt"), dtype="str")
        self.files

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        timestamp = self.timestamps[index]

        irc = cv2.imread(self.get_item_path(timestamp, "irc"))
        irc = cv2.cvtColor(irc, cv2.COLOR_BGR2GRAY)
        label = np.load(self.get_item_path(timestamp, "label")).astype(np.long)

        if self.transform:
            irc = self.transform(irc)
            label = self.transform(label)

        assert irc is not None, "Could not load irc for timestamp {}".format(timestamp)
        assert label is not None, "Could not load label for timestamp {}".format(
            timestamp
        )

        return {"id": index, "timestamp": timestamp, "irc": irc, "label": label}

    def get_item_path(self, timestamp, modality):
        modality_suffix = {"label": "label.npy", "irc": "irc.tif"}
        return os.path.join(
            self.dataset_root,
            self.split,
            timestamp[:8],
            "{}_{}".format(timestamp, modality_suffix[modality]),
        )


class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.

    Input params
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, dataset_root, split, transform=None, nth_sample=1):
        super().__init__()
        assert split in ["train", "val", "test"], "Invalid split {}".format(split)

        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.days = np.loadtxt(os.path.join(dataset_root, split + ".txt"), dtype="str")
        self.nth_sample = nth_sample

        self.data_info = []
        self.transform = transform
        self.length = 0
        self.offsets = []

        self.files = [
            os.path.join(os.path.join(dataset_root, day + ".h5")) for day in self.days
        ]
        for h5dataset_fp in self.files:
            self._add_data_infos(h5dataset_fp)

    def __getitem__(self, index):
        timestamp, irc, label = self.get_data(index)

        if self.transform:
            irc = self.transform(irc)
            label = self.transform(label)

        return {"index": index, "timestamp": timestamp, "irc": irc, "label": label}

    def __len__(self):
        return self.length

    def _add_data_infos(self, file_path):
        with h5py.File(file_path) as h5_file:
            size = math.floor(h5_file["timestamp"].shape[0] / self.nth_sample)
            self.offsets.append(self.length)
            self.length += size

    def get_data_infos(self, type):
        data_info_type = [di for di in self.data_info if di["type"] == type]
        return data_info_type

    def get_data(self, index):
        file_index = bisect_right(self.offsets, index) - 1
        file_offset = self.offsets[file_index]
        i = (index - file_offset) * self.nth_sample
        with h5py.File(self.files[file_index]) as h5_file:
            return (
                h5_file["timestamp"][i],
                np.nan_to_num(h5_file["irc"][i], nan=255) / 255,  # scale to range [0,1]
                h5_file["labels1"][i].astype(np.long),
            )


if __name__ == "__main__":
    dataset = HDF5Dataset("../../data/datasets/dataset_v1", "train")
    print(dataset.days)
    print(len(dataset))
    print(dataset[23]["timestamp"])
