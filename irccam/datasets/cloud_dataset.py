from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from bisect import bisect_right

"""
Implemented the dataset from daily H5 files

Inspiration https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
"""


class HDF5Dataset(Dataset):
    def __init__(self, dataset_root, split, transform=None, use_clear_sky=False, mask_sun=True):
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

        self.mask_sun = mask_sun

        self.files = [os.path.join(os.path.join(dataset_root, day + ".h5")) for day in self.days]
        for h5dataset_fp in self.files:
            self._add_data_infos(h5dataset_fp)

    def __getitem__(self, index):
        timestamp, irc_raw, label, clear_sky, sun_mask = self.get_data(index)

        if self.use_clear_sky:
            irc_raw[irc_raw == 0] = 255.0
            irc = irc_raw - clear_sky

            # Scale to [0,1]
            irc[irc_raw == 255] = -30.0
            irc += 30.0
            irc[irc > 130.0] = 130.0
            irc /= 130.0
        else:
            # Scale to [0,1]
            irc = irc_raw / 255.0
            irc[irc_raw == 255] = 0.0

        if self.transform:
            irc = self.transform(irc)
            label = self.transform(label)

        if self.mask_sun:
            irc[sun_mask] = np.nan
            label[sun_mask] = -1

        return {"index": index, "timestamp": timestamp, "irc": irc, "label": label}

    def __len__(self):
        return self.length

    def _add_data_infos(self, file_path):
        with h5py.File(file_path) as h5_file:
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
                h5_file["sun_mask"][i] if self.mask_sun else None,
            )


if __name__ == "__main__":
    dataset = HDF5Dataset("../../data/datasets/main_2", "train")
    print(len(dataset))
    print(dataset[23]["label"])

    dataset = HDF5Dataset("../../data/datasets/optimized_2", "train")
    print(len(dataset))
    print(dataset[23]["label"])
