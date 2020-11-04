from torch.utils.data import Dataset
import cv2
import os
from glob import glob
import numpy as np


class CloudDataset(Dataset):
    def __init__(self, dataset_root, split):
        assert split in ["train", "val", "test"], "Invalid split {}".format(split)

        self.dataset_root = dataset_root
        self.split = split
        self.timestamps = [
            file.replace("_irc.tif", "").split("/")[-1]
            for file in glob(os.path.join(dataset_root, split, "**/*_irc.tif"))
        ]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        timestamp = self.timestamps[index]

        irc = cv2.imread(self.get_item_path(timestamp, "irc"))
        label = np.load(self.get_item_path(timestamp, "label"))["arr_0"]

        assert irc is not None, "Could not load irc for timestamp {}".format(timestamp)
        assert label is not None, "Could not load label for timestamp {}".format(
            timestamp
        )

        return {"id": index, "timestamp": timestamp, "irc": irc, "label": label}

    def get_item_path(self, timestamp, modality):
        modality_suffix = {"label": "labels.npz", "irc": "irc.tif"}
        return os.path.join(
            self.dataset_root,
            self.split,
            timestamp[:8],
            "{}_{}".format(timestamp, modality_suffix[modality]),
        )


if __name__ == "__main__":
    dataset = CloudDataset(
        "/Users/elrich/code/eth/irccam-pmodwrc/data/datasets/dataset_v1", "train"
    )
    print(dataset.timestamps)
    print(len(dataset))
    print(dataset[1]["label"])
