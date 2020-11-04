from torch.utils.data import Dataset
import cv2

import os
from glob import glob


class CloudDataset(Dataset):
    def __init__(self, dataset_root, split):
        assert split in ["train", "val", "test"], "Invalid split {}".format(split)

        self.dataset_root = dataset_root
        self.split = split
        self.timestamps = [
            file.replace("_irc.tif", "").split("/")[-1]
            for file in glob(
                os.path.join(dataset_root, split, "images", "**/*_irc.tif")
            )
        ]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        timestamp = self.timestamps[index]

        irc = cv2.imread(self.get_item_path(timestamp, "irc"))
        label = cv2.imread(self.get_item_path(timestamp, "label"))

        return {"id": index, "timestamp": timestamp, "irc": irc, "label": label}

    def get_item_path(self, timestamp, modality):
        modality_suffix = {"label": "labels.npz", "irc": "irc.tif"}
        return os.path.join(
            self.dataset_root,
            self.split,
            "images",
            timestamp[:8],
            "{}_{}".format(timestamp, modality_suffix[modality]),
        )


if __name__ == "__main__":
    dataset = CloudDataset(
        "/Users/elrich/code/eth/irccam-pmodwrc/data/datasets/dataset_v1", "train"
    )
    print(dataset.timestamps)
    print(len(dataset))
    print(dataset[1])
