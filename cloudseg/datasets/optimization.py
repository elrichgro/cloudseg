import numpy as np
import h5py
from shutil import copyfile
from joblib import Parallel, delayed
from tqdm import tqdm

from cloudseg.utils.constants import *


def optimize_dataset(in_name, out_name=None):
    """
    Optimize H5 dataset for use in training.

    Read in the dataset `in_name` and create an optimized, compressed version at
    `out_name`. Optimization is done using lzf compression and by chunking the H5 files.
    """
    if not out_name:
        out_name = "optimized_" + in_name

    print("Optimizing dataset")
    in_root = os.path.join(DATASET_PATH, in_name)
    out_root = os.path.join(DATASET_PATH, out_name)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    train_days = np.loadtxt(os.path.join(in_root, "train.txt"), dtype="str")
    test_days = np.loadtxt(os.path.join(in_root, "test.txt"), dtype="str")
    val_days = np.loadtxt(os.path.join(in_root, "val.txt"), dtype="str")
    process_set(np.concatenate((train_days, test_days, val_days)), in_root, out_root)

    copyfile(os.path.join(in_root, "train.txt"), os.path.join(out_root, "train.txt"))
    copyfile(os.path.join(in_root, "test.txt"), os.path.join(out_root, "test.txt"))
    copyfile(os.path.join(in_root, "val.txt"), os.path.join(out_root, "val.txt"))
    copyfile(os.path.join(in_root, "changes.txt"), os.path.join(out_root, "changes.txt"))


def process_set(days, in_root, out_root):
    Parallel(n_jobs=6)(delayed(process_day)(d, in_root, out_root) for d in tqdm(days))


def process_day(day, in_root, out_root):
    name = "{}.h5".format(day)
    chunk_s = (1, 420, 420)
    with h5py.File(os.path.join(in_root, name), "r") as f_in, h5py.File(os.path.join(out_root, name), "w") as f_out:
        f_out.create_dataset("irc", data=f_in["irc"], chunks=chunk_s, compression="lzf")
        f_out.create_dataset("clear_sky", data=f_in["clear_sky"], chunks=chunk_s, compression="lzf")
        f_out.create_dataset(
            "selected_label", data=f_in["selected_label"], chunks=chunk_s, compression="lzf", dtype=LABEL_DATATYPE
        )
        f_out.create_dataset("sun_mask", data=f_in["sun_mask"], chunks=chunk_s, compression="lzf")
        f_out.create_dataset("timestamp", data=f_in["timestamp"])


if __name__ == "__main__":
    print("Which dataset to optimize: ", end="")
    in_name = input().strip()
    print("Output dataset name (default optimized_...): ", end="")
    out_name = input().strip()
    optimize_dataset(in_name, out_name)
