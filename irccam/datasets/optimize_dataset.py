"""
Takes a single label dataset and optimizes it for training
"""

import numpy as np
import h5py
from irccam.datasets.rgb_labeling import create_rgb_label_julian, create_label_adaptive

from irccam.utils.constants import *


def optimize_dataset(in_name, out_name=None):
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
    merge(train_days, in_root, os.path.join(out_root, "train.h5"))
    merge(test_days, in_root, os.path.join(out_root, "test.h5"))
    merge(val_days, in_root, os.path.join(out_root, "val.h5"))


def merge(days, in_root, out_name):
    with h5py.File(out_name, "w") as f_out:
        irc = f_out.create_dataset("irc", (0, 420, 420), chunks=(1, 420, 420), maxshape=(None, 420, 420), compression="lzf", dtype='float32')
        clear_sky = f_out.create_dataset("clear_sky", (0, 420, 420), chunks=(1, 420, 420), maxshape=(None, 420, 420), compression="lzf", dtype='float32')
        ir_labels = f_out.create_dataset("ir_labels", (0, 420, 420), chunks=(1, 420, 420), maxshape=(None, 420, 420), compression="lzf", dtype='int8')
        rgb_labels = f_out.create_dataset("rgb_labels", (0, 420, 420), chunks=(1, 420, 420), maxshape=(None, 420, 420), compression="lzf", dtype='int8')
        dt = h5py.string_dtype(encoding='ascii')
        timestamps = f_out.create_dataset("timestamps", (0,), maxshape=(None,), dtype=dt)
        for i, day in enumerate(days):
            with h5py.File(os.path.join(in_root, "{}.h5".format(day)), "r") as f_in:
                n = timestamps.shape[0]
                m = f_in["timestamp"].shape[0]
                irc.resize(n + m, axis=0)
                irc[n:] = f_in["irc"]
                clear_sky.resize(n + m, axis=0)
                clear_sky[n:] = f_in["clear_sky"]
                ir_labels.resize(n + m, axis=0)
                ir_labels[n:] = f_in["ir_label"]
                timestamps.resize(n + m, axis=0)
                timestamps[n:] = f_in["timestamp"]
                rgb_labels.resize(n + m, axis=0)
                rgb_labels[n:] = f_in["selected_label"]
            print("Finished processing day {}/{}. Added {} timestamps".format(i, len(days), m))


if __name__ == "__main__":
    print("Which dataset to optimize: ", end='')
    in_name = input().strip()
    print("Output dataset name (default optimized_...): ", end='')
    out_name = input().strip()
    optimize_dataset(in_name, out_name)
