""" 
Loading the .mat files with the IRCCAM data becomes a bottleneck for
creating a dataset, because there is a lot of unused data in those files.

Here, we extract the raw components of the .mat files, to speed up
dataset creation.

Changes made in here to prevent big time memory leakage, still won't run without at least 20G of RAM.
If all else fails temporarily increase swapfile size to stupid levels
"""

import os
import mat73
import numpy as np
import sys
import math
import datetime
from tqdm import tqdm
import time
import cv2

from datasets.image_processing import process_irccam_img

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data/raw/davos")


def extract_data(file=None):
    if file is None:
        irccam_files = get_contained_files(os.path.join(RAW_DATA_PATH, "irccam"))
    else:
        irccam_files = [file]

    irccam_files = [file for file in irccam_files if file.endswith(".mat")]
    extract_path = os.path.join(RAW_DATA_PATH, "irccam_extract")
    for file in irccam_files:
        day = file.split("_")[1]
        if os.path.exists(os.path.join(extract_path, day)) and has_completed(day):
            print("ignoring day {}, already processed".format(day))
            continue

        print("processing day {}".format(day))
        start = time.time()
        irccam_data = mat73.loadmat(os.path.join(RAW_DATA_PATH, "irccam", file))
        end = time.time()
        print("loaded data, time: {} seconds".format(round(end - start)))

        bt_path = os.path.join(extract_path, day, "bt")
        img_path = os.path.join(extract_path, day, "img")
        jpg_path = os.path.join(extract_path, day, "jpg")

        if not os.path.exists(bt_path):
            os.makedirs(bt_path)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(jpg_path):
            os.makedirs(jpg_path)
        print("saving BT")
        for i in tqdm(range(0, irccam_data["BT"].shape[2])):
            timestamp = convert_timestamp(day, irccam_data["TM"][i])
            filename = os.path.join(bt_path, "{}.npy".format(timestamp))
            np.save(filename, irccam_data["BT"][:, :, i])
        print("saving img")
        for i in tqdm(range(0, irccam_data["img"].shape[2])):
            timestamp = convert_timestamp(day, irccam_data["TM"][i])
            filename = os.path.join(img_path, "{}.npy".format(timestamp))
            np.save(filename, irccam_data["img"][:, :, i])
        # Save jpg so that we can inspect which timestamps to filter out.
        print("saving jpg")
        for i in tqdm(range(0, irccam_data["img"].shape[2])):
            timestamp = convert_timestamp(day, irccam_data["TM"][i])
            filename = os.path.join(jpg_path, "{}.jpg".format(timestamp))
            img_data = np.array(irccam_data["BT"][:, :, i])
            img = process_irccam_img(img_data, dtype=np.uint8)
            saved = cv2.imwrite(filename, img)
            assert saved, "Failed to save {}".format(filename)

        irccam_data = None  # don't know what worked...but at some point python started releasing memory on time
        save_completed(day)


def get_completed_filename():
    return os.path.join(RAW_DATA_PATH, "irccam_extract", "completed.txt")


def has_completed(day):
    completed_filename = get_completed_filename()
    if not os.path.exists(completed_filename):
        open(completed_filename, "a").close()
    with open(completed_filename) as f:
        completed_days = f.readlines()
    completed_days = [ts.strip() for ts in completed_days]
    return day in completed_days


def save_completed(day):
    with open(get_completed_filename(), "a") as f:
        f.write(day + "\n")


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def convert_timestamp(day, timestamp):
    """
    Converts irccam timestamps in double format (e.g. 737653.55976907) to
    timestamps capped to the nearest second (e.g. 20190816132643)
    """
    seconds = round(24 * 60 * 60 * (timestamp - math.floor(timestamp)))
    seconds_delta = datetime.timedelta(0, seconds)
    day_timestamp = datetime.datetime.strptime(day, "%Y%m%d")
    return (day_timestamp + seconds_delta).strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_data(sys.argv[1])
    else:
        extract_data()
