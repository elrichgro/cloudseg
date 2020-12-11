"""
Take the raw IRCCAM data and RGB data and create train, val, and test sets
for model training. 

Basic flow:
- Split into train, val, test sets by day
- Get all timestamps for each set
- For each timestamp:
    - Get ir image, RGB image
    - Preprocess
    - Create label
    - Filter out some images
    - Save files into HDF5 format

Still to do:
- better RGB detection
- filter out bad days
"""

import cv2
import math
import numpy as np
import h5py
import sys
import uuid

from joblib import Parallel, delayed
from bisect import bisect_left

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split

from datasets.helpers import get_contained_dirs, get_contained_files
from datasets.optimize_dataset import optimize_dataset
from irccam.datasets.image_processing import process_irccam_img, process_vis_img, sun_correction, process_irccam_label, \
    apply_common_mask
from irccam.datasets.dataset_filter import (
    filter_sun,
    filter_sparse, filter_manual,
)
from irccam.datasets.rgb_labeling import create_rgb_label_julian, create_label_adaptive

from irccam.utils.constants import *


def create_dataset(dataset_name, test=False, sizes=(0.6, 0.2, 0.2), changelog="", use_manual_filter=True):
    assert sum(sizes) == 1, "Split sizes to not sum up to 1"

    print("Creating dataset")
    days = get_days()

    if test:
        days = days[:6]
        dataset_name = dataset_name + "_" + str(uuid.uuid4())
        path = os.path.join(DATASET_PATH, "../test/", dataset_name)
    else:
        path = os.path.join(DATASET_PATH, dataset_name)

    # Create data/previews directories if it doesn't exist yet
    previews_path = os.path.join(path, "previews")
    if not os.path.exists(previews_path):
        os.makedirs(previews_path)

    if not test:
        with open(os.path.join(path, "changes.txt"), "w") as f:
            f.write(changelog)

    success = Parallel(n_jobs=2)(
        delayed(process_day)(path, d, i, len(days), use_manual_filter) for i, d in enumerate(days))
    print("Successfully added {}/{} days to the dataset".format(sum(success), len(days)))
    # Save splits
    days_ok = [x for x, y in zip(days, success) if y]
    train, test, val = sizes
    days_train, days_testval = train_test_split(days_ok, test_size=test + val, train_size=train)
    days_val, days_test = train_test_split(days_testval, test_size=test / (test + val), train_size=val / (test + val))
    np.savetxt(os.path.join(path, "train.txt"), days_train, fmt="%s")
    np.savetxt(os.path.join(path, "test.txt"), days_test, fmt="%s")
    np.savetxt(os.path.join(path, "val.txt"), days_val, fmt="%s")


def process_day(data_path, day, i, n, use_manual_filter):
    print("Processing day {} - {}/{}".format(day, i + 1, n))

    # create output directory
    data_filename = os.path.join(data_path, "{}.h5".format(day))
    if os.path.exists(data_filename):
        return True

    with h5py.File(os.path.join(RAW_DATA_PATH, "irccam", "irccam_{}_rad.mat".format(day)), "r") as fr:
        irc_raw = fr["BT"]
        clear_sky_raw = fr["TB"]
        ir_labels_raw = fr["CLOUDS"]
        irc_timestamps = get_irc_timestamps(day, fr)
        vis_timestamps = get_vis_timestamps(day)

        matching_timestamps = match_timestamps(irc_timestamps, vis_timestamps)
        filtered_timestamps = filter_sun(matching_timestamps, day)
        filtered_timestamps = filter_sparse(filtered_timestamps)

        label_selected = 3
        if use_manual_filter:
            filtered_timestamps, label_selected = filter_manual(day, filtered_timestamps)

        n = len(filtered_timestamps)
        if n == 0:
            return False

        timestamps = []
        vis_images = np.empty((n, 420, 420, 3), dtype="float32")
        irc_images = np.empty((n, 420, 420), dtype="float32")
        clear_skies = np.empty((n, 420, 420), dtype="float32")
        labels_out = [np.empty((n, 420, 420), dtype="byte") for _ in range(4)]
        ir_labels = np.empty((n, 420, 420), dtype="byte")

        # We'll output a preview video
        preview_filename = os.path.join(data_path, "previews", "{}_preview.mp4".format(day))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Be sure to use lower case
        video_out = cv2.VideoWriter(preview_filename, fourcc, 3, (420 * 4, 420 * 2))

        print("Processing images")
        for i, (vis_ts, (irc_ts, irc_idx)) in enumerate(filtered_timestamps):
            # lets just keep one timestamp, time sync will have to be done in this file anyway
            timestamps.append(irc_ts.strftime(TIMESTAMP_FORMAT))

            irc_img = irc_raw[irc_idx, :, :]
            irc_img = process_irccam_img(irc_img)

            vis_img = get_vis_img(vis_ts)
            vis_img = process_vis_img(vis_img)

            clear_sky = clear_sky_raw[irc_idx, :, :]
            clear_sky = process_irccam_img(clear_sky)

            ir_label = ir_labels_raw[irc_idx, :, :]
            ir_label = process_irccam_label(ir_label)

            # Create labels
            labels = [create_rgb_label_julian(vis_img, cloud_ref=2.35),
                      create_rgb_label_julian(vis_img, cloud_ref=2.7),
                      create_rgb_label_julian(vis_img, cloud_ref=3),
                      create_label_adaptive(vis_img)]

            sun_correction(vis_img, irc_img, clear_sky, labels)
            label_images = [create_label_image(i) for i in labels]
            ir_label_image = create_label_image(ir_label)

            # apply common mask to vis, cannot do it before to not mess with adaptive labeling
            apply_common_mask(vis_img)

            comparison_image = concat_images({
                "irccam": irc_img,
                "rgb": vis_img,
                "ir_label": ir_label_image,
                "tresh 2.35": label_images[0],
                "tresh 2.7": label_images[1],
                "tresh 3": label_images[2],
                "adaptive": label_images[3],
                "selected " + str(label_selected): label_images[label_selected]
            })

            cv2.putText(comparison_image, irc_ts.strftime(PRETTY_FORMAT), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            video_out.write(comparison_image)  # Write out frame to video

            vis_images[i, :, :, :] = vis_img
            irc_images[i, :, :] = irc_img
            clear_skies[i, :, :] = clear_sky
            ir_labels[i, :, :] = ir_label

            for j, label in enumerate(labels):
                labels_out[j][i, :, :] = label

        video_out.release()

        print("Saving data")
        with h5py.File(data_filename, "w") as fw:
            fw.create_dataset("timestamp", data=timestamps)
            fw.create_dataset("irc", data=irc_images, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("vis", data=vis_images, chunks=(1, 420, 420, 3), compression="lzf")
            fw.create_dataset("clear_sky", data=clear_skies, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("ir_label", data=ir_labels, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("selected_label", data=labels_out[label_selected], chunks=(1, 420, 420), compression="lzf")
            for j, data in enumerate(labels_out):
                fw.create_dataset("labels" + str(j), data=data, chunks=(1, 420, 420), compression="lzf")

        return True


def save_arrays_to_dataset(data, path, timestamp):
    filename = os.path.join(path, "{}.npz".format(timestamp[0].strftime(TIMESTAMP_FORMAT_MINUTE)))
    np.savez(filename, **data)


def save_image_to_dataset(img, path, timestamp, extension):
    filename = os.path.join(path, "{}_{}.jpg".format(timestamp.strftime(TIMESTAMP_FORMAT_MINUTE), extension))
    saved = cv2.imwrite(filename, img)
    if not saved:
        raise Exception("Failed to save image {}".format(filename))


def concat_images(images):
    processed = []
    for name, img in images.items():
        i = np.nan_to_num(img, copy=True, nan=255)
        if i.ndim == 2:
            i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        i = i.astype(np.uint8)
        cv2.putText(i, name, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 50), 2)
        processed.append(i)

    n = len(processed) // 2
    a, b = processed[:n], processed[n:]
    return cv2.vconcat((cv2.hconcat(a), cv2.hconcat(b)))


def create_label_image(labels):
    img = np.zeros((labels.shape[0], labels.shape[1], 3))
    img[:, :, 0] = labels * 255
    img[np.where(labels == -1)] = float("nan")
    return img


def match_timestamps(ir_ts, vis_ts):
    """
    intersection is not enough since the timestamps do not match exactly
    """
    valid = []
    for t_vis in vis_ts:
        # find closes timestamp
        idx = take_closest(ir_ts, t_vis)
        if abs(t_vis - ir_ts[idx]) < timedelta(seconds=25):
            valid.append((t_vis, (ir_ts[idx], idx)))

    return valid


def get_days():
    vis_days = get_contained_dirs(os.path.join(RAW_DATA_PATH, "rgb"))
    ir_days = [f[7:-8] for f in get_contained_files(os.path.join(RAW_DATA_PATH, "irccam"))]

    valid = list(sorted(set(vis_days).intersection(ir_days)))
    return valid


def get_vis_timestamps(day):
    filenames = [
        file
        for file in get_contained_files(os.path.join(RAW_DATA_PATH, "rgb", day))
        if file.endswith("_0.jpg")
    ]
    timestamps = [TIMEZONE.localize(datetime.strptime(filename[:-6], TIMESTAMP_FORMAT))
                  for filename in filenames]
    timestamps.sort()
    return timestamps


def get_irc_timestamps(day, irc_file):
    return [convert_timestamp(day, x) for x in irc_file["TM"][0, :]]


def convert_timestamp(day, timestamp):
    """
    Converts irccam timestamps in double format (e.g. 737653.55976907) to
    timestamps capped to the nearest second (e.g. 20190816132643)
    """
    seconds = round(24 * 60 * 60 * (timestamp - math.floor(timestamp)))
    seconds_delta = timedelta(0, seconds)
    day_timestamp = datetime.strptime(day, "%Y%m%d")
    return TIMEZONE.localize(day_timestamp + seconds_delta)


def get_vis_img(timestamp):
    file_path = os.path.join(
        RAW_DATA_PATH,
        "rgb",
        timestamp.strftime(TIMESTAMP_FORMAT_DAY),
        "{}_0.jpg".format(timestamp.strftime(TIMESTAMP_FORMAT)),
    )
    img_vis = cv2.imread(file_path)
    if img_vis is None:
        raise FileNotFoundError("Image {} not found".format(file_path))
    return img_vis


# https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
def take_closest(array, number):
    pos = bisect_left(array, number)
    if pos == 0:
        return 0
    if pos == len(array):
        return len(array) - 1
    before = array[pos - 1]
    after = array[pos]
    if after - number < number - before:
        return pos
    else:
        return pos - 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        create_dataset(dataset_name="test", test=True)
    else:
        name = "main_"
        optimize = False

        print("Dataset classifier: ", end='')
        classifier = input().strip()
        print("Dataset changes: ", end='')
        changes = input().strip()

        print("Would you also like to generate an training optimized version: (y)/n", end='')
        if input().strip() != "n":
            optimize = True

        create_dataset(dataset_name=name + classifier, changelog=changes)
        if optimize:
            optimize_dataset(name + classifier, "optimized_" + classifier)
