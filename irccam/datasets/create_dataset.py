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
    - Save files

Still to do:
- Fix irccam processing (see todo note below)
- rgb image horizon mask (currently parts of horizon get marked as clouds)

Considerations:
- Save images as numpy array instead of as images to make life easier/more uniform for import/export
- irccam data is between approx. -500 and 60, although most images at between -60, 60. What kind of grayscale to user?
"""

import cv2
import math
import numpy as np
import h5py

from tqdm import tqdm
from bisect import bisect_left

from datetime import datetime, timedelta
from pytz import timezone

from sklearn.model_selection import train_test_split

from datasets.filesystem import get_contained_dirs, get_contained_files
from irccam.datasets.image_processing import process_irccam_img, process_vis_img
from irccam.datasets.dataset_filter import filter_sun, filter_ignored_days
from irccam.datasets.rgb_labeling import create_rgb_label_julian

from irccam.utils.definitions import *

tz = timezone("Europe/Zurich")


def create_dataset(dataset_name="dataset_v1", sizes=(0.6, 0.2, 0.2)):
    assert sum(sizes) == 1, "Split sizes to not sum up to 1"

    print("Creating dataset")
    days = valid_days()

    # Create data directory if it doesn't exist yet
    path = os.path.join(DATASET_PATH, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

        # Save splits
    train, test, val = sizes
    days_train, days_testval = train_test_split(days, test_size=test + val, train_size=train)
    days_val, days_test = train_test_split(days_testval, test_size=test / (test + val), train_size=val / (test + val))
    np.savetxt(os.path.join(path, "train.txt"), days_train, fmt="%s")
    np.savetxt(os.path.join(path, "test.txt"), days_test, fmt="%s")
    np.savetxt(os.path.join(path, "val.txt"), days_val, fmt="%s")

    for i, day in enumerate(days):
        print("Processing day {} - {}/{}".format(day, i + 1, len(days)))
        process_day(day, dataset_name)


def process_day(day, dataset_name):
    # create output directory
    data_path = os.path.join(DATASET_PATH, dataset_name)
    data_filename = os.path.join(data_path, "{}.h5".format(day))
    if os.path.exists(data_filename):
        return

    previews_path = os.path.join(data_path, "previews")
    preview_filename = os.path.join(previews_path, "{}_preview.mp4".format(day))
    if not os.path.exists(previews_path):
        os.makedirs(previews_path)

    with h5py.File(os.path.join(RAW_DATA_PATH, "irccam", "irccam_{}_rad.mat".format(day)), "r") as fr:
        irc_raw = fr["BT"]
        irc_timestamps = get_irc_timestamps(day, fr)
        vis_timestamps = get_vis_timestamps(day)

        matching_timestamps = match_timestamps(irc_timestamps, vis_timestamps)
        filtered_timestamps = filter_sun(matching_timestamps, day)
        n = len(filtered_timestamps)
        if n == 0:
            return False

        timestamps = []
        vis_images = np.empty((n, 420, 420, 3), dtype="float32")
        irc_images = np.empty((n, 420, 420), dtype="float32")
        labels1 = np.empty((n, 420, 420), dtype="bool")
        labels2 = np.empty((n, 420, 420), dtype="bool")
        labels3 = np.empty((n, 420, 420), dtype="bool")

        # We'll output a preview video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        video_out = cv2.VideoWriter(preview_filename, fourcc, 30.0, (2100, 420))

        print("Processing images")
        for i, (vis_ts, (irc_ts, irc_idx)) in enumerate(tqdm(filtered_timestamps)):
            # lets just keep one timestamp, time sync will have to be done in this file anyway
            timestamps.append(irc_ts.strftime(TIMESTAMP_FORMAT))

            irc_img = irc_raw[irc_idx, :, :]
            irc_img = process_irccam_img(irc_img)
            vis_img = get_vis_img(vis_ts)
            vis_img = process_vis_img(vis_img)

            # Create labels
            label1 = create_rgb_label_julian(vis_img, cloud_ref=2.35)
            label1_image = create_label_image(label1)
            label2 = create_rgb_label_julian(vis_img, cloud_ref=2.7)
            label2_image = create_label_image(label2)
            label3 = create_rgb_label_julian(vis_img, cloud_ref=3)
            label3_image = create_label_image(label3)

            comparison_image = concat_images(irc_img, vis_img, label1_image, label2_image, label3_image)
            video_out.write(comparison_image)  # Write out frame to video
            # save_image_to_dataset(comparison_image, previews_path, vis_ts, "preview")

            vis_images[i, :, :, :] = vis_img
            irc_images[i, :, :] = irc_img
            labels1[i, :, :] = label1
            labels2[i, :, :] = label2
            labels3[i, :, :] = label3

        video_out.release()

        print("Saving data")
        with h5py.File(data_filename, "w") as fw:
            fw.create_dataset("timestamp", data=timestamps)
            fw.create_dataset("irc", data=irc_images, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("vis", data=vis_images, chunks=(1, 420, 420, 3), compression="lzf")
            fw.create_dataset("labels1", data=labels1, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("labels2", data=labels2, chunks=(1, 420, 420), compression="lzf")
            fw.create_dataset("labels3", data=labels3, chunks=(1, 420, 420), compression="lzf")

        return True


def save_arrays_to_dataset(data, path, timestamp):
    filename = os.path.join(path, "{}.npz".format(timestamp[0].strftime(TIMESTAMP_FORMAT_MINUTE)))
    np.savez(filename, **data)


def save_image_to_dataset(img, path, timestamp, extension):
    filename = os.path.join(path, "{}_{}.jpg".format(timestamp.strftime(TIMESTAMP_FORMAT_MINUTE), extension))
    saved = cv2.imwrite(filename, img)
    if not saved:
        raise Exception("Failed to save image {}".format(filename))


def concat_images(*images):
    processed = []
    for i in images:
        i = np.nan_to_num(i, copy=True, nan=255)
        if i.ndim == 2:
            i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        i = i.astype(np.uint8)
        processed.append(i)

    return np.concatenate(processed, axis=1)


def create_label_image(labels):
    img = np.zeros((labels.shape[0], labels.shape[1], 3))
    img[np.where(np.isnan(labels))] = float('nan')
    img[:, :, 0] = labels * 255
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


def valid_days():
    vis_days = get_contained_dirs(os.path.join(RAW_DATA_PATH, "rgb"))
    ir_days = [f[7:-8] for f in get_contained_files(os.path.join(RAW_DATA_PATH, "irccam"))]

    valid = list(sorted(set(vis_days).intersection(ir_days)))
    valid = filter_ignored_days(valid)
    return valid


def get_vis_timestamps(day):
    filenames = [
        file
        for file in get_contained_files(os.path.join(RAW_DATA_PATH, "rgb", day))
        if file.endswith("_0.jpg")
    ]
    timestamps = [tz.localize(datetime.strptime(filename[:-6], TIMESTAMP_FORMAT))
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
    return tz.localize(day_timestamp + seconds_delta)


def get_vis_img(timestamp):
    file_path = os.path.join(
        RAW_DATA_PATH, "rgb", timestamp.strftime(TIMESTAMP_FORMAT_DAY),
        "{}_0.jpg".format(timestamp.strftime(TIMESTAMP_FORMAT))
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
    create_dataset()
