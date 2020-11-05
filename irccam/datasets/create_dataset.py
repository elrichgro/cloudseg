"""
Take the raw IRCCAM data and RGB data and create train, val, and test sets
for model training. 

This requires the `extract_irccam_data.py` script to have been run already. That
script reads the huge matlab file and stores the data for each image 
individually. This makes this script a lot faster and easier to tweak and 
experiment with. 

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

import os
import mat73
import datetime
import cv2
import numpy as np
from tqdm import tqdm

from datasets.dataset_filter import is_almost_black, filter_ignored
from datasets.rgb_labeling import create_rgb_label, create_label_image

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data/raw/davos")
DATASET_PATH = os.path.join(PROJECT_PATH, "data/datasets")


def create_dataset(dataset_name="dataset_v1", train_val_test_split=[0.6, 0.2, 0.2]):
    assert (
        sum(train_val_test_split) == 1
    ), "Invalid train_val_test_split: must sum to 1."

    # Split by day to minimize data leak between sets
    days = valid_days()
    train_days, val_days, test_days = split_subsets(days, train_val_test_split)

    train_ts = valid_timestamps_for_days(train_days)
    val_ts = valid_timestamps_for_days(val_days)
    test_ts = valid_timestamps_for_days(test_days)

    create_set("train", train_ts, dataset_name)
    create_set("val", val_ts, dataset_name)
    create_set("test", test_ts, dataset_name)


def split_subsets(data, train_val_test_split):
    size = len(data)
    rand_idx = np.random.permutation(np.arange(size))
    train_size = round(size * train_val_test_split[0])
    val_size = round(size * train_val_test_split[1])
    test_size = round(size * train_val_test_split[2])

    train_idx = rand_idx[:train_size]
    val_idx = rand_idx[train_size : train_size + val_size]
    test_idx = rand_idx[train_size + val_size :]

    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]

    return train_data, val_data, test_data


def create_set(subset, timestamps, dataset_name):
    assert subset in ("train", "val", "test")
    print("Creating {} set".format(subset))
    count = 0
    for timestamp in tqdm(timestamps[:100]):
        count += process_timestamp(timestamp, dataset_name, subset)
    print("{} datapoints in {} set".format(count, subset))


def process_timestamp(timestamp, dataset_name, subset):
    irccam_raw = get_irccam_data(timestamp)
    irccam_img = process_irccam_img(irccam_raw)

    vis_img_raw = get_vis_img(timestamp)
    vis_img = process_vis_img(vis_img_raw)

    # Ignore if filtered out
    if vis_img is None or irccam_img is None:
        return False

    # Create label
    label = create_rgb_label(vis_img)
    label = transform_perspective(label, (irccam_img.shape[0], irccam_img.shape[1]))
    label_image = create_label_image(label)

    # Create day directory if it doesn't exist yet
    img_path = os.path.join(DATASET_PATH, dataset_name, subset, timestamp[:8])
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # Save data
    save_image_to_dataset(irccam_img, img_path, timestamp, "irc")
    save_image_to_dataset(vis_img, img_path, timestamp, "vis")
    save_image_to_dataset(label_image, img_path, timestamp, "label")
    save_array_to_dataset(label, img_path, timestamp, "label")

    return True


def save_array_to_dataset(data, path, timestamp, extension):
    filename = os.path.join(path, "{}_{}.npz".format(timestamp, extension))
    np.savez(filename, data)


def save_image_to_dataset(img, path, timestamp, extension):
    filename = os.path.join(path, "{}_{}.tif".format(timestamp, extension))
    saved = cv2.imwrite(filename, img)
    if not saved:
        raise Exception("Failed to save image {}".format(filename))


def get_contained_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


# from https://stackoverflow.com/a/23316542
def rotate_image(image, angle):
    row, col, _ = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def process_irccam_img(img):
    processed_ir = normalize_irccam_image(img)
    processed_ir = cv2.flip(processed_ir, -1)
    processed_ir = processed_ir[110:530, 80:500]
    return processed_ir


# need to add masking too, but unsure about the rotations
def process_vis_img(img):
    if is_almost_black(img):
        return None
    processed_vis = cv2.resize(img, (640, 480))
    processed_vis = processed_vis[50:470, 105:525]
    processed_vis = cv2.flip(processed_vis, 1)
    processed_vis = rotate_image(processed_vis, -130)
    return processed_vis


def vis_to_irccam_timestamp(timestamp):
    vis_ts = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    ir_ts = vis_ts
    return ir_ts


def normalize_irccam_image(img_ir_raw):
    """
    TODO:
    Find appropriate min and max temperature thresholds for IRCCAM data. The threshold
    is to remove outliers.

    Using 16 bit tif instead of rgb, to prevent data loss on images with outlying
    pixels, should rethink this too.
    Set the actual image to 0-60000. reserve completly white for the mask
    """
    # Threshold to remove outliers
    ir_threshold = (-80.0, 30.0)
    mi = ir_threshold[0]
    ma = ir_threshold[1]
    lower = img_ir_raw < mi
    higher = img_ir_raw > ma
    gray_ir = img_ir_raw - mi
    gray_ir *= 60000 / (ma - mi)
    np.nan_to_num(gray_ir, copy=False, nan=(2 ** 16 - 1))
    gray_ir[lower] = 0
    gray_ir[higher] = 60000
    return gray_ir.astype(np.uint16)


def get_irccam_data(timestamp, data_type="bt"):
    assert data_type in ["bt", "img"], "Unrecognized IRCCAM data type: {}".format(
        data_type
    )
    ir_ts = vis_to_irccam_timestamp(timestamp)
    filename = os.path.join(
        RAW_DATA_PATH,
        "irccam_extract",
        ir_ts.strftime("%Y%m%d"),
        data_type,
        "{}.npz".format(ir_ts.strftime("%Y%m%d%H%M")),
    )
    return np.load(filename)["arr_0"]


def valid_timestamps_for_days(days):
    return np.concatenate([valid_timestamps_for_day(day) for day in days])


def valid_timestamps_for_day(day):
    vis_ts = vis_timestamps_for_day(day)
    ir_ts = irccam_timestamps_for_day(day)
    ts = filter_ignored(np.intersect1d(vis_ts, ir_ts))
    return ts


def valid_days():
    vis_days = np.array(get_contained_dirs(os.path.join(RAW_DATA_PATH, "rgb")))
    ir_days = np.array(
        get_contained_dirs(os.path.join(RAW_DATA_PATH, "irccam_extract"))
    )
    days = filter_ignored(np.intersect1d(vis_days, ir_days))
    return days


def vis_timestamps_for_day(day):
    filenames = [
        file
        for file in get_contained_files(os.path.join(RAW_DATA_PATH, "rgb", day))
        if file.endswith("_0.jpg")
    ]
    timestamps = [filename.replace("_0.jpg", "") for filename in filenames]
    return np.array(timestamps)


def irccam_timestamps_for_day(day, data_type="bt"):
    assert data_type in ["bt", "img"], "Unrecognized IRCCAM data type: {}".format(
        data_type
    )
    filenames = [
        file
        for file in get_contained_files(
            os.path.join(RAW_DATA_PATH, "irccam_extract", day, data_type)
        )
        if file.endswith(".npz")
    ]
    timestamps = [filename.replace(".npz", "") + "00" for filename in filenames]
    return np.array(timestamps)


def get_vis_img(timestamp):
    img_time = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    file_path = os.path.join(
        RAW_DATA_PATH, "rgb", img_time.strftime("%Y%m%d"), "{}_0.jpg".format(timestamp)
    )
    img_vis = cv2.imread(file_path)
    if img_vis is None:
        raise FileNotFoundError("Image {} not found".format(file_path))
    return img_vis


def transform_perspective(img, shape):
    matrix_file = os.path.join(PROJECT_PATH, "irccam/datasets/trans_matrix.csv")
    M = np.loadtxt(matrix_file, delimiter=",")
    return cv2.warpPerspective(img, M, shape, cv2.INTER_NEAREST)


if __name__ == "__main__":
    create_dataset()
