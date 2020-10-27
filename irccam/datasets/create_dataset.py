"""
Take the raw IRCCAM data and RGB data and create train, val, and test sets
for model training. 

This requires the `extract_irccam_data.py` script to have been run already. That
script reads the huge matlab file and stores the data for each image 
individually. This makes this script a lot faster and easier to tweak and 
experiment with. 

Basic flow:
- For each day of rgb data:
    - Read all timestamps,
    - Get irccam image corresponding to timestamp, preprocess and save
    - Get rgb image for timestamp, preprocess, create label, and save
    - Filter out black and ignored images

Still to do:
- Fix irccam processing (see todo note below)
- rgb image horizon mask (currently parts of horizon get marked as clouds)
- Split into train, val, and test folders (currently just stored as one set)
"""

import os
import mat73
import datetime
import cv2
import numpy as np

from datasets.dataset_filter import is_almost_black, filter_ignored
from datasets.rgb_labeling import create_rgb_label

PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../..')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw/davos')
DATASET_PATH = os.path.join(PROJECT_PATH, 'data/datasets')


def create_dataset(dataset_name='dataset_v1'):
    vis_days = get_contained_dirs(os.path.join(RAW_DATA_PATH, 'rgb'))
    vis_days = filter_ignored(vis_days)
    offset = 0
    for day in vis_days:
        count = process_day_data(day, dataset_name, 'train', offset)
        offset += count


def process_day_data(day, dataset_name, subset, offset):
    print('Processing data for {}'.format(day))
    image_filenames = get_contained_files(os.path.join(RAW_DATA_PATH, 'rgb', day))
    image_filenames = [file for file in image_filenames if file.endswith('_0.jpg')]
    image_timestamps = [filename.replace('_0.jpg', '') for filename in image_filenames]
    image_timestamps = filter_ignored(image_timestamps)
    img_dir = os.path.join(DATASET_PATH, dataset_name, subset)
    count = 0
    image_timestamps.sort()
    cloud_labels = []
    for idx, timestamp in enumerate(image_timestamps):
        # Remove this to process all data
        if count > 5:
            break
        count += 1

        irccam_idx = timestamp_to_idx(timestamp)
        try:
            irccam_raw = get_irccam_bt_data(day, irccam_idx)
        except FileNotFoundError:
            print("Skipping the rest of {} after".format(day, irccam_idx))
            # the irccam data does not exist for this image (sometimes the data is not complete for a day eg. irccam_20180511_rad.mat)
            break
        irccam_img = process_irccam_img(irccam_raw)

        vis_img_raw = get_vis_img(timestamp)
        vis_img = process_vis_img(vis_img_raw)

        # save if all filtering was OK
        if vis_img is not None and irccam_img is not None:
            img_path = os.path.join(img_dir, 'images', day)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            irccam_img_filename = os.path.join(img_path, '{}_irc.tif'.format(irccam_idx))
            saved = cv2.imwrite(irccam_img_filename, irccam_img)
            if not saved:
                raise Exception('Failed to save image {}'.format(irccam_img_filename))
            vis_img_filename = os.path.join(img_path, '{}_vis.tif'.format(irccam_idx))
            saved = cv2.imwrite(vis_img_filename, vis_img)
            if not saved:
                raise Exception('Failed to save image {}'.format(vis_img_filename))

            label_filename = os.path.join(img_path, '{}_labels.npz'.format(irccam_idx))
            np.savez(label_filename, create_rgb_label(vis_img))

    print('Processed {} images for day {}'.format(count, day))

    return count


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
    processed_vis = rotate_image(processed_vis, -120)
    return processed_vis


def timestamp_to_idx(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    start_of_day = datetime.datetime.combine(img_time.date(), datetime.time(0, 0, 0, 0))
    return round(((img_time - start_of_day).total_seconds() / 60.0)) - 1


"""
TODO:
this function currently reads raw irccam data and then maps to rgb range 
by taking the min and max values in the image. This is wrong though, because
it uses a different range for each image.

We should instead figure out what are suitable min and max ir values to use
over all these images. Then for each one we cap at those min and max
values and map to grayscale range.

Using 16 bit tif instead of rgb, to prevent data loss on images with outlying pixels, should rethink this too
Set the actual image to 0-60000. reserve completly white for the mask
"""


def normalize_irccam_image(img_ir_raw):
    mi = np.nanmin(img_ir_raw)
    ma = np.nanmax(img_ir_raw)
    gray_ir = img_ir_raw - mi
    gray_ir *= (60000 / (ma - mi))
    np.nan_to_num(gray_ir, copy=False, nan=(2 ** 16 - 1))
    return gray_ir.astype(np.uint16)


def get_irccam_bt_data(day, idx):
    filename = os.path.join(RAW_DATA_PATH, 'irccam_extract', day, 'bt', '{}.npz'.format(idx))
    return np.load(filename)['arr_0']


def get_vis_img(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    file_path = os.path.join(RAW_DATA_PATH, 'rgb', img_time.strftime('%Y%m%d'), '{}_0.jpg'.format(timestamp))
    img_vis = cv2.imread(file_path)
    if img_vis is None:
        raise FileNotFoundError('Image {} not found'.format(file_path))
    return img_vis


if __name__ == "__main__":
    create_dataset()
