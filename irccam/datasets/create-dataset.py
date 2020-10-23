"""
Take the raw IRCCAM data and RGB data and create train, val, and test sets
for model training. 

This requires the `extract-irccam-data.py` script to have been run already. That
script reads the huge matlab file and stores the data for each image 
individually. This makes this script a lot faster and easier to tweak and 
experiment with. 

Basic flow:
- For each day of rgb data:
    - Read all timestamps,
    - Get irccam image corresponding to timestamp, preprocess and save
    - Get rgb image for timestamp, preprocess, create label, and save

Still to do:
- Fix irccam processing (see todo note below)
- Create label from rgb images (currently rgb image itself is saved)
- Split into train, val, and test folders (currently just stored as one set)
"""

import os
import mat73
import datetime
import cv2
import numpy as np


PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../..')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw/davos')
DATASET_PATH = os.path.join(PROJECT_PATH, 'data/datasets')

def create_dataset(dataset_name='dataset_v1'):
    ignored_timestamps = get_ignored_timestamps()
    vis_days = get_contained_dirs(os.path.join(RAW_DATA_PATH, 'rgb'))
    vis_days = filter_ignored(vis_days, ignored_timestamps)

    offset = 0
    for day in vis_days:
        count = process_day_data(day, dataset_name, ignored_timestamps, 'train', offset)
        offset += count

def process_day_data(day, dataset_name, ignored_timestamps, subset, offset):
    print('Processing data for {}'.format(day))
    image_filenames = get_contained_files(os.path.join(RAW_DATA_PATH, 'rgb', day))
    image_filenames = [file for file in image_filenames if file.endswith('_0.jpg')]
    image_timestamps = [filename.replace('_0.jpg', '') for filename in image_filenames]
    image_timestamps = filter_ignored(image_timestamps, ignored_timestamps)
    img_dir = os.path.join(DATASET_PATH, dataset_name, subset)
    count = 0
    image_timestamps.sort()
    for idx, timestamp in enumerate(image_timestamps):
        # Remove this to process all data
        if count > 5:
            break
        count+=1

        irccam_idx = timestamp_to_idx(timestamp)
        irccam_raw = get_irccam_bt_data(day, irccam_idx)
        irccam_img = process_irccam_img(irccam_raw)
        irccam_img_path = os.path.join(img_dir, 'irccam')
        if not os.path.exists(irccam_img_path):
            os.makedirs(irccam_img_path)
        irccam_img_filename = os.path.join(irccam_img_path, '{}.jpg'.format(offset+idx))
        saved = cv2.imwrite(irccam_img_filename, irccam_img)
        if saved == False:
            raise Exception('Failed to save image {}'.format(irccam_img_path))

        vis_img_raw = get_vis_img(timestamp)
        vis_img = process_vis_img(vis_img_raw)
        vis_img_path = os.path.join(img_dir, 'labels')
        if not os.path.exists(vis_img_path):
            os.makedirs(vis_img_path)
        vis_img_filename = os.path.join(vis_img_path, '{}.jpg'.format(offset+idx))
        saved = cv2.imwrite(vis_img_filename, vis_img)
        if saved == False:
            raise Exception('Failed to save image {}'.format(vis_img_path))

    print('Processed {} images for day {}'.format(count, day))
    return count


def filter_ignored(items, ignore_list):
    return [i for i in items if i not in ignore_list]

def get_contained_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# from https://stackoverflow.com/a/23316542
def rotate_image(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def process_irccam_img(img):
    processed_ir = cv2.flip(img, -1)
    processed_ir = processed_ir[110:530, 80:500]
    return processed_ir

def process_vis_img(img):
    processed_vis = cv2.resize(img, (640, 480))
    processed_vis = processed_vis[50:470, 105:525]
    processed_vis = cv2.flip(processed_vis, 1)
    processed_vis = rotate_image(processed_vis, -120)
    return processed_vis

def timestamp_to_idx(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    start_of_day = datetime.datetime.combine(img_time.date(), datetime.time(0,0,0,0))
    return round(((img_time - start_of_day).total_seconds() / 60.0)) - 1

"""
TODO:
this function currently reads raw irccam data and then maps to rgb range 
by taking the min and max values in the image. This is wrong though, because
it uses a different range for each image.

We should instead figure out what are suitable min and max ir values to use
over all these images. Then for each one we cap at those min and max
values and map to grayscale range.
"""
def get_irccam_bt_data(day, idx):
    filename = os.path.join(RAW_DATA_PATH, 'irccam_extract', day, 'bt', '{}.npz'.format(idx))
    img_ir_raw = np.load(filename)['arr_0']
    img_ir = np.nan_to_num(img_ir_raw)
    gray_ir = img_ir - img_ir.min()
    gray_ir *= (255.0/gray_ir.max())
    gray_ir = np.array(gray_ir.round(), dtype = np.uint8)
    return gray_ir

def get_vis_img(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    file_path = os.path.join(RAW_DATA_PATH, 'rgb', img_time.strftime('%Y%m%d'), '{}_0.jpg'.format(timestamp))
    img_vis = cv2.imread(file_path)
    if img_vis is None:
        raise Exception('Image {} not found'.format(file_path))
    return img_vis

def get_ignored_timestamps():
    filename = os.path.join(PROJECT_PATH, 'irccam','datasets', 'ignored_timestamps.txt')
    with open(filename) as f:
        content = f.readlines()
    content = [ts.strip() for ts in content]
    return content

if __name__ == "__main__":
    create_dataset()

