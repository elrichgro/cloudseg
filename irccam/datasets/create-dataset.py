import os
import mat73
import datetime
import cv2
import numpy as np

RAW_DATA_DIR = 'data/remote'
DATASET_DIR = 'data/datasets'

PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../..')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, RAW_DATA_DIR)
DATASET_PATH = os.path.join(PROJECT_PATH, DATASET_DIR)

def create_dataset():
    # Read raw data folder to get vis days
    # Read ignore file
    # For each vis day:
    # - read all image names
    # - ignore ignored ones
    # - for each one: read irccam one, 
    ignored_timestamps = []
    vis_days = get_contained_dirs(os.path.join(RAW_DATA_PATH, 'vis', 'images'))
    vis_days = filter_ignored(vis_days, ignored_timestamps)

    for day in vis_days:
        process_day_data(day, ignored_timestamps, 'train', 0)

def process_day_data(day, ignored_timestamps, subset, offset):
    print('Processing data for {}'.format(day))
    image_filenames = get_contained_files(os.path.join(RAW_DATA_PATH, 'vis', 'images', day))
    image_timestamps = [filename.replace('.jpg', '') for filename in image_filenames]
    # print(image_timestamps)
    image_timestamps = filter_ignored(image_timestamps, ignored_timestamps)
    irccam_data_path = os.path.join(RAW_DATA_PATH, 'irccam', 'irccam_{}_rad.mat'.format(day))
    print('loading ir data')
    irccam_data = mat73.loadmat(irccam_data_path)
    print('done!')
    img_dir = os.path.join(DATASET_PATH, subset)
    count = 0
    for idx, timestamp in enumerate(image_timestamps):
        if count > 5:
            break
        count+=1

        irccam_idx = timestamp_to_idx(timestamp)
        irccam_raw = get_irccam_img(irccam_data, irccam_idx)
        irccam_img = process_irccam_img(irccam_raw)
        irccam_img_path = os.path.join(img_dir, 'irccam', '{}.jpg'.format(offset+idx))
        saved = cv2.imwrite(irccam_img_path, irccam_img)
        if saved == False:
            raise Exception('Failed to save image {}'.format(irccam_img_path))

        # print(irccam_img_path)

        vis_img_raw = get_vis_img(timestamp)
        vis_img = process_vis_img(vis_img_raw)
        vis_img_path = os.path.join(img_dir, 'vis', '{}.jpg'.format(offset+idx))
        saved = cv2.imwrite(vis_img_path, vis_img)
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

def crop_irrcam_img(img):
    crop_img = img[20:440, 120:540]
    return crop_img

def crop_vis_img(img):
    crop_img = img[50:470, 105:525]
    return crop_img

def scale_vis_img(img):
    scale_img = cv2.resize(img, (640, 480))
    return scale_img

# from https://stackoverflow.com/a/23316542
def rotate_image(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def flip_and_rotate_vis_img(img):
    flip_img = cv2.flip(img, 1)
    rotate_img = rotate_image(flip_img, -130)
    return rotate_img

def process_irccam_img(img):
    cropped_ir = crop_irrcam_img(img)
    return cropped_ir

def process_vis_img(img):
    processed_vis = scale_vis_img(img)
    processed_vis = crop_vis_img(processed_vis)
    processed_vis = flip_and_rotate_vis_img(processed_vis)
    return processed_vis

def timestamp_to_idx(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    start_of_day = datetime.datetime.combine(img_time.date(), datetime.time(0,0,0,0))
    return round(((img_time - start_of_day).total_seconds() / 60.0)) - 1

def get_irccam_img(irccam_data, idx):
    img_ir = np.nan_to_num(irccam_data['BT'][:,:,idx])
    gray_ir = img_ir - img_ir.min()
    gray_ir *= (255.0/gray_ir.max())
    gray_ir = np.array(gray_ir.round(), dtype = np.uint8)
    return gray_ir

def get_vis_img(timestamp):
    img_time = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    file_path = os.path.join(RAW_DATA_PATH, 'vis', 'images', img_time.strftime('%Y%m%d'), '{}.jpg'.format(timestamp))
    img_vis = cv2.imread(file_path)
    if img_vis is None:
        raise Exception('Image {} not found'.format(file_path))
    return img_vis

if __name__ == "__main__":
    create_dataset()

