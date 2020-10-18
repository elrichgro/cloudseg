import os
import mat73

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
        process_day_data(day, ignored_timestamps)

def process_day_data(day, ignored_timestamps):
    print('Processing data for {}'.format(day))
    image_filenames = get_contained_files(os.path.join(RAW_DATA_PATH, 'vis', 'images', day))
    image_timestamps = [filename.replace('.jpg', '') for filename in image_filenames]
    image_timestamps = filter_ignored(image_timestamps, ignored_timestamps)
    irccam_data_path = os.path.join(RAW_DATA_PATH, 'irccam', 'irccam_{}_rad.mat'.format(day))
    irccam_data = mat73.loadmat(irccam_data_path)
    for timestamp in image_timestamps:
        # process and save irccam data
        # process and save vis data


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
    row,col = image.shape
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

def get_irccam_idx(timestamp):
    

if __name__ == "__main__":
    create_dataset()

