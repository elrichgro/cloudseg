import cv2
import numpy as np

from datasets.dataset_filter import is_almost_black

# from https://stackoverflow.com/a/23316542
def rotate_image(image, angle):
    row, col, _ = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def process_irccam_img(img, dtype=np.uint16):
    processed_ir = normalize_irccam_image(img, dtype=dtype)
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


def normalize_irccam_image(img_ir_raw, dtype=np.uint16):
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
    out_min = np.iinfo(dtype).min
    out_max = np.iinfo(dtype).max
    gray_ir = img_ir_raw - mi
    gray_ir *= out_max / (ma - mi)
    np.nan_to_num(gray_ir, copy=False, nan=out_max)
    gray_ir[lower] = out_min
    gray_ir[higher] = out_max
    img = gray_ir.astype(dtype)
    return img
