import cv2
import numpy as np
import os

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data/raw/davos")
DATASET_PATH = os.path.join(PROJECT_PATH, "data/datasets")

# a bit ugly global, don't want to carry it around in parameters or load it a million times
# to fix if causes problems when importing this file
# 255 to remove, 0 to keep
MASK = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/common_mask.bmp"), -1)


# from https://stackoverflow.com/a/23316542
def rotate_image(image, angle):
    row, col, _ = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def process_irccam_img(img):
    processed_ir = np.swapaxes(img, 0, 1)
    processed_ir = cv2.flip(processed_ir, -1)
    processed_ir = processed_ir[110:530, 80:500]
    normalize_irccam_image(processed_ir)
    processed_ir = transform_perspective(processed_ir, (processed_ir.shape[0], processed_ir.shape[1]))
    apply_mask(processed_ir, MASK)

    return processed_ir


def process_vis_img(img):
    processed_vis = cv2.resize(img, (640, 480))
    processed_vis = processed_vis[50:470, 105:525]
    processed_vis = cv2.flip(processed_vis, 1)
    processed_vis = rotate_image(processed_vis, -130)
    processed_vis = processed_vis.astype("float32")

    apply_mask(processed_vis, MASK)

    return processed_vis


def normalize_irccam_image(img_ir):
    """
    Using floats from 0-255 since we are saving into np arrays anyway. Modifies the input array in place.

    TODO:
    Find appropriate min and max temperature thresholds for IRCCAM data. The threshold
    is to remove outliers.

    Found quite a few images that have values higher than 30, but still make sense
    """
    # Threshold to remove outliers
    mi, ma = -80.0, 60.0
    img_ir[img_ir < mi] = mi
    img_ir[img_ir > ma] = ma
    img_ir -= mi
    img_ir *= 255 / (ma - mi)


def apply_mask(image, mask):
    """
    Applies the common RGB/IRC mask to the image in place
    """
    image[mask == 255] = np.nan


def transform_perspective(img, shape):
    matrix_file = os.path.join(PROJECT_PATH, "irccam/datasets/trans_matrix.csv")
    M = np.loadtxt(matrix_file, delimiter=",")
    return cv2.warpPerspective(img, M, shape, cv2.INTER_NEAREST)
