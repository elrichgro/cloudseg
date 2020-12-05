import cv2
import numpy as np
import os
from irccam.utils.definitions import *
from datasets.masks import common_mask, background_mask


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
    apply_common_mask(processed_ir)

    return processed_ir


def process_vis_img(img):
    processed_vis = cv2.resize(img, (640, 480))
    processed_vis = processed_vis[50:470, 105:525]
    processed_vis = cv2.flip(processed_vis, 1)
    processed_vis = rotate_image(processed_vis, -130)
    processed_vis = processed_vis.astype("float32")
    processed_vis = transform_perspective(processed_vis, (processed_vis.shape[0], processed_vis.shape[1]))

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


def apply_background_mask(img):
    img[background_mask == 255] = np.nan


def apply_common_mask(img):
    img[common_mask == 255] = np.nan


def transform_perspective(img, shape):
    matrix_file = os.path.join(PROJECT_PATH, "irccam/datasets/trans_matrix.csv")
    M = np.loadtxt(matrix_file, delimiter=",")
    return cv2.warpPerspective(img, M, shape, cv2.INTER_NEAREST)
