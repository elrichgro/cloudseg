import cv2
import numpy as np
import datetime
from pysolar.solar import get_azimuth, get_altitude

from irccam.utils.constants import *
from irccam.datasets.masking import apply_full_mask, apply_background_mask


def rotate_image(image, angle):
    """
    Rotate image by given angle. Adapted from https://stackoverflow.com/a/23316542.
    """
    row, col, _ = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def process_irccam_img(img):
    """
    Flip, crop, normalize and apply mask to raw irccam image.
    """
    processed_ir = np.swapaxes(img, 0, 1)
    processed_ir = cv2.flip(processed_ir, -1)
    processed_ir = processed_ir[110:530, 80:500]
    normalize_irccam_image(processed_ir)
    apply_full_mask(processed_ir)
    return processed_ir


def process_irccam_label(img):
    """
    Flip, crop, and apply mask to raw IRCCAM threshold label. 
    """
    processed_ir = np.swapaxes(img, 0, 1)
    processed_ir = cv2.flip(processed_ir, -1)
    processed_ir = processed_ir[110:530, 80:500]
    mask = np.isnan(processed_ir)
    processed_ir[mask] = 0
    processed_ir[np.invert(mask)] = 1
    apply_full_mask(processed_ir, fill=-1)
    return processed_ir


def process_vis_img(img, transform=True, mask=True):
    """
    Resize, flip, rotate, align, and apply mask to raw RGB image.
    
    Parameters
    ----------
    transform : bool
        Apply transformation to align perspective to IRCCAM
    mask: bool
        Apply background mask to output image
    """
    processed_vis = cv2.resize(img, (640, 480))
    processed_vis = processed_vis[50:470, 105:525]
    processed_vis = cv2.flip(processed_vis, 1)
    processed_vis = rotate_image(processed_vis, -130)
    processed_vis = processed_vis.astype("float32")
    if transform:
        processed_vis = transform_perspective(processed_vis, (processed_vis.shape[0], processed_vis.shape[1]))
    if mask:
        apply_background_mask(processed_vis)
    return processed_vis


def sun_correction(vis_img, ir_img, threshold=235):
    """
    Naive approach for filtering out sun flare on IRCCAM images. Pixels over the given 
    threshold in the IRCCAM image are considered sun pixels, and are filtered out
    with a circle on the RGB image.

    Parameters
    ----------
    vis_img : np.array
        Input RGB image
    ir_img: np.array
        Input IRCCAM image
    threshold: 
        Temperature threshold used to detect sun pixels
    """
    img = ir_img.copy()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
    if maxVal > threshold:
        # lets call this a sun
        cv2.circle(img, maxLoc, 40, -1337, -1)
        # draw circle on vis
        cv2.circle(vis_img, maxLoc, 40, (0, 0, 255), 3)
        return img == -1337

    return np.zeros(img.shape, dtype=bool)


def normalize_irccam_image(img_ir):
    """
    Normalize IRCCAM image data.

    Pixel values (temperatures) outside the range [-80, 60] are clipped. The pixel data 
    is then normalized to the range [0, 255]. 
    """
    mi, ma = -80.0, 60.0
    np.nan
    img_ir[np.less(img_ir, mi, where=np.isnan(img_ir) == False)] = mi
    img_ir[np.greater(img_ir, ma, where=np.isnan(img_ir) == False)] = ma
    img_ir -= mi
    img_ir *= 255 / (ma - mi)


def transform_perspective(img, shape):
    """
    Transform RGB image to match the perspective of the IRCCAM. The transformation 
    matrix used was computed by matching SIFT features in IRRCAM-RGB image pairs. 
    """
    matrix_file = os.path.join(PROJECT_PATH, "irccam/datasets/resources/trans_matrix.csv")
    M = np.loadtxt(matrix_file, delimiter=",")
    return cv2.warpPerspective(img, M, shape, cv2.INTER_NEAREST)

