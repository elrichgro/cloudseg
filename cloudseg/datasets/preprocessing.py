import cv2
import numpy as np

from cloudseg.utils.constants import *
from cloudseg.datasets.masking import apply_mask, apply_background_mask, full_mask


def rotate_image(img, angle):
    """
    Rotate image by given angle. Adapted from https://stackoverflow.com/a/23316542.
    """
    row, col, _ = img.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_img = cv2.warpAffine(img, rot_mat, (col, row))
    return new_img


def get_cropping_indices(img, crop_size=(420, 420)):
    """
    Get indices to crop a raw irccam image (640 x 640) to the size used by
    our models (420 x 420). The indices are calculated by finding the leftmost
    and uppermost non-nan values in the raw image.
    """
    left_idx = max(0, np.isnan(img).min(0).argmin() - 10)
    right_idx = left_idx + crop_size[1]
    upper_idx = max(0, np.isnan(img).min(1).argmin() - 10)
    lower_idx = upper_idx + crop_size[0]

    return (upper_idx, lower_idx), (left_idx, right_idx)


def crop_image(img, crop_idx):
    """
    Returns a cropped version of img with the provided crop indices
    """
    upper, lower = crop_idx[0]
    left, right = crop_idx[1]
    return img[upper:lower, left:right]


def process_irccam_img(img, crop_idx=((110, 530), (80, 500)), flip=True, mask=full_mask):
    """
    Flip, crop, normalize and apply mask to raw irccam image.
    """
    processed_ir = img
    if flip:
        processed_ir = np.swapaxes(img, 0, 1)
        processed_ir = cv2.flip(processed_ir, -1)
    if crop_idx:
        processed_ir = crop_image(processed_ir, crop_idx)
    normalize_irccam_image(processed_ir)
    apply_mask(processed_ir, mask)
    return processed_ir


def process_irccam_label(img, crop_idx=((110, 530), (80, 500)), flip=True, mask=full_mask):
    """
    Flip, crop, and apply mask to raw IRCCAM threshold label.
    """
    processed_ir = img
    if flip:
        processed_ir = np.swapaxes(img, 0, 1)
        processed_ir = cv2.flip(processed_ir, -1)
    if crop_idx:
        processed_ir = crop_image(processed_ir, crop_idx)
    nan_mask = np.isnan(processed_ir)
    processed_ir[nan_mask] = 0
    processed_ir[np.invert(nan_mask)] = 1
    apply_mask(processed_ir, mask, fill=-1)
    return processed_ir


def create_mask(mask, crop_idx):
    """
    Create a mask from the raw mask in an irccam matlab file.
    """
    return crop_image(np.array(mask), crop_idx) * 255


def apply_clear_sky(img, clear_sky):
    """
    Subtract clear sky reference to IRCCAM image, clip to range [-30,100]
    and normalize to [0,1] range.
    """
    output = img - clear_sky

    # Scale to [0,1]
    mi = -30
    ma = 100
    np.nan_to_num(output, copy=False, nan=mi)
    output[img == 255] = mi
    output[output < mi] = mi
    output[output > ma] = ma
    output -= mi
    output /= ma - mi
    return output


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
    Transform RGB image to match the perspective of the irccam. The transformation
    matrix used was computed by matching SIFT features in IRRCAM-RGB image pairs.
    """
    matrix_file = os.path.join(PROJECT_PATH, "cloudseg/datasets/resources/trans_matrix.csv")
    M = np.loadtxt(matrix_file, delimiter=",")
    return cv2.warpPerspective(img, M, shape, cv2.INTER_NEAREST)
