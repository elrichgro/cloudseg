import cv2
import numpy as np
from skimage.filters import threshold_otsu

from cloudseg.datasets.labelling_helpers import (
    Pipeline,
    SimpleRatio,
    ApplyMask,
    GaussianBlur,
    CombineLabels,
    AdaptiveThreshold,
    MaskedOtsu,
    FixedThreshold,
)
from cloudseg.datasets.masking import full_mask, background_mask


def create_label_adaptive(img):
    """
    Generate a segmentation label for the given RGB image using an adaptive method.
    This method is a combination of:
        - Pixel level mean thresholding (good at detecting cirrus clouds)
        - Otsu thresholding on the whole image (handles mixed sky conditions)
        - Fixed threshold (handles days where there is complete cloud coverage)

    Output: int array with the same size as the input image. 0 indicates non-cloud
    pixels, 1 indicates cloud pixels, and -1 indicates masked areas.
    """
    pipeline = Pipeline(
        [
            SimpleRatio(),
            ApplyMask(background_mask, mask_val=40),  # Workaround for making adaptive threshold work for masked images
            GaussianBlur(size=(5, 5)),
            CombineLabels(
                [
                    CombineLabels(
                        [
                            AdaptiveThreshold(
                                C=1,
                                block_size=111,
                                morph_size=(3, 3),
                                method=cv2.ADAPTIVE_THRESH_MEAN_C,
                            ),  # Picks up individual cirrus clouds
                            MaskedOtsu(background_mask),  # Good for mixed condition days
                            FixedThreshold(threshold=48),  # Takes care of very cloudy days
                        ],
                        operation="or",
                    ),
                    FixedThreshold(threshold=23),  # Helps with completely open days
                ],
                operation="and",
            ),
            ApplyMask(full_mask, mask_val=-1),
        ]
    )
    result = pipeline.apply(img)
    result[result == 255] = 1
    return result


def create_label_rb_threshold(image, cloud_ref=2.35):
    """
    Generate a segmentation mask for the given RGB image, based on a simple
    red/blue ratio threshold. Input `cloud_ref` is the threshold value to apply.

    Output: int array with the same size as the input image. 0 indicates non-cloud
    pixels, 1 indicates cloud pixels, and -1 indicates masked areas.
    """
    zeros = np.logical_or(image[:, :, 1] == 0, image[:, :, 2] == 0)
    nans = np.logical_or(np.logical_or(np.isnan(image[:, :, 0]), np.isnan(image[:, :, 1])), np.isnan(image[:, :, 2]))

    ok = np.logical_not(np.logical_or(zeros, nans))

    rat = np.zeros(image.shape[:2])
    rat[ok] = image[:, :, 0][ok] / image[:, :, 2][ok] + image[:, :, 0][ok] / image[:, :, 1][ok]

    result = np.zeros(rat.shape, dtype="byte")
    result[rat < cloud_ref] = 1
    result[nans] = -1
    result[full_mask == 255] = -1

    return result


########################################################
############## Pipeline ################################
########################################################


class Pipeline:
    def __init__(self, methods):
        self.methods = methods

    def process(self, imgs):
        results = []
        for img in imgs:
            results.append(self.process_one(img))
        return results

    def process_one(self, img):
        results = [img]
        for method in self.methods:
            img = method.apply(img)
            results.append(img)
        return results

    def apply(self, img):
        for method in self.methods:
            img = method.apply(img)
        return img


########################################################
############## Ratios ##################################
########################################################


class RBRatio:
    def __init__(self):
        pass

    def apply(self, image):
        zeros = np.logical_or(image[:, :, 1] == 0, image[:, :, 2] == 0)
        nans = np.logical_or(
            np.logical_or(np.isnan(image[:, :, 0]), np.isnan(image[:, :, 1])),
            np.isnan(image[:, :, 2]),
        )

        ok = np.logical_not(np.logical_or(zeros, nans))
        rat = np.zeros(image.shape[:2])
        rat[ok] = image[:, :, 0][ok] / image[:, :, 2][ok] + image[:, :, 0][ok] / image[:, :, 1][ok]

        img = rat
        img -= img.min()
        # img *= 255.0/img.max()
        img *= 255.0 / 4
        img[np.where(img > 255.0)] = 255.0
        img[np.where(img < 0.0)] = 0.0
        img = img.astype(np.uint8)

        return img


class SimpleRatio:
    def apply(self, image):
        zeros = image[:, :, 0] == 0
        nans = np.logical_or(
            np.logical_or(np.isnan(image[:, :, 0]), np.isnan(image[:, :, 1])),
            np.isnan(image[:, :, 2]),
        )

        ok = np.logical_not(np.logical_or(zeros, nans))
        rat = np.zeros(image.shape[:2])
        rat[ok] = image[:, :, 2][ok] / image[:, :, 0][ok]

        img = rat
        img -= img.min()
        # img *= 255.0/img.max()
        img *= 255.0 / 4
        img[np.where(img > 255.0)] = 255.0
        img[np.where(img < 0.0)] = 0.0
        img = img.astype(np.uint8)

        return img


########################################################
############## Thresholds ##############################
########################################################


class AdaptiveThreshold:
    def __init__(
        self,
        max=255,
        method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type=cv2.THRESH_BINARY,
        block_size=191,
        C=2,
        morph_shape=cv2.MORPH_ELLIPSE,
        morph_size=(3, 3),
        morph_type=cv2.MORPH_OPEN,
    ):
        self.max = max
        self.method = method
        self.thresh_type = thresh_type
        self.block_size = block_size
        self.C = C
        self.morph_shape = morph_shape
        self.morph_size = morph_size
        self.morph_type = morph_type

    def apply(self, img):
        img = cv2.adaptiveThreshold(img, self.max, self.method, self.thresh_type, self.block_size, self.C)
        # Morphology to remove noise
        if self.morph_shape:
            kernel = cv2.getStructuringElement(self.morph_shape, self.morph_size)
            img = cv2.morphologyEx(img, self.morph_type, kernel)

        return img


class FixedThreshold:
    def __init__(self, threshold=160, max=255, thresh_type=cv2.THRESH_BINARY):
        self.threshold = threshold
        self.max = max
        self.thresh_type = thresh_type

    def apply(self, img):
        _, result = cv2.threshold(img, self.threshold, self.max, self.thresh_type)
        return result


class OtsuThreshold:
    def apply(self, img):
        _, label = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return label


class CombineLabels:
    def __init__(self, methods, operation="or"):
        self.methods = methods
        self.operation = operation

    def apply(self, img):
        labels = [method.apply(img) for method in self.methods]
        label = np.zeros(img.shape)
        if self.operation == "or":
            label[np.logical_or.reduce([l == 255 for l in labels])] = 255
        elif self.operation == "and":
            label[np.logical_and.reduce([l == 255 for l in labels])] = 255
        return label


########################################################
############## Blurring ################################
########################################################


class Blur:
    def __init__(self, size=(30, 30)):
        self.size = size

    def apply(self, img):
        return cv2.blur(img, self.size)


class Identity:
    def apply(self, img):
        return img


class BilateralFilter:
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img):
        return cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)


class GaussianBlur:
    def __init__(self, size=(5, 5), sigma=0):
        self.size = size
        self.sigma = sigma

    def apply(self, img):
        return cv2.GaussianBlur(img, self.size, self.sigma)


class MedianBlur:
    def __init__(self, size=5):
        self.size = size

    def apply(self, img):
        return cv2.medianBlur(img, self.size)


########################################################
############## Masks ###################################
########################################################


class ApplyMask:
    def __init__(self, mask, mask_val=255):
        self.mask = mask / 255
        self.mask_val = mask_val

    def apply(self, img):
        result = img.copy()
        result[np.where(self.mask == 1)] = self.mask_val
        return result


class MaskedOtsu:
    def __init__(self, mask):
        self.mask = mask / 255

    def apply(self, img):
        masked_img = np.ma.masked_array(img, self.mask)
        thr = threshold_otsu(masked_img.compressed())
        return FixedThreshold(threshold=thr).apply(img)
