"""
Takes the RGB image and produces labels for each pixel using the formula Julian provided
"""
"""
Filtering of images based on various criteria
"""

import os
import cv2
import numpy as np

from irccam.datasets.labelling_helpers import (
    Pipeline,
    SimpleRatio,
    ApplyMask,
    GaussianBlur,
    CombineLabels,
    AdaptiveThreshold,
    MaskedOtsu,
    FixedThreshold,
)
from irccam.datasets.masks import common_mask, background_mask


def create_label_adaptive(img):
    pipeline = Pipeline(
        [
            SimpleRatio(),
            ApplyMask(background_mask, mask_val=40),  # Workaround for making adaptive threshold fork for masked images
            GaussianBlur(size=(5, 5)),
            CombineLabels(
                [
                    CombineLabels(
                        [
                            AdaptiveThreshold(
                                C=1, block_size=111, morph_size=(3, 3), method=cv2.ADAPTIVE_THRESH_MEAN_C,
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
            ApplyMask(common_mask, mask_val=-1),
        ]
    )
    result = pipeline.apply(img)
    result[result == 255] = 1
    return result


def create_rgb_label_julian(image, cloud_ref=2.35):
    # For the pixel cloud treshhold, this is the relevant part:
    #
    # rat=imid(:,:,3)./imid(:,:,1) + imid(:,:,3)./imid(:,:,2);
    #
    # imi_cloud = rat<cloud_ref;   % the smaller the whiter
    #
    # with cloud_ref=2.15 for this camera.
    # use ratios by Julian
    # careful RGB -> BGR

    zeros = np.logical_or(image[:, :, 1] == 0, image[:, :, 2] == 0)
    nans = np.logical_or(np.logical_or(np.isnan(image[:, :, 0]), np.isnan(image[:, :, 1])), np.isnan(image[:, :, 2]))

    ok = np.logical_not(np.logical_or(zeros, nans))

    rat = np.zeros(image.shape[:2])
    rat[ok] = image[:, :, 0][ok] / image[:, :, 2][ok] + image[:, :, 0][ok] / image[:, :, 1][ok]

    result = np.zeros(rat.shape, dtype="byte")
    result[rat < cloud_ref] = 1
    result[nans] = -1
    result[common_mask == 255] = -1

    return result


def create_rgb_label_alt(image, cloud_ref=2.15):
    # For the pixel cloud treshhold, this is the relevant part:
    #
    # rat=imid(:,:,3)./imid(:,:,1) + imid(:,:,3)./imid(:,:,2);
    #
    # imi_cloud = rat<cloud_ref;   % the smaller the whiter
    #
    # with cloud_ref=2.15 for this camera.
    # use ratios by Julian
    # careful RGB -> BGR

    # ugly fix for division with 0
    zeros = np.logical_or(image[:, :, 1] == 0, image[:, :, 2] == 0)
    image[:, :, 1][zeros] = 1
    image[:, :, 2][zeros] = 1

    rat = image[:, :, 0] / image[:, :, 2] + image[:, :, 0] / image[:, :, 1]
    rat[zeros] = float("inf")

    cloud = np.array(rat < cloud_ref, dtype=np.uint8)
    return cloud

