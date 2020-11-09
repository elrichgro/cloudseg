"""
Filtering of images based on various criteria
"""

import os
import cv2
import numpy as np

from irccam.utils.definitions import *


def get_ignored_timestamps():
    filename = os.path.join(
        PROJECT_PATH, "irccam", "datasets", "ignored_timestamps.txt"
    )
    with open(filename) as f:
        content = f.readlines()
    content = [ts.strip() for ts in content]
    return content


def filter_ignored(items, ignore_list=None):
    if ignore_list is None:
        ignore_list = get_ignored_timestamps()
    return np.array([i for i in items if i not in ignore_list])


def is_almost_black(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    all = gray.shape[0] * gray.shape[1]
    almost_black = np.sum(gray < 15)
    return almost_black > all * 0.9
