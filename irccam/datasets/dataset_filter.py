"""
Filtering of images based on various criteria
"""

import numpy as np

from irccam.utils.definitions import *
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime

location = LocationInfo("Davos", "Switzerland", "Europe/Zurich", 46.813492, 9.844433)


def get_ignored_days():
    filename = os.path.join(
        PROJECT_PATH, "irccam", "datasets", "ignored_days.txt"
    )
    with open(filename) as f:
        content = f.readlines()
    content = [ts.strip() for ts in content]
    return content


def filter_ignored_days(items, ignore_list=None):
    if ignore_list is None:
        ignore_list = get_ignored_days()
    return [item for item in items if item not in ignore_list]


def filter_sun(timestamps, day):
    date = datetime.strptime(day, TIMESTAMP_FORMAT_DAY)
    astral_data = sun(location.observer, date=date)
    return [(vis_ts, ir_ts) for vis_ts, ir_ts in timestamps if astral_data["sunrise"] < vis_ts < astral_data["sunset"]]
