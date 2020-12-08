"""
Filtering of images based on various criteria
"""

import numpy as np

from irccam.utils.definitions import *
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime, timedelta

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


def filter_sparse(timestamps):
    if len(timestamps) == 0:
        return []

    ok = [timestamps[0]]

    for vis_ts, ir_ts in sorted(timestamps, key=lambda x: x[0]):
        if (vis_ts - ok[-1][0]) > timedelta(minutes=9):
            ok.append((vis_ts, ir_ts))

    print(len(timestamps), len(ok))
    return ok

def filter_manual(data, day, timestamps):
    row = data[data['Name'] == day + "_preview.mp4"]
    if row.shape[0] > 0:
        info = row.iloc[0]
        bad = info.BAD == 1
        tresh = 4 if np.isnan(info.TRESH) else int(info.TRESH)
        start= 0 if np.isnan(info.START) else int(info.START * 3)
        end = len(timestamps) - 1 if np.isnan(info.END) else int(info.END * 3)
        return bad, start, end, tresh
    return True, None, None, None
