import pandas as pd

from irccam.utils.constants import *
from astral.sun import sun
from datetime import datetime, timedelta


def filter_sun(timestamps, day):
    """
    Filter out timestamps before sunrise and after sunset on a given day.
    """
    date = datetime.strptime(day, TIMESTAMP_FORMAT_DAY)
    astral_data = sun(LOCATION.observer, date=date)
    return [(vis_ts, ir_ts) for vis_ts, ir_ts in timestamps if astral_data["sunrise"] < vis_ts < astral_data["sunset"]]


def filter_sparse(timestamps):
    """
    Create a sparse set of timestamps, by keeping only 1 timestamp for every 10 minute 
    interval. 
    """
    if len(timestamps) == 0:
        return []

    ok = [timestamps[0]]

    for vis_ts, ir_ts in sorted(timestamps, key=lambda x: x[0]):
        if (vis_ts - ok[-1][0]) > timedelta(minutes=9):
            ok.append((vis_ts, ir_ts))

    return ok


def filter_manual(day, timestamps):
    """
    Filter out timestamps on a given day by using the manually prepared filtering
    data stored in the file `irccam/datasets/resources/filter_manual.csv`. This 
    data was compiled by manually checking each day for bad weather conditions 
    or equipment malfunction. 
    """
    filename = os.path.join(PROJECT_PATH, "irccam/datasets/resources/filter_manual.csv")
    data = pd.read_csv(
        filename,
        dtype={"DAY": "str", "START_TS": "str", "END_TS": "str", "DELETED": "str"},
        na_values={"DELETED": []},
        keep_default_na=False,
    )

    row = data[data["DAY"] == day]
    if row.shape[0] > 0 and timestamps:
        info = row.iloc[0]
        bad = info.BAD == 1
        if bad:
            return [], None

        start = TIMEZONE.localize(datetime.strptime(info.START_TS, TIMESTAMP_FORMAT))
        end = TIMEZONE.localize(datetime.strptime(info.END_TS, TIMESTAMP_FORMAT))
        timestamps = [x for x in timestamps if start <= x[0] < end]

        if info.DELETED:
            deleted = set(info.DELETED.strip().split(";"))
            timestamps = [x for x in timestamps if x[0].strftime("%H:%M:%S") not in deleted]
        return timestamps, info.LABEL_INDEX
    return [], None
