""" 
Loading the .mat files with the IRCCAM data becomes a bottleneck for
creating a dataset, because there is a lot of unused data in those files.

Here, we extract the raw components of the .mat files, to speed up
dataset creation.

Changes made in here to prevent big time memory leakage, still won't run without at least 20G of RAM.
If all else fails temporarily increase swapfile size to stupid levels
"""

import os
import mat73
import numpy as np
import sys
import math
import datetime

PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw/davos')


def extract_data(file=None):
    if file is None:
        irccam_files = get_contained_files(os.path.join(RAW_DATA_PATH, 'irccam'))
    else:
        irccam_files = [file]

    irccam_files = [file for file in irccam_files if file.endswith('.mat')]
    extract_path = os.path.join(RAW_DATA_PATH, 'irccam_extract')
    for file in irccam_files:
        day = file.split('_')[1]
        if os.path.exists(os.path.join(extract_path, day)):
            print('ignoring day {}, already processed'.format(day))
            continue

        print('processing day {}'.format(day))
        irccam_data = mat73.loadmat(os.path.join(RAW_DATA_PATH, 'irccam', file))
        print('loaded data')

        # might be the cause of memory troubles
        # bt_data = irccam_data['BT']
        # img_data = irccam_data['img']

        bt_path = os.path.join(extract_path, day, 'bt')
        img_path = os.path.join(extract_path, day, 'img')

        if not os.path.exists(bt_path):
            os.makedirs(bt_path)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        print('saving BT')
        for i in range(0, irccam_data['BT'].shape[2]):
            timestamp = convert_timestamp(day, irccam_data['TM'][i])
            filename = os.path.join(bt_path, '{}.npz'.format(timestamp))
            np.savez_compressed(filename, irccam_data['BT'][:, :, i])
        print('saving img')
        for i in range(0, irccam_data['img'].shape[2]):
            timestamp = convert_timestamp(day, irccam_data['TM'][i])
            filename = os.path.join(img_path, '{}.npz'.format(timestamp))
            np.savez_compressed(filename, irccam_data['img'][:, :, i])
        print('done')
        irccam_data = None  # don't know what worked...but at some point python started releasing memory on time


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def convert_timestamp(day, timestamp):
    """
    Converts irccam timestamps in double format (e.g. 737653.55976907) to 
    timestamps capped to the nearest minute (e.g. 201908161326)
    """
    seconds = round(24*60*60*(timestamp-math.floor(timestamp))/60)*60
    seconds_delta = datetime.timedelta(0,seconds)
    day_timestamp = datetime.datetime.strptime(day, '%Y%m%d')
    return (day_timestamp + seconds_delta).strftime('%Y%m%d%H%M')
    



if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_data(sys.argv[1])
    else:
        extract_data()
