""" 
Loading the .mat files with the IRCCAM data becomes a bottleneck for
creating a dataset, because there is a lot of unused data in those files.

Here, we extract the raw components of the .mat files, to speed up
dataset creation.
"""

import os
import mat73
import numpy as np
import sys

PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw/davos')


def extract_data(file=None):
    if file == None:
        irccam_files = get_contained_files(os.path.join(RAW_DATA_PATH, 'irccam'))
    else:
        irccam_files = [file]

    irccam_files = [file for file in irccam_files if file.endswith('.mat')]
    extract_path = os.path.join(RAW_DATA_PATH, 'irccam_extract')
    for file in irccam_files:
        timestamp = file.split('_')[1]
        if os.path.exists(os.path.join(extract_path, timestamp)):
            print('ignoring day {}, already processed'.format(timestamp))
            continue

        print('processing day {}'.format(timestamp))
        irccam_data = mat73.loadmat(os.path.join(RAW_DATA_PATH, 'irccam', file))
        print('loaded data')

        # might be the cause of memory troubles
        # bt_data = irccam_data['BT']
        # img_data = irccam_data['img']

        bt_path = os.path.join(extract_path, timestamp, 'bt')
        img_path = os.path.join(extract_path, timestamp, 'img')

        if not os.path.exists(bt_path):
            os.makedirs(bt_path)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        print('saving BT')
        for i in range(0, irccam_data['BT'].shape[2]):
            filename = os.path.join(bt_path, '{}.npz'.format(i))
            np.savez_compressed(filename, irccam_data['BT'][:, :, i])
        print('saving img')
        for i in range(0, irccam_data['img'].shape[2]):
            filename = os.path.join(img_path, '{}.npz'.format(i))
            np.savez_compressed(filename, irccam_data['img'][:, :, i])
        print('done')
        irccam_data = None  # don't know what worked...but at some point python started releasing memory on time


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_data(sys.argv[1])
    else:
        extract_data()
