""" 
Loading the .mat files with the IRCCAM data becomes a bottleneck for
creating a dataset, because there is a lot of unused data in those files.

Here, we extract the raw components of the .mat files, to speed up
dataset creation.
"""

import os
import mat73
import numpy as np

PROJECT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../..')
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/remote')

def extract_data():
    irccam_files = get_contained_files(os.path.join(RAW_DATA_PATH, 'irccam'))
    extract_path = os.path.join(RAW_DATA_PATH, 'irccam_extract')
    for file in irccam_files:
        timestamp = file.split('_')[1]
        print('processing day {}'.format(timestamp))
        irccam_data = mat73.loadmat(os.path.join(RAW_DATA_PATH, 'irccam', file))
        bt_data = irccam_data['BT']
        img_data = irccam_data['img']
        bt_path = os.path.join(extract_path, timestamp, 'bt')
        img_path = os.path.join(extract_path, timestamp, 'img')
        if not os.path.exists(bt_path):
            os.makedirs(bt_path)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        for i in range(0, bt_data.shape[2]):
            filename = os.path.join(bt_path, '{}.npz'.format(i))
            np.savez_compressed(filename, bt_data[:,:,i])
        for i in range(0, img_data.shape[2]):
            filename = os.path.join(img_path, '{}.npz'.format(i))
            np.savez_compressed(filename, img_data[:,:,i])

def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

if __name__ == "__main__":
    extract_data()
