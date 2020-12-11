import os
from irccam.datasets.cloud_dataset import HDF5Dataset, OptimizedDataset


def get_dataset_class(name):
    datasets = {"hdf5": HDF5Dataset, "optimized": OptimizedDataset}
    return datasets[name]


def get_contained_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
