from irccam.datasets.cloud_dataset import CloudDataset, HDF5Dataset


def get_dataset_class(name):
    datasets = {"cloud": CloudDataset, "hdf5": HDF5Dataset}
    return datasets[name]
