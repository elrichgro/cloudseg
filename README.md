# irccam-pmodwrc
Cloud detection using the IRCCAM at PMOD/WRC

## Create environment
We are using good old Anaconda now... **** you Windows 
```
conda env create --name=dslab -f environment.yml 
```

## Update environment
```
conda env update --name=dslab -f environment.yml 
```

## Running notebooks
To run jupyter notebook with the project virtual env:
```
conda activate dslab
jupyter notebook
```


## Data
The `data` folder is ignored by git, but we should use a consistent structure
locally to make it easy to work with data in the code. The structure is 
currently:
```
.
└── data/
    ├── raw/
    │   ├── davos/
    │   │   ├── irccam
    │   │   └── rgb
    │   └── geneva/
    │       ├── irccam
    │       └── rgb
    └── datasets/
        ├── dataset_1/
        │   ├── previews/
        │   │   ├── (day).mp4 # daily preview video
        │   │   └── ...
        │   ├── train.txt # days from train
        │   ├── test.txt # days for test
        │   ├── val.txt # days for val
        │   ├── (day).h5 # all daily data in HDF5 format
        │   └── ...
        └── ...
```
