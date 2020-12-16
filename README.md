# irccam-pmodwrc
Cloud segmentation using the Infrared all-sky Cloud Camera (IRCCAM) at PMOD/WRC in Davos, Switzerland.

## Create environment
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

# Dataset variables
### main
 - timestamps - array of timestamp is format `%Y%m%d%H%M%S`
 - irc - normalized irc image, 0-255 float32
 - vis - reference rgb image
 - clear_sky - normalized clear sky model, 0-255 float32
 - ir_label - labels produced by Julians treshold algo
 - selected_label - manually selected label
 - sun_mask - boolean array, true where the sun is
 - label(0-3) - rgb produced label, in order (2.35, 2.75, 3, adaptive)
 
 
### optimized
 - timestamps
 - irc
 - selected_label
 - sun_mask
 
All labels have the format of: -1 mask, 0 sky, 1 clouds

All images have nan for mask 
