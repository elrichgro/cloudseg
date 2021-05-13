import os
import h5py
from pytz import timezone
from astral import LocationInfo

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data/raw/davos")
DATASET_PATH = os.path.join(PROJECT_PATH, "data/datasets")
MODELS_PATH = os.path.join(PROJECT_PATH, "pretrained_models")
AVAILABLE_MODELS_FILE = os.path.join(PROJECT_PATH, "irccam/inference/models.yaml")

TIMESTAMP_FORMAT_DAY = "%Y%m%d"
TIMESTAMP_FORMAT_MINUTE = "%Y%m%d%H%M"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
PRETTY_FORMAT = "%d.%m.%Y %H:%M:%S"

LOCATION = LocationInfo("Davos", "Switzerland", "Europe/Zurich", 46.813492, 9.844433)
TIMEZONE = timezone("Europe/Zurich")

LABEL_DATATYPE = h5py.enum_dtype({"CLOUD": 0, "SKY": 1, "MASK": -1}, basetype='i')
