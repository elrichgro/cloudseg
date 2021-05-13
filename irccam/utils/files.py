
import os
import yaml

def get_contained_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_contained_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def load_yaml_file(path):
    with open(path) as f:
        return yaml.load(f)
