import os
import gdown
from cloudseg.utils.constants import MODELS_PATH, AVAILABLE_MODELS_FILE
from cloudseg.training.cloud_segmentation import CloudSegmentation
from cloudseg.utils.files import load_yaml_file


def load_model(model_name="model_1"):
    """
    Load pretrained model from stored checkpoint. Download if necessary.
    """
    # Validate model_name
    available_models = load_yaml_file(AVAILABLE_MODELS_FILE)
    assert (
        model_name in available_models
    ), f"Model \"{model_name}\" does not exist. Available models: {', '.join(available_models.keys())}"

    # Download if needed
    if not is_model_downloaded(model_name):
        print(f"Downloading model {model_name}")
        download_model(model_name, available_models[model_name])

    # Load model
    print(f"Loading model {model_name}")
    model_path = get_model_path(model_name)
    model = CloudSegmentation.load_from_checkpoint(model_path)
    return model


def download_model(model_name, file_id):
    """
    Download pretrained cloud segmentation model
    """
    output_path = get_model_path(model_name)
    model_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(model_url, output_path)


def is_model_downloaded(model_name):
    """
    Check if model has already been downloaded
    """
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)


def get_model_path(model_name):
    """
    Get the location where model with name `model_name` should be stored
    """
    return os.path.join(MODELS_PATH, f"{model_name}.ckpt")


if __name__ == "__main__":
    model = load_model()
    # print(model)
