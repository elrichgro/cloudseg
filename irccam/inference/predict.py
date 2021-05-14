from irccam.inference.load_model import load_model
from irccam.datasets.preprocessing import apply_clear_sky, process_irccam_img
import h5py
from torchvision import transforms
import torch
import hdf5storage


def load_data(input_path, limit=None):
    """
    Load data from IRCCAM .mat file. Convert to tensor for input to PyTorch model.
    """
    data = []

    with h5py.File(input_path, "r") as f:
        n = len(f["BT"])
        if limit is not None:
            n = limit
        for i in range(n):
            img = process_irccam_img(f["BT"][i])
            clear_sky = process_irccam_img(f["TB"][i])
            img = apply_clear_sky(img, clear_sky)
            img_tensor = transforms.ToTensor()(img)
            data.append(img_tensor)

    return torch.stack(data)


def save_predictions(input_path, predictions, output_path=None):
    # TODO: works on windows?
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = input_filename + "_pred.mat"
    output_dir = os.path.dirname(output_path) if output_path is not None else os.getcwd()
    output_file = os.path.join(output_dir, output_filename)
    output_data = {"predictions": predictions}
    hdf5storage.savemat(output_file, output_data)


def predict(model, input_path, limit=None):
    model.eval()
    data = load_data(input_path, limit)
    pred = model.model(data)
    pred = torch.argmax(pred, 1)
    return pred.numpy()


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from irccam.utils.files import load_yaml_file
    from irccam.utils.constants import AVAILABLE_MODELS_FILE

    available_models = load_yaml_file(AVAILABLE_MODELS_FILE)
    parser = ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to input file")
    parser.add_argument("--output_path", type=str, help="Path to save output file(s)")
    parser.add_argument(
        "--model", type=str, default="model_1", help="The model to predict with", choices=available_models.keys()
    )
    parser.add_argument("--limit", type=int, help="Limit the number of predictions to make")

    args = parser.parse_args()

    assert os.path.isfile(args.input_path), f'Invalid input file: "{args.input_path}"'

    model = load_model(args.model)
    predictions = predict(model, args.input_path, args.limit)
    save_predictions(args.input_path, predictions)
