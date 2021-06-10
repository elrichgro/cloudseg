from cloudseg.datasets.preprocessing import apply_clear_sky, process_irccam_img
import h5py
from torchvision import transforms
import torch
import hdf5storage
from tqdm import tqdm
import numpy as np
import os


def load_data(input_path, limit=None):
    """
    Load data from IRCCAM .mat file. Convert to tensor for input to PyTorch model.
    """
    data = []

    with h5py.File(input_path, "r") as f:
        n = len(f["BT"])
        if limit is not None:
            n = limit
        print("Loading data")
        for i in tqdm(range(n)):
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


def predict(model, input_path, limit=None, batch_size=8):
    model.eval()
    data = load_data(input_path, limit)
    data = data.split(batch_size)
    predictions = np.array([])
    print(f"Making predictions (batch_size={batch_size})")
    with torch.no_grad():
        for batch in tqdm(data):
            pred = model.model(batch)
            pred = torch.argmax(pred, 1)
            np.append(predictions, pred.numpy())
    return predictions
