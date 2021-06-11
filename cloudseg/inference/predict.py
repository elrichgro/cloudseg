from cloudseg.datasets.masking import apply_mask
from cloudseg.datasets.preprocessing import apply_clear_sky, process_irccam_img, get_cropping_indices, create_mask
import h5py
from torchvision import transforms
import torch
import hdf5storage
from tqdm import tqdm
import os
import numpy as np


def load_data(input_path, limit=None):
    """
    Load data from IRCCAM .mat file. Convert to tensor for input to PyTorch model.
    """
    data = []

    with h5py.File(input_path, "r") as f:
        n = len(f["BT"])
        if limit is not None:
            n = min(n, limit)
        print("Loading data")
        raw_mask = f["mask"]
        original_shape = f["BT"][0].shape
        crop_idx = get_cropping_indices(f["BT"][0])
        mask = create_mask(raw_mask, crop_idx)
        for i in tqdm(range(n)):
            raw_img = f["BT"][i]
            raw_clear_sky = f["TB"][i]
            img = process_irccam_img(raw_img, crop_idx, flip=False, mask=mask)
            clear_sky = process_irccam_img(raw_clear_sky, crop_idx, flip=False, mask=mask)
            img = apply_clear_sky(img, clear_sky)
            img_tensor = transforms.ToTensor()(img)
            data.append(img_tensor)

    return torch.stack(data), mask, crop_idx, original_shape


def save_predictions(input_path, preds, output_path=None, output_file=None):
    # TODO: works on windows?
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = output_file if output_file is not None else (input_filename + "_pred.mat")
    if not output_filename.endswith(".mat"):
        output_filename = output_filename + ".mat"
    output_dir = output_path if output_path is not None else os.getcwd()
    print("output dir", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, output_filename)
    print(preds.shape)
    output_data = {"preds": preds.T}
    hdf5storage.savemat(output_file, output_data)


def expand_batch_preds(preds, original_shape, crop_idx):
    expanded_preds = torch.zeros((preds.shape[0], *original_shape))
    upper, lower = crop_idx[0]
    left, right = crop_idx[1]
    expanded_preds[:, upper:lower, left:right] = preds
    return expanded_preds


def predict(model, input_path, limit=None, batch_size=8):
    model.eval()
    data, mask, crop_idx, original_shape = load_data(input_path, limit)
    data = data.split(batch_size)
    preds = torch.tensor([])
    print(f"Making predictions (batch_size={batch_size})")
    with torch.no_grad():
        for batch in tqdm(data):
            batch_preds = model.model(batch)
            batch_preds = torch.argmax(batch_preds, 1)
            batch_preds = apply_mask(batch_preds, torch.tensor(mask).repeat((batch_preds.shape[0], 1, 1)), 0)
            batch_preds = expand_batch_preds(batch_preds, original_shape, crop_idx)
            preds = torch.cat((preds, batch_preds))
    return preds.numpy()
