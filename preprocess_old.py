import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import nibabel as nib

from scipy import ndimage

import torch

import warnings

warnings.simplefilter("ignore")


def resize_volume(img, desired_depth=64, desired_width=512, desired_height=512):
    """Resize across z-axis"""
    # Get current depth
    current_width, current_height, current_depth = img.shape

    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def normalize(sample, WL=45, WW=300):
    """Normalize the volume"""
    CT_min = -200  # WL - WW / 2
    CT_max = WL + WW / 2

    sample = (sample - CT_min) / (CT_max - CT_min)
    sample[sample > 1] = 1.0
    sample[sample < 0] = 0.0
    return sample


if __name__ == "__main__":
    trainVal = pd.read_csv("./data/TrainValid_split.csv")
    gtVal = pd.read_csv("./data/TrainValid_ground_truth.csv")
    SAVE_TO = "./data/preprocess/256x256x50_v2"

    min_x = 185
    max_x = 361
    min_y = 59
    max_y = 291
    S = 12 - 1
    E = 61
    N = E - S

    s = 0
    o = 0
    vs = 0
    vo = 0

    for idx, id in tqdm(enumerate(trainVal["id"]), total=len(trainVal["id"])):
        filepath_sample = f"./data/1_Train,Valid_Image/{id}.nii.gz"
        sample = nib.load(filepath_sample)
        sample_arr = np.asanyarray(sample.dataobj)
        sample = sample_arr[min_y:max_y, min_x:max_x, S:E]

        labels = gtVal[gtVal["id"].str.contains(rf"{id}_.*", regex=True)][
            "label"
        ].tolist()
        label = labels[S:E]

        # if sample.shape[2] != N:
        #     z_zeros = N - sample.shape[2]

        #     S = max(S - (N - len(label)), 0)

        #     label = labels[S:E]
        #     sample = sample_arr[min_y:max_y, min_x:max_x, S:E]

        # factor = N / len(label)
        # label = torch.tensor(label)
        # label = ndimage.zoom(label, (factor), order=1)

        if trainVal["group"][idx] == "Train":
            s += len(labels) - len(label)
            o += sum(labels[:S]) + sum(labels[E:])
        else:
            vs += len(labels) - len(label)
            vo += sum(labels[:S]) + sum(labels[E:])

        sample = resize_volume(
            sample, desired_depth=N, desired_width=256, desired_height=256
        )

        sample = normalize(sample)

        sample_tensor = torch.from_numpy(sample).float()
        label_tensor = torch.tensor(label).float()

        torch.save(
            {"ct": sample_tensor, "label": label_tensor},
            f"{SAVE_TO}/1_Train,Valid_Image/{id}.pt",
        )

    print(f"Train: {s}, {o}")
    print(f"Valid: {vs}, {vo}")
    ################################################################################################################################
    # Valid
    ################################################################################################################################
    valid_files = glob("./data/3_Test1_Image/*.nii.gz")

    for idx, filepath_sample in tqdm(enumerate(valid_files), total=len(valid_files)):
        id = os.path.basename(filepath_sample).split(".")[0]
        sample = nib.load(filepath_sample)
        sample_arr = np.asanyarray(sample.dataobj)
        sample = sample_arr[min_y:max_y, min_x:max_x, S:E]

        # if sample.shape[2] != N:
        #     S = max(S - (N - sample.shape[2]), 0)
        #     sample = sample_arr[min_y:max_y, min_x:max_x, S:E]

        sample = resize_volume(
            sample, desired_depth=N, desired_width=256, desired_height=256
        )

        sample = normalize(sample)

        sample_tensor = torch.from_numpy(sample).float()

        torch.save({"ct": sample_tensor}, f"{SAVE_TO}/3_Test1_Image/{id}.pt")
