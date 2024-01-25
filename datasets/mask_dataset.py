import os
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Any
from scipy import ndimage
from skimage.transform import resize

import monai
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import albumentations as A
from albumentations import Compose, HorizontalFlip


class SegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_path: str,
        phase: str = "Valid",
        transform=None,
    ):
        if phase not in ["Valid", "Train", "ALL", "Test", "ValidTest"]:
            raise ValueError("Invalid phase. Only 'Valid' and 'Train' are allowed.")

        if phase == "Test":
            self.df = df[df["id"].str.contains(r"^[^_]*$", regex=True)].reset_index(
                drop=True
            )  # without slice-level scan
            self.root_path = root_path + "/3_Test1_Image/"
        elif phase == "ValidTest":
            self.df = df[df["group"] == "Valid"].reset_index(drop=True)
            self.root_path = root_path + "/1_Train,Valid_Image_Test/"
        elif phase == "ALL":
            self.df = df
            # self.df = df[df["scan-level label"] == 1].reset_index(drop=True)
            self.root_path = root_path + "/1_Train,Valid_Image/"
        else:
            self.df = df[df["group"] == phase].reset_index(drop=True)
            # self.df = df[df["scan-level label"] == 1].reset_index(drop=True)
            self.root_path = root_path + "/1_Train,Valid_Image/"
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "id"]
        data = torch.load(self.root_path + f"{id_}.pt")

        ct = data["ct"]
        num_slices = ct.shape[-1]

        if self.phase not in ["Test", "ValidTest"]:
            mask = data["mask"].float()
            # mask = torch.unsqueeze(mask, 0)  # expand

            if self.transform:
                trans_data = self.transform({"image": ct, "mask": mask})
                return id_, trans_data["image"], trans_data["mask"]
            else:
                # padding_width = ((0, 24), (0, 16), (0, 14))
                # ct = np.pad(ct, padding_width, mode="constant", constant_values=0)
                # mask = np.pad(mask, padding_width, mode="constant", constant_values=0)
                # ct = torch.from_numpy(ct)  # Convert NumPy array to PyTorch tensor
                # mask = torch.from_numpy(mask)  # Convert NumPy array to PyTorch tensor

                # print(ct.shape)
                # ct = monai.transforms.Resize((96, 96, 50))(ct)
                # print(ct.shape)
                # exit()
                # mask = monai.transforms.Resized((96, 96, 50))(mask)

                ct = torch.unsqueeze(ct, 0)
                mask = torch.unsqueeze(mask, 0)

                return id_, ct, mask

        # padding_width = ((0, 24), (0, 8), (0, 14))
        # ct = np.pad(ct, padding_width, mode="constant", constant_values=0)
        # ct = torch.from_numpy(ct)  # Convert NumPy array to PyTorch tensor

        # ct = monai.transforms.Resize((96, 96, 96))(ct)
        # ct = monai.transforms.Resize((96, 96, 50))(ct)
        ct = torch.unsqueeze(ct, 0)

        return id_, ct, data["num_slice"], data["s"], data["e"]


def build_dataloader(
    csv_file_path,
    root_directory,
    transform=None,
    batch_size: int = 10,
    distributed: bool = False,
    num_workers: int = 0,
    phase: str = "Train",
) -> torch.utils.data.DataLoader:
    df = pd.read_csv(csv_file_path)
    train_dataset = SegDataset(df, root_directory, phase, transform=transform)

    train_sampler: Optional[DistributedSampler[Any]] = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) if phase == "Train" else False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    return data_loader


if __name__ == "__main__":
    from tqdm import tqdm

    # Set your CSV file path and root directory containing images
    csv_file_path = "data/TrainValid_split.csv"
    # root_directory = "data/preprocess/232x176x50_m"
    root_directory = "data/preprocess/232x176x50_v18"
    test_file_path = "data/sample_submission.csv"

    train_loader = build_dataloader(
        csv_file_path, root_directory, batch_size=4, num_workers=0, phase="Train"
    )

    print("train_loader", len(train_loader))
    for i, (id, ct, mask) in tqdm(enumerate(train_loader)):
        if i == 0:
            print(ct.shape)

    val_loader = build_dataloader(
        csv_file_path,
        root_directory,
        batch_size=4,
        num_workers=0,
        phase="Valid",
    )

    print("val_loader", len(val_loader))
    for i, (id, ct, mask) in tqdm(enumerate(val_loader)):
        if i == 0:
            print(ct.shape)

    test_loader = build_dataloader(
        csv_file_path,
        root_directory,
        batch_size=4,
        num_workers=0,
        phase="Test",
    )

    print("test_loader", len(test_loader))
    for i, (id, ct) in tqdm(enumerate(test_loader)):
        if i == 0:
            print(ct.shape)
