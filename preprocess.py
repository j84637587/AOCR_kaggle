import pandas as pd
import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
from scipy import ndimage

import torch
from torch.utils.data import Dataset, DataLoader

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
    CT_min = WL - WW / 2  # -200
    CT_max = WL + WW / 2

    sample = (sample - CT_min) / (CT_max - CT_min)
    sample[sample > 1] = 1.0
    sample[sample < 0] = 0.0
    return sample


class CustomDataset(Dataset):
    def __init__(
        self, dataframe, gt_df, save_to, min_x, max_x, min_y, max_y, s, e, phase="Train"
    ):
        self.dataframe = dataframe
        self.gt_df = gt_df
        self.save_to = save_to
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.init_s = s
        self.init_e = e
        self.n = e - s
        self.phase = phase
        self.sub_dir = (
            "1_Train,Valid_Image" if self.phase != "Test" else "3_Test1_Image"
        )

    def __len__(self):
        return len(self.dataframe)

    def load_nii(self, nii_path):
        volume = nib.load(nii_path)
        volume = np.asanyarray(volume.dataobj)
        return volume, volume.shape

    def get_label(self, id, s, e):
        labels = self.gt_df[self.gt_df["id"].str.contains(rf"{id}_.*", regex=True)][
            "label"
        ].tolist()
        return labels[s:e]

    def get_slice(self, nii_path, s=None, e=None, resize=False):
        if s is None:
            s = self.init_s
        if e is None:
            e = self.init_e

        volume, shape = self.load_nii(nii_path)
        volume = volume[self.min_y : self.max_y, self.min_x : self.max_x, ...]

        while True:
            slices = volume[..., s:e]
            if slices.shape[2] < self.n:
                s = s - (self.n - slices.shape[2])
                if s < 0:
                    e = e + abs(s)
                    s = 0
                    continue
            else:
                break

        if resize:
            slices = resize_volume(
                slices, desired_depth=self.n, desired_width=512, desired_height=512
            )
        return slices, shape, s, e

    def __getitem__(self, idx):
        id = self.dataframe["id"][idx]
        slices, shape, s, e = self.get_slice(f"./data/{self.sub_dir}/{id}.nii.gz")
        label = self.get_label(id, s, e)

        # slices = normalize(slices)
        sample_tensor = torch.from_numpy(slices).float()
        label_tensor = torch.tensor(label).float()

        if self.phase == "Train":
            mask = self.get_slice(f"./data/2_Train,Valid_Mask/{id}_label.nii.gz", s, e)
            mask = mask[0]
            mask_tensor = torch.from_numpy(mask.astype(np.uint8)).float()
            torch.save(
                {"ct": sample_tensor, "mask": mask_tensor, "label": label_tensor},
                f"{self.save_to}/1_Train,Valid_Image/{id}.pt",
            )

            torch.save(
                {"ct": sample_tensor, "num_slice": shape[2], "s": s, "e": e},
                f"{self.save_to}/1_Train,Valid_Image_Test/{id}.pt",
            )
        else:
            torch.save(
                {"ct": sample_tensor, "num_slice": shape[2], "s": s, "e": e},
                f"{self.save_to}/3_Test1_Image/{id}.pt",
            )
        if idx == 0:
            print(f"\nshape: {sample_tensor.shape}")
        return torch.tensor([])


if __name__ == "__main__":
    trainVal = pd.read_csv("./data/TrainValid_split.csv")
    # trainVal = trainVal.tail(1000 - 656).reset_index(drop=True)
    testVal = pd.read_csv("./data/sample_submission.csv")
    testVal = testVal[testVal["id"].str.contains(r"^[^_]*$", regex=True)].reset_index(
        drop=True
    )
    gtVal = pd.read_csv("./data/TrainValid_ground_truth.csv")
    SAVE_TO = "./data/preprocess/232x176x50_v5"
    min_x = 185
    max_x = 361
    min_y = 59
    max_y = 291
    S = 12 - 1
    E = 61

    # min_x = 185
    # max_x = 361
    # min_y = 59 - 4
    # max_y = 291 + 4
    # S = 12 - 1 + 1
    # E = 61 - 1
    # SAVE_TO = "./data/preprocess/240x176x48"

    for sub_dir in [
        "1_Train,Valid_Image",
        "1_Train,Valid_Image_Test",
        "3_Test1_Image",
    ]:
        if not os.path.exists(os.path.join(SAVE_TO, sub_dir)):
            os.makedirs(os.path.join(SAVE_TO, sub_dir))

        sub_dir = os.path.join(SAVE_TO, sub_dir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    train_dataset = CustomDataset(
        dataframe=trainVal,
        gt_df=gtVal,
        save_to=SAVE_TO,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        s=S,
        e=E,
    )

    test_dataset = CustomDataset(
        dataframe=testVal,
        gt_df=gtVal,
        save_to=SAVE_TO,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        s=S,
        e=E,
        phase="Test",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # 02:57
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    for itr, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        pass

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,  # 02:57
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    for itr, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        pass
