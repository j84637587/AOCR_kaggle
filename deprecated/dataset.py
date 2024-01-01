import os
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Any
from scipy import ndimage


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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_valid=False):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[
            self.data_frame["group"] == ("Valid" if is_valid else "Train")
        ]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        ct_path = os.path.join(self.root_dir, f"{self.data_frame.at[idx, 'id']}.nii")

        ct_img = nib.load(ct_path)
        ct_img = np.asanyarray(ct_img.dataobj)
        ct_img = resize_volume(
            ct_img, desired_depth=100, desired_width=512, desired_height=512
        )
        ct_img = (ct_img + 1024) / 255  # -1024 ~ 1600 normalize to 0 ~ 255

        label = int(self.data_frame.at[idx, "scan-level label"])

        if self.transform:
            ct_img = self.transform(ct_img)

        return ct_img.astype("float32"), label


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    from tqdm import tqdm

    # Set your CSV file path and root directory containing images
    csv_file_path = "data_unzip/TrainValid_split.csv"
    root_directory = "data_unzip/1_Train,Valid_Image"

    # Create custom datasets for training and validation
    train_dataset = CustomDataset(
        csv_file=csv_file_path, root_dir=root_directory  # , transform=transform
    )

    print("train_dataset", len(train_dataset))

    distributed = False
    train_sampler: Optional[DistributedSampler[Any]] = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    print("train_loader", len(train_loader))

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        batch_time,
        data_time,
        losses,
        top1,
        top5,
        prefix="Epoch: [{}]".format(1),
    )

    for i, (images, target) in tqdm(enumerate(train_loader)):
        print(images.shape)
        progress.print(i)
