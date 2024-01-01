import os
import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from typing import Optional, Any


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_valid=False):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[
            self.data_frame["group"] == ("Valid" if is_valid else "Train")
        ].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_valid = is_valid

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        id = self.data_frame.at[idx, "id"]
        ct_path = os.path.join(self.root_dir, f"{id}.pt")
        data = torch.load(ct_path)

        if self.transform:
            data["ct"] = self.transform(data["ct"])
        return data["ct"].float(), data["label"].float()


def build_dataloader(
    csv_file_path,
    root_directory,
    transform=None,
    batch_size=10,
    distributed=False,
    num_workers=0,
    is_valid=False,
):
    train_dataset = CustomDataset(
        csv_file=csv_file_path,
        root_dir=root_directory,
        is_valid=is_valid,
        transform=transform,
    )

    train_sampler: Optional[DistributedSampler[Any]] = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) if not is_valid else False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    return data_loader


def permute_and_expand_with_zeros(x):
    n_zero = torch.zeros((x.shape[0], x.shape[1], 14), dtype=torch.float32)
    x = torch.cat((x, n_zero), dim=-1)
    x = x.permute(0, 2, 1)
    return x


if __name__ == "__main__":
    from tqdm import tqdm
    import progressmeter as pm
    from torchvision import transforms

    # Set your CSV file path and root directory containing images
    csv_file_path = "data/TrainValid_split.csv"
    root_directory = "data/preprocess/232x176x50/1_Train,Valid_Image"
    csv_ground_truth = "data/TrainValid_ground_truth.csv"

    transform = transforms.Compose([permute_and_expand_with_zeros])

    train_loader = build_dataloader(
        csv_file_path,
        root_directory,
        csv_ground_truth=csv_ground_truth,
        batch_size=1,
        distributed=False,
        transform=transform,
    )

    print("train_dataset", len(train_loader.dataset))
    input("Press Enter to continue...")

    progress = pm.build_train_progress(len(train_loader))

    for i, (images, target) in tqdm(enumerate(train_loader)):
        print(images.shape)
        print(torch.unsqueeze(images, 1).shape)
        print(target.shape)
        if images.shape != torch.Size([1, 232, 176, 50]):
            print("error")
            print(images.shape)
            print(torch.Size([1, 232, 176, 50]))
            exit()
        progress.print(i)
        break

    test_loader = build_dataloader(
        csv_file_path,
        root_directory,
        csv_ground_truth=csv_ground_truth,
        batch_size=1,
        distributed=False,
        is_valid=True,
        transform=transform,
    )

    print("test_dataset", len(test_loader.dataset))
    input("Press Enter to continue...")

    progress = pm.build_train_progress(len(test_loader))

    for i, (images, target) in tqdm(enumerate(test_loader)):
        print(images.shape)
        print(torch.unsqueeze(images, 1).shape)
        print(target.shape)
        progress.print(i)
        break

# 80it [11:36,  8.71s/it]
