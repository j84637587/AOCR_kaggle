import os
import json
import shutil
import tempfile
import time
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai import data
from monai.data import decollate_batch
from functools import partial
import monai.networks.nets as nets

from models.UNet3d import UNet3d

import torch
from torch import nn
from torchinfo import summary

from compose.LoadPreprocessedImaged import LoadPreprocessedImaged
from logger.logger import create_logger


log_dir, logger = create_logger()
print_config()


class AverageMeter(object):
    def __init__(self):
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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(train_csv):
    df = pd.read_csv(train_csv)
    tr = df[df["group"] == "Train"].reset_index(drop=True)["id"].tolist()
    val = df[df["group"] == "Valid"].reset_index(drop=True)["id"].tolist()
    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=log_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


# data = torch.load(self.root_path + f"{id_}.pt")


def get_loader(batch_size, data_dir, train_csv, fold=None):
    train_files, validation_files = datafold_read(train_csv=train_csv)
    train_transform = transforms.Compose(
        [
            LoadPreprocessedImaged(data_dir=data_dir),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(256, 192, 64)),
            # transforms.CropForegroundd(
            #     keys=["image", "label"],
            #     source_key="image",
            #     k_divisible=[roi[0], roi[1], roi[2]],
            # ),
            # transforms.RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=[roi[0], roi[1], roi[2]],
            #     random_size=False,
            # ),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            LoadPreprocessedImaged(data_dir=data_dir),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(256, 192, 64)),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader


data_dir = "data/preprocess/232x176x50_v5"
train_data_dir = os.path.join(data_dir, "1_Train,Valid_Image")
train_csv = "./data/TrainValid_split.csv"
batch_size = 4
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 5
print_interval = 10
train_loader, val_loader = get_loader(batch_size, train_data_dir, train_csv)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # set order of gpu based on pci bus id
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" # set visible gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = nets.SwinUNETR(
#     img_size=(256, 192, 64),
#     in_channels=1,
#     out_channels=1,
#     feature_size=48,
#     drop_rate=0.0,
#     attn_drop_rate=0.0,
#     dropout_path_rate=0.0,
#     use_checkpoint=True,
# ).to(device)

# model = nets.UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128),
#     strides=(1, 1, 1),
#     kernel_size=3,
#     up_kernel_size=3,
#     num_res_units=2,
#     act="PRELU",
#     norm="BATCH",
#     dropout=0.2,
#     bias=True,
# ).to(device)

model = UNet3d(in_channels=1, n_classes=1, n_channels=32).to(device)
model = nn.DataParallel(model)

logger.info(model)
logger.info(summary(model, (1, 1, 64, 64, 64), verbose=0))

torch.backends.cudnn.benchmark = True
dice_loss = DiceCELoss(sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)

dice_acc = DiceMetric(
    include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
)
model_inferer = partial(
    sliding_window_inference,
    roi_size=None,
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        if idx % print_interval == 0:
            logger.info(
                f"Epoch {epoch}/{max_epochs} {idx}/{len(loader)}, "
                f"loss: {run_loss.avg:.4f}, "
                f"time {time.time() - start_time:.2f}s"
            )
            start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            if idx % print_interval == 0:
                logger.info(
                    f"Val {epoch}/{max_epochs} {idx}/{len(loader)}, "
                    f"dice: {run_acc.avg}, "
                    f"time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()

    return run_acc.avg


def trainer(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        logger.info(
            f"Final training {epoch}/{max_epochs - 1}, "
            f"loss: {train_loss:.4f}, "
            f"time {time.time() - epoch_time:.2f}s"
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            val_avg_acc = np.mean(val_acc)
            logger.info(
                f"Final validation stats {epoch}/{max_epochs - 1}, "
                f"Dice_Avg: {val_avg_acc}, "
                f"time {time.time() - epoch_time:.2f}s"
            )
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                logger.info(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )

            optimizer.zero_grad()
            scheduler.step()

        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Epoch Average Loss\nBest: {np.min(loss_epochs)}")
        plt.xlabel("epoch")
        plt.plot(trains_epoch, loss_epochs, color="red")
        plt.subplot(1, 2, 2)
        plt.title(f"Val Mean Dice\nBest: {val_acc_max}")
        plt.xlabel("epoch")
        plt.plot(trains_epoch, dices_avg, color="green")
        plt.savefig(os.path.join(log_dir, "history.png"))

    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max}")
    return (
        val_acc_max,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


start_epoch = 0

(
    val_acc_max,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
)
logger.info(f"train completed, best average dice: {val_acc_max:.4f} ")
