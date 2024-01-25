import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from postprocessing import find_longest_sequence, postprocessing

import torch
import torch.nn as nn

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from torch.cuda.amp import autocast, GradScaler
from warmup_scheduler import GradualWarmupScheduler

from sklearn.metrics import confusion_matrix

from models.UNet3d import UNet3d
from models.R2UNet3D import Net
from models.UNETR import UNETR
from datasets.mask_dataset import build_dataloader

import nibabel as nib

import segmentation_models_pytorch as smp
import monai
from monai import data, transforms
from compose.LoadPreprocessedImaged import LoadPreprocessedImaged

import warnings
import logging

warnings.simplefilter("ignore")


def dice_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    """factory for storing and updating iou and dice scores."""

    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.contiguous().view(num, -1)
        assert probability.shape == targets.shape

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        # print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert (
            logits.shape == targets.shape
        ), f"Error the shape of logits {logits.shape} != shape of targets{targets.shape}"
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets.float())

        return bce_loss + dice_loss


# helper functions for testing.
def dice_coef_metric_per_classes(
    probabilities: np.ndarray,
    truth: np.ndarray,
    treshold: float = 0.5,
    eps: float = 1e-9,
    classes: list = ["WT", "TC", "ET"],
) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert predictions.shape == truth.shape

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def jaccard_coef_metric_per_classes(
    probabilities: np.ndarray,
    truth: np.ndarray,
    treshold: float = 0.5,
    eps: float = 1e-9,
    classes: list = ["WT", "TC", "ET"],
) -> np.ndarray:
    """
    Calculate Jaccard index for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert predictions.shape == truth.shape

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def get_loader(data_dir, batch_size=8, train_csv="./data/TrainValid_split.csv"):
    data_dir = os.path.join(data_dir, "1_Train,Valid_Image")

    def datafold_read(train_csv):
        df = pd.read_csv(train_csv)
        tr = df[df["group"] == "Train"].reset_index(drop=True)["id"].tolist()
        val = df[df["group"] == "Valid"].reset_index(drop=True)["id"].tolist()
        return tr, val

    train_files, validation_files = datafold_read(train_csv=train_csv)
    train_transform = transforms.Compose(
        [
            LoadPreprocessedImaged(data_dir=data_dir),
            # transforms.Resized(keys=["image", "label"], spatial_size=(256, 192, 64)),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            LoadPreprocessedImaged(data_dir=data_dir),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
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


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


def get_scheduler(epochs, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-7
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine
    )

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """

    def __init__(
        self,
        net: nn.Module,
        dataloaders,
        criterion: nn.Module,
        lr: float,
        accumulation_steps: int,
        batch_size: int,
        fold: int,
        num_epochs: int,
        logger,
        log_path,
        display_plot: bool = True,
    ):
        """Initialization."""
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"device: {self.device}")
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, mode="min", patience=2, verbose=True
        # )
        self.scheduler = get_scheduler(num_epochs, self.optimizer)
        # self.scaler = GradScaler(enabled=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["Train", "Valid"]
        self.num_epochs = num_epochs
        self.log_path = log_path

        self.dataloaders = dataloaders
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        # with autocast(True):
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        # print learning rate
        for param_group in self.optimizer.param_groups:
            self.logger.info(f"epoch: {epoch} | lr: {param_group['lr']}")
        self.logger.info(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "Train" else self.net.eval()
        meter = Meter()

        if phase == "Train":
            dataloaders = [self.dataloaders["Train"], self.dataloaders["TrainExtra"]]
            total_batches = len(dataloaders[0]) + len(dataloaders[1])
        else:
            dataloaders = [self.dataloaders[phase]]
            total_batches = len(dataloaders[0])

        running_loss = 0.0
        self.optimizer.zero_grad()

        itr_c = 0
        for dataloader in dataloaders:
            for _ in range(len(dataloader)):
                itr_c += 1
                id, images, targets = next(iter(dataloader))

                images = images.permute(0, 1, 4, 2, 3)
                targets = targets.permute(0, 1, 4, 2, 3)

                images = images.float()
                targets = targets.long()
                # images, targets = batch_data["image"].to(self.device), batch_data[
                #     "label"
                # ].to(self.device)

                loss, logits = self._compute_loss_and_outputs(images, targets)
                loss = loss / self.accumulation_steps
                if phase == "Train":
                    loss.backward()
                    # self.scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    self.optimizer.zero_grad()

                    # if (itr_c + 1) % self.accumulation_steps == 0:
                    #     self.optimizer.step()
                    #     self.optimizer.zero_grad()

                    self.logger.info(
                        f"{phase} epoch: {epoch} | batch: {itr_c}/{total_batches} | "
                        f"loss: {loss.item():.4f}"
                    )
                running_loss += loss.item()
                meter.update(logits.detach().cpu(), targets.detach().cpu())

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "Train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "Valid")
                self.scheduler.step(val_loss)

            if self.display_plot:
                self._plot_train_history()

            if val_loss < self.best_loss:
                self.logger.info(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(
                    self.net.state_dict(), os.path.join(self.log_path, "best_model.pth")
                )
            self._save_train_history()

    def test_submit(
        self,
        threshold=0.7,
        valid=False,
        filter_num_slice=0,
        do_connected_ones=False,
        gen_nii=False,
    ):
        mask_result = os.path.join(self.log_path, "mask_result")
        os.makedirs(mask_result, exist_ok=True)

        def get_confusion_matrix_df(y_pred, y):
            # y_pred_tensor = torch.tensor(y_pred.astype(int).to_numpy())
            # y_true_tensor = torch.tensor(y.astype(int).to_numpy())
            confusion = confusion_matrix(
                y.astype(int).to_numpy(), y_pred.astype(int).to_numpy()
            )
            return confusion.ravel()

        def connected_ones(y_pred):
            pred_connected = []
            current_group = []

            for label in y_pred:
                if label == 1:
                    current_group.append(label)
                else:
                    if current_group:
                        pred_connected.extend(current_group)
                        current_group = []
                    pred_connected.append(label)

            if current_group:
                pred_connected.extend(current_group)
            return pred_connected

        self.net.eval()
        with torch.no_grad():
            submit = pd.DataFrame(columns=["id", "label"])
            for _, (ids, images, num_slices, start, end) in tqdm(
                enumerate(self.dataloaders["Test" if not valid else "ValidTest"])
            ):
                images = images.float()
                outputs = self.net(images)
                outputs = outputs.cpu().numpy()
                binary_predictions = (outputs > threshold) * 1

                for id_itr, id in enumerate(ids):
                    image = images[id_itr]
                    logger.info(f"Processing {id}")

                    num_slice = num_slices[id_itr]
                    pred = binary_predictions[id_itr]
                    logger.info(f"pred {np.any(pred, axis=(0, 1, 2))}")

                    post_processed = postprocessing(pred, image, logger=logger)

                    pred = np.any(post_processed, axis=(0, 1, 2))
                    logger.info(f"post_processed {pred}")
                    if np.sum(pred) < filter_num_slice:
                        pred = np.zeros_like(pred)

                    if do_connected_ones:
                        pred = connected_ones(pred)
                    logger.info(f"do_connected_ones {pred}")

                    # if find_longest_sequence(pred) < 2:
                    #     pred = np.zeros_like(pred)

                    s = start[id_itr]
                    e = end[id_itr]

                    if gen_nii:
                        mask = np.zeros((512, 512, num_slice))
                        mask[59:291, 185:361, s:e] = post_processed
                        mask = nib.Nifti1Image(
                            mask.astype(np.float64), affine=np.eye(4)
                        )

                        nib.save(mask, os.path.join(mask_result, f"{id}_pred.nii.gz"))

                    # set slice-level cls
                    rlt = np.zeros(num_slice)
                    rlt[s:e] = pred

                    # save np array as a image
                    scan_level_cls = "0" if (np.sum(pred) == 0) else "1"

                    submit = pd.concat(
                        [
                            submit,
                            pd.DataFrame(
                                {
                                    "id": [id]
                                    + [f"{id}_{i}" for i in range(num_slice)],
                                    "label": [scan_level_cls]
                                    + np.where(rlt, "1", "0").tolist(),
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
            submit_path = os.path.join(self.log_path, "submit.csv")
            submit.to_csv(submit_path, index=False)
            if valid:
                df_scan = submit[
                    submit["id"].str.contains(r"^[^_]*$", regex=True)
                ].reset_index(drop=True)
                df_slice = submit[
                    ~submit["id"].str.contains(r"^[^_]*$", regex=True)
                ].reset_index(drop=True)

                gt_df = pd.read_csv("data/TrainValid_ground_truth.csv")
                gt_df_scan = gt_df[gt_df["id"].isin(df_scan["id"])]
                gt_df_slice = gt_df[gt_df["id"].isin(df_slice["id"])]

                # get ids that labels not are not correct
                gt_df_scan_reset = gt_df_scan.reset_index(drop=True)
                df_scan_reset = df_scan.reset_index(drop=True)

                mismatched_ids = gt_df_scan_reset[
                    gt_df_scan_reset["label"].astype(str)
                    != df_scan_reset["label"].astype(str)
                ]["id"].tolist()
                self.logger.info(
                    f"Scan ids that labels not are not correct: {mismatched_ids}"
                )
                if gen_nii:
                    os.makedirs(os.path.join(mask_result, "wrong_masks"), exist_ok=True)
                    for id in mismatched_ids:
                        os.system(
                            f"cp {os.path.join(mask_result, f'{id}_pred.nii.gz')} {os.path.join(mask_result, 'wrong_masks')}"
                        )
                for id in mismatched_ids:
                    a = gt_df_slice[gt_df_slice["id"].str.contains(id)][
                        "label"
                    ].tolist()
                    b = (
                        df_slice[df_slice["id"].str.contains(id)]["label"]
                        .astype(int)
                        .tolist()
                    )
                    print(f"{id}: \n{a} -> \n{b}")

                tn, fp, fn, tp = get_confusion_matrix_df(
                    y_pred=df_scan["label"], y=gt_df_scan["label"]
                )
                precision_scan = tp / (tp + fp)
                recall_scan = tp / (tp + fn)
                f1_scan = 2 * tp / (2 * tp + fp + fn)

                self.logger.info("Scan-level confusion matrix: ")
                self.logger.info(
                    f"TP: {tp.item()}, FP: {fp.item()}, TN: {tn.item()}, FN: {fn.item()}"
                )
                self.logger.info(f"Precision: {precision_scan}")
                self.logger.info(f"Recall: {recall_scan}")
                self.logger.info(f"F1: {f1_scan}")

                tn, fp, fn, tp = get_confusion_matrix_df(
                    y_pred=df_slice["label"], y=gt_df_slice["label"]
                )
                precision_slice = tp / (tp + fp)
                recall_slice = tp / (tp + fn)
                f1_slice = 2 * tp / (2 * tp + fp + fn)

                self.logger.info("Slice-level confusion matrix: ")
                self.logger.info(
                    f"TP: {tp.item()}, FP: {fp.item()}, TN: {tn.item()}, FN: {fn.item()}"
                )
                self.logger.info(f"Precision: {precision_slice}")
                self.logger.info(f"Recall: {recall_slice}")
                self.logger.info(f"F1: {f1_slice}")

                self.logger.info(f"Final F1: {(f1_scan + f1_slice) / 2}")

                self.logger.info("This is for valid test only.")
            else:
                self.logger.info("To submit the result, please run:")
                self.logger.info(
                    f"kaggle competitions submit -c aocr2024 -f {submit_path} -m [Message]"
                )

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ["deepskyblue", "crimson"]
        labels = [
            f"""
            train loss {self.losses["Train"][-1]}
            val loss {self.losses["Valid"][-1]}
            """,
            f"""
            train dice score {self.dice_scores["Train"][-1]}
            val dice score {self.dice_scores["Valid"][-1]}
            """,
            f"""
            train jaccard score {self.jaccard_scores["Train"][-1]}
            val jaccard score {self.jaccard_scores["Valid"][-1]}
            """,
        ]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        for i, ax in enumerate(axes):
            ax.plot(data[i]["Valid"], c=colors[0], label="Valid")
            ax.plot(data[i]["Train"], c=colors[-1], label="Train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path, "history.png"))

    def load_pretrain_model(self, state_path: str):
        self.net.load_state_dict(torch.load(state_path)["state_dict"])
        # self.net.load_state_dict(torch.load(state_path))
        self.logger.info(f"Pretrain model loaded {state_path}")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(
            self.net.state_dict(), os.path.join(self.log_path, "last_epoch_model.pth")
        )

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_))) for key in logs_[i]]
        log_names = [
            key + log_names_[i] for i in list(range(len(logs_))) for key in logs_[i]
        ]
        pd.DataFrame(dict(zip(log_names, logs))).to_csv(
            os.path.join(self.log_path, "train_log.csv"), index=False
        )


def create_logger():
    i = 0
    while True:
        try:
            log_path = f"logs/{time.strftime('%Y_%m_%d')}_{i}"
            os.mkdir(log_path)
            break
        except FileExistsError:
            i += 1
            pass
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_path, "run.log")),  # Log to a file
            logging.StreamHandler(),  # Log to the console
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"=> Log path: {log_path}")
    return log_path, logger


if __name__ == "__main__":
    log_path, logger = create_logger()

    batch_size = 2
    accumulation_steps = 8
    workers = 2
    csv_file_path = "data/TrainValid_split.csv"
    csv_extra_file_path = "data/TrainValid_split_extra.csv"
    test_file_path = "data/sample_submission.csv"
    root_directory = "data/preprocess/232x176x50_v5"
    # root_directory = "data/preprocess/232x176x70_v1"  # unet3d

    logger.info(f"=> batch_size: {batch_size}")
    logger.info(f"=> accumulation_steps: {accumulation_steps}")
    logger.info(f"=> root_directory: {root_directory}")

    # train_loader, val_loader = get_loader(
    #     root_directory, batch_size=batch_size, train_csv=csv_file_path
    # )

    dataloaders = {
        # "Train": train_loader,
        # "Valid": val_loader,
        "Train": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=batch_size,
            num_workers=workers,
            phase="Train",
        ),
        "TrainExtra": build_dataloader(
            csv_extra_file_path,
            root_directory,
            batch_size=batch_size,
            num_workers=workers,
            phase="TrainExtra",
        ),
        "Valid": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=batch_size,
            num_workers=workers,
            phase="Valid",
        ),
        "ValidTest": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=batch_size,
            num_workers=workers,
            phase="ValidTest",
        ),
        "Test": build_dataloader(
            test_file_path,
            root_directory,
            batch_size=batch_size,
            num_workers=workers,
            phase="Test",
        ),
    }
    net = UNet3d(in_channels=1, n_classes=1, n_channels=32).to("cuda")
    # net = smp.Unet(
    #     encoder_name="efficientnet-b0",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=1,
    #     activation=None,
    # )

    # aux_params = dict(
    #     pooling="avg",  # one of 'avg', 'max'
    #     dropout=0.5,  # dropout ratio, default is None
    #     activation="sigmoid",  # activation function, default is None
    #     classes=1,  # define number of output labels
    # )
    # net = smp.Unet("resnet34", in_channels=1, classes=1, aux_params=aux_params)
    # net = UNETR(input_dim=1, output_dim=1)

    # net = Net().to("cuda")  # in_channels=1, out_channels=1
    logger.info(f"=> model: {net}")
    net = torch.nn.DataParallel(net).cuda()

    trainer = Trainer(
        net=net,
        dataloaders=dataloaders,
        criterion=BCEDiceLoss(),
        lr=1e-4,  # 2e-4, #
        accumulation_steps=accumulation_steps,
        batch_size=batch_size,
        fold=0,
        num_epochs=50,
        logger=logger,
        log_path=log_path,
    )
    trainer.load_pretrain_model(
        "/mnt/data/M11217002/AOCR_Tung_MIPL/logs/2023_12_31_8_unet3d_best_v2_bad/model_best.pth"
    )
    # trainer.load_pretrain_model(
    #     "/mnt/data/M11217002/AOCR_Tung_MIPL/logs/2024_01_21_11/best_model.pth"
    # )
    trainer.run()
    # trainer.test_submit(
    #     valid=True,
    #     threshold=0.7,
    #     filter_num_slice=3,
    #     do_connected_ones=True,
    #     gen_nii=True,
    # )