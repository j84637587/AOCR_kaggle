import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logging import Logger
from progressmeter import Meter

import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

from postprocessing import postprocessing


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
        dataloaders: dict[str, torch.utils.data.DataLoader],
        criterion: nn.Module,
        lr: float,
        accumulation_steps: int,
        batch_size: int,
        num_epochs: int,
        logger: Logger,
        log_path: str,
        log_freq: int = 10,
        display_plot: bool = True,
    ):
        """Initialization."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.log_path = log_path
        self.logger.info(f"=> Log Path: {self.log_path}")
        self.logger.info(f"=> Device: {self.device}")
        self.log_freq = log_freq
        self.display_plot = display_plot
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, verbose=True
        )
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["Train", "Valid"]
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.f1_scores = {phase: [] for phase in self.phases}
        self.aurocs_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        self.net.train() if phase == "Train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()

        # Running Batch
        start = time.time()
        for itr, (id, images, targets) in enumerate(dataloader):
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps

            if phase == "Train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())  # 更新 dice, iou
            if itr % self.log_freq == 0:
                self.logger.info(
                    f"{phase} epoch: {epoch} | batch: {itr}/{total_batches} | "
                    f"time: {(time.time() - start):.1f}s"
                )

                last_dice, last_iou, last_f1, last_auroc = meter.get_last_metrics()
                self.logger.info(
                    (
                        f"lr: {self.optimizer.param_groups[0]['lr']:.4f} | "
                        f"dice: {last_dice:.4f} ({np.mean(self.dice_scores[phase])}) | "
                        f"iou: {last_iou:.4f} ({np.mean(self.jaccard_scores[phase])}) | "
                        f"f1: {last_f1:.4f} ({np.mean(self.f1_scores[phase])}) | "
                        f"auroc: {last_auroc:.4f}({np.mean(self.aurocs_scores[phase])}) "
                    )
                )

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou, epoch_f1, epoch_auroc = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.f1_scores[phase].append(epoch_f1)
        self.aurocs_scores[phase].append(epoch_auroc)

        return epoch_loss

    def run(self, validate: bool = False):
        for epoch in range(self.num_epochs):
            # 訓練
            if not validate:
                self._do_epoch(epoch, "Train")

            # 驗證
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "Valid")
                self.scheduler.step(val_loss)

            if not validate:
                # 製圖
                if self.display_plot:
                    self._plot_train_history()

                # 以 loss 保存最佳模型
                if val_loss < self.best_loss:
                    self._save_weights(epoch, val_loss)
                    self.best_loss = val_loss

                self._save_train_history()
            else:
                dice_mean, iou_mean, f1_mean, auroc_mean = self.get_mean_metrics()
                self.logger.info(
                    f"\ndice:\t{dice_mean:.2f}\n"
                    f"iou:\t{iou_mean:.2f}\n"
                    f"f1:\t{f1_mean:.2f}\n"
                    f"auroc:\t{auroc_mean:.2f}"
                )

    def get_mean_metrics(self):
        dice_mean = (
            np.mean(self.dice_scores["Valid"])
            if len(self.dice_scores["Valid"]) > 0
            else 0.0
        )
        iou_mean = (
            np.mean(self.jaccard_scores["Valid"])
            if len(self.jaccard_scores["Valid"]) > 0
            else 0.0
        )
        f1_mean = (
            np.mean(self.f1_scores["Valid"])
            if len(self.f1_scores["Valid"]) > 0
            else 0.0
        )
        auroc_mean = (
            np.mean(self.aurocs_scores["Valid"])
            if len(self.aurocs_scores["Valid"]) > 0
            else 0.0
        )

        return dice_mean, iou_mean, f1_mean, auroc_mean

    def test_submit(self, threshold=0.7):
        self.net.eval()
        with torch.no_grad():
            submit = pd.DataFrame(columns=["id", "label"])
            for _, (ids, images, num_slices, start, end) in tqdm(
                enumerate(self.dataloaders["Test"])
            ):
                outputs = self.net(images)
                outputs = outputs.cpu().numpy()
                binary_predictions = (outputs > threshold) * 1

                for id_itr, id in enumerate(ids):
                    num_slice = num_slices[id_itr]
                    pred = binary_predictions[id_itr]

                    post_processed = postprocessing(
                        pred, area_thresholding=41, connectivity=8
                    )

                    pred = np.any(post_processed, axis=(0, 1, 2))
                    s = start[id_itr]
                    e = end[id_itr]

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
            self.logger.info("To submit the result, please run:")
            self.logger.info(
                f"kaggle competitions submit -c aocr2024 -f {submit_path} -m [Message]"
            )
            # "First submit with model UNETR_Net / threshold: 0.5 / date: 12_26_23"
            # "2ed submit with model UNETR_Net / threshold: 0.7 / date: 12_26_23"

    def _save_weights(self, epoch: int, val_loss):
        filename = os.path.join(self.log_path, "model_best.pth.tar")
        self.logger.info(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
        torch.save(
            {
                "epoch": epoch + 1,
                "arch": "Unet3d",
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loss": val_loss,
            },
            filename,
        )

    def _plot_train_history(self):
        data = [
            self.losses,
            self.dice_scores,
            self.jaccard_scores,
            self.f1_scores,
            self.aurocs_scores,
        ]
        colors = ["deepskyblue", "crimson"]
        labels = [
            f"""
            train loss {self.losses['Train'][-1]:.4f}
            val loss {self.losses['Valid'][-1]:.4f}
            """,
            f"""
            train dice score {self.dice_scores['Train'][-1]:.4f}
            val dice score {self.dice_scores['Valid'][-1]:.4f}
            """,
            f"""
            train jaccard score {self.jaccard_scores['Train'][-1]:.4f}
            val jaccard score {self.jaccard_scores['Valid'][-1]:.4f}
            """,
            f"""
            train f1 score {self.f1_scores['Train'][-1]:.4f}
            val f1 score {self.f1_scores['Valid'][-1]:.4f}
            """,
            f"""
            train auroc score {self.aurocs_scores['Train'][-1]:.4f}
            val auroc score {self.aurocs_scores['Valid'][-1]:.4f}
            """,
        ]

        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        for i, ax in enumerate(axes.flat):
            if i >= len(data):
                ax.axis("off")
                break

            ax.plot(data[i]["Valid"], c=colors[0], label="Valid")
            ax.plot(data[i]["Train"], c=colors[-1], label="Train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path, "history.png"))

        history = {
            "loss": self.losses,
            "dice": self.dice_scores,
            "jaccard": self.jaccard_scores,
            "f1": self.f1_scores,
            "auroc": self.aurocs_scores,
        }
        json.dump(str(history), open(os.path.join(self.log_path, "history.json"), "w"))

    def load_pretrained_model(self, state_path: str):
        checkpoint_path = f"{state_path}/model_best.pth.tar"
        self.logger.info(f"=> loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.logger.info("=> pretrained model loaded successfully")
        self.logger.info(f"=> best train/valid loss: {checkpoint['loss']}")
        self.logger.info(f"=> have trained for {checkpoint['epoch']} epochs")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(
            self.net.state_dict(),
            os.path.join(self.log_path, "last_epoch_model.pth"),
        )

        logs_ = [
            self.losses,
            self.dice_scores,
            self.jaccard_scores,
            self.f1_scores,
            self.aurocs_scores,
        ]
        log_names_ = ["_loss", "_dice", "_jaccard", "_f1", "_auroc"]
        logs = [logs_[i][key] for i in list(range(len(logs_))) for key in logs_[i]]
        log_names = [
            key + log_names_[i] for i in list(range(len(logs_))) for key in logs_[i]
        ]
        pd.DataFrame(dict(zip(log_names, logs))).to_csv(
            os.path.join(self.log_path, "train_log.csv"), index=False
        )
