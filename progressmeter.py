import torch
import numpy as np

from monai.metrics import DiceMetric, ConfusionMatrixMetric, get_confusion_matrix


class Meter:
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.f1_scores: list = []

        self.TPs: list = []
        self.FPs: list = []
        self.TNs: list = []
        self.FNs: list = []

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.f1_metric = ConfusionMatrixMetric(
            metric_name="f1 score", reduction="mean_batch"
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs, targets = probs.detach().cpu(), targets.detach().cpu()
        self.dice_metric(probs, targets)
        self.f1_metric(probs, targets)

        # Calculate TP, FP, TN, FN
        predictions = probs >= self.threshold
        confusion = get_confusion_matrix(y_pred=predictions, y=targets)
        confusion = torch.sum(confusion, dim=0)

        tp, fp, tn, fn = (
            confusion[..., 0],
            confusion[..., 1],
            confusion[..., 2],
            confusion[..., 3],
        )

        self.TPs.append(tp.item())
        self.FPs.append(fp.item())
        self.TNs.append(tn.item())
        self.FNs.append(fn.item())

    def get_last_metrics(self) -> np.ndarray:
        dice = self.dice_metric.aggregate().item()
        f1 = self.f1_metric.aggregate()[0].item()
        tp = self.TPs[-1]
        fp = self.FPs[-1]
        tn = self.TNs[-1]
        fn = self.FNs[-1]
        return tp, fp, tn, fn, dice, f1

    def get_metrics(self) -> np.ndarray:
        dice = self.dice_metric.aggregate().item()
        f1 = self.f1_metric.aggregate()[0].item()

        self.dice_metric.reset()
        self.f1_metric.reset()
        return dice, f1
